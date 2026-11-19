"""LLM provider clients behind ``LLMClient`` (ADR 0005, HLD 010).

Both clients keep the network call behind an injectable seam (an SDK client object or an
HTTP ``post_json`` function), so unit tests never call a real API.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from functools import cached_property
from typing import Any, Final

from rag.errors import GenerationError
from rag.generation.base import Completion, LLMClient, Prompt

REFUSAL_FALLBACK_BETA: Final = "server-side-fallback-2026-07-01"
_REFUSAL: Final = "refusal"


class AnthropicLLMClient(LLMClient):
    """Claude via the official Anthropic Python SDK.

    - Thinking is always on for current Opus models; depth is set with ``effort``.
    - Structured output (``output_config.format``) when the prompt carries a schema.
    - Server-side refusal fallbacks (``fallbacks="default"``) unless disabled; a final
      ``stop_reason == "refusal"`` is still reported as ``Completion.refused``.
    - Credentials are resolved by the SDK from the environment (``ANTHROPIC_API_KEY``).
    """

    def __init__(
        self,
        model: str = "claude-sonnet-5-5",
        max_tokens: int = 16000,
        effort: str = "medium",
        refusal_fallbacks: bool = True,
        client: Any | None = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        self._model = model
        self._max_tokens = max_tokens
        self._effort = effort
        self._refusal_fallbacks = refusal_fallbacks
        self._injected_client = client

    @property
    def model_id(self) -> str:
        return self._model

    @cached_property
    def _client(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        import anthropic

        return anthropic.Anthropic()

    def request_params(self, prompt: Prompt) -> dict[str, Any]:
        """The exact request body (minus credentials); public so it can be inspected."""
        output_config: dict[str, Any] = {"effort": self._effort}
        if prompt.response_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": prompt.response_schema}
        params: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": prompt.system,
            "messages": [{"role": "user", "content": prompt.user}],
            "output_config": output_config,
        }
        if self._refusal_fallbacks:
            params["betas"] = [REFUSAL_FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params

    def complete(self, prompt: Prompt) -> Completion:
        import anthropic

        try:
            response = self._client.beta.messages.create(**self.request_params(prompt))
        except anthropic.APIError as exc:
            raise GenerationError(f"Anthropic API error: {exc}") from exc

        usage = response.usage
        refused = response.stop_reason == _REFUSAL
        text = (
            ""
            if refused
            else next((block.text for block in response.content if block.type == "text"), "")
        )
        return Completion(
            text=text,
            model_id=str(response.model),
            input_tokens=int(usage.input_tokens),
            output_tokens=int(usage.output_tokens),
            stop_reason=str(response.stop_reason),
            refused=refused,
        )


PostJson = Callable[..., dict[str, Any]]


def urllib_post_json(
    url: str,
    payload: dict[str, Any],
    timeout_s: float,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """POST JSON with the standard library; no HTTP dependency is required."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body: dict[str, Any] = json.loads(response.read())
            return body
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise GenerationError(f"LLM request to {url} failed: {exc}") from exc


class OllamaLLMClient(LLMClient):
    """A local model served by Ollama (``/api/chat``), with JSON-schema constrained output."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        timeout_s: float = 300.0,
        post_json: PostJson = urllib_post_json,
    ) -> None:
        self._model = model
        self._url = base_url.rstrip("/") + "/api/chat"
        self._timeout_s = timeout_s
        self._post_json = post_json

    @property
    def model_id(self) -> str:
        return self._model

    def request_payload(self, prompt: Prompt) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "stream": False,
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            "options": {"temperature": 0},
        }
        if prompt.response_schema is not None:
            payload["format"] = prompt.response_schema
        return payload

    def complete(self, prompt: Prompt) -> Completion:
        body = self._post_json(self._url, self.request_payload(prompt), self._timeout_s)
        try:
            text = str(body["message"]["content"])
        except (KeyError, TypeError) as exc:
            raise GenerationError(f"unexpected Ollama response: {body!r:.200}") from exc
        return Completion(
            text=text,
            model_id=str(body.get("model", self._model)),
            input_tokens=int(body.get("prompt_eval_count", 0)),
            output_tokens=int(body.get("eval_count", 0)),
            stop_reason=str(body.get("done_reason", "stop")),
        )


class BifrostLLMClient(LLMClient):
    """Claude or another model through Bifrost's OpenAI-compatible API."""

    def __init__(
        self,
        model: str,
        base_url: str = "https://bifrost-llm-proxy-alb-0.cmd.hotstar-prod.com/v1",
        api_key_env: str = "BIFROST_API_KEY",
        max_tokens: int = 16000,
        timeout_s: float = 300.0,
        post_json: PostJson = urllib_post_json,
    ) -> None:
        self._model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._api_key_env = api_key_env
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        self._max_tokens = max_tokens
        self._timeout_s = timeout_s
        self._post_json = post_json

    @property
    def model_id(self) -> str:
        return self._model

    def request_payload(self, prompt: Prompt) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            "max_tokens": self._max_tokens,
        }
        if prompt.response_schema is not None:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": "grounded_answer",
                        "description": "Return the grounded answer in the required schema.",
                        "parameters": prompt.response_schema,
                        "strict": True,
                    },
                },
            ]
            payload["tool_choice"] = {
                "type": "function",
                "function": {"name": "grounded_answer"},
            }
        return payload

    def complete(self, prompt: Prompt) -> Completion:
        api_key = os.environ.get(self._api_key_env)
        if not api_key:
            raise GenerationError(f"Bifrost credential missing: set {self._api_key_env}")
        try:
            body = self._post_json(
                self._url,
                self.request_payload(prompt),
                self._timeout_s,
                headers={"x-bf-vk": api_key},
            )
        except urllib.error.HTTPError as exc:
            raise GenerationError(f"Bifrost request failed with HTTP {exc.code}") from exc
        try:
            choice = body["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GenerationError(f"unexpected Bifrost response: {body!r:.200}") from exc
        tool_calls = message.get("tool_calls", [])
        grounded_call = next(
            (
                call
                for call in tool_calls
                if call.get("function", {}).get("name") == "grounded_answer"
            ),
            None,
        )
        if grounded_call is not None:
            arguments = grounded_call["function"].get("arguments", "")
            text = arguments if isinstance(arguments, str) else json.dumps(arguments)
        elif message.get("content") is not None:
            text = str(message["content"])
        else:
            raise GenerationError(f"unexpected Bifrost response: {body!r:.200}")
        usage = body.get("usage", {})
        return Completion(
            text=text,
            model_id=str(body.get("model", self._model)),
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            stop_reason=str(choice.get("finish_reason", "stop")),
        )
