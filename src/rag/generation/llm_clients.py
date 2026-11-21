"""LLM provider clients behind ``LLMClient`` (ADR 0005, HLD 010).

Both clients keep the network call behind an injectable seam (an SDK client object or an
HTTP ``post_json`` function), so unit tests never call a real API.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from functools import cached_property
from typing import Any, Final, Protocol

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
        model: str = "claude-opus-5-5",
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


PostJson = Callable[[str, dict[str, Any], float], dict[str, Any]]


def urllib_post_json(
    url: str,
    payload: dict[str, Any],
    timeout_s: float,
    headers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """POST JSON with the standard library (no extra dependency).

    Header *values* (which may hold a key) never appear in error messages. An HTTP error
    includes the start of the server's reply, which says *why* (e.g. an unsupported
    ``response_format``).
    """
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
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:300].decode("utf-8", "replace")
        raise GenerationError(f"request to {url} failed: HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise GenerationError(f"request to {url} failed: {exc}") from exc


def urllib_get_json(url: str, timeout_s: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            body: dict[str, Any] = json.loads(response.read())
            return body
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise GenerationError(f"request to {url} failed: {exc}") from exc


GetJson = Callable[[str, float], dict[str, Any]]


def check_ollama_model(
    model: str,
    base_url: str = "http://localhost:11434",
    timeout_s: float = 5.0,
    get_json: GetJson = urllib_get_json,
) -> None:
    """Fail fast, with instructions, if Ollama is not running or the model is not pulled.

    Without this, a planner whose every call fails silently falls back to the rules and
    the run looks like a real "no improvement" result.
    """
    try:
        tags = get_json(base_url.rstrip("/") + "/api/tags", timeout_s)
    except GenerationError as exc:
        raise GenerationError(
            f"Ollama is not reachable at {base_url} ({exc}). Install it from "
            "https://ollama.com/download and open the app (it serves on port 11434)."
        ) from exc
    names = {str(m.get("name", "")) for m in tags.get("models", []) if isinstance(m, dict)}
    wanted = model if ":" in model else f"{model}:latest"
    if wanted not in names:
        raise GenerationError(f"Ollama model {model!r} is not pulled: run `ollama pull {model}`")


class OllamaLLMClient(LLMClient):
    """A local model served by Ollama (``/api/chat``), with JSON-schema constrained output."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        timeout_s: float = 300.0,
        post_json: PostJson = urllib_post_json,
        max_output_tokens: int | None = None,
        context_tokens: int | None = None,
    ) -> None:
        """``max_output_tokens`` (Ollama ``num_predict``) bounds a runaway generation.
        ``context_tokens`` (``num_ctx``) sets the context window: Ollama's default is small
        and *silently truncates* longer prompts from the front."""
        if max_output_tokens is not None and max_output_tokens < 1:
            raise ValueError("max_output_tokens must be >= 1")
        if context_tokens is not None and context_tokens < 1:
            raise ValueError("context_tokens must be >= 1")
        self._context_tokens = context_tokens
        self._model = model
        self._url = base_url.rstrip("/") + "/api/chat"
        self._timeout_s = timeout_s
        self._post_json = post_json
        self._max_output_tokens = max_output_tokens

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
        if self._max_output_tokens is not None:
            payload["options"]["num_predict"] = self._max_output_tokens
        if self._context_tokens is not None:
            payload["options"]["num_ctx"] = self._context_tokens
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


class PostJsonWithHeaders(Protocol):
    def __call__(
        self,
        url: str,
        payload: dict[str, Any],
        timeout_s: float,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]: ...


class OpenAICompatibleLLMClient(LLMClient):
    """Any OpenAI-compatible ``/chat/completions`` endpoint: an LLM gateway (e.g. Bifrost,
    LiteLLM), vLLM, or Ollama's ``/v1`` (HLD 018).

    Lessons from an earlier gateway attempt (PR #14, ``experiments/analysis/2026-11-19-…``):

    - Structured output goes through ``response_format`` with our JSON schema, **never a
      forced tool call**. Translating tool calls through a gateway lost 78% of answers.
    - The URL and key come from the environment, never from code or configs, so no
      company-internal address is ever committed.
    - The answer is read from ``message.content``. A reply with only ``tool_calls``, or a
      ``refusal``, is reported as such and never parsed as an answer. The integrity checks
      (HLD 010 §2.1a) catch any remaining mangling.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "",
        key_header: str = "Authorization",
        timeout_s: float = 300.0,
        max_tokens: int = 2048,
        temperature: float | None = 0.0,
        use_response_format: bool = True,
        post_json: PostJsonWithHeaders = urllib_post_json,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        self._model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = _auth_headers(api_key, key_header)
        self._timeout_s = timeout_s
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._use_response_format = use_response_format
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
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        if prompt.response_schema is not None:
            if self._use_response_format:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "grounded_answer",
                        "schema": prompt.response_schema,
                        "strict": True,
                    },
                }
            else:
                schema = json.dumps(prompt.response_schema, ensure_ascii=False)
                payload["messages"][0]["content"] += (
                    "\n\nReturn only one valid JSON object matching this schema. "
                    "Do not wrap it in Markdown. Schema: "
                    f"{schema}"
                )
        return payload

    def complete(self, prompt: Prompt) -> Completion:
        body = self._post_json(
            self._url, self.request_payload(prompt), self._timeout_s, headers=self._headers
        )
        try:
            choice = body["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GenerationError(f"unexpected response: {str(body)[:200]}") from exc
        usage = body.get("usage") or {}
        common = {
            "model_id": str(body.get("model") or self._model),
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
            "stop_reason": str(choice.get("finish_reason") or "stop"),
        }
        if message.get("refusal"):
            return Completion(text="", refused=True, **common)
        content = message.get("content")
        if content is None:
            kind = "tool_calls instead of content" if message.get("tool_calls") else "no content"
            fields = ",".join(sorted(str(key) for key in message)) or "none"
            finish_reason = str(choice.get("finish_reason") or "unknown")
            refusal = "present" if message.get("refusal") else "absent"
            raise GenerationError(
                f"gateway returned {kind} (finish_reason={finish_reason}, "
                f"message_fields={fields}, refusal={refusal})"
            )
        return Completion(text=str(content), **common)


def _auth_headers(api_key: str, key_header: str) -> dict[str, str]:
    """``Authorization`` gets the standard ``Bearer`` prefix; any other header (e.g. a
    gateway's virtual-key header) carries the key as-is."""
    if not api_key:
        return {}
    if key_header.lower() == "authorization":
        return {"Authorization": f"Bearer {api_key}"}
    return {key_header: api_key}
