import re
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest

from rag.errors import GenerationError
from rag.generation.base import Prompt
from rag.generation.llm_clients import (
    REFUSAL_FALLBACK_BETA,
    AnthropicLLMClient,
    OllamaLLMClient,
    check_ollama_model,
)

SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}
PROMPT = Prompt(system="sys", user="question", chunk_ids=("c1",), response_schema=SCHEMA)


class FakeMessages:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response, self.error = response, error
        self.calls: list[dict[str, Any]] = []

    def create(self, **params: Any) -> Any:
        self.calls.append(params)
        if self.error:
            raise self.error
        return self.response


def _fake_client(messages: FakeMessages) -> Any:
    return SimpleNamespace(beta=SimpleNamespace(messages=messages))


def _response(
    stop_reason: str = "end_turn", text: str = '{"answerable": true, "answer": "x"}'
) -> Any:
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        usage=SimpleNamespace(input_tokens=1200, output_tokens=340),
        model="claude-opus-5-5",
        stop_reason=stop_reason,
    )


class TestAnthropic:
    def test_request_uses_effort_schema_and_refusal_fallbacks(self) -> None:
        messages = FakeMessages(_response())
        completion = AnthropicLLMClient(effort="high", client=_fake_client(messages)).complete(
            PROMPT
        )

        (params,) = messages.calls
        assert params["model"] == "claude-opus-5-5"
        assert params["system"] == "sys"
        assert params["messages"] == [{"role": "user", "content": "question"}]
        assert params["output_config"] == {
            "effort": "high",
            "format": {"type": "json_schema", "schema": SCHEMA},
        }
        assert params["betas"] == [REFUSAL_FALLBACK_BETA]
        assert params["fallbacks"] == "default"
        assert completion.text == '{"answerable": true, "answer": "x"}'
        assert (completion.input_tokens, completion.output_tokens) == (1200, 340)
        assert completion.model_id == "claude-opus-5-5"
        assert not completion.refused

    def test_fallbacks_can_be_disabled_and_schema_is_optional(self) -> None:
        client = AnthropicLLMClient(refusal_fallbacks=False)
        params = client.request_params(PROMPT.model_copy(update={"response_schema": None}))
        assert "betas" not in params and "fallbacks" not in params
        assert params["output_config"] == {"effort": "medium"}

    def test_refusal_is_reported_without_reading_content(self) -> None:
        messages = FakeMessages(_response(stop_reason="refusal", text="partial"))
        completion = AnthropicLLMClient(client=_fake_client(messages)).complete(PROMPT)
        assert completion.refused
        assert completion.text == ""
        assert completion.stop_reason == "refusal"

    def test_api_errors_become_generation_errors(self) -> None:
        error = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://x.test"))
        client = AnthropicLLMClient(client=_fake_client(FakeMessages(error=error)))
        with pytest.raises(GenerationError, match="Anthropic API error"):
            client.complete(PROMPT)

    def test_rejects_invalid_max_tokens(self) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            AnthropicLLMClient(max_tokens=0)


class TestOllama:
    def test_posts_chat_request_with_schema_and_reads_usage(self) -> None:
        calls: list[tuple[str, dict[str, Any], float]] = []

        def post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
            calls.append((url, payload, timeout))
            return {
                "model": "llama3.1:8b",
                "message": {
                    "role": "assistant",
                    "content": '{"answerable": false, "answer": "n/a"}',
                },
                "prompt_eval_count": 900,
                "eval_count": 40,
                "done_reason": "stop",
            }

        client = OllamaLLMClient(base_url="http://host:11434/", timeout_s=9, post_json=post)
        completion = client.complete(PROMPT)

        url, payload, timeout = calls[0]
        assert url == "http://host:11434/api/chat"
        assert timeout == 9
        assert payload["format"] == SCHEMA
        assert payload["stream"] is False
        assert [m["role"] for m in payload["messages"]] == ["system", "user"]
        assert (completion.input_tokens, completion.output_tokens) == (900, 40)
        assert client.model_id == "llama3.1:8b"

    def test_unexpected_response_is_a_generation_error(self) -> None:
        client = OllamaLLMClient(post_json=lambda url, payload, timeout: {"error": "no model"})
        with pytest.raises(GenerationError, match="unexpected Ollama response"):
            client.complete(PROMPT)


class TestOllamaPreflight:
    def test_passes_when_the_model_is_pulled(self) -> None:
        tags = {"models": [{"name": "llama3.1:8b"}, {"name": "qwen2.5:latest"}]}
        check_ollama_model("llama3.1:8b", get_json=lambda url, t: tags)
        check_ollama_model("qwen2.5", get_json=lambda url, t: tags)  # implicit :latest

    def test_missing_model_says_how_to_pull_it(self) -> None:
        with pytest.raises(GenerationError, match=re.escape("ollama pull llama3.1:8b")):
            check_ollama_model("llama3.1:8b", get_json=lambda url, t: {"models": []})

    def test_unreachable_server_says_how_to_install(self) -> None:
        def down(url: str, timeout: float) -> dict[str, Any]:
            raise GenerationError("connection refused")

        with pytest.raises(GenerationError, match=re.escape("ollama.com/download")):
            check_ollama_model("llama3.1:8b", base_url="http://localhost:11434", get_json=down)


def test_ollama_output_cap_is_sent_as_num_predict() -> None:
    capped = OllamaLLMClient(max_output_tokens=300).request_payload(PROMPT)
    assert capped["options"] == {"temperature": 0, "num_predict": 300}
    assert "num_predict" not in OllamaLLMClient().request_payload(PROMPT)["options"]
    with pytest.raises(ValueError, match="max_output_tokens"):
        OllamaLLMClient(max_output_tokens=0)


def test_ollama_context_window_is_sent_as_num_ctx() -> None:
    payload = OllamaLLMClient(context_tokens=4096).request_payload(PROMPT)
    assert payload["options"]["num_ctx"] == 4096
    with pytest.raises(ValueError, match="context_tokens"):
        OllamaLLMClient(context_tokens=0)
