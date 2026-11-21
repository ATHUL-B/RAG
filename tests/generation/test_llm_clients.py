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


class TestOpenAICompatible:
    @staticmethod
    def _client(reply: dict[str, Any], **kw: Any) -> tuple[Any, list[dict[str, Any]]]:
        calls: list[dict[str, Any]] = []

        def post(
            url: str, payload: dict[str, Any], timeout_s: float, headers: Any = None
        ) -> dict[str, Any]:
            calls.append({"url": url, "payload": payload, "headers": headers, "timeout": timeout_s})
            return reply

        from rag.generation.llm_clients import OpenAICompatibleLLMClient

        client = OpenAICompatibleLLMClient(
            model="gw-model", base_url="https://gw.test/v1/", post_json=post, **kw
        )
        return client, calls

    @staticmethod
    def _reply(
        content: Any = '{"answerable": true, "working": "", "answer": "x"}', **message: Any
    ) -> dict[str, Any]:
        return {
            "model": "provider/gw-model",
            "choices": [{"message": {"content": content, **message}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1500, "completion_tokens": 200},
        }

    def test_uses_response_format_json_schema_never_a_tool_call(self) -> None:
        client, calls = self._client(self._reply(), api_key="k", max_tokens=777)
        completion = client.complete(PROMPT)
        (call,) = calls
        assert call["url"] == "https://gw.test/v1/chat/completions"
        payload = call["payload"]
        assert payload["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "grounded_answer", "schema": SCHEMA, "strict": True},
        }
        assert "tools" not in payload and "tool_choice" not in payload
        assert payload["temperature"] == 0 and payload["max_tokens"] == 777
        assert completion.text.startswith('{"answerable"')
        assert (completion.input_tokens, completion.output_tokens) == (1500, 200)
        assert completion.model_id == "provider/gw-model"

    def test_temperature_can_be_omitted_for_models_that_reject_it(self) -> None:
        client, calls = self._client(self._reply(), temperature=None)
        client.complete(PROMPT)
        (call,) = calls
        assert "temperature" not in call["payload"]

    def test_can_request_json_with_schema_instructions_not_response_format(self) -> None:
        client, calls = self._client(self._reply(), use_response_format=False)
        client.complete(PROMPT)
        (call,) = calls
        assert "response_format" not in call["payload"]
        system = call["payload"]["messages"][0]["content"]
        assert "valid JSON object matching this schema" in system
        assert '"additionalProperties": false' in system

    def test_no_content_error_reports_response_diagnostics(self) -> None:
        reply = {
            "choices": [
                {"message": {"role": "assistant", "content": None}, "finish_reason": "stop"}
            ]
        }
        client, _ = self._client(reply)
        with pytest.raises(
            GenerationError,
            match=r"finish_reason=stop, message_fields=content,role, refusal=absent",
        ):
            client.complete(PROMPT)

    @staticmethod
    def _sequence(*replies: dict[str, Any], **kw: Any) -> tuple[Any, list[dict[str, Any]]]:
        calls: list[dict[str, Any]] = []
        queue = list(replies)

        def post(
            url: str, payload: dict[str, Any], timeout_s: float, headers: Any = None
        ) -> dict[str, Any]:
            calls.append(payload)
            return queue.pop(0)

        from rag.generation.llm_clients import OpenAICompatibleLLMClient

        client = OpenAICompatibleLLMClient(
            model="gw-model", base_url="https://gw.test/v1", post_json=post, **kw
        )
        return client, calls

    def _filtered(self, content: Any = '{"answerable": true, "work') -> dict[str, Any]:
        reply = self._reply(content)
        reply["choices"][0]["finish_reason"] = "content_filter"
        reply["usage"] = {"prompt_tokens": 1500, "completion_tokens": 30}
        return reply

    def test_content_filter_is_retried_and_all_attempts_are_billed(self) -> None:
        client, calls = self._sequence(
            self._filtered(), self._filtered(None), self._reply(), filter_retries=2
        )
        completion = client.complete(PROMPT)
        assert len(calls) == 3
        assert completion.stop_reason == "stop"
        assert completion.text.startswith('{"answerable": true, "working"')
        assert (completion.input_tokens, completion.output_tokens) == (4500, 260)

    def test_filtered_after_all_retries_returns_the_filtered_reply(self) -> None:
        client, calls = self._sequence(self._filtered(), self._filtered(None), filter_retries=1)
        completion = client.complete(PROMPT)
        assert len(calls) == 2
        assert completion.stop_reason == "content_filter" and completion.text == ""
        assert completion.output_tokens == 60

    def test_no_retry_by_default_and_none_on_success(self) -> None:
        client, calls = self._sequence(self._filtered())
        assert client.complete(PROMPT).stop_reason == "content_filter"
        client, calls = self._sequence(self._reply(), filter_retries=2)
        client.complete(PROMPT)
        assert len(calls) == 1
        with pytest.raises(ValueError, match="filter_retries"):
            self._sequence(filter_retries=-1)

    @pytest.mark.parametrize(
        ("key", "header", "expected"),
        [
            ("k", "Authorization", {"Authorization": "Bearer k"}),
            ("k", "x-bf-vk", {"x-bf-vk": "k"}),
            ("", "Authorization", {}),
        ],
    )
    def test_key_goes_in_the_configured_header(
        self, key: str, header: str, expected: dict[str, str]
    ) -> None:
        client, calls = self._client(self._reply(), api_key=key, key_header=header)
        client.complete(PROMPT)
        assert calls[0]["headers"] == expected

    def test_refusal_and_tool_calls_are_never_parsed_as_answers(self) -> None:
        refused, _ = self._client(self._reply(content=None, refusal="I can't help with that"))
        assert refused.complete(PROMPT).refused
        tools, _ = self._client(self._reply(content=None, tool_calls=[{"id": "t"}]))
        with pytest.raises(GenerationError, match="tool_calls instead of content"):
            tools.complete(PROMPT)
        broken, _ = self._client({"error": "bad model"})
        with pytest.raises(GenerationError, match="unexpected response"):
            broken.complete(PROMPT)

    def test_rejects_bad_settings(self) -> None:
        from rag.generation.llm_clients import OpenAICompatibleLLMClient

        with pytest.raises(ValueError, match="base_url"):
            OpenAICompatibleLLMClient(model="m", base_url="")
        with pytest.raises(ValueError, match="max_tokens"):
            OpenAICompatibleLLMClient(model="m", base_url="u", max_tokens=0)


def test_http_errors_show_the_server_reason_but_never_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import io
    import urllib.error
    import urllib.request

    from rag.generation.llm_clients import urllib_post_json

    def fail(request: Any, timeout: float) -> Any:
        raise urllib.error.HTTPError(
            "https://gw.test",
            400,
            "Bad Request",
            {},  # type: ignore[arg-type]
            io.BytesIO(b'{"error": "response_format not supported"}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    with pytest.raises(GenerationError) as info:
        urllib_post_json("https://gw.test", {}, 1.0, headers={"x-bf-vk": "SECRET-KEY"})
    assert "HTTP 400" in str(info.value) and "response_format not supported" in str(info.value)
    assert "SECRET-KEY" not in str(info.value)
