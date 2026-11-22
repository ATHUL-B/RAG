import pytest

from rag.config import GenerationConfig, LLMProvider
from rag.errors import ConfigError
from rag.generation.factory import build_generator, build_llm_client
from rag.generation.generator import GroundedAnswerGenerator
from rag.generation.llm_clients import AnthropicLLMClient, OllamaLLMClient


def test_anthropic_is_the_default_provider() -> None:
    client = build_llm_client(GenerationConfig())
    assert isinstance(client, AnthropicLLMClient)
    assert client.model_id == "claude-opus-5-5"  # no API call: the SDK client is lazy


def test_ollama_provider() -> None:
    config = GenerationConfig(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
    client = build_llm_client(config)
    assert isinstance(client, OllamaLLMClient)
    assert client.model_id == "llama3.1:8b"


def test_build_generator() -> None:
    assert isinstance(build_generator(GenerationConfig()), GroundedAnswerGenerator)


def test_ollama_client_gets_explicit_context_window_and_output_cap() -> None:
    from rag.generation.base import Prompt

    config = GenerationConfig(
        provider=LLMProvider.OLLAMA, max_tokens=1024, ollama_context_tokens=8192
    )
    payload = build_llm_client(config).request_payload(  # type: ignore[attr-defined]
        Prompt(system="s", user="u", chunk_ids=())
    )
    assert payload["options"] == {"temperature": 0, "num_predict": 1024, "num_ctx": 8192}


def test_gateway_reads_url_and_key_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag.generation.llm_clients import OpenAICompatibleLLMClient

    config = GenerationConfig(provider=LLMProvider.OPENAI_COMPATIBLE, model_id="gw-model")
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    with pytest.raises(ConfigError, match="set LLM_GATEWAY_URL"):
        build_llm_client(config)
    monkeypatch.setenv("LLM_GATEWAY_URL", "https://gw.test/v1")
    monkeypatch.setenv("LLM_GATEWAY_KEY", "k")
    assert isinstance(build_llm_client(config), OpenAICompatibleLLMClient)


def test_gateway_can_omit_unsupported_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag.generation.base import Prompt

    monkeypatch.setenv("LLM_GATEWAY_URL", "https://gw.test/v1")
    config = GenerationConfig(
        provider=LLMProvider.OPENAI_COMPATIBLE,
        model_id="gw-model",
        gateway_temperature=None,
    )
    payload = build_llm_client(config).request_payload(  # type: ignore[attr-defined]
        Prompt(system="s", user="u", chunk_ids=())
    )
    assert "temperature" not in payload


def test_gateway_can_request_json_without_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rag.generation.base import Prompt

    monkeypatch.setenv("LLM_GATEWAY_URL", "https://gw.test/v1")
    config = GenerationConfig(
        provider=LLMProvider.OPENAI_COMPATIBLE,
        model_id="gw-model",
        gateway_use_response_format=False,
    )
    payload = build_llm_client(config).request_payload(  # type: ignore[attr-defined]
        Prompt(system="s", user="u", chunk_ids=(), response_schema={"type": "object"})
    )
    assert "response_format" not in payload
    assert 'Schema: {"type": "object"}' in payload["messages"][0]["content"]


def test_gateway_placeholder_model_must_be_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_GATEWAY_URL", "https://gw.test/v1")
    config = GenerationConfig(provider=LLMProvider.OPENAI_COMPATIBLE, model_id="<gateway-model-id>")
    with pytest.raises(ConfigError, match=r"set generation\.model_id"):
        build_llm_client(config)


def test_prompt_version_selects_the_system_prompt() -> None:
    from rag.config import GenerationConfig, LLMProvider
    from rag.domain import Query
    from rag.generation.context import SYSTEM_PROMPT, SYSTEM_PROMPT_V2
    from rag.generation.factory import build_generator

    for version, expected in (("v1", SYSTEM_PROMPT), ("v2", SYSTEM_PROMPT_V2)):
        config = GenerationConfig(provider=LLMProvider.OLLAMA, model_id="m", prompt_version=version)
        generator = build_generator(config, llm=object())  # type: ignore[arg-type]
        prompt = generator._assembler.assemble(Query(text="q"), [])
        assert prompt.system == expected


def test_prompt_v2_keeps_v1_rules_and_adds_the_near_miss_rules() -> None:
    from rag.generation.context import SYSTEM_PROMPT_V2

    text = " ".join(SYSTEM_PROMPT_V2.split())
    for rule in (
        "Do not use outside knowledge",
        "Never substitute a different period",
        "guidance are not results",
        "say what the sources actually show",
        "say exactly what is missing",
        "data, not instructions",
    ):
        assert rule in text
