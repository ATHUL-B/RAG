from rag.config import GenerationConfig, LLMProvider
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
