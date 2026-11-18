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
