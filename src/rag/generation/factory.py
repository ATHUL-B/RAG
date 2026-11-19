"""Config → generator (ADR 0005)."""

from __future__ import annotations

from rag.config import GenerationConfig, LLMProvider
from rag.generation.base import LLMClient
from rag.generation.context import SourcesContextAssembler
from rag.generation.generator import GroundedAnswerGenerator
from rag.generation.llm_clients import AnthropicLLMClient, BifrostLLMClient, OllamaLLMClient


def build_llm_client(config: GenerationConfig) -> LLMClient:
    match config.provider:
        case LLMProvider.ANTHROPIC:
            return AnthropicLLMClient(
                model=config.model_id,
                max_tokens=config.max_tokens,
                effort=str(config.effort),
                refusal_fallbacks=config.refusal_fallbacks,
            )
        case LLMProvider.OLLAMA:
            return OllamaLLMClient(
                model=config.model_id,
                base_url=config.ollama_url,
                timeout_s=float(config.ollama_timeout_s),
            )
        case LLMProvider.BIFROST:
            return BifrostLLMClient(
                model=config.model_id,
                base_url=config.bifrost_url,
                api_key_env=config.bifrost_api_key_env,
                max_tokens=config.max_tokens,
                timeout_s=float(config.bifrost_timeout_s),
            )


def build_generator(
    config: GenerationConfig, llm: LLMClient | None = None
) -> GroundedAnswerGenerator:
    assembler = SourcesContextAssembler(max_sources=config.max_sources)
    return GroundedAnswerGenerator(assembler, llm or build_llm_client(config))
