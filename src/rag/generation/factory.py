"""Config → generator (ADR 0005)."""

from __future__ import annotations

import os

from rag.config import GenerationConfig, LLMProvider
from rag.errors import ConfigError
from rag.generation.base import LLMClient
from rag.generation.context import SourcesContextAssembler
from rag.generation.generator import GroundedAnswerGenerator
from rag.generation.llm_clients import (
    AnthropicLLMClient,
    OllamaLLMClient,
    OpenAICompatibleLLMClient,
)


def build_llm_client(config: GenerationConfig) -> LLMClient:
    match config.provider:
        case LLMProvider.ANTHROPIC:
            return AnthropicLLMClient(
                model=config.model_id,
                max_tokens=config.max_tokens,
                effort=str(config.effort),
                refusal_fallbacks=config.refusal_fallbacks,
            )
        case LLMProvider.OPENAI_COMPATIBLE:
            if config.model_id.startswith("<"):
                raise ConfigError(f"set generation.model_id (still {config.model_id!r})")
            base_url = os.environ.get(config.gateway_url_env, "")
            if not base_url:
                raise ConfigError(f"set {config.gateway_url_env} in .env (the gateway's base URL)")
            return OpenAICompatibleLLMClient(
                model=config.model_id,
                base_url=base_url,
                api_key=os.environ.get(config.gateway_key_env, ""),
                key_header=config.gateway_key_header,
                timeout_s=float(config.gateway_timeout_s),
                max_tokens=config.max_tokens,
                temperature=config.gateway_temperature,
                use_response_format=config.gateway_use_response_format,
                filter_retries=config.gateway_filter_retries,
            )
        case LLMProvider.OLLAMA:
            return OllamaLLMClient(
                model=config.model_id,
                base_url=config.ollama_url,
                timeout_s=float(config.ollama_timeout_s),
                max_output_tokens=config.max_tokens,
                context_tokens=config.ollama_context_tokens,
            )


def build_generator(
    config: GenerationConfig, llm: LLMClient | None = None
) -> GroundedAnswerGenerator:
    assembler = SourcesContextAssembler(max_sources=config.max_sources)
    return GroundedAnswerGenerator(assembler, llm or build_llm_client(config))
