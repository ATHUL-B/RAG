"""Domain-specific exceptions.

One base class lets callers catch "anything this library raised" while the
subclasses say *which stage* failed, which is the first question the debugging
ladder asks (study guide, Part 13).
"""


class RagError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(RagError):
    """Configuration is missing, malformed or inconsistent."""


class DatasetError(RagError):
    """A dataset could not be downloaded or its records are malformed."""


class StorageError(RagError):
    """A pipeline artifact could not be read or written."""


class IngestionError(RagError):
    """Parsing, cleaning, enrichment or chunking failed."""


class IndexingError(RagError):
    """Embedding or writing to an index failed."""


class RetrievalError(RagError):
    """Query rewriting, retrieval, fusion or reranking failed."""


class GenerationError(RagError):
    """Context assembly or the LLM call failed."""


class UnusableAnswerError(GenerationError):
    """The model replied, but the reply cannot be used as an answer.

    Examples: it breaks the answer schema, or the answer field is empty, a bare
    ``true``/``false`` or leaked tool-call markup. ``kind`` names the problem and
    ``raw_output`` keeps the exact text, so a run can say *why* it failed and the failure
    can be inspected later (the first full run lost 78% of its answers this way, see
    ``experiments/analysis/2026-11-19-first-generation-run.md``).
    """

    def __init__(
        self, message: str, kind: str, raw_output: str, prompt_chunk_ids: tuple[str, ...] = ()
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.raw_output = raw_output
        self.prompt_chunk_ids = prompt_chunk_ids
