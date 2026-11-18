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
