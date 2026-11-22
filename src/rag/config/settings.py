"""Typed, validated configuration.

Every tunable number (chunk size, top-K, ...) lives here and in ``configs/*.yaml``,
never as a literal in code, so an experiment is fully described by its config
file plus the git SHA (CLAUDE.md §2). Sections are added as components land.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)


class StrictConfig(BaseModel):
    """Unknown keys are errors: a typo in YAML must fail loudly, not be ignored."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class DatasetName(StrEnum):
    FINANCEBENCH = "financebench"


class ParserName(StrEnum):
    PYPDF = "pypdf"
    PYPDF_LAYOUT = "pypdf-layout"  # layout mode: table rows stay on one line (HLD 021)


class ChunkingStrategy(StrEnum):
    RECURSIVE = "recursive"


class PathsConfig(StrictConfig):
    data_dir: Path = Path("data")
    runs_dir: Path = Path("runs")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def parsed_dir(self) -> Path:
        """Parsed ``Document`` artifacts of the baseline parser, one JSON per document."""
        return self.processed_dir / "parsed"

    def parsed_dir_for(self, parser: str) -> Path:
        """Per-parser artifacts, so parsers can be compared side by side (HLD 021). The
        baseline keeps its original directory, so existing artifacts stay valid."""
        return self.parsed_dir if parser == "pypdf" else self.processed_dir / f"parsed-{parser}"

    @property
    def judge_cache_dir(self) -> Path:
        """Cached judge verdicts (HLD 016); safe to delete."""
        return self.data_dir / "cache" / "judges"

    @property
    def rerank_cache_dir(self) -> Path:
        """Cached LLM rerank grades (HLD 014); safe to delete."""
        return self.data_dir / "cache" / "rerank"

    @property
    def planner_cache_dir(self) -> Path:
        """Cached LLM query plans (HLD 012); safe to delete, rebuilt on the next run."""
        return self.data_dir / "cache" / "planner"

    def chunks_dir(self, chunker_fingerprint: str) -> Path:
        """Chunk artifacts for one chunker configuration (configs coexist for ablations)."""
        return self.processed_dir / "chunks" / chunker_fingerprint


class ParsingConfig(StrictConfig):
    parser: ParserName = ParserName.PYPDF


class ContextConfig(StrictConfig):
    """Contextual chunk headers (HLD 009): prepended to the indexed text, not to ``text``."""

    enabled: bool = False
    document_fields: tuple[str, ...] = ("company", "doc_type", "year")
    page_title: bool = True
    """Include the title block (statement/table/note name, column years) of the page."""
    max_title_lines: PositiveInt = 8


class ChunkingConfig(StrictConfig):
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_size: PositiveInt = 400
    chunk_overlap: NonNegativeInt = 0
    tokenizer: str = Field(default="cl100k_base", min_length=1)
    """tiktoken encoding used to measure chunk_size and chunk_overlap."""
    respect_pages: bool = False
    """Never let a chunk cross a page boundary (HLD 009)."""
    context: ContextConfig = ContextConfig()

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class EmbeddingConfig(StrictConfig):
    """Defaults follow ADR 0003."""

    model_id: str = Field(default="BAAI/bge-small-en-v1.5", min_length=1)
    revision: str | None = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    """Pinned model revision: the same config always means the same weights."""
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    batch_size: PositiveInt = 64
    device: str = Field(default="auto", pattern=r"^(auto|cpu|mps|cuda(:\d+)?)$")


class FilterField(StrEnum):
    COMPANY = "company"
    YEAR = "year"


class FilterConfig(StrictConfig):
    """Metadata filters extracted from the question (HLD 006)."""

    enabled: bool = False
    fields: tuple[FilterField, ...] = (FilterField.COMPANY, FilterField.YEAR)
    year_window: NonNegativeInt = 0
    """Widen each mentioned year by this many years on both sides."""
    relax_order: tuple[FilterField, ...] = (FilterField.YEAR, FilterField.COMPANY)
    """Filters dropped one at a time, in this order, when a search finds nothing."""
    short_fiscal_years: bool = False
    """Read "FY22" / "FY 20" as 2022 / 2020 (rule-based planner, HLD 011)."""
    forward_looking_prior_year: bool = False
    """Forecast questions ("expected in FY2023") also allow the previous year's filing."""


class LLMPlannerConfig(StrictConfig):
    """Local LLM query planner via Ollama: decomposition into sub-queries (HLD 012)."""

    enabled: bool = False
    model_id: str = Field(default="llama3.1:8b", min_length=1)
    ollama_url: str = "http://localhost:11434"
    timeout_s: PositiveInt = 120
    max_sub_queries: PositiveInt = 4
    max_chars: PositiveInt = 200
    """Longest sub-query: enforced in the output schema, then truncated as a backstop."""
    max_output_tokens: PositiveInt = 300
    """Cap on planner output (Ollama ``num_predict``): a runaway plan fails fast."""
    cache: bool = True
    """Reuse plans from data/cache/planner/<model>-<prompt hash>.json (Part 7, rule 5)."""


class RewriteConfig(StrictConfig):
    """Rule-based query rewriting before search (Part 7, HLD 011)."""

    strip_boilerplate: bool = False
    """Drop instruction sentences; name statements by their filing titles."""
    expand_abbreviations: bool = False
    """Append spelled-out names of finance abbreviations (PP&E, COGS, ...)."""
    keep_original: bool = True
    """Also search the original question and fuse with RRF (Part 7, rule 1)."""
    original_weight: float = Field(default=1.0, gt=0.0)
    """RRF weight of the original question's list; every rewrite list has weight 1 (HLD 013)."""
    llm_planner: LLMPlannerConfig = LLMPlannerConfig()

    @property
    def rules_enabled(self) -> bool:
        return self.strip_boilerplate or self.expand_abbreviations

    @property
    def enabled(self) -> bool:
        return self.rules_enabled or self.llm_planner.enabled


class RetrievalMode(StrEnum):
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"


class BM25Config(StrictConfig):
    k1: float = Field(default=1.2, ge=0.0)
    """Term-frequency saturation: higher = repeated terms keep adding score for longer."""
    b: float = Field(default=0.75, ge=0.0, le=1.0)
    """Length normalization: 0 = ignore chunk length, 1 = fully normalize."""


class FusionConfig(StrictConfig):
    rrf_k: NonNegativeInt = 60
    """RRF constant: small k rewards top ranks, large k rewards agreement across lists."""
    weights: dict[str, float] = Field(default_factory=lambda: {"dense": 1.0, "bm25": 1.0})


class JudgeConfig(StrictConfig):
    """LLM judges for answers (HLD 016). Default: a different family from the generator."""

    model_id: str = Field(default="qwen2.5:7b", min_length=1)
    ollama_url: str = "http://localhost:11434"
    timeout_s: PositiveInt = 120
    context_tokens: PositiveInt = 8192
    """Faithfulness prompts carry the 5 sources (~2.4k tokens) plus the answer."""
    max_output_tokens: PositiveInt = 300
    cache: bool = True
    min_kappa: float = Field(default=0.6, ge=-1.0, le=1.0)
    """Acceptance bar vs hand labels, fixed before calibrating."""


class RerankerKind(StrEnum):
    CROSS_ENCODER = "cross_encoder"
    LLM = "llm"


class LLMRerankConfig(StrictConfig):
    """Pointwise grading of each candidate by a local Ollama model (HLD 014)."""

    model_id: str = Field(default="llama3.1:8b", min_length=1)
    ollama_url: str = "http://localhost:11434"
    timeout_s: PositiveInt = 120
    context_tokens: PositiveInt = 4096
    """Ollama ``num_ctx``: must fit prompt + passage, or Ollama silently truncates."""
    max_passage_chars: PositiveInt = 3000
    max_output_tokens: PositiveInt = 20
    cache: bool = True
    """Reuse grades from data/cache/rerank/<model>-<prompt hash>.json."""
    grade_format: Literal["object", "integer"] = "object"
    """``object`` writes {"grade": 3} (HLD 014); ``integer`` writes a bare 3, ~350 ms
    faster per call (HLD 019). Each format has its own grade cache."""
    time_budget_s: PositiveFloat | None = None
    """Stop starting new grading calls after this many seconds for a question; the
    ungraded candidates keep their first-stage order (HLD 020). None = no budget."""
    concurrency: PositiveInt = 1
    """Grading calls in flight at once (HLD 019). Helps only if the server has parallel
    slots (``OLLAMA_NUM_PARALLEL > 1``); a single-slot Ollama just queues them."""


class RerankConfig(StrictConfig):
    """Second-stage reranking: cross-encoder (ADR 0004, HLD 008) or local LLM (HLD 014)."""

    enabled: bool = False
    kind: RerankerKind = RerankerKind.CROSS_ENCODER
    llm: LLMRerankConfig = LLMRerankConfig()
    model_id: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", min_length=1)
    revision: str | None = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    depth: PositiveInt = 50
    """First-stage candidates given to the reranker: bounds recall, drives latency."""
    batch_size: PositiveInt = 16
    max_length: PositiveInt = 512
    device: str = Field(default="auto", pattern=r"^(auto|cpu|mps|cuda(:\d+)?)$")


class ParentChildConfig(StrictConfig):
    """Search small child chunks, return their parent chunks (HLD 023).

    The parents are the chunks of ``chunking``. The children use the same chunking with
    these size and overlap values, and have their own index (``make data`` with a config
    whose chunking matches them, e.g. ``chunks-200-o0.yaml``)."""

    enabled: bool = False
    child_chunk_size: PositiveInt = 200
    child_chunk_overlap: NonNegativeInt = 0
    fanout: PositiveInt = 4
    """Children fetched per parent requested (several children share one parent)."""


class RetrievalConfig(StrictConfig):
    top_k: PositiveInt = 5
    mode: RetrievalMode = RetrievalMode.DENSE
    candidate_depth: PositiveInt = 50
    """Candidates each retriever returns before fusion (hybrid mode)."""
    bm25: BM25Config = BM25Config()
    fusion: FusionConfig = FusionConfig()
    rerank: RerankConfig = RerankConfig()
    filters: FilterConfig = FilterConfig()
    rewrite: RewriteConfig = RewriteConfig()
    parent_child: ParentChildConfig = ParentChildConfig()


class EvaluationConfig(StrictConfig):
    ks: tuple[PositiveInt, ...] = Field(default=(1, 3, 5, 10), min_length=1)
    """Cut-offs for recall@K, precision@K, hit@K and nDCG@K."""
    parse_audit_min_coverage: float = Field(default=0.8, ge=0.0, le=1.0)
    """Evidence passes the parse audit when this share of its tokens is on the parsed page."""
    split_file: Path = Path("evals/financebench-split.json")
    """Committed dev/test split (``scripts/make_split.py``); scripts default to ``dev``."""


class LLMProvider(StrEnum):
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"


class Effort(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class PricingConfig(StrictConfig):
    """USD per million tokens, used for cost accounting and the pre-run budget check."""

    input_per_mtok: float = Field(default=4.0, ge=0.0)
    output_per_mtok: float = Field(default=20.0, ge=0.0)


class GenerationConfig(StrictConfig):
    """Grounded answer generation (ADR 0005, HLD 010)."""

    provider: LLMProvider = LLMProvider.ANTHROPIC
    model_id: str = Field(default="claude-opus-5-5", min_length=1)
    effort: Effort = Effort.MEDIUM
    """Claude only: thinking depth (thinking is always on for current Opus models)."""
    max_tokens: PositiveInt = 16000
    max_sources: PositiveInt = 5
    prompt_version: Literal["v1", "v2"] = "v1"
    """System prompt (generation/context.py): v1 (HLD 010/015) or v2 with exact-period,
    no-guidance, false-premise and explained-refusal rules (HLD 027)."""
    refusal_fallbacks: bool = True
    """Claude only: server-side fallback when a request is declined."""
    ollama_url: str = "http://localhost:11434"
    ollama_timeout_s: PositiveInt = 300
    ollama_context_tokens: PositiveInt = 8192
    """Ollama ``num_ctx``. Its default is small and truncates long prompts from the FRONT,
    which is where the grounding rules live. Dev prompts are at most ~2.4k tokens."""
    gateway_url_env: str = "LLM_GATEWAY_URL"
    """openai_compatible: env var holding the base URL (ends in /v1). Never in configs."""
    gateway_key_env: str = "LLM_GATEWAY_KEY"
    gateway_key_header: str = "Authorization"
    """Header the key goes in: ``Authorization`` (Bearer) or a gateway's own, e.g. ``x-bf-vk``."""
    gateway_temperature: float | None = 0.0
    """OpenAI-compatible sampling temperature; set to null for models that reject it."""
    gateway_use_response_format: bool = True
    """Use JSON-schema response_format; disable if the gateway rejects or filters it."""
    gateway_filter_retries: NonNegativeInt = 2
    """Retries when the gateway's content filter cuts a reply off (it fires intermittently)."""
    gateway_timeout_s: PositiveInt = 300
    pricing: PricingConfig = PricingConfig()
    expected_output_tokens: PositiveInt = 1500
    """Per-question output (incl. thinking) assumed by the pre-run cost estimate."""


class AppConfig(StrictConfig):
    experiment_name: str = Field(min_length=1)
    dataset: DatasetName = DatasetName.FINANCEBENCH
    paths: PathsConfig = PathsConfig()
    parsing: ParsingConfig = ParsingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
