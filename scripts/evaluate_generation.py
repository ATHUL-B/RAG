"""Evaluate grounded answer generation on FinanceBench (HLD 010).

Usage:
    uv run python scripts/evaluate_generation.py --config configs/generation-claude.yaml \
        [--limit 10] [--offset 0] [--budget-usd 1.0] [--dry-run]

Paid providers cost money: the script estimates the cost first and stops if the estimate
exceeds --budget-usd. --dry-run prints the estimate and one assembled prompt without
calling any model. Writes runs/<run_id>/generations.jsonl and summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from rag.app import load_retrieval_stack
from rag.config import LLMProvider, load_config
from rag.config.env import load_env_file
from rag.datasets.financebench import FinanceBenchDataset
from rag.domain import Query
from rag.evaluation.generation_runner import GenerationEvalRunner, Pricing, estimate_cost
from rag.evaluation.retrieval_report import OVERALL
from rag.generation.context import SourcesContextAssembler
from rag.generation.factory import build_generator
from rag.ingestion.tokenization import TiktokenCounter
from rag.observability import configure_logging
from rag.storage import atomic_write_bytes

SHOWN = ("answered", "evidence_in_prompt", "has_citation", "citation_valid", "cites_evidence")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/generation-claude.yaml"))
    parser.add_argument("--limit", type=int, default=10, help="number of questions (0 = all)")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--budget-usd", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true", help="estimate only, call no model")
    args = parser.parse_args(argv)
    configure_logging("WARNING")
    load_env_file()

    config = load_config(args.config)
    gen = config.generation
    items = FinanceBenchDataset(config.paths.raw_dir / "financebench").golden_items()
    items = items[args.offset :]
    items = items[: args.limit] if args.limit > 0 else items

    stack = load_retrieval_stack(config)
    pricing = Pricing(
        input_per_mtok=gen.pricing.input_per_mtok, output_per_mtok=gen.pricing.output_per_mtok
    )
    assembler = SourcesContextAssembler(max_sources=gen.max_sources)
    prompts = [
        assembler.assemble(q, stack.retriever.retrieve(q, gen.max_sources).candidates)
        for q in (Query(text=item.question) for item in items)
    ]
    estimate = estimate_cost(prompts, TiktokenCounter(), pricing, gen.expected_output_tokens)
    print(
        f"{len(items)} questions, provider={gen.provider} model={gen.model_id} "
        f"effort={gen.effort}: estimated cost ≤ ${estimate:.2f} (budget ${args.budget_usd:.2f})"
    )
    if args.dry_run:
        print(f"\n--- example prompt (system) ---\n{prompts[0].system}")
        print(f"\n--- example prompt (user, first 1500 chars) ---\n{prompts[0].user[:1500]}")
        return 0
    if gen.provider is not LLMProvider.OLLAMA and estimate > args.budget_usd:
        print("estimate exceeds budget: lower --limit or raise --budget-usd explicitly")
        return 2

    runner = GenerationEvalRunner(
        stack.retriever, build_generator(gen), pricing, depth=gen.max_sources
    )
    result = runner.run(items)

    run_id = f"{config.experiment_name}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    out = config.paths.runs_dir / run_id
    lines = "".join(t.model_dump_json() + "\n" for t in result.traces)
    atomic_write_bytes(out / "generations.jsonl", lines.encode("utf-8"))
    summary = result.model_dump(exclude={"traces"}, mode="json")
    summary |= {"run_id": run_id, "config": config.model_dump(mode="json")}
    atomic_write_bytes(out / "summary.json", json.dumps(summary, indent=2).encode("utf-8"))

    totals = result.totals
    print(
        f"\nrun {run_id}: errors={result.errors}  numeric match "
        f"{result.numeric['matched']:.0f}/{result.numeric['n']:.0f}  "
        f"tokens in/out {totals['input_tokens']:.0f}/{totals['output_tokens']:.0f}  "
        f"cost ${totals['cost_usd']:.3f}"
    )
    gen_ms = result.latency_ms.get("generate", {})
    print(f"generation latency ms: { {k: round(v) for k, v in gen_ms.items()} }")
    if result.report is not None:
        overall = result.report.segments[OVERALL]
        print("  ".join(f"{name}={overall.means[name]:.2f}" for name in SHOWN))
    print(f"\ngenerations: {out / 'generations.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
