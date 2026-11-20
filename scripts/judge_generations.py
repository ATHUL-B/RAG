"""Judge a generation run's answers for correctness and faithfulness (HLD 016).

Usage:
    uv run python scripts/judge_generations.py runs/<generation-run> [--judges configs/judges.yaml]

Reads <run>/generations.jsonl and <run>/summary.json (for the run's own config, so sources
are rebuilt from the same index), then writes <run>/judgements.jsonl and
<run>/judge_summary.json. Needs Ollama with the judge model. Verdicts are cached, so
re-judging is free.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from rag.config import AppConfig, JudgeConfig
from rag.errors import GenerationError
from rag.evaluation.generation_runner import GenerationTrace
from rag.evaluation.judge_runner import JudgeRunner
from rag.evaluation.judges import JudgeKind, LLMJudge, VerdictCache
from rag.generation.llm_clients import OllamaLLMClient, check_ollama_model
from rag.indexing.factory import build_embedder, dense_index_dir
from rag.indexing.vector_store import ExactVectorStore
from rag.ingestion.factory import build_chunker
from rag.observability import configure_logging
from rag.storage import atomic_write_bytes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path, help="a runs/<generation-run> directory")
    parser.add_argument("--judges", type=Path, default=Path("configs/judges.yaml"))
    args = parser.parse_args(argv)
    configure_logging("WARNING")

    judge_config = JudgeConfig.model_validate(yaml.safe_load(args.judges.read_text()))
    config = AppConfig.model_validate(json.loads((args.run / "summary.json").read_text())["config"])
    try:
        check_ollama_model(judge_config.model_id, judge_config.ollama_url)
    except GenerationError as exc:
        print(f"judge model unavailable: {exc}")
        return 2

    embedder = build_embedder(config.embedding)  # lazy: only its model id is used
    index_dir = dense_index_dir(config, build_chunker(config.chunking).fingerprint)
    store = ExactVectorStore.load(index_dir, expected_model_id=embedder.model_id)
    chunks = {c.chunk_id: c for c in store.chunks}

    llm = OllamaLLMClient(
        model=judge_config.model_id,
        base_url=judge_config.ollama_url,
        timeout_s=float(judge_config.timeout_s),
        max_output_tokens=judge_config.max_output_tokens,
        context_tokens=judge_config.context_tokens,
    )
    judges = [
        LLMJudge(
            kind,
            llm,
            VerdictCache.for_judge(config.paths.judge_cache_dir, kind, judge_config.model_id)
            if judge_config.cache
            else None,
        )
        for kind in JudgeKind
    ]
    traces = [
        GenerationTrace.model_validate_json(line)
        for line in (args.run / "generations.jsonl").read_text().splitlines()
        if line.strip()
    ]
    judged, summary = JudgeRunner(judges, chunks).run(traces)

    lines = "".join(a.model_dump_json() + "\n" for a in judged)
    atomic_write_bytes(args.run / "judgements.jsonl", lines.encode("utf-8"))
    report = summary.model_dump(mode="json") | {
        "judge_model": judge_config.model_id,
        "calls": {str(j.kind): {"calls": j.calls, "cache_hits": j.cache_hits} for j in judges},
    }
    atomic_write_bytes(args.run / "judge_summary.json", json.dumps(report, indent=2).encode())

    print(
        f"judged {summary.judged} answers (skipped {summary.skipped}), "
        f"judge failures {summary.failures}"
    )
    print("  ".join(f"{k}={v:.2f}" for k, v in summary.rates.items()))
    print(f"{'evidence reach':<16}{'n':>5}" + "".join(f"{k:>14}" for k in summary.rates))
    for reach, row in summary.by_reach.items():
        print(f"{reach:<16}{row['n']:>5.0f}" + "".join(f"{row[k]:>14.2f}" for k in summary.rates))
    na = summary.numeric_agreement
    print(
        f"correctness judge vs numeric_match on {na['n']:.0f} numeric answers: "
        f"{na['agreement']:.2f} agree"
    )
    print(f"\njudgements: {args.run / 'judgements.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
