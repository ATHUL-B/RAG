"""Calibrate the LLM judges against your own labels (CLAUDE.md §2, Part 13; HLD 016).

Usage:
    uv run python scripts/judge_calibration.py sample runs/<gen-run-a> runs/<gen-run-b> [--n 50]
    uv run python scripts/judge_calibration.py label     # interactive, resumable, BLIND
    uv run python scripts/judge_calibration.py report    # agreement + Cohen's kappa per judge

`sample` picks answers stratified by run x evidence reach x numeric/text (fixed seed) and
writes evals/judge_calibration/sample-v1.jsonl. `label` shows one answer at a time and never
shows the judge's verdict, so your label is not anchored to it. `report` compares your labels
with the judges' verdicts (from each run's judgements.jsonl) against the bar in
configs/judges.yaml (min_kappa), which is fixed before calibrating.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from rag.config import AppConfig, JudgeConfig
from rag.evaluation.calibration import agreement
from rag.evaluation.generation_runner import AnswerOutcome, GenerationTrace
from rag.evaluation.judge_runner import rebuild_sources
from rag.indexing.factory import build_embedder, dense_index_dir
from rag.indexing.vector_store import ExactVectorStore
from rag.ingestion.factory import build_chunker
from rag.storage import atomic_write_bytes

DIR = Path("evals/judge_calibration")
SAMPLE = DIR / "sample-v1.jsonl"
LABELS = DIR / "labels-v1.jsonl"
SEED = 0
CHOICES = {
    "correctness": {"c": "correct", "i": "incorrect"},
    "faithfulness": {"f": "faithful", "u": "unfaithful"},
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    atomic_write_bytes(path, body.encode("utf-8"))


def _chunks(run: Path) -> dict[str, Any]:
    config = AppConfig.model_validate(json.loads((run / "summary.json").read_text())["config"])
    index_dir = dense_index_dir(
        config, build_chunker(config.chunking, parser=config.parsing.parser).fingerprint
    )
    model_id = build_embedder(config.embedding).model_id
    return {c.chunk_id: c for c in ExactVectorStore.load(index_dir, model_id).chunks}


def sample(runs: list[Path], n: int) -> int:
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        chunks = _chunks(run)
        for line in (run / "generations.jsonl").read_text().splitlines():
            trace = GenerationTrace.model_validate_json(line)
            if trace.outcome is AnswerOutcome.ERROR or trace.refused:
                continue
            kind = "numeric" if trace.numeric_match is not None else "text"
            strata[(run.name, str(trace.evidence_reach), kind)].append(
                {
                    "key": f"{run.name}:{trace.item_id}",
                    "run": run.name,
                    "item_id": trace.item_id,
                    "question": trace.question,
                    "reference_answer": trace.reference_answer,
                    "answer": trace.answer,
                    "working": trace.working,
                    "sources": rebuild_sources(trace.prompt_chunk_ids, chunks),
                }
            )
    rng = random.Random(SEED)
    for rows in strata.values():
        rng.shuffle(rows)
    picked: list[dict[str, Any]] = []
    while len(picked) < n and any(strata.values()):  # round-robin: every stratum is represented
        for key in sorted(strata):
            if strata[key] and len(picked) < n:
                picked.append(strata[key].pop())
    _write_jsonl(SAMPLE, picked)
    print(f"wrote {len(picked)} answers to {SAMPLE} from {len(strata)} strata")
    return 0


def _wrap(text: str, width: int = 100) -> str:
    return textwrap.fill(text, width) if text else "(empty)"


def _show(row: dict[str, Any], position: int, total: int) -> None:
    wrap = _wrap
    print("\n" + "=" * 100)
    print(f"[{position}/{total}] {row['key']}")
    print(f"\nQUESTION:\n{wrap(row['question'])}")
    print(f"\nREFERENCE ANSWER:\n{wrap(row['reference_answer'])}")
    print(f"\nMODEL ANSWER:\n{wrap(row['answer'])}")
    print(f"\nMODEL WORKING:\n{wrap(row['working'])}")
    print("\nSOURCES SHOWN TO THE MODEL:")
    for block in row["sources"].split("\n\n"):
        print(textwrap.indent(textwrap.fill(block, 96), "  ") + "\n")


def _ask(question: str, options: dict[str, str]) -> str | None:
    keys = "/".join(f"{k}={v}" for k, v in options.items())
    while True:
        reply = input(f"{question} [{keys}, s=skip, q=quit]: ").strip().lower()
        if reply in options:
            return options[reply]
        if reply == "s":
            return ""
        if reply == "q":
            return None


def label() -> int:
    rows = _read_jsonl(SAMPLE)
    if not rows:
        print(f"no sample: run `{Path(__file__).name} sample <runs…>` first")
        return 1
    labels = {r["key"]: r for r in _read_jsonl(LABELS)}
    todo = [r for r in rows if r["key"] not in labels]
    print(f"{len(labels)} labelled, {len(todo)} to go. Judge verdicts are hidden on purpose.")
    print("correct   = same number/conclusion as the reference (rounding, correct unit/scale)")
    print("faithful  = every claim in answer + working is in the sources (or computed from them)")
    for position, row in enumerate(todo, start=len(labels) + 1):
        _show(row, position, len(rows))
        correctness = _ask("Is the answer CORRECT vs the reference?", CHOICES["correctness"])
        if correctness is None:
            break
        faithfulness = _ask("Is it FAITHFUL to the sources?", CHOICES["faithfulness"])
        if faithfulness is None:
            break
        note = input("optional note (enter to skip): ").strip()
        labels[row["key"]] = {
            "key": row["key"],
            "correctness": correctness,
            "faithfulness": faithfulness,
            "note": note,
        }
        _write_jsonl(LABELS, list(labels.values()))  # save after every answer
    print(f"\n{len(labels)}/{len(rows)} labelled → {LABELS}")
    return 0


def report(judges_config: Path) -> int:
    config = JudgeConfig.model_validate(yaml.safe_load(judges_config.read_text()))
    labels = _read_jsonl(LABELS)
    sample_rows = {r["key"]: r for r in _read_jsonl(SAMPLE)}
    verdicts: dict[str, dict[str, str]] = {}
    for run in {r["run"] for r in sample_rows.values()}:
        for judged in _read_jsonl(Path("runs") / run / "judgements.jsonl"):
            verdicts[f"{run}:{judged['item_id']}"] = {
                kind: v["label"] for kind, v in judged["verdicts"].items()
            }
    passed_all = True
    for kind in CHOICES:
        pairs = [
            (label[kind], verdicts[label["key"]][kind])
            for label in labels
            if label.get(kind) and kind in verdicts.get(label["key"], {})
        ]
        if not pairs:
            print(f"{kind}: no labelled answers with a verdict yet")
            passed_all = False
            continue
        result = agreement([h for h, _ in pairs], [j for _, j in pairs])
        ok = result.kappa >= config.min_kappa
        passed_all &= ok
        print(
            f"{kind:<13} n={result.n:<3} agreement={result.observed:.2f} "
            f"kappa={result.kappa:.2f} (bar {config.min_kappa}) → {'PASS' if ok else 'FAIL'}"
        )
        print(f"{'':<13} human→judge: {result.confusion}")
    return 0 if passed_all else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sample_cmd = sub.add_parser("sample")
    sample_cmd.add_argument("runs", type=Path, nargs="+")
    sample_cmd.add_argument("--n", type=int, default=50)
    sub.add_parser("label")
    report_cmd = sub.add_parser("report")
    report_cmd.add_argument("--judges", type=Path, default=Path("configs/judges.yaml"))
    args = parser.parse_args(argv)
    if args.command == "sample":
        return sample(args.runs, args.n)
    if args.command == "label":
        return label()
    return report(args.judges)


if __name__ == "__main__":
    sys.exit(main())
