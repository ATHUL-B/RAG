"""Evaluation regression gate on a small golden subset (backlog 4.3; HLD 024).

Usage:
    uv run python scripts/eval_gate.py                     # check against evals/gate-v1.json
    uv run python scripts/eval_gate.py --update            # rewrite the expectations (deliberate)
    uv run python scripts/eval_gate.py --copy-pdfs-from data/raw/financebench/pdfs   # no download

Builds a small corpus under data/gate/: only the filings of the gate companies. It runs
the normal pipeline scripts on it (parse, chunk, index), evaluates the gate questions with
the gate's retrieval config, and compares each question's recall@5 with the committed
expectations. It exits 1 if the gate fails. Needs no LLM, so it runs on a CI runner.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from rag.app import load_golden_items, load_retrieval_stack
from rag.config import load_config
from rag.datasets.financebench import FinanceBenchDataset, FinanceBenchDownloader
from rag.datasets.http import UrllibFetcher
from rag.evaluation.gate import check_gate, load_gate
from rag.evaluation.retrieval_report import RetrievalEvaluator
from rag.evaluation.retrieval_runner import RetrievalEvalRunner, indexed_units
from rag.evaluation.splits import SplitName
from rag.observability import configure_logging
from rag.storage import atomic_write_bytes

PIPELINE = ("parse_documents.py", "chunk_documents.py", "build_index.py")


def gate_config(spec_config: Path, data_dir: Path, device: str | None) -> Path:
    """The gate's retrieval config, pointed at the small corpus in ``data_dir``."""
    raw = yaml.safe_load(spec_config.read_text(encoding="utf-8"))
    raw["experiment_name"] = "eval-gate"
    raw.setdefault("paths", {})["data_dir"] = str(data_dir)
    if device is not None:
        raw.setdefault("embedding", {})["device"] = device
    path = data_dir / "gate-config.yaml"
    data_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def fetch_corpus(doc_names: tuple[str, ...], raw_root: Path, copy_from: Path | None) -> int:
    downloader = FinanceBenchDownloader(UrllibFetcher(), raw_root)
    if downloader.download_metadata().failed:
        print("metadata download failed")
        return 1
    dataset = FinanceBenchDataset(raw_root)
    if copy_from is not None:  # local runs: reuse PDFs already downloaded for the full corpus
        for name in doc_names:
            target = dataset.pdf_path(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(copy_from / target.name, target)
        return 0
    report = downloader.download_pdfs(doc_names)
    if report.failed:
        print(f"pdf download failed: {report.failed}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gate", type=Path, default=Path("evals/gate-v1.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/gate"))
    parser.add_argument("--update", action="store_true", help="write expectations, don't check")
    parser.add_argument("--copy-pdfs-from", type=Path, default=None)
    parser.add_argument("--device", default=None, help="override embedding.device (e.g. cpu)")
    args = parser.parse_args(argv)
    configure_logging("WARNING")

    spec = load_gate(args.gate)
    config_path = gate_config(Path(spec.config), args.data_dir, args.device)
    config = load_config(config_path)
    raw_root = config.paths.raw_dir / "financebench"
    if fetch_corpus(spec.doc_names, raw_root, args.copy_pdfs_from):
        return 1
    for script in PIPELINE:
        step = [sys.executable, str(Path(__file__).parent / script), "--config", str(config_path)]
        if subprocess.run(step, check=False, capture_output=True).returncode:
            print(f"pipeline step failed: {script} (re-run it with --config {config_path})")
            return 1

    stack = load_retrieval_stack(config)
    wanted = set(spec.expected) if not args.update else None
    companies = set(spec.companies)
    items = [
        item
        for item in load_golden_items(config, SplitName(spec.split))
        if (wanted is None and item.metadata.get("company") in companies)
        or (wanted is not None and item.item_id in wanted)
    ]
    runner = RetrievalEvalRunner(
        retriever=stack.retriever,
        evaluator=RetrievalEvaluator(config.evaluation.ks),
        indexed=indexed_units(stack.store.chunks),
        top_k=config.retrieval.top_k,
    )
    result = runner.run(items)
    recall = f"recall@{config.retrieval.top_k}"
    actual = {t.item_id: t.metrics[recall] for t in result.traces}

    if args.update:
        updated = spec.model_copy(update={"expected": dict(sorted(actual.items()))})
        body = json.dumps(updated.model_dump(mode="json"), indent=2) + "\n"
        atomic_write_bytes(args.gate, body.encode("utf-8"))
        mean = sum(actual.values()) / len(actual)
        print(f"wrote {len(actual)} expectations to {args.gate} (mean {recall} {mean:.3f})")
        return 0

    gate = check_gate(spec.expected, actual, spec.max_regressions, spec.max_mean_drop)
    print(
        f"eval gate {'PASSED' if gate.passed else 'FAILED'}: {len(actual)} questions, "
        f"mean {recall} {gate.mean_actual:.3f} (expected {gate.mean_expected:.3f})"
    )
    for qid, (before, after) in gate.regressions.items():
        print(f"  regression  {qid}: {before:.2f} -> {after:.2f}")
    for qid, (before, after) in gate.improvements.items():
        print(f"  improvement {qid}: {before:.2f} -> {after:.2f}")
    for reason in gate.reasons:
        print(f"  FAIL: {reason}")
    if gate.improvements and gate.passed:
        print("  (gains: if intended, run --update and commit the new expectations)")
    return 0 if gate.passed else 1


if __name__ == "__main__":
    sys.exit(main())
