# HLD 005: Retrieval evaluation runner, traces and failure ladder

## Goal

Backlog 1.6 + 1.7: run every golden question through a `Retriever`, score it with the
HLD 001 metrics via the HLD 003 relevance units, and record **why** each miss happened
(study guide, Part 13 debugging ladder, rungs 1-2), plus per-query latency.

## Design

`RetrievalEvalRunner(retriever, evaluator, indexed_units, top_k, clock)` in
`evaluation/retrieval_runner.py`:

For each answerable `GoldenItem` (unanswerable items are listed as skipped):

1. `Query(text=item.question)`: no filters, no rewriting (the naive baseline, Part 6).
2. Retrieve `depth = max(ks ∪ {top_k})` candidates, timed with `StageTimer` (`retrieve`).
3. `relevance_ids` → metrics from `RetrievalEvaluator`, plus `doc_hit@top_k`: did any of
   the top_k chunks come from an evidence *document*? (separates "wrong filing" from
   "right filing, wrong page").
4. Failure classification at `top_k`:

| Status | Meaning | Ladder rung → where to fix |
| --- | --- | --- |
| `not_in_index` | no evidence page is covered by any indexed chunk | 1 → parsing/chunking |
| `not_retrieved` | evidence indexed, none in top_k | 2 → query, filters, embeddings, hybrid |
| `partial` | some but not all evidence pages in top_k | 2 (multi-evidence) |
| `hit` | all evidence pages in top_k | - |

5. A `QueryTrace`: question, tags, evidence units, each retrieved chunk (rank, id, doc,
   pages, score, relevance unit, relevant?), timings, status, metrics.

`indexed_units` = every `<doc>#p<page>` covered by a chunk in the index (from the store).

Result: `RetrievalRunResult` (traces, `MetricsReport` overall + per tag, status counts,
latency p50/p95/p99/max). The script writes `runs/<run_id>/traces.jsonl` and
`runs/<run_id>/metrics.json` and prints a summary; `run_id = <experiment>-<UTC timestamp>`.

## Test plan

Fake retriever returning scripted chunks: each status, doc_hit, skipped unanswerable,
latency from a fake clock, metrics wiring. Percentile helper edge cases.
