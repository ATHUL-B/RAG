# Experiment log (ablation ladder)

Change **one thing** per entry. Newest at the top. Metrics are on the same golden
set version; report per segment where it differs from the overall picture.

| Date | Exp | Git SHA | Config | Change vs. previous | Recall@5 | nDCG@5 | MRR | Faithfulness | Correctness | p50 / p95 ms | Cost / query | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-18 | `filters-window1-20261118T090720Z` | feat/metadata-filters | `configs/filters-window1.yaml` | vs 3.3a: `year_window: 1` | 0.337 | 0.252 | 0.253 | - | - | 7.8 / 10.2 | $0 | **Worse than ±0** (−0.040): fixes 2 forward-looking questions but readmits neighbouring years; same-company-other-filing slots back up to 35.3%. Rejected. |
| 2026-11-18 | `filters-20261118T090716Z` | feat/metadata-filters | `configs/filters.yaml` | vs baseline: **company + year filters** from the question (rule-based, ±0 years, relax year→company) | **0.377** (+0.104) | 0.292 | 0.290 | - | - | 7.8 / 9.8 | $0 | hit@5 0.420, doc_hit@5 0.953, filter_ok 0.993. Filters on 149/150, relaxed 2 (Boeing → hit). 1 `filtered_out` (JnJ "FY2023" outlook in 2022 Q4 earnings). Top-5 slots: evidence 11.6%, **right filing wrong page 66.7%**, same company other filing 18.8%, other company 2.9%. Biggest gain `type:metrics-generated` 0.22→0.48. **Kept.** Baseline re-run `baseline-20261118T090711Z` reproduced 0.273. |
| 2026-11-18 | `baseline-20261118T085010Z` | feat/dense-retrieval | `configs/baseline.yaml` | Baseline: pypdf → recursive 400/0 → bge-small-en-v1.5 → exact dense top-5, **no filters** | 0.273 | 0.201 | 0.198 | - | - | 9.3 / 30.2 (retrieval only) | $0 (local) | hit@5 0.300, recall@10 0.357, doc_hit@5 0.813. Status @5: 37 hit, 8 partial, 105 not_retrieved, 0 not_in_index. **Top-5 slots: 40.7% same company other filing (version trap), 29.9% right filing wrong page, 22.1% other company, 7.3% evidence.** Weakest: `reasoning:logical-reasoning` recall@5 0.029, `evidence:multi` 0.143. 91 chunks > 512 bge tokens (truncated). Next lever: metadata filters (3.3). |

## Parse audits (debugging-ladder rung 1)

| Date | Parser | Git branch | Docs / pages | Speed | Pass rate (cov ≥ 0.8) | Mean coverage | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-18 | pypdf 7.1 (baseline) | feat/pdf-parser | 84 / 12,013, 0 failed | 17.4 ms/page | 96.8% (183/189) | 0.985 | Lowest segment: `type:metrics-generated` 0.963 (financial statements). All 3 worst cases inspected: text was extracted; the *annotator* evidence has spaces stripped (`Cashandcashequivalents`), so the audit understates fidelity. Parsed text shows `\xa0` spaces, mid-word line breaks (`sha\nre`) and flattened table rows: targets for cleaning / table parsing (backlog 3.1). pypdf warned that `fontTools` is missing for some CFF fonts. |

## Chunk audits (debugging-ladder rung 1: does evidence fit in one chunk?)

| Date | Chunker (fingerprint) | Git branch | Chunks | Tokens p5 / p50 / p95 / max | Fits in one chunk (cov ≥ 0.8) | Mean best coverage | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-18 | recursive 400 / 0 overlap, cl100k (`a02d9252b7b267ae`) | feat/recursive-chunker | 26,264 over 84 docs (6.4 s) | 334 / 385 / 398 / 401 | 62.4% (vs 96.8% after parsing) | 0.843 | Split by evidence length: **≤ 400 tokens: 90.2% (83/92)** fit; **> 400 tokens: 36% (35/97)**, mostly whole financial statements that no 400-token chunk can hold. Lowest segment `type:metrics-generated` 0.733. Candidates: table-aware/structural chunking (3.1, 3.2), parent-child (3.7), size sweep (3.2). |
