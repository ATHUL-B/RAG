# Experiment log (ablation ladder)

Change **one thing** per entry. Newest at the top. Metrics are on the same golden
set version; report per segment where it differs from the overall picture.

| Date | Exp | Git SHA | Config | Change vs. previous | Recall@5 | nDCG@5 | MRR | Faithfulness | Correctness | p50 / p95 ms | Cost / query | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | | | | | | No runs yet: baseline is backlog item 1.7 |

## Parse audits (debugging-ladder rung 1)

| Date | Parser | Git branch | Docs / pages | Speed | Pass rate (cov ≥ 0.8) | Mean coverage | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-18 | pypdf 7.1 (baseline) | feat/pdf-parser | 84 / 12,013, 0 failed | 17.4 ms/page | 96.8% (183/189) | 0.985 | Lowest segment: `type:metrics-generated` 0.963 (financial statements). All 3 worst cases inspected: text was extracted; the *annotator* evidence has spaces stripped (`Cashandcashequivalents`), so the audit understates fidelity. Parsed text shows `\xa0` spaces, mid-word line breaks (`sha\nre`) and flattened table rows: targets for cleaning / table parsing (backlog 3.1). pypdf warned that `fontTools` is missing for some CFF fonts. |

## Chunk audits (debugging-ladder rung 1: does evidence fit in one chunk?)

| Date | Chunker (fingerprint) | Git branch | Chunks | Tokens p5 / p50 / p95 / max | Fits in one chunk (cov ≥ 0.8) | Mean best coverage | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-18 | recursive 400 / 0 overlap, cl100k (`a02d9252b7b267ae`) | feat/recursive-chunker | 26,264 over 84 docs (6.4 s) | 334 / 385 / 398 / 401 | 62.4% (vs 96.8% after parsing) | 0.843 | Split by evidence length: **≤ 400 tokens: 90.2% (83/92)** fit; **> 400 tokens: 36% (35/97)**, mostly whole financial statements that no 400-token chunk can hold. Lowest segment `type:metrics-generated` 0.733. Candidates: table-aware/structural chunking (3.1, 3.2), parent-child (3.7), size sweep (3.2). |
