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
