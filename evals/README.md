# Golden sets

- FinanceBench's 150 questions are loaded directly from `data/raw/financebench`
  via `rag.datasets.financebench.FinanceBenchDataset.golden_items()`.
- **Dev/test split:** `financebench-split.json` lists question IDs per split (96 dev / 54
  test). It is grouped by company (all of a company's questions are on one side) and
  balanced by question type x single/multi evidence. Made once by
  `scripts/make_split.py` (seed 0, fixed in advance). Tune on dev, confirm on test.
- **Our own** questions live here as JSONL in the `GoldenItem` schema
  (`rag.domain.GoldenItem`), versioned in git, e.g. `custom_v1.jsonl`.

What to add (study guide, Part 13):
- version traps (same company, different year), exact-identifier lookups,
  multi-part questions, vague follow-ups,
- **unanswerable** questions (`evidence: []`): the right behaviour is to refuse.

Tags follow `<dimension>:<value>`, e.g. `type:version-trap`, `evidence:multi`.
Every failure found later becomes a new item here.
