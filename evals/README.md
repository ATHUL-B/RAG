# Golden sets

- FinanceBench's 150 questions are loaded directly from `data/raw/financebench`
  via `rag.datasets.financebench.FinanceBenchDataset.golden_items()`.
- **Our own** questions live here as JSONL in the `GoldenItem` schema
  (`rag.domain.GoldenItem`), versioned in git, e.g. `custom_v1.jsonl`.

What to add (study guide, Part 13):
- version traps (same company, different year), exact-identifier lookups,
  multi-part questions, vague follow-ups,
- **unanswerable** questions (`evidence: []`): the right behaviour is to refuse.

Tags follow `<dimension>:<value>`, e.g. `type:version-trap`, `evidence:multi`.
Every failure found later becomes a new item here.
