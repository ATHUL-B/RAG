# HLD 024: An evaluation regression gate in CI (backlog 4.3)

## Goal

CLAUDE.md §2: *"no retrieval/generation change without an eval run."* Until now, nothing
enforced it: `make check` runs unit tests only, so a PR could quietly make retrieval worse.
Study guide Part 14: *"version, never overwrite ... evaluate on the golden set"*, and *"CI
evaluation"* as a guard against a "sudden recall drop". Make every PR prove that retrieval
didn't regress, on a runner with no GPU and no LLM.

## Non-goals

- LLM stages (planner, reranker, generation): they need Ollama and minutes per question.
  The gate checks the **default** retrieval (`planner-rules`), which every other config
  builds on.
- Measuring improvements: the gate reports gains, but deciding them is still the full dev
  run (and test, rarely).

## The key idea: a small corpus that gives *identical* results

A full evaluation needs 84 PDFs, about 31k chunks, and ~20 minutes of embedding. But the
default retrieval **filters by company**: a question only ever searches its own company's
filings, unless the filter relaxes. So a corpus holding only a few companies' filings
gives those companies' questions **exactly the same rankings** as the full corpus.

- **Gate set** (`evals/gate-v1.json`): 7 dev companies (Ulta Beauty, Amazon, Activision
  Blizzard, Johnson & Johnson, Microsoft, American Express, Block), **14 filings, 32
  questions**. They mix outcomes: 16 hits, 15 misses, 1 partial.
  - Companies whose questions relax the company filter (Pfizer) were left out.
  - **Dev questions only**: a check that runs on every PR would otherwise leak the
    held-out test split.
- **Verified:** built on CPU, the gate reproduced the full-corpus run (built on the Mac's
  GPU) for **all 32 questions**, to the decimal. Mean recall@5 is 0.516.

## Design

- **`evaluation/gate.py`** (pure, tested):
  - `GateSpec`: the config, companies, filings, per-question expected recall@5, and the
    tolerances;
  - `check_gate(expected, actual)`: passes unless more than `max_regressions` (1) questions
    lose recall, mean recall@5 falls by more than `max_mean_drop` (0.05), or an expected
    question wasn't evaluated. **Gains never fail**; they're reported.
  - One allowed flip absorbs a near-tie ranking that changes across CPU and GPU arithmetic.
- **`scripts/eval_gate.py`**:
  1. writes a gate config (the spec's config, with `data_dir: data/gate`);
  2. downloads the metadata and the 14 PDFs (`--copy-pdfs-from` reuses local ones);
  3. runs the **normal pipeline scripts** (parse, chunk, index) as subprocesses, so the gate
     exercises exactly what a user runs;
  4. evaluates the gate questions and compares them with the expectations; exit 1 on failure.

  `--update` rewrites the expectations, deliberately.
- **CI:** a separate `eval-gate` job runs `make gate` (`--device cpu`), caching the
  embedding model and the gate PDFs.
- **Changing the expectations** is a deliberate act:
  - a PR that intentionally moves gate numbers runs `make gate` locally (or
    `eval_gate.py --update`) and commits the new `evals/gate-v1.json`;
  - the diff shows exactly which questions changed;
  - it is reviewed like code.

## Verification

| Check | Result |
| --- | --- |
| Gate on the current code (CPU) | **PASSED**: 32 questions, mean recall@5 0.516 = expected |
| Gate vs full-corpus run, per question | 32/32 identical |
| Gate on a weaker config (`pages-context`, no query planner) | **FAILED**: mean 0.469; `00651` 1.00 → 0.00, `07966` 1.00 → 0.50; exit 1 |
| Local build time (Mac, CPU) | ~2 min (parse, chunk, embed ~3.8k chunks, evaluate) |

## Risks

- **CPU/GPU numeric differences** could flip a near-tie on another machine. This is
  absorbed by `max_regressions: 1`, and so far 0 of 32 differ.
- **Network:** the job downloads PDFs from the FinanceBench repository and the model from
  Hugging Face. Both are cached; a download failure fails the job loudly.
- **Coverage:** 32 questions from 7 companies can't catch everything (e.g. a regression only
  in relaxed-filter questions). It's a tripwire, not a replacement for the full dev run.

## Test plan

Unit tests for `check_gate`:

- identical results pass;
- one flip passes and two fail;
- a mean-drop failure within the flip allowance;
- gains reported and never failing;
- missing questions failing;
- empty expectations being a config error;
- strict spec parsing.

The script was run end to end both ways: pass on current code, fail on a weaker config.
