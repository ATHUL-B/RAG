# HLD 016: LLM judges for correctness and faithfulness (backlog 2.2)

## Goal

`numeric_match` scores only answers whose reference is a plain number: 30 of 62 answered
dev questions in HLD 015's default run. Text answers ("what drove…", "is X improving…")
were unscored. This adds judges (study guide Part 13), so every answer gets a verdict.

## Design

- **Two judges, one job each** (`evaluation/judges.py`, `LLMJudge`):
  - **Correctness** sees question + reference + answer, *not* the sources: "same fact or
    conclusion as the reference?" (rounding and scale rules are in the rubric).
  - **Faithfulness** sees the sources + answer + working, *not* the reference: "is every
    claim supported by, or correctly computed from, the sources?" It needs no labels, so
    it catches hallucination even where FinanceBench's page labels are incomplete.
- **Guards** (CLAUDE.md §2, Part 13):
  - Each answer is judged alone against a rubric, so there is no position bias.
  - "Length, style… do not matter", against verbosity bias.
  - Reason **before** verdict, in a schema with `maxLength` on the reason.
  - Answer and sources are treated as data, not instructions.
  - **A different family judges:** `qwen2.5:7b` (Alibaba) judges `llama3.1:8b` (Meta),
    both local.
- **Offline over a finished run** (`evaluation/judge_runner.py`,
  `scripts/judge_generations.py`). Sources are rebuilt from `prompt_chunk_ids` with the
  generator's own assembler, so the judge sees the exact `[S#]` labels. Errors and
  refusals are skipped and counted. Verdicts are cached per (kind, model, rubric,
  content).
- **Summary:**
  - pass rate per judge;
  - pass rate per evidence reach (the debugging ladder, now covering text answers);
  - **agreement with `numeric_match`** on numeric questions, a free, automatic
    calibration of the correctness judge.
- **Human calibration** (`evaluation/calibration.py`, `scripts/judge_calibration.py`):
  - `sample` picks 50 answers stratified by run × reach × numeric/text with a fixed seed.
  - `label` is an interactive, resumable labeller that **hides the judge's verdict**
    (blind labelling).
  - `report` gives agreement and **Cohen's κ** per judge against a bar fixed in advance
    (`configs/judges.yaml: min_kappa: 0.6`).

  κ, not raw agreement: a judge that always says "correct" on 80%-correct data agrees 80%
  but has κ = 0 (a hand-computed test).

## Results (dev, both HLD 015 runs, `qwen2.5:7b`)

| Run (retrieval) | judged | correct | faithful | correct when in prompt | faithful when not retrieved | judge vs `numeric_match` |
| --- | --- | --- | --- | --- | --- | --- |
| `generation-ollama` (`planner-rules`) | 62 | 0.35 | 0.74 | 0.47 (n=38) | 0.58 (n=24) | **0.96** (n=25) |
| `generation-ollama-best` (LLM planner + reranker) | 76 | 0.32 | 0.72 | 0.34 (n=65) | 0.73 (n=11) | **0.90** (n=31) |

Paired over all 96 questions (refusals and errors count as failures):

| | default | best | Δ (95% CI) |
| --- | --- | --- | --- |
| judged correct | 0.229 | 0.250 | +0.021 [-0.073, +0.115], tie |
| judged faithful | 0.479 | 0.573 | +0.094 [-0.010, +0.208], borderline |

- **It confirms HLD 015 on text answers too.** Better retrieval makes answers *more
  grounded*, but not *more correct*. The 8B generator is the bottleneck.
- **The ladder pattern is visible:** with default retrieval, answers given *without* the
  evidence are rarely correct (0.17) and much less faithful (0.58 vs 0.84). The
  faithfulness judge catches the model making things up.

## Calibration status: provisional

- **Automatic:** the correctness judge agrees with the exact `numeric_match` on 90-96% of
  numeric answers. On plain numbers it is reliable.
- **Human:** 4 of 50 labels. κ = 0 on both judges, but that comes from the sample: all 4
  human labels are "incorrect / unfaithful", and κ is undefined-to-zero when a rater
  uses one label. The disagreements are informative:
  1. **The judge is lenient on derived numbers.** JnJ "US growth 2.2 points higher"
     (truth: 3.0 − (−0.6) = 3.6). The judge missed the sign of the decline and passed it
     on *both* judges.
  2. **The rubric is ambiguous on answering a different quantity.** Pfizer "$700 million"
     is the *total* separation cost in the sources; the question asks the *remaining*
     cost. The judge said faithful (the figure is in the sources); the human said
     unfaithful (it misrepresents them).

**Decision:** the judges are built and run, but **their numbers are indicative, not decision
grade**. No keep/reject decision rests on judge scores alone until κ ≥ 0.6 on at least
~30 labels. The tooling resumes where labelling stopped.

## Next

- Rubric v2, measured against labels:
  - "derived numbers must be recomputed from the sources' values, signs included";
  - "answering a different quantity than asked is unfaithful / incorrect".
- A stronger judge (Claude, a different family again) once a key is available. Compare
  it with qwen on the same labels.
- More labels: even 15-20 mixed ones give a usable κ.
