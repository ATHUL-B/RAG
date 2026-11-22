# HLD 027: System prompt v2: exact periods, no guidance as results, explained refusals

## Goal

HLD 026's near-miss set found four failures of llama3.1:8b that v1 never showed:

| Failure | Example |
| --- | --- |
| **period substitution** | Pfizer's Q2 2023 revenue reported as Q3; Ulta's fiscal 2022 sales as fiscal 2023 |
| **outside knowledge** | Foot Locker "$9 billion", with no Foot Locker financials in the corpus |
| **guidance risk** | outlook-only years (Ulta FY2023, Amcor FY2024) |
| **bare refusals** | "None", "Unknown": 2-4 of ~25 refusals per run explain anything |

Study guide Part 11 ("generate with discipline") and HLD 017's open item: *a refusal must
say what is missing.* One change: the system prompt.

## Design

- **`SYSTEM_PROMPT_V2`** (`generation/context.py`) keeps every v1 rule (sources only, citations,
  `working` first, injection guard, concise) and adds four:
  - **rule 4, exact match:** answer only for the exact company, period (fiscal year or
    quarter) and figure asked. A Q2 figure is not a Q3 answer, and last year's result is not
    this year's;
  - **rule 5, guidance:** outlooks, forecasts and guidance are not results;
  - **rule 6, false premise:** if the question assumes something the sources contradict,
    set `answerable` false and say what the sources actually show;
  - **rule 7, explained refusal:** say exactly what is missing (company, period or figure),
    and which periods or figures the sources do cover.

  Rule 6 deliberately routes premise corrections through `answerable: false`. The runner
  scores a correction as a refusal, so "the sources show sales *rose* 18%" counts as correct
  on the unanswerable sets.
- **Versioned, not edited:** `generation.prompt_version: v1 | v2` (default v1), with
  `SYSTEM_PROMPTS` keyed by version. Every earlier run stays reproducible, and
  `generation-ollama-pv2.yaml` / `generation-ollama-best-pv2.yaml` differ from their bases
  only in this field (a config test enforces it).

## Evaluation plan

Same generator (llama3.1:8b) and the same cached retrieval, so only the prompt changes. Six
runs: {default, best retrieval} × {dev 96, unanswerable v1 26, v2 24}.

| Question | Metric | Risk |
| --- | --- | --- |
| Does it stop substitution and invention? | v2 refusal recall, hallucinations | — |
| Does it still refuse the easy traps? | v1 refusal recall | — |
| Are refusals informative? | refusals with > 4 words, and naming a period or figure | — |
| **Does it refuse answerable questions more?** | dev over-refusal; numeric correct (paired) | the trade-off HLD 026 warned about |

## Results

Six runs on 2026-11-22 (AC power, `caffeinate`), each paired question by question with the
v1-prompt run on the same cached retrieval. **Errors are counted separately**: the runner's
refusal recall leaves them out, which hides the main effect.

| Set | Retrieval | v1 prompt: refused / hallucinated / errors | v2 prompt | Explained refusals (> 4 words) v1 → v2 |
| --- | --- | --- | --- | --- |
| unanswerable v1 (26) | default | 25 / 0 / 1 | 22 / 0 / 4 | 4 → 1 |
| unanswerable v1 (26) | best | 24 / 0 / 2 | 23 / 0 / 3 | 2 → 1 |
| near-miss v2 (24) | default | 23 / 1 / 0 | 15 / 1 / **8** | 2 → 2 |
| near-miss v2 (24) | best | 17 / 3 / 4 | 13 / 2 / **9** | 3 → 2 |

| Dev (96) | Refusals | Over-refusal | Numeric correct | Paired numeric (v2 wins / losses) | Errors |
| --- | --- | --- | --- | --- | --- |
| default, v1 → v2 | 28 → 22 | 0.19 → **0.13** | 13/30 → 14/32 | 1 / 1 | 6 → 7 |
| best, v1 → v2 | 16 → 10 | 0.11 → **0.07** | 10/32 → 13/31 | 5 / 2 (n.s.) | 4 → 4 |

Generation p50 rose from ~7-8 s to 9-11 s: the model writes longer `working`.

### What happened

1. **The reasoning improved, but it landed in the wrong field.** Across the six v2 runs,
   15 `empty_answer` errors carry exactly the refusal we wanted, inside `working`, with
   `"answer": ""`. Examples: Pfizer *"the sources cover the second quarter of 2023, but not
   the third quarter"*; Adobe *"Creative Cloud revenue, which includes Photoshop, but not
   Photoshop separately"*; Ulta *"S2: fiscal 2023 outlook"*. The generator rejects an empty
   answer (HLD 015: a refusal must say something), so these count as errors. v1 had one
   such case.
2. **Repetition loops nearly doubled** (13 → 20 `stop_reason=length` across the six runs).
   The model loops on rule 7's own words, *"The sources do not contain enough information
   to answer."*, until it runs out of tokens. That phrase is what rule 7 itself says, so an
   8B model copies it.
3. **Outside knowledge is not fixed.** Foot Locker *"$9 billion"* appears in both v2
   near-miss runs. Rule 1 already forbade it, and repeating it in other words doesn't help.
   One best-retrieval period substitution became an error rather than a correct refusal.
4. **On answerable questions, v2 helps.** It refuses less (over-refusal down by a third in
   both configs), with no extra errors and numeric correctness flat to slightly up. None
   of the changes is significant at n ≈ 30.

## Decision

**Not adopted; v1 stays the default.** The goal metrics went the wrong way. Near-miss
refusals fell from 23 to 15 (default) and 17 to 13 (best), and explained refusals did not
rise. The cause is format compliance, not judgement: the explanations exist but sit in
`working`. `prompt_version: v2` and its configs stay in the repo, so the runs stay
reproducible.

**Next (v2.1, backlog 2.3c):** one change on top of v2. Say where the refusal text goes
(*"when `answerable` is false, put the explanation in `answer`"*), and replace the quotable
sentence in rule 7 with a description, so there is no phrase for the model to loop on.
Measure on the same six runs. Adopt it only if near-miss refusals plus errors beat v1 and
dev over-refusal does not rise. The outside-knowledge failure needs a different tool (an
evidence check on answers, or a stronger generator), not more prompt text.
