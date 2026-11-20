# HLD 017: Unanswerable questions and refusal accuracy (backlog 2.3)

## Goal

Every FinanceBench question is answerable, so until now we could measure only one side of
refusal: **over-refusal**, saying "not answerable" when the evidence was in the prompt. The
other side is **under-refusal**: answering (hallucinating) when the corpus holds no answer.
It needs questions that *should* be refused (study guide Parts 11 and 13). A good RAG system
needs both: refuse what it can't answer, answer what it can.

## Design

- **Our own set**, `evals/unanswerable-v1.jsonl`: 26 `GoldenItem`s with `evidence: []`,
  tagged `type:unanswerable` plus the *reason* (`unanswerable:<reason>`), so a failure can
  be traced to its trap:

  | Reason | n | Trap | Example |
  | --- | --- | --- | --- |
  | `company-in-catalog-no-filings` | 4 | Alias known, no filings: the company filter finds nothing, relaxes, and search runs over *other* companies | Apple FY2022 net sales |
  | `company-not-in-catalog` | 5 | No company filter at all | Tesla FY2022 automotive revenue |
  | `period-not-covered` | 9 | Future years, or years older than any five-year table (never a year a comparative column could contain) | Boeing FY2026 net income; Coca-Cola FY2005 |
  | `not-disclosed` | 5 | Company and period covered, figure never broken out | Costco rotisserie-chicken revenue |
  | `speculative` | 3 | Unknowable from any filing | PepsiCo share price on 31 Dec 2026 |

  The set was **verified against the corpus, not from memory**. The document catalog
  confirms which companies and periods exist. A text search of the parsed filings
  found only incidental near-misses, which make good traps: "Paris" inside "comparison",
  the Brazilian *real*, MGM's "CravingsBuffet" venue, exhibit dates and bond maturities.
- **Runner** (`evaluation/generation_runner.py`):
  - New evidence reach `no_evidence`.
  - New outcomes `correct_refusal` / `hallucinated`.
  - The summary gains `refusal`:
    - **refusal recall:** unanswerable questions refused, out of those with a usable
      decision;
    - **over-refusal rate:** answerable questions refused while their evidence was in the
      prompt.
- **`--golden PATH`** on `evaluate_generation.py` runs one of our sets
  (`datasets/golden_file.py`) instead of FinanceBench.

## Results (llama3.1:8b; both retrieval configs)

| | default retrieval | best retrieval |
| --- | --- | --- |
| **Refusal recall** (unanswerable set, n=26) | **25/25 = 1.00** (1 error) | **24/24 = 1.00** (2 errors) |
| Hallucinated | **0** | **0** |
| **Over-refusal** (dev, evidence in prompt; HLD 015 runs) | **9/47 = 0.19** | **8/73 = 0.11** |

- **All 3 "errors" are correct refusals with no explanation**
  (`{"answerable": false, "answer": ""}`). The integrity check rejects them on purpose,
  because a refusal must say what is missing. Counting the decision alone, both runs
  refuse 26/26.
- **Refusal explanations are minimal:** "No answer", "Not available", "Not answerable".
  The prompt asks the model to say what is missing; the 8B model doesn't. A user cannot
  tell "we have no Apple filings" from "the filings don't disclose this".

## What it shows

1. **The model is cautious, not reckless.** It never invented a figure for a missing
   company, a future year or an undisclosed number. This held even when the filter relaxed
   and it was shown other companies' financials.
2. **The real refusal problem is over-refusal.** 11-19% of answerable questions are
   refused *with the evidence in the prompt*. Better retrieval lowers it, because clearer
   evidence makes the model more willing to answer.
3. **v1 traps are too easy.** Every question is plainly out of scope, so a cautious model
   scores 100%. That is a limitation of the test, not proof of robustness.

## Next

- **`unanswerable-v2`, near-miss traps:** questions that look answerable but aren't. For
  example, a figure for a segment the company doesn't report, a false premise ("why did
  X's revenue fall 40%?" when it rose), or a metric for an adjacent year or quarter. These
  test *reading*, not just "is this company in the corpus".
- An integrity rule or prompt change for refusal explanations (must name the missing
  company, period or figure), measured on this set.
- Track over-refusal and refusal recall together on every generation change: lowering one
  usually raises the other.
