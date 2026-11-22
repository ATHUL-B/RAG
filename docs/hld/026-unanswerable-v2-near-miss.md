# HLD 026: Near-miss unanswerable questions (backlog 2.3, v2)

## Goal

HLD 017's v1 set (26 questions) asked about companies with no filings, companies not in
the catalog, far-away years, undisclosed trivia and speculation. Every question was
*plainly* out of scope, so a cautious model scored 100%: **"a limitation of the test, not
proof of robustness."** v2 tests *reading*. The company and the filing exist, and the
question looks answerable, but the corpus does not support an answer (study guide
Part 11: refuse when the evidence is insufficient; Part 13: unanswerable as a segment).

## The set: `evals/unanswerable-v2.jsonl` (24 questions)

| Trap | n | Built how | Examples |
| --- | --- | --- | --- |
| `false-premise` | 8 | asserts a direction the filing contradicts | "Why did Costco's net sales decline in fiscal 2021?" (+18%); "Why did Microsoft report a net loss in FY2023?" (net income $72,361M); "Why did Verizon suspend its dividend in 2022?" (raised 2.0%) |
| `not-reported` | 8 | company and year covered, figure never broken out | Xbox console units; Netflix revenue from Japan; Boeing 787 program revenue; Sprite revenue; Walmart+ members in FY2020 (the program didn't exist yet) |
| `adjacent-period` | 8 | the neighbouring period is in the corpus, the asked one isn't, and no comparative column contains it | Q3 2023 when only the Q2 10-Q exists (3M, Pfizer, JPMorgan, MGM, Best Buy); **guidance traps**: Ulta FY2023 and Amcor FY2024 exist only as *outlooks*; Foot Locker is in the corpus but its two 8-Ks have no financials |

**Every question was verified against the parsed corpus**, by text search of that company's
filings. `metadata.verified_in` records where (e.g. `COSTCO_2021_10K p23-24`, or "no match"),
and a data-contract test requires it. The adjacent-quarter filings mention the next
quarter only as forward-looking statements ("expected to close in the third quarter"),
never as results.

## Results (llama3.1:8b, the same two retrieval configs as v1)

| | v1 default | v1 best | **v2 default** | **v2 best** |
| --- | --- | --- | --- | --- |
| refused correctly | 25/25 (1 error) | 24/24 (2 errors) | **23/24** | **17/24** |
| hallucinated | 0 | 0 | **1** | **3** |
| errors (schema) | 1 | 2 | 0 | 4 (2 of them were answering) |
| refusal recall (decided only) | 1.00 | 1.00 | **0.96** | **0.85** |
| refusals that explain what's missing (> 4 words) | 4/25 | 2/24 | 2/23 | 3/17 |

### What the model got wrong

| Question | Answer | Failure |
| --- | --- | --- |
| Foot Locker FY2021 revenue (both runs) | "$9 billion" | **outside knowledge**: there are no Foot Locker financials in the corpus at all |
| Pfizer Q3 2023 revenue (best) | $12,734 million | **period substitution**: the Q2 2023 figure reported as Q3 |
| Ulta *actual* fiscal 2023 net sales (best) | $10,208,580 (thousand) | **period substitution**: fiscal 2022's actual sales (the guidance itself was not used) |
| Best Buy, MGM Q3 (best; schema errors) | working labels Q2 figures "(Q3 2023)" | the same substitution, cut off by a repetition loop before the answer |

## What it shows

1. **v2 finds real failures where v1 found none.** Near-miss questions get hallucinated
   answers that look plausible: a real-looking revenue figure, the right company, the
   wrong period.
2. **Better retrieval makes this worse.** Best retrieval puts the *neighbouring* period's
   tables in the prompt, and the model relabels them: 3 hallucinations plus 2 attempted,
   vs 1 with default retrieval. This is the flip side of HLD 017, where better retrieval
   *lowered* over-refusal. **Refusal recall and over-refusal must be tracked together.**
3. **False premises are refused, not corrected.** All 16 false-premise answers refused.
   That's safe, but only 2 said *why* (e.g. "Microsoft did not report a net loss in fiscal
   year 2023"). The rest say "None", "Unknown", "No answer".
4. **Refusals are mostly bare in both sets** (2-4 per run explain anything). A user can't
   tell "the corpus ends at Q2" from "the filing doesn't break this out".

## Decision and next step

- **v2 is kept as a standing evaluation set.** Every generation change is run on dev
  (over-refusal), v1 and v2 (refusal recall).
- **Next (one change):** a prompt rule, then measured on all three sets:
  - "Answer only for the exact period and entity asked; never substitute another period";
  - "If the premise contradicts the sources, say so";
  - "When refusing, name what is missing (company, period or figure)".

  Expected effect: fewer v2 hallucinations and more explained refusals. The risk is more
  over-refusal on dev, which is exactly what the paired runs will show.
