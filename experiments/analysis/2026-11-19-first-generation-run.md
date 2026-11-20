# First full generation run: what it shows (2026-11-19)

**Run:** `generation-claude-20261119T031125Z`. All 150 FinanceBench questions, Claude
Sonnet 5.5 reached through an OpenAI-compatible proxy (code not merged), retrieval
`pages-context` (top 5 sources). Reported cost: $1.38.

**Verdict:** the headline numbers are **not usable as a quality measurement**. 78% of
questions came back without a usable answer because of a structured-output transport bug.
The answers that did arrive intact are strong. This run is therefore not recorded as a
row in `experiments/LOG.md`; this note records what we learned from it.

## 1. Output integrity: most answers were lost in transit

The proxy route asked for the answer as a forced, strict function call. The runner
counted 144 of 150 as valid. Reading every answer shows otherwise:

| What came back as the `answer` field | Questions |
| --- | --- |
| Empty string | 46 |
| Just `true` or `false` | 51 |
| Tool-call markup (`<parameter name="answerable">false`, `<br>`) | 14 |
| **Real answer text** | **33** |
| Prose with no schema at all (counted as errors) | 6 |

- The `answerable` flag is unreliable on broken outputs: 23 are marked *answered* with
  an empty answer. So the reported "answered 48.6%" and "refused 51%" are mostly noise.
- Broken outputs had a median of 89 output tokens, intact ones 358. The loss happens
  before our parser (the provider or proxy translation), but we cannot prove where,
  because the client stores only the parsed arguments, not the raw response.
- **Our own gap:** `generation/generator.py::_parse` accepted `""` and `"true"` as valid
  answers. A check that the output matches the schema *shape* is not a check that it
  is *usable*. Empty or trivial answers must be errors, so a run like this fails loudly
  instead of looking 96% valid.

## 2. Intact answers: generation is not the weak link

Of the 33 intact answers:

- **Evidence in the prompt (13):** all 13 answered, all cited the labelled evidence, and
  **all 6 numeric ones matched**: Nike current assets 16,525 ✓, MGM accounts payable
  302.6M ≈ 303 ✓, General Mills working capital ratio 0.68 ✓, Adobe operating income
  +65.4% ✓, Best Buy margin 2.8% ✓, Block operating cash flow 381.6M ≈ 382 ✓. All six
  numeric matches in the whole run come from here: the reported "numeric 12%" is really
  "6 of 6 that arrived".
- **Evidence not in the prompt, refused (14):** all correct refusals. The refusal path
  works.
- **Evidence not in the prompt, answered (6):** 4 are **correct anyway** (J&J US +3.0% vs
  international −0.6%; American Express tax rate 24.6% → 21.6%; Pfizer's Consumer
  Healthcare JV gain; AmEx geographic segments). The fact also appears on pages
  FinanceBench did not label. 2 are wrong: PepsiCo net capex $4.5B vs $4.60B reference,
  and Pfizer $175M vs a 77.78 reference. Both are near-miss figures from the wrong line.
- A few answers start with `<![CDATA[`, another transport artifact.

## 3. Where each question failed (debugging ladder)

The prompts equal the top 5 of the matching retrieval run
(`pages-context-20261118T100713Z`) for 144 of 150 questions, so retrieval behaved as
evaluated.

| Ladder step | Questions | Notes |
| --- | --- | --- |
| 1. Evidence not in the index | 0 | Parse and chunk audits |
| 2. Not in the top 5 (retrieval) | 79 | recall@5 = 0.427; 80 `not_retrieved` in the retrieval run |
| 2. Removed by the filter | 1 | |
| 5. In the prompt, output lost (transport) | ~50 | Not measurable until section 1 is fixed |
| 5. In the prompt, intact answer | 13 | 13/13 correct and cited |

## Follow-up: the same run replayed through the 2.1a checks

Replaying the saved answers offline through the new usability checks and ladder report
(no API calls) flags **114** answers rather than the 111 counted above. The difference is
the 3 `<![CDATA[`-wrapped answers: their text was correct, but the wrapper is a transport
leak, so they are now errors too. Together with the 6 schema breaks:

| Evidence reach | error | refused | correct | wrong | unscored | total |
| --- | --- | --- | --- | --- | --- | --- |
| not_retrieved | 62 | 13 | 0 | 2 | 4 | 81 |
| in_prompt | 58 | 0 | 4 | 0 | 7 | 69 |

Error kinds: 51 `bare_boolean`, 46 `empty_answer`, 17 `markup_leak`, 6 `schema_mismatch`.

## Lessons (feed into the backlog)

1. **Validate usability, not just schema:** reject empty or trivial answers and markup
   leaks. Store the raw provider response in the trace.
2. **The runner should report output integrity and the ladder step per question**
   (CLAUDE.md §2). This analysis was done by hand; it should be a report.
3. **Retrieval is the bottleneck.** Retrieval missed 79 of 150 questions, while the model
   was correct on every intact answer that had the evidence. Retrieval work (3.1 tables,
   3.6 query rewriting, 3.7 contextual chunks) matters more than prompt work.
4. **Page labels undercount correct answers** (4 correct "no evidence" answers). Answer-level
   judges (2.2) are needed before generation metrics can be trusted.
5. Before the next paid run: a 10-question pilot whose acceptance criterion is
   **0 broken outputs**.
