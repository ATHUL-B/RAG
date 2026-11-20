# HLD 015: First valid generation runs with a local model (backlog 2.1)

## Goal

The only full generation run so far was invalidated by a transport bug
(`experiments/analysis/2026-11-19-first-generation-run.md`). This is the first valid
answer-level measurement. It is free, using a local Ollama model (`llama3.1:8b`), and it
asks Phase 2's first question: **does better retrieval become better answers?**

## Changes before running

1. **Explicit Ollama context window** (`generation.ollama_context_tokens: 8192` →
   `num_ctx`). Ollama truncates prompts beyond its runtime default *from the front*,
   which is where the grounding rules are. Dev prompts measure at most ~2.4k tokens, so
   8192 has a wide margin.
2. **Output cap** (`max_tokens: 1024` → `num_predict`), so a looping answer stops.
3. **Shared preflight** (`app.local_models` / `check_local_models`): both eval scripts
   check every local model a config needs (planner, reranker, generator) before running.
4. **A `working` field before `answer` in the answer schema**, measured first on 8
   numeric dev questions with the evidence in the prompt:

   | Answer format | Numeric correct | Time / question |
   | --- | --- | --- |
   | `{answerable, answer}` (before) | 1/8 | 3.9 s |
   | **`{answerable, working, answer}`** | **5/8** | 3.5 s |
   | free text ending in `ANSWER:` | 2/8 | 13.1 s |

   Under a JSON grammar, a small model must commit to a number at once. The `working` field
   gives it room to list line items, values with citations and the arithmetic *first*.
   **Only `answer` is scored.** `working` is kept separately in the trace so its input
   numbers can never count as a match (a test enforces this). Citations are taken from
   both fields, because the inputs are what get cited.

## Runs (dev n=96, same generator, retrieval is the only difference)

| | `generation-ollama` (retrieval: `planner-rules`) | `generation-ollama-best` (retrieval: LLM planner + LLM reranker) |
| --- | --- | --- |
| evidence in prompt | 0.52 | **0.79** |
| answered | 0.69 | 0.83 |
| cites the evidence | 0.22 | **0.42** |
| has any citation (answered) | 0.51 | 0.55 |
| numeric correct | **13/30** | 10/32 |
| errors (broken output) | 6 | 4 |
| generation p50 / p95 | 10.9 s / 44.6 s | 10.9 s / 27.3 s |

Ladder (evidence reach × outcome):

```
generation-ollama        error refused correct wrong unscored total
  not_retrieved              3      19       0     2       22    46
  in_prompt                  3       9      13    10       15    50

generation-ollama-best
  not_retrieved              2       8       0     3        8    21
  in_prompt                  2       8      10    18       37    75
```

## What it shows

1. **Retrieval improvements reach the prompt.** The evidence is in the prompt for 79% of
   questions instead of 52%, and answers cite the evidence twice as often. The number of
   questions that fail at retrieval fell from 46 to 21.
2. **The 8B generator is now the bottleneck (ladder rung 5).** "In prompt, but wrong" is the
   largest failure group with the best retrieval (18). On numeric questions the paired
   outcomes are: 7 correct → wrong, 4 wrong/refused → correct, 6 correct in both,
   11 wrong in both. That is not a significant difference (7 vs 4 discordant), and better
   retrieval did **not** produce better numeric answers with this generator.
3. **The lost answers show how the small model fails** when the context changes:
   - Scale: `$381,603 million` for $382M; the source table is in thousands.
   - Ratio vs percent: `0.7` for 65.4%.
   - Picking the wrong table: `$7,444 million` for $5,409M.
   - Near-misses just outside the 1% tolerance: 0.82 for 0.83.

   A different set of 5 sources changes which table the model reads, so its answers are
   unstable even at temperature 0.
4. **Refusals behave sensibly.** Most refusals happen when the evidence was not
   retrieved (19 of 28 with the default retrieval), and only 2-3 questions are answered
   wrongly without evidence.
5. **Citations land in `working`, almost never in `answer`** (0-1 answers carry an `[S#]`).
   Counting both fields, about half the answers cite something. The rest are uncited
   claims, which CLAUDE.md treats as a bug.
6. **The remaining errors (4-6) are repetition loops** inside `working` that hit the
   1024-token cap mid-JSON (e.g. "[S1] and [S5] and [S2] and …"). Error traces record 0
   tokens because the usage is lost when parsing fails; a small fix for later.
7. **Most answers can't be scored yet:** 37 of the best run's in-prompt answers are text
   ("what drove…", "is X improving…"), and `numeric_match` only covers plain-number
   references. **Correctness needs the judges (2.2).**

## Decision

- 2.1 is done with a local model. `generation-ollama` and `generation-ollama-best` are the
  first valid answer-level baselines.
- The `working` field stays (it is part of the shared schema, so Claude will use it too).
- Next, in order:
  1. **2.2 judges**, so text answers count.
  2. **A stronger generator**: Claude (needs a key, about $1-6 per full run) or a larger
     local model, compared on the same best retrieval.
  3. A citation-in-answer check, and a `maxLength` on `working` for the Ollama path.
