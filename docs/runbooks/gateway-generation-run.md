# Runbook: generation run through your LLM gateway

**For:** someone with access to an OpenAI-compatible LLM gateway (Bifrost, LiteLLM, …) and
a strong model behind it (e.g. Claude Sonnet / Opus).
**Why:** in our runs the local 8B model is now the bottleneck: the evidence is in the
prompt but the answer is wrong (HLD 015-017). Your run answers *"how much does a stronger
generator improve answers, with the same retrieval?"*
**Your effort:** about 30-45 minutes, mostly waiting. You send back two run folders. We
do all the judging and analysis.

Please follow the steps in order. **Steps 5 and 6 are stop points:** if they fail, send
us what they produced instead of continuing. A broken setup must never turn into a full
run whose numbers look real.

---

## 0. Before you start

- **Permission.** Check that you may send this data through your gateway. Every request
  contains a question plus 5 excerpts from public SEC filings (FinanceBench, CC BY-NC
  4.0). No personal or company-internal data.
- **Cost.** About 96 + 26 = 122 requests of about 2.5k input and up to 2k output tokens.
  Step 4 prints an estimate for your model's price before anything is sent.
- **Never commit** your `.env`, gateway URL, key, or your local edits to
  `configs/generation-gateway.yaml`.

## 1. Set up the repository

Needs macOS or Linux with [uv](https://docs.astral.sh/uv/) installed.

```bash
git clone https://github.com/ATHUL-B/RAG.git && cd RAG
make install      # Python 3.12 + dependencies
make data         # downloads 84 public PDFs (~160 MB), parses, chunks, embeds (~10-20 min, CPU)
git rev-parse --short HEAD   # note this SHA: send it back with the results
```

`make data` builds the same chunks and index as ours (chunking is deterministic), so your
prompts contain the same sources ours did.

## 2. Configure the gateway (secrets go in `.env`, never in configs)

```bash
cp .env.example .env
```

Edit `.env`:

```
LLM_GATEWAY_URL=https://<your-gateway-host>/v1     # base URL, ending in /v1
LLM_GATEWAY_KEY=<your key or virtual key>
```

Edit three lines in `configs/generation-gateway.yaml` (local only, don't commit):

| Line | Set to |
| --- | --- |
| `model_id: "<gateway-model-id>"` | the model name your gateway expects (exactly as you would send it) |
| `gateway_key_header: Authorization` | keep `Authorization` for a standard Bearer key; for Bifrost virtual keys use `x-bf-vk` |
| `pricing: {input_per_mtok: …, output_per_mtok: …}` | your model's USD price per million tokens (for the cost estimate) |

If the model rejects `temperature` (for example, Claude Sonnet 5.5 on Bedrock), add
`gateway_temperature: null` under `generation`; otherwise the default is deterministic `0`.

Do **not** change any other section. The retrieval settings must stay identical to our
runs, so the comparison isolates the generator.

## 3. Check the tests pass

```bash
make check        # lint, types, ~530 unit tests; no network, ~10 s
```

## 4. Dry run: cost estimate, nothing is sent

```bash
uv run python scripts/evaluate_generation.py --config configs/generation-gateway.yaml --limit 0 --dry-run
```

It prints `96 questions … estimated cost ≤ $X` and one example prompt. If $X is more than
you are allowed to spend, stop and tell us.

## 5. Smoke test: 3 questions (STOP POINT)

```bash
uv run python scripts/evaluate_generation.py --config configs/generation-gateway.yaml --limit 3 --budget-usd 1
```

**Pass** only if *all* of these hold:

- the summary line says **`errors=0`**;
- the ladder table shows outcomes such as `correct`, `wrong` or `refused`, and no `error`;
- each answer in `runs/generation-gateway-dev-<timestamp>/generations.jsonl` has a
  non-empty `answer` **and** a non-empty `working` (open the file and look).

**If it fails, stop.** Zip that run folder and send it with the printed output. The
`raw_output` and `error` fields tell us exactly what the gateway returned. Common causes:

| Symptom | Likely cause |
| --- | --- |
| `HTTP 401/403` | wrong key, or wrong `gateway_key_header` |
| `HTTP 404` | `LLM_GATEWAY_URL` must end in `/v1`, or the model name is wrong |
| `HTTP 400 … response_format` | the gateway or model doesn't support JSON-schema output; tell us, we'll adapt |
| `gateway returned tool_calls instead of content` | the gateway rewrote the request into a tool call; tell us |
| `errors` with kind `empty_answer` / `bare_boolean` / `markup_leak` | the reply was mangled in transit: the same failure class as an earlier gateway run, please send the folder |

## 6. Full runs

Both runs need a budget above the step 4 estimate (the guard refuses otherwise):

```bash
# 96 dev questions (same questions and retrieval as our local runs)
uv run python scripts/evaluate_generation.py --config configs/generation-gateway.yaml --limit 0 --budget-usd <limit>

# 26 questions that must be refused (no answer exists in the corpus)
uv run python scripts/evaluate_generation.py --config configs/generation-gateway.yaml \
    --golden evals/unanswerable-v1.jsonl --limit 0 --budget-usd <limit>
```

**Sanity check (STOP POINT):** in each run's printed summary, `errors` should be 0 or
very few (≤ 3). Many errors mean something broke mid-run (rate limits, timeouts); send
the folder rather than re-running blindly.

## 7. Send back

```bash
zip -r gateway-runs.zip runs/generation-gateway-dev-* runs/generation-gateway-unanswerable-v1-*
```

Send `gateway-runs.zip` plus:

1. the git SHA from step 1,
2. the exact `model_id` you used,
3. anything unusual (rate limits, retries, warnings).

The folders contain the questions, answers, token counts and the config. The config
records only the *names* of the environment variables (`LLM_GATEWAY_URL`,
`LLM_GATEWAY_KEY`), never their values. Check `summary.json` yourself if in doubt.

## What we do with it (no action needed)

- **Judge offline** with our local judges (HLD 016). They rebuild each prompt's sources
  from the shared index, so you don't need Ollama.
- **Compare with our llama3.1:8b runs**, question by question:
  - numeric accuracy;
  - judged correctness and faithfulness;
  - the debugging ladder;
  - refusal recall and over-refusal;
  - citations;
  - latency and cost per question.
- Record it in `experiments/LOG.md` with your model name, and credit you in the PR.
