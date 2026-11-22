# HLD 025: Cross-checking retrieval code on BEIR (backlog 4.4)

## Goal

CLAUDE.md §6: *"a wrong metric silently invalidates every experiment."* Our metric
functions are tested on hand-computed examples, and every FinanceBench number in this
project rests on four pieces of code: the embedder wrapper, the exact vector store, dense
retrieval and the metrics. None of that had been compared with an *independent* result.
BEIR (Thakur et al., 2021) is the standard retrieval benchmark, and MTEB publishes
nDCG@10 and recall@100 for our exact embedding model. **Run our code on the same data
with the same model, and the published numbers must come out.**

## Design

- **Reference numbers** (`evals/beir-published.json`): from the `BAAI/bge-small-en-v1.5`
  model card's MTEB results, at **revision `5c38ec7`**, the revision pinned in every
  config. They were read from the model card cached locally with the model.
- **Data:** MTEB's own copies (`huggingface.co/datasets/mteb/<name>`), pinned to a commit
  per dataset. That's the exact data the published scores were computed on.
  - The original BEIR mirror (`public.ukp.informatik.tu-darmstadt.de`) timed out on
    2026-11-22.
  - The `BeIR/*` Hugging Face copies are Parquet, which would have needed a new dependency.
- **Datasets:**
  - **SciFact** and **FiQA-2018**, as the backlog asked;
  - **NFCorpus**, which is small and has **graded** relevance (1 and 2), so it also tests
    the graded nDCG gain.
- **`datasets/beir.py`:**
  - `BeirDataset` reads the corpus, the judged queries and the graded qrels;
  - each document becomes one `Chunk` with `title + " " + text`, the text MTEB embeds;
  - `download_beir` fetches three files per dataset at a pinned revision, and rejects
    malformed revisions.
- **`scripts/evaluate_beir.py`:**
  - uses **our** pieces: the config's embedder (same model, revision and query prefix),
    `ExactVectorStore`, `DenseRetriever`, `ndcg_at_k` and `recall_at_k`;
  - caches corpus embeddings under `data/indexes/beir/`;
  - exits 1 if any metric differs from the published value by more than **±0.01**.

### A detail that mattered: empty documents

FiQA has **38 empty documents** (no title, no text), and one of them (`117276`) is judged
relevant for a test query. `Chunk` forbids empty text by design. MTEB embeds them as `""`.
We store them as `" "`: both tokenize to `[CLS][SEP]`, which was checked, so the
embedding is identical. Dropping them would have changed the corpus.

## Results (test split; Mac, MPS)

| Dataset | Docs | Queries | **nDCG@10** ours / published | **recall@100** ours / published | Verdict |
| --- | --- | --- | --- | --- | --- |
| SciFact | 5,183 | 300 | **0.7127 / 0.7127** (±0.0000) | **0.9417 / 0.9417** (±0.0000) | OK |
| NFCorpus (graded) | 3,633 | 323 | 0.3438 / 0.3431 (+0.0007) | 0.3110 / 0.3114 (−0.0003) | OK |
| FiQA-2018 | 57,638 | 648 | 0.4035 / 0.4034 (+0.0000) | 0.6963 / 0.6955 (+0.0008) | OK |

Embedding time: SciFact 47 s, NFCorpus 35 s, FiQA 307 s. Re-runs use the cached embeddings.

## What it shows

1. **Our retrieval stack and metrics are correct.** SciFact matches to four decimals. Our
   embedder wrapper (normalisation, query prefix, 512-token truncation), exact
   inner-product search, nDCG (linear gain, ideal from all judged documents) and recall
   give the same answers as MTEB's `pytrec_eval` pipeline.
2. **Graded nDCG is right.** NFCorpus, with grades 1 and 2, is within 0.001.
3. **The small differences are numeric, not logical.** Up to 0.0008 on NFCorpus and FiQA
   is what different hardware (Apple GPU vs MTEB's) does to near-tie rankings. SciFact,
   with fewer near-ties, is exact.
4. **Every FinanceBench number in `experiments/LOG.md` was computed with the same
   functions,** so this backs all of them, from the 0.273 baseline to 0.781.

## Not done

- **BM25 against the BEIR paper's BM25 numbers.** Our BM25 (own analyzer, k1 1.2, b 0.75) is
  not Elasticsearch's multi-field BM25, so the paper's numbers aren't a like-for-like
  reference. Dense retrieval is what every configuration builds on.
- Running this in CI: FiQA takes 5 minutes of GPU embedding. It's a one-off validation,
  re-run when the embedder, the store or a metric changes.
