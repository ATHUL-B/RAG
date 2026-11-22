# data/ (contents gitignored)

```
raw/financebench/      downloaded by scripts/download_financebench.py
raw/beir/<name>/       downloaded by scripts/download_beir.py (HLD 025)
processed/<stage>/     artifacts written by each ingestion stage
indexes/               vector and keyword indexes
```

| Dataset | Source | Version | Licence |
| --- | --- | --- | --- |
| FinanceBench | https://github.com/patronus-ai/financebench | `main`, fetched 2026-11-18 | No LICENSE file upstream; cite arXiv:2311.11944; personal learning use only (ADR 0001) |
| BEIR SciFact, NFCorpus, FiQA-2018 | MTEB copies: https://huggingface.co/datasets/mteb/{scifact,nfcorpus,fiqa} (originals: https://github.com/beir-cellar/beir) | pinned commits in `evals/beir-published.json` (`mteb_revision`), fetched 2026-11-22 | Hugging Face cards list the licence as unknown; the original datasets' licences are listed in the BEIR repository. Used only to validate our code, never redistributed |
