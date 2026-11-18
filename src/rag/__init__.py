"""rag-lab: a production-grade RAG system built step by step.

Package map (mirrors the pipeline in the study guide, Part 2):

- ``domain``        plain data shared by every stage (Document, Chunk, ...)
- ``ingestion``     offline: parse, clean, enrich, chunk         (Parts 3-4)
- ``indexing``      offline: embed, vector store, keyword index   (Part 5)
- ``retrieval``     online: rewrite, filter, retrieve, fuse, rerank (Parts 6-10)
- ``generation``    online: assemble context, call the LLM        (Part 11)
- ``evaluation``    golden sets, metrics, judges                  (Part 13)
- ``observability`` timing, tracing, logging                      (Part 14)
- ``datasets``      adapters that turn public datasets into domain objects
"""

__version__ = "0.1.0"
