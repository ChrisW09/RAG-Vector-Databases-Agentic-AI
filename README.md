# RAG-Vector-Databases-Agentic-AI

A collection of small proof-of-concept projects exploring Retrieval-Augmented
Generation (RAG), vector databases, and agentic AI patterns.

## Projects

- [poc1_rag_pdf/](poc1_rag_pdf/) — Streamlit app that performs RAG over a
  user-uploaded PDF using `sentence-transformers` (`all-MiniLM-L6-v2`),
  an in-memory FAISS index (cosine via inner product on normalised
  embeddings), and an LLM served through OpenRouter. See
  [poc1_rag_pdf/README.md](poc1_rag_pdf/README.md) for setup and run
  instructions.

## Requirements

- Python 3.10+
- An API key for the LLM provider used by each PoC (e.g. OpenRouter),
  stored in a local `.env` file (git-ignored).
