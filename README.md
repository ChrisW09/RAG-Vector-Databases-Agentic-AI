# RAG-Vector-Databases-Agentic-AI

A collection of small proof-of-concept projects exploring Retrieval-Augmented
Generation (RAG), vector databases, and Agentic AI patterns.


Three small, self-contained prototypes that map directly to the lecture
*"Building with Large Language Models — LLMs, RAG, Vector Databases &
Agentic AI"* (Prof. Dr. Christoph Weisser, HSBI / IBSEN, April 2026):

| # | PoC | What it shows | Stack |
|---|-----|---------------|-------|
| 1 | [poc1_rag_pdf/](poc1_rag_pdf/) | **RAG** over a user-uploaded PDF with grounded citations | Streamlit, pypdf, sentence-transformers, FAISS, OpenRouter |
| 2 | [poc2_vector_search/](poc2_vector_search/) | **Persistent vector DB** for semantic product search with metadata filtering | Streamlit, sentence-transformers, Chroma |
| 3 | [poc3_react_agent/](poc3_react_agent/) | **Agentic AI**: a ReAct loop with three tools (search / calculator / final_answer) | Python CLI, OpenRouter |

Each PoC has its own `README.md` with setup, run, and architecture notes.

## The three levels of LLM usage

```
   LLM ──Extend──► LLM + RAG ──Enable action──► LLM + RAG + Agent
   (text)         (grounded facts)              (autonomous tasks)
   PoC 1 / 2                                    PoC 3
```

## Requirements

- Python 3.10+
- An [OpenRouter](https://openrouter.ai/keys) API key for PoCs 1 and 3,
  stored in a local `.env` file (git-ignored). PoC 2 runs fully offline.

## Quick start

```bash
git clone https://github.com/ChrisW09/RAG-Vector-Databases-Agentic-AI.git
cd RAG-Vector-Databases-Agentic-AI/<poc-folder>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# follow the per-PoC README from here
```

## References

- Vaswani et al. (2017) — *Attention Is All You Need.*
- Lewis et al. (2020) — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.*
- Yao et al. (2023) — *ReAct: Synergizing Reasoning and Acting in Language Models.*
- Alammar & Grootendorst (2024) — *Hands-On Large Language Models*, O'Reilly.
