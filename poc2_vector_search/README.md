# PoC 2 — Semantic Product Search with Chroma

A Streamlit app that performs **semantic search** over a synthetic product
catalog using a persistent [Chroma](https://docs.trychroma.com/) vector store.
Demonstrates the difference between exact keyword search and meaning-based
retrieval — a query like *"cosy office sweater"* will match a *"Soft
breathable hoodie perfect for the office"* even though the words don't
overlap.

## What this PoC shows

| Concept                | Where to look                                    |
| ---------------------- | ------------------------------------------------ |
| Vector embeddings      | `app.py` → `load_embedder()`, `index_catalog()`  |
| Persistent vector DB   | `app.py` → `get_chroma_collection()` (path `./db`) |
| Cosine similarity      | Chroma collection metadata `{"hnsw:space": "cosine"}` |
| Metadata filtering     | `collection.query(..., where={"category": ...})` |
| Index-once / query-many | First run builds the index; later runs reuse `./db` |

## Architecture

```
products.csv ──► MiniLM embeddings ──► Chroma collection ──► ./db (persisted)
                                                  ▲
            Streamlit search box ──► query embed ─┘
                                                  ▼
                                         Top-k + category filter
                                                  ▼
                                            Ranked results
```

## Setup

```bash
cd poc2_vector_search
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No API key required — embeddings run locally via `sentence-transformers`,
and similarity search is done by Chroma in-process.

## Run

```bash
streamlit run app.py
```

- The first launch generates `data/products.csv` (≈2,000 synthetic
  products) and builds the Chroma index under `./db/`. This takes a minute.
- Subsequent launches reuse the persisted index and start instantly.
- Use the sidebar to filter by **Category** and adjust **Top-k** /
  **min similarity threshold**.

## Try these queries

| Query                                    | Why it's interesting                                                             |
| ---------------------------------------- | -------------------------------------------------------------------------------- |
| `cosy office sweater`                    | Matches hoodies / sweaters described as "soft" or "breathable", not exact words. |
| `something to listen to music on the go` | Finds wireless headphones / portable speakers without the word "music".          |
| `gift for a runner`                      | Surfaces running shoes, fitness trackers, etc.                                   |
| `quiet bedroom decoration`               | Pulls candles, lamps, throw blankets from the **Home** category.                 |

## Tuning knobs

- **Embedding model** (`EMBED_MODEL_NAME` in `app.py`) — swap for
  `BAAI/bge-small-en-v1.5` or `intfloat/e5-small-v2` to compare quality.
- **Distance metric** — change `hnsw:space` to `l2` or `ip` in
  `get_chroma_collection()`.
- **Catalog size** — edit `n` in `sample_data.get_products(n=...)`.
- **Re-index** — delete the `db/` and `data/` folders and restart.

## Files

- [app.py](app.py) — the Streamlit application.
- [sample_data.py](sample_data.py) — synthetic catalog generator.
- [requirements.txt](requirements.txt)
- [.gitignore](.gitignore) — excludes `db/`, `data/`, `.venv/`.
