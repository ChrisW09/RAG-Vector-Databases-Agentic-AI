"""Streamlit semantic product search backed by a persistent Chroma collection.

Pipeline:
  1. Generate / load a synthetic product catalog (data/products.csv).
  2. Embed each product description with sentence-transformers all-MiniLM-L6-v2.
  3. Store vectors + metadata (category, price, title) in a Chroma collection
     persisted under ./db.
  4. On each query: embed the query, run a top-k similarity search with an
     optional category filter, and display ranked results.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import chromadb
import pandas as pd
import streamlit as st
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

import sample_data

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "products"
DB_PATH = "./db"
CSV_PATH = Path("data/products.csv")


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_chroma_collection() -> chromadb.api.models.Collection.Collection:
    client = chromadb.PersistentClient(
        path=DB_PATH, settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


@st.cache_data(show_spinner="Generating sample catalog…")
def load_catalog() -> pd.DataFrame:
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH)
    return sample_data.get_products()


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
def index_catalog(
    collection, embedder: SentenceTransformer, df: pd.DataFrame, batch_size: int = 256
) -> None:
    ids = df["id"].astype(str).tolist()
    docs = df["description"].astype(str).tolist()
    metadatas = [
        {
            "title": str(row.title),
            "category": str(row.category),
            "price": float(row.price),
        }
        for row in df.itertuples(index=False)
    ]

    progress = st.progress(0.0, text="Embedding catalog…")
    total = len(docs)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        vectors = embedder.encode(
            docs[start:end],
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        collection.add(
            ids=ids[start:end],
            documents=docs[start:end],
            embeddings=vectors,
            metadatas=metadatas[start:end],
        )
        progress.progress(end / total, text=f"Embedded {end}/{total}")
    progress.empty()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Semantic Product Search", page_icon="🔎", layout="wide")
    st.title("🔎 Semantic Product Search")
    st.caption(
        "Find catalog items by **meaning**, not exact keywords. "
        "Embeddings via `all-MiniLM-L6-v2`, vector store: Chroma (persistent)."
    )

    df = load_catalog()
    embedder = load_embedder()
    collection = get_chroma_collection()

    # Build the index on first run (or if the catalog grew).
    if collection.count() < len(df):
        with st.status("Building Chroma index (first run only)…", expanded=False):
            # Reset to keep ids consistent if the CSV changed.
            if collection.count() > 0:
                collection.delete(ids=collection.get()["ids"])
            index_catalog(collection, embedder, df)

    categories: List[str] = ["All"] + sorted(df["category"].unique().tolist())

    with st.sidebar:
        st.header("Filters")
        category = st.selectbox("Category", categories, index=0)
        top_k = st.slider("Top-k results", 1, 25, 10)
        min_score = st.slider(
            "Min similarity (cosine)", 0.0, 1.0, 0.0, step=0.05,
            help="Hide results below this score.",
        )
        st.caption(f"Indexed items: **{collection.count()}**")

    query = st.text_input("Search", placeholder="e.g. cosy office sweater")
    if not query:
        st.info("Type a natural-language description of what you're looking for.")
        return

    q_vec = embedder.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).tolist()

    where = None if category == "All" else {"category": category}
    results = collection.query(
        query_embeddings=q_vec, n_results=top_k, where=where
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    rows = []
    for doc, meta, dist in zip(docs, metas, distances):
        # Chroma returns cosine distance in [0, 2]; similarity = 1 - distance.
        similarity = 1.0 - float(dist)
        if similarity < min_score:
            continue
        rows.append(
            {
                "title": meta.get("title", ""),
                "category": meta.get("category", ""),
                "price": meta.get("price", float("nan")),
                "similarity": round(similarity, 3),
                "description": doc,
            }
        )

    if not rows:
        st.warning("No results above the similarity threshold. Try lowering it.")
        return

    st.subheader(f"Top {len(rows)} results")
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        column_config={
            "price": st.column_config.NumberColumn(format="$%.2f"),
            "similarity": st.column_config.ProgressColumn(
                "similarity", min_value=0.0, max_value=1.0, format="%.3f"
            ),
        },
    )


if __name__ == "__main__":
    main()
