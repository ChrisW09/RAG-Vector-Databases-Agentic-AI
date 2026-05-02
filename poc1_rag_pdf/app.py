"""Streamlit RAG app over a user-uploaded PDF.

Pipeline:
  1. Extract text with pypdf.
  2. Split into ~500-token chunks with 50-token overlap.
  3. Embed with sentence-transformers all-MiniLM-L6-v2.
  4. Store normalised vectors in an in-memory FAISS index (cosine via IP).
  5. On query: embed -> top-k retrieve -> build cited prompt -> Anthropic API.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from io import BytesIO
from typing import List

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Load environment variables from a local .env file (if present) so the
# OPENROUTER_API_KEY does not have to be exported manually each session.
load_dotenv()

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "anthropic/claude-sonnet-4"  # any OpenRouter model slug works
CHUNK_TOKENS = 500
CHUNK_OVERLAP = 50


@dataclass
class Chunk:
    text: str
    page: int  # 1-indexed page where the chunk starts


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_llm_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set.")
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


# ---------------------------------------------------------------------------
# PDF -> chunks
# ---------------------------------------------------------------------------
def extract_pages(pdf_bytes: bytes) -> List[str]:
    reader = PdfReader(BytesIO(pdf_bytes))
    return [(page.extract_text() or "") for page in reader.pages]


def _tokenize(text: str) -> List[str]:
    # Simple whitespace tokenisation; "tokens" here = whitespace-delimited words.
    return text.split()


def chunk_pages(pages: List[str], chunk_size: int = CHUNK_TOKENS,
                overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    chunks: List[Chunk] = []
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than chunk_size")

    for page_idx, page_text in enumerate(pages, start=1):
        tokens = _tokenize(page_text)
        if not tokens:
            continue
        for start in range(0, len(tokens), step):
            window = tokens[start:start + chunk_size]
            if not window:
                break
            chunks.append(Chunk(text=" ".join(window), page=page_idx))
            if start + chunk_size >= len(tokens):
                break
    return chunks


# ---------------------------------------------------------------------------
# Embeddings + FAISS index
# ---------------------------------------------------------------------------
def embed_texts(embedder: SentenceTransformer, texts: List[str]) -> np.ndarray:
    vectors = embedder.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.astype("float32")


def build_index(vectors: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


# ---------------------------------------------------------------------------
# Prompt + LLM call
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a careful assistant answering questions about a user-provided document. "
    "Answer ONLY using the provided context. If the answer is not contained in the "
    "context, say you don't know. Cite supporting chunks inline using bracketed "
    "numbers like [1], [2] that match the chunk numbering in the context."
)


def build_user_prompt(question: str, retrieved: List[Chunk]) -> str:
    context_blocks = []
    for i, c in enumerate(retrieved, start=1):
        context_blocks.append(f"[{i}] (page {c.page})\n{c.text}")
    context = "\n\n".join(context_blocks)
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above and cite chunks as [1], [2], etc."
    )


def call_llm(client: OpenAI, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="PDF RAG", page_icon="📄", layout="wide")
    st.title("📄 RAG over your PDF")

    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Top-k chunks", min_value=1, max_value=10, value=4)
        st.caption(f"Embedding model: `{EMBED_MODEL_NAME}`")
        st.caption(f"LLM: `{LLM_MODEL}` via OpenRouter")
        if not os.environ.get("OPENROUTER_API_KEY"):
            st.warning("Set OPENROUTER_API_KEY in your environment or .env file.")

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded is None:
        st.info("Upload a PDF to get started.")
        return

    file_bytes = uploaded.getvalue()
    file_key = (uploaded.name, len(file_bytes))

    if st.session_state.get("file_key") != file_key:
        with st.spinner("Extracting and indexing PDF…"):
            pages = extract_pages(file_bytes)
            chunks = chunk_pages(pages)
            if not chunks:
                st.error("Could not extract any text from this PDF.")
                return
            embedder = load_embedder()
            vectors = embed_texts(embedder, [c.text for c in chunks])
            index = build_index(vectors)

        st.session_state.file_key = file_key
        st.session_state.chunks = chunks
        st.session_state.index = index

    st.success(
        f"Indexed **{len(st.session_state.chunks)}** chunks from "
        f"`{uploaded.name}`."
    )

    question = st.text_input("Ask a question about the PDF")
    if not question:
        return

    if st.button("Answer", type="primary"):
        embedder = load_embedder()
        q_vec = embed_texts(embedder, [question])
        scores, ids = st.session_state.index.search(q_vec, top_k)
        retrieved = [st.session_state.chunks[i] for i in ids[0] if i != -1]

        try:
            client = get_llm_client()
            with st.spinner("Asking the LLM…"):
                answer = call_llm(
                    client, build_user_prompt(question, retrieved)
                )
        except Exception as e:  # noqa: BLE001
            st.error(f"LLM call failed: {e}")
            return

        st.subheader("Answer")
        st.markdown(answer)

        st.subheader("Retrieved context")
        for i, (chunk, score) in enumerate(zip(retrieved, scores[0]), start=1):
            with st.expander(f"[{i}] page {chunk.page} — score {score:.3f}"):
                st.write(chunk.text)


if __name__ == "__main__":
    main()
