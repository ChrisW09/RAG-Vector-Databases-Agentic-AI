# PDF RAG (PoC 1)

A single-file Streamlit app that does Retrieval-Augmented Generation over a
user-uploaded PDF using `sentence-transformers` (`all-MiniLM-L6-v2`), an
in-memory FAISS index (cosine similarity via inner product on normalised
embeddings), and an LLM served through [OpenRouter](https://openrouter.ai)
(default model: `anthropic/claude-sonnet-4`).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit .env and set OPENROUTER_API_KEY
```

The app loads `OPENROUTER_API_KEY` from a local `.env` file via
`python-dotenv` (falling back to a normal environment variable). `.env` is
git-ignored so the key never ends up in the repo. To switch models, edit
`LLM_MODEL` in `app.py` to any [OpenRouter model slug](https://openrouter.ai/models)
(e.g. `openai/gpt-4o-mini`, `meta-llama/llama-3.1-70b-instruct`).

## Run

```bash
streamlit run app.py
```

Upload a PDF, adjust **Top-k** in the sidebar (default 4), and ask questions.
The model is instructed to answer only from the retrieved context and to cite
chunks inline as `[1]`, `[2]`, ….
