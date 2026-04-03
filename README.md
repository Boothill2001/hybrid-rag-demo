# 🔍 Hybrid Search RAG API

> Production-grade Retrieval-Augmented Generation system with **Hybrid Search** (Dense Vector + BM25 Sparse), **Reciprocal Rank Fusion**, and **Groq LLM** generation.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green) ![ChromaDB](https://img.shields.io/badge/ChromaDB-Persistent-orange) ![Groq](https://img.shields.io/badge/Groq-llama--3.1--8b-purple)

---

## Architecture

```
User Query
    │
    ├──► Dense Retrieval  (ChromaDB · all-MiniLM-L6-v2 · cosine similarity)
    │
    └──► Sparse Retrieval (BM25Okapi · keyword TF-IDF matching)
              │
              ▼
        RRF Fusion        (Reciprocal Rank Fusion · k=60)
              │
              ▼
        Groq LLM          (llama-3.1-8b-instant · context-grounded generation)
              │
              ▼
        JSON Response     (answer + sources + debug scores)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| Dense Retrieval | ChromaDB (persistent) + sentence-transformers |
| Sparse Retrieval | BM25Okapi (rank_bm25) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| LLM | Groq API — llama-3.1-8b-instant |
| Orchestration | Python async/await |

---

## Quick Start

### 1. Clone & setup
```bash
git clone <repo-url>
cd upwork_portfolio

python -m venv venv
.\venv\Scripts\activate      # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add your Groq API key:
# GROQ_API_KEY=gsk_xxxxxxxxxxxx
```
Get a free key at https://console.groq.com/keys

### 3. Run the server
```bash
.\venv\Scripts\uvicorn app:app --reload --port 8000
```

Server starts at **http://localhost:8000**

---

## API Reference

### `GET /health`
```json
{
  "status": "ok",
  "version": "2.0.0 — Hybrid Search (Dense + Sparse + RRF)",
  "dense_docs": 10,
  "sparse_docs": 10,
  "embedding_model": "all-MiniLM-L6-v2",
  "llm": "llama-3.1-8b-instant (Groq)"
}
```

### `POST /chat`

**Request:**
```json
{
  "query": "Hybrid Search hoạt động như thế nào?",
  "top_k": 3
}
```

**Response:**
```json
{
  "answer": "Hybrid Search kết hợp...",
  "query": "Hybrid Search hoạt động như thế nào?",
  "retrieved_context": ["...top docs after RRF..."],
  "dense_results": ["...ChromaDB results..."],
  "sparse_results": ["...BM25 results..."],
  "rrf_scores_preview": {
    "Hybrid Search kết hợp Dense...": 0.016129,
    "RAG (Retrieval-Augmented...":   0.015873
  }
}
```

### Interactive Docs
Open **http://localhost:8000/docs** for Swagger UI.  
Open **`index.html`** in a browser for the live visual demo.

---

## How RRF Works

```
score(doc) = 1/(60 + rank_dense) + 1/(60 + rank_sparse)
```

Documents appearing high in **both** lists get the highest combined score — giving the best of semantic similarity and keyword matching.

---

## Project Structure

```
upwork_portfolio/
├── app.py              ← FastAPI application (Hybrid RAG core)
├── requirements.txt    ← Python dependencies
├── index.html          ← Frontend demo (calls real API)
├── .env.example        ← Environment template
├── .gitignore
└── README.md
```
