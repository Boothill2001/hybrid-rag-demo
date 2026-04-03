"""
Enterprise Hybrid Search RAG API
─────────────────────────────────────────────────────────────────────────────
Pipeline: Query → [Dense (ChromaDB) ‖ Sparse (BM25)] → RRF Fusion → Groq LLM
─────────────────────────────────────────────────────────────────────────────
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE — Sample Vietnamese documents
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_DOCS = [
    "RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with large language models to generate accurate and grounded answers.",
    "ChromaDB is an open-source vector database that stores embeddings and performs cosine similarity search, supporting both in-memory and persistent disk execution.",
    "Sentence Transformers is a Python library that generates high-quality dense embeddings from text, supporting multiple languages through models like all-MiniLM-L6-v2.",
    "LangChain is an open-source framework for building AI applications with LLMs, providing abstractions for chains, agents, memory, and integrating with hundreds of data sources.",
    "Groq is an ultra-fast hardware inference platform for large LLMs like LLaMA-3 and Mixtral, providing a free API with extremely low latency powered by Custom LPU chips.",
    "Hybrid Search combines Dense Retrieval (vector similarity) and Sparse Retrieval (BM25 keyword search), then uses Reciprocal Rank Fusion to merge results for higher accuracy than either method alone.",
    "LangGraph is an extension of LangChain that allows building agentic workflows as directed graphs, supporting cyclic loops and self-correction mechanisms.",
    "FastAPI is a high-performance Python web framework that automatically generates OpenAPI docs, supports async/await natively, and is widely used for production AI APIs.",
    "BM25 (Best Match 25) is a ranking algorithm based on an improved TF-IDF formula, popular in sparse retrieval due to its high efficiency with keyword matching without requiring GPUs.",
    "Reciprocal Rank Fusion (RRF) is an algorithm that merges multiple ranked result lists, calculating scores via the 1/(k + rank) formula, boosting documents that rank highly across multiple lists.",
]

# ─────────────────────────────────────────────────────────────────────────────
# Global singletons (khởi tạo tại startup, dùng lại cho mọi request)
# ─────────────────────────────────────────────────────────────────────────────
embedder: SentenceTransformer = None
chroma_collection: chromadb.Collection = None
bm25_index: BM25Okapi = None          # ← SPARSE INDEX
bm25_corpus: list[str] = None         # ← raw docs mapped 1-1 với BM25 index
llm: ChatGroq = None


# ─────────────────────────────────────────────────────────────────────────────
# RRF — Reciprocal Rank Fusion
# ─────────────────────────────────────────────────────────────────────────────
def reciprocal_rank_fusion(
    dense_docs: list[str],
    sparse_docs: list[str],
    k: int = 60,
) -> list[str]:
    """
    Gộp 2 danh sách kết quả (Dense + Sparse) bằng thuật toán RRF.

    Công thức: score(d) = Σ  1 / (k + rank(d, list_i))
                          i

    Args:
        dense_docs : list tài liệu từ ChromaDB (thứ tự = rank vector similarity)
        sparse_docs: list tài liệu từ BM25     (thứ tự = rank BM25 score)
        k          : hằng số smoothing, thường = 60 (theo paper gốc)

    Returns:
        list tài liệu đã được rerank theo RRF score (descending), đã dedup.
    """
    scores: dict[str, float] = {}

    # Score từ Dense list (rank bắt đầu từ 1)
    for rank, doc in enumerate(dense_docs, start=1):
        scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)

    # Score từ Sparse list (cộng dồn nếu doc đã xuất hiện ở Dense)
    for rank, doc in enumerate(sparse_docs, start=1):
        scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)

    # Sort descending theo tổng RRF score → doc liên quan nhất lên đầu
    reranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked]


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — khởi tạo tất cả resources khi server start
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, chroma_collection, bm25_index, bm25_corpus, llm

    # ── 1. Load embedding model ───────────────────────────────────────────────
    print("⏳ Loading SentenceTransformer (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Embedding model ready.")

    # ── 2. DENSE INDEX — ChromaDB ─────────────────────────────────────────────
    print("⏳ Setting up ChromaDB (persistent on ./chroma_db)...")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_or_create_collection(
        name="rag_hybrid_en",
        metadata={"hnsw:space": "cosine"},
    )
    if chroma_collection.count() == 0:
        embeddings = embedder.encode(SAMPLE_DOCS).tolist()
        chroma_collection.add(
            documents=SAMPLE_DOCS,
            embeddings=embeddings,
            ids=[f"doc_{i}" for i in range(len(SAMPLE_DOCS))],
        )
        print(f"✅ ChromaDB: ingested {len(SAMPLE_DOCS)} docs.")
    else:
        print(f"✅ ChromaDB: loaded {chroma_collection.count()} existing docs from disk.")
    print(f"   Dense index ready — {chroma_collection.count()} docs.")

    # ── 3. SPARSE INDEX — BM25 ───────────────────────────────────────────────
    print("⏳ Building BM25 index...")
    bm25_corpus = SAMPLE_DOCS
    tokenized_corpus = [doc.lower().split() for doc in bm25_corpus]
    bm25_index = BM25Okapi(tokenized_corpus)
    print(f"✅ BM25 (Sparse) ready — {len(bm25_corpus)} docs indexed.")

    # ── 4. Groq LLM ───────────────────────────────────────────────────────────
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError("❌ GROQ_API_KEY not set. Create a .env file with GROQ_API_KEY=your_key")
    llm = ChatGroq(
        api_key=groq_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
    )
    print("✅ Groq LLM ready (llama-3.1-8b-instant)")
    print("\n🚀 Hybrid RAG Server ready! Listening on http://localhost:8000\n")

    yield

    print("Server shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Hybrid Search RAG API",
    description=(
        "Enterprise RAG pipeline: ChromaDB Dense Search + BM25 Sparse Search "
        "→ Reciprocal Rank Fusion → Groq LLM (llama-3.1-8b-instant)"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Allow browser (index.html) to call this API cross-origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    top_k: int = 3


class ChatResponse(BaseModel):
    answer: str
    query: str
    retrieved_context: list[str]   # top docs sau RRF
    dense_results: list[str]       # raw kết quả từ ChromaDB
    sparse_results: list[str]      # raw kết quả từ BM25
    rrf_scores_preview: dict       # top-3 docs + RRF score để debug/demo


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "2.0.0 — Hybrid Search (Dense + Sparse + RRF)",
        "dense_docs": chroma_collection.count() if chroma_collection else 0,
        "sparse_docs": len(bm25_corpus) if bm25_corpus else 0,
        "embedding_model": "all-MiniLM-L6-v2",
        "llm": "llama-3.1-8b-instant (Groq)",
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Hybrid RAG pipeline:
    ① Embed query  → Dense Search (ChromaDB cosine similarity)
    ② Tokenize     → Sparse Search (BM25 keyword matching)
    ③ RRF Fusion   → merge + rerank cả 2 list
    ④ LLM Generate → Groq Llama-3 trả lời dựa trên fused context
    """
    try:
        n = min(req.top_k, len(SAMPLE_DOCS))

        # ── ① DENSE RETRIEVAL — ChromaDB ────────────────────────────────────
        query_embedding = embedder.encode(req.query).tolist()
        dense_res = chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
        )
        dense_docs: list[str] = dense_res["documents"][0]
        # dense_docs đã được sắp theo cosine similarity (cao → thấp)

        # ── ② SPARSE RETRIEVAL — BM25 ────────────────────────────────────────
        tokenized_query = req.query.lower().split()
        bm25_scores = bm25_index.get_scores(tokenized_query)
        # Lấy top-n index theo BM25 score (cao → thấp)
        top_n_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:n]
        sparse_docs: list[str] = [bm25_corpus[i] for i in top_n_indices]
        # sparse_docs đã được sắp theo BM25 score (cao → thấp)

        # ── ③ RRF FUSION — merge Dense + Sparse ─────────────────────────────
        fused_docs = reciprocal_rank_fusion(
            dense_docs=dense_docs,
            sparse_docs=sparse_docs,
            k=60,
        )
        # Lấy top-k docs sau RRF làm context cho LLM
        context_docs = fused_docs[:n]

        # Build preview scores để trả về cho client debug
        _temp_scores: dict[str, float] = {}
        for rank, doc in enumerate(dense_docs, 1):
            _temp_scores[doc] = _temp_scores.get(doc, 0.0) + 1.0 / (60 + rank)
        for rank, doc in enumerate(sparse_docs, 1):
            _temp_scores[doc] = _temp_scores.get(doc, 0.0) + 1.0 / (60 + rank)
        rrf_preview = {
            doc[:60] + "...": round(score, 6)
            for doc, score in sorted(_temp_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        }

        # ── ④ LLM GENERATION — Groq Llama-3 ─────────────────────────────────
        context_block = "\n".join(
            f"[{i+1}] {doc}" for i, doc in enumerate(context_docs)
        )
        messages = [
            SystemMessage(content=(
                "You are an expert AI assistant. "
                "Based precisely on the provided context passages (optimized via Hybrid Search + RRF), "
                "answer the user's question accurately and concisely in English. "
                "If the context does not contain the answer, say you do not know."
            )),
            HumanMessage(content=(
                f"Context (Hybrid Search + RRF):\n{context_block}\n\n"
                f"Question: {req.query}"
            )),
        ]
        response = llm.invoke(messages)

        return ChatResponse(
            answer=response.content,
            query=req.query,
            retrieved_context=context_docs,
            dense_results=dense_docs,
            sparse_results=sparse_docs,
            rrf_scores_preview=rrf_preview,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
