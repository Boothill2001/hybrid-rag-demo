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
    "RAG (Retrieval-Augmented Generation) là kỹ thuật kết hợp tìm kiếm thông tin với mô hình ngôn ngữ lớn để sinh ra câu trả lời chính xác và có căn cứ hơn.",
    "ChromaDB là vector database mã nguồn mở, lưu trữ embedding và tìm kiếm theo độ tương đồng cosine, hỗ trợ chạy hoàn toàn in-memory hoặc persistent trên disk.",
    "Sentence Transformers là thư viện Python tạo ra dense embedding chất lượng cao từ văn bản, hỗ trợ đa ngôn ngữ bao gồm tiếng Việt thông qua mô hình all-MiniLM-L6-v2.",
    "LangChain là framework mã nguồn mở giúp xây dựng ứng dụng AI với LLM, cung cấp abstraction cho chains, agents, memory và tích hợp với hàng trăm data sources.",
    "Groq là nền tảng inference phần cứng tốc độ cao cho các mô hình LLM lớn như LLaMA-3 và Mixtral, cung cấp API miễn phí với độ trễ cực thấp nhờ chip LPU.",
    "Hybrid Search kết hợp Dense Retrieval (vector similarity) và Sparse Retrieval (BM25 keyword) rồi dùng Reciprocal Rank Fusion để merge kết quả, cho độ chính xác cao hơn từng phương pháp riêng lẻ.",
    "LangGraph là extension của LangChain cho phép xây dựng các agentic workflow dưới dạng đồ thị có hướng, hỗ trợ vòng lặp cyclic và self-correction loop.",
    "FastAPI là web framework Python hiệu năng cao, tự động sinh ra OpenAPI docs, hỗ trợ async/await và được dùng rộng rãi trong production AI APIs.",
    "BM25 (Best Match 25) là thuật toán ranking dựa trên TF-IDF cải tiến, phổ biến trong sparse retrieval vì hiệu quả cao với keyword matching và không cần GPU.",
    "Reciprocal Rank Fusion (RRF) là thuật toán gộp nhiều danh sách kết quả ranked, tính điểm theo công thức 1/(k + rank) rồi cộng lại, giúp boost các tài liệu xuất hiện cao ở nhiều list.",
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
        name="rag_hybrid",
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
                "Bạn là trợ lý AI chuyên nghiệp. "
                "Dựa vào các đoạn context được cung cấp (đã được tối ưu bằng Hybrid Search + RRF), "
                "hãy trả lời câu hỏi một cách chính xác, ngắn gọn bằng tiếng Việt. "
                "Nếu context không đủ, hãy nói thật."
            )),
            HumanMessage(content=(
                f"Context (Hybrid Search + RRF):\n{context_block}\n\n"
                f"Câu hỏi: {req.query}"
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
