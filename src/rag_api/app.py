"""
RAG API
-------
Query endpoint: embeds the query, retrieves from ChromaDB (vector)
and PostgreSQL (full-text), reranks, generates answer with citations.

Endpoints:
  POST /query   - answer a question
  GET  /health
"""

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import chromadb
import psycopg2
import psycopg2.extras
import structlog
import uvicorn
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

# ── Logging ───────────────────────────────────────────────────────────────────

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = structlog.get_logger()

# ── Config ────────────────────────────────────────────────────────────────────

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "documents")
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
DATABASE_URL = (
    f"postgresql://{os.getenv('POSTGRES_USER', 'rag')}:"
    f"{os.getenv('POSTGRES_PASSWORD', 'ragpass')}@"
    f"{os.getenv('POSTGRES_HOST', 'postgres')}:"
    f"{os.getenv('POSTGRES_PORT', '5432')}/"
    f"{os.getenv('POSTGRES_DB', 'rag_pipeline')}"
)

# ── Clients ───────────────────────────────────────────────────────────────────

client = OpenAI(api_key=OPENAI_KEY)
# Lazy initialization — not connected until first use
_chroma = None
collection = None


def get_chroma_client():
    global _chroma
    if _chroma is None:
        _chroma = chromadb.HttpClient(
            host=CHROMA_HOST,
            port=CHROMA_PORT,
            tenant="default_tenant",
            database="default_database",
        )
    return _chroma


def get_collection():
    global collection
    if collection is None:
        collection = get_chroma_client().get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return collection


def get_db():
    return psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)


# ── Retrieval ─────────────────────────────────────────────────────────────────


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def embed_query(query: str) -> list[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    return response.data[0].embedding


def vector_search(query_embedding: list[float], k: int = TOP_K) -> list[dict]:
    """Semantic search via ChromaDB."""
    results = get_collection().query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append(
            {
                "content": doc,
                "filename": meta.get("filename", "unknown"),
                "doc_id": meta.get("document_id", ""),
                "chunk_idx": meta.get("chunk_index", 0),
                "score": round(1 - dist, 4),  # cosine similarity
                "source": "vector",
            }
        )
    return chunks


def keyword_search(query: str, k: int = TOP_K) -> list[dict]:
    """Full-text search via PostgreSQL."""
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.content,
                       d.filename,
                       c.document_id::text as doc_id,
                       c.chunk_index as chunk_idx,
                       ts_rank(c.content_tsv, plainto_tsquery('english', %s)) as score
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.content_tsv @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
                """,
                (query, query, k),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    return [{**dict(r), "source": "keyword"} for r in rows]


def deduplicate(chunks: list[dict]) -> list[dict]:
    """Remove duplicate chunks keeping highest score."""
    seen = {}
    for c in chunks:
        key = (c["doc_id"], c["chunk_idx"])
        if key not in seen or c["score"] > seen[key]["score"]:
            seen[key] = c
    return sorted(seen.values(), key=lambda x: x["score"], reverse=True)


# ── Generation ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise research assistant. Answer the user's question
using ONLY the provided context chunks. For each claim, cite the source filename
in square brackets like [filename.txt].

If the context doesn't contain enough information, say so clearly.
Do not hallucinate or add information not present in the context."""


def generate_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant documents found. Please upload some documents first."

    context = "\n\n".join(
        f"[{c['filename']}] (chunk {c['chunk_idx']}, score {c['score']}):\n{c['content']}"
        for c in chunks
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content


def log_query(query: str, chunks: list[dict], answer: str, latency_ms: int):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO query_logs (id, query, retrieved_ids, answer, latency_ms, num_chunks)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    str(uuid.uuid4()),
                    query,
                    [c["doc_id"] for c in chunks],
                    answer,
                    latency_ms,
                    len(chunks),
                ),
            )
        conn.commit()
    finally:
        conn.close()


# ── App ───────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("rag_api_starting")
    get_collection()  # warm up ChromaDB connection
    yield
    log.info("rag_api_stopped")


app = FastAPI(title="RAG API", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K
    search_mode: str = "hybrid"  # vector | keyword | hybrid


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict]
    latency_ms: int
    num_chunks: int


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Answer a question using RAG."""
    t0 = time.time()
    log.info("query_received", query=req.query, mode=req.search_mode)

    chunks = []

    if req.search_mode in ("vector", "hybrid"):
        embedding = embed_query(req.query)
        chunks += vector_search(embedding, req.top_k)

    if req.search_mode in ("keyword", "hybrid"):
        chunks += keyword_search(req.query, req.top_k)

    # Deduplicate and keep top-k
    chunks = deduplicate(chunks)[: req.top_k]
    answer = generate_answer(req.query, chunks)
    latency = round((time.time() - t0) * 1000)

    log_query(req.query, chunks, answer, latency)
    log.info("query_complete", latency_ms=latency, num_chunks=len(chunks))

    return QueryResponse(
        query=req.query,
        answer=answer,
        sources=[
            {"filename": c["filename"], "score": c["score"], "source": c["source"]} for c in chunks
        ],
        latency_ms=latency,
        num_chunks=len(chunks),
    )


@app.get("/health")
def health():
    try:
        get_chroma_client()
        get_collection().count()
        chroma_ok = True
    except Exception:
        chroma_ok = False

    try:
        conn = get_db()
        conn.close()
        db_ok = True
    except Exception:
        db_ok = False

    return {
        "status": "ok" if chroma_ok and db_ok else "degraded",
        "chromadb": chroma_ok,
        "db": db_ok,
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
