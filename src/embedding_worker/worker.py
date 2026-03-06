"""
Embedding Worker
----------------
Consumes document events from Kafka, chunks text, generates embeddings
via OpenAI, stores vectors in ChromaDB, updates metadata in PostgreSQL.

Flow:
  Kafka event → chunk text → embed chunks → store ChromaDB → update Postgres
"""

import json
import logging
import os
import time
import uuid

import chromadb
import psycopg2
import psycopg2.extras
import structlog
import tiktoken
from confluent_kafka import Consumer, KafkaError
from openai import OpenAI
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

KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC_DOCUMENTS", "documents")
KAFKA_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "embedding-workers")
CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
DATABASE_URL = (
    f"postgresql://{os.getenv('POSTGRES_USER', 'rag')}:"
    f"{os.getenv('POSTGRES_PASSWORD', 'ragpass')}@"
    f"{os.getenv('POSTGRES_HOST', 'postgres')}:"
    f"{os.getenv('POSTGRES_PORT', '5432')}/"
    f"{os.getenv('POSTGRES_DB', 'rag_pipeline')}"
)

# ── Clients ───────────────────────────────────────────────────────────────────

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
enc = tiktoken.encoding_for_model("text-embedding-3-small")


def get_chroma() -> chromadb.HttpClient:
    client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        tenant="default_tenant",
        database="default_database",
    )
    return client


def get_collection():
    client = get_chroma()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def get_db():
    return psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)


# ── Chunking ──────────────────────────────────────────────────────────────────


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping token-based chunks."""
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = enc.decode(tokens[start:end])
        chunks.append(chunk)
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks


# ── Embedding ─────────────────────────────────────────────────────────────────


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of text chunks."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=chunks,
    )
    return [item.embedding for item in response.data]


# ── Processing ────────────────────────────────────────────────────────────────


def process_document(event: dict):
    """Full pipeline: chunk → embed → store."""
    doc_id = event["document_id"]
    filename = event["filename"]
    content = event["content"]

    log.info("processing_document", doc_id=doc_id, filename=filename)
    t0 = time.time()

    # Mark as processing
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE documents SET status='processing', updated_at=NOW() WHERE id=%s",
                (doc_id,),
            )
        conn.commit()
    finally:
        conn.close()

    # Chunk
    chunks = chunk_text(content)
    log.info("chunks_created", doc_id=doc_id, count=len(chunks))

    # Embed (batch to avoid rate limits)
    batch_size = 20
    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        embeddings = embed_chunks(batch)
        all_embeddings.extend(embeddings)

    # Store in ChromaDB
    collection = get_collection()
    chroma_ids = [str(uuid.uuid4()) for _ in chunks]
    collection.add(
        ids=chroma_ids,
        embeddings=all_embeddings,
        documents=chunks,
        metadatas=[
            {"document_id": doc_id, "filename": filename, "chunk_index": i}
            for i in range(len(chunks))
        ],
    )

    # Store chunks in Postgres
    conn = get_db()
    try:
        with conn.cursor() as cur:
            for i, (chunk, chroma_id) in enumerate(zip(chunks, chroma_ids)):
                token_count = len(enc.encode(chunk))
                cur.execute(
                    """
                    INSERT INTO chunks
                    (id, document_id, chunk_index, content, token_count, chroma_id)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (str(uuid.uuid4()), doc_id, i, chunk, token_count, chroma_id),
                )
            cur.execute(
                """
                UPDATE documents
                SET status='complete', chunk_count=%s, updated_at=NOW()
                WHERE id=%s
                """,
                (len(chunks), doc_id),
            )
        conn.commit()
    finally:
        conn.close()

    duration = round((time.time() - t0) * 1000)
    log.info(
        "document_processed",
        doc_id=doc_id,
        chunks=len(chunks),
        duration_ms=duration,
    )


def mark_failed(doc_id: str, error: str):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE documents SET status='failed', updated_at=NOW() WHERE id=%s",
                (doc_id,),
            )
        conn.commit()
    finally:
        conn.close()
    log.error("document_failed", doc_id=doc_id, error=error)


# ── Consumer Loop ─────────────────────────────────────────────────────────────


def run():
    consumer = Consumer(
        {
            "bootstrap.servers": KAFKA_SERVERS,
            "group.id": KAFKA_GROUP,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([KAFKA_TOPIC])
    log.info("embedding_worker_started", topic=KAFKA_TOPIC, group=KAFKA_GROUP)

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                log.error("kafka_error", error=str(msg.error()))
                continue

            try:
                event = json.loads(msg.value().decode("utf-8"))
                etype = event.get("event_type", "ingest")
                doc_id = event.get("document_id", "unknown")

                if etype == "delete":
                    delete_vectors(event.get("chroma_ids", []), doc_id)
                elif etype == "reingest":
                    delete_vectors(event.get("chroma_ids", []), doc_id)
                else:
                    process_document(event)

                consumer.commit(message=msg)
            except Exception as exc:
                doc_id = event.get("document_id", "unknown") if "event" in dir() else "unknown"
                mark_failed(doc_id, str(exc))
                # still commit to avoid infinite retry on bad messages
                consumer.commit(message=msg)

    except KeyboardInterrupt:
        log.info("embedding_worker_stopping")
    finally:
        consumer.close()


if __name__ == "__main__":
    run()


def delete_vectors(chroma_ids: list[str], doc_id: str):
    """Remove vectors from ChromaDB by their IDs."""
    if not chroma_ids:
        return
    try:
        collection = get_collection()
        collection.delete(ids=chroma_ids)
        log.info("vectors_deleted", doc_id=doc_id, count=len(chroma_ids))
    except Exception as e:
        log.error("vector_delete_failed", doc_id=doc_id, error=str(e))
