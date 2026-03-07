"""
Upload Service
--------------
Accepts TXT/Markdown file uploads, saves metadata to PostgreSQL,
publishes document events to Kafka for async processing.

Endpoints:
  POST /upload          - upload a file
  GET  /documents       - list all documents
  GET  /documents/{id}  - get document status
  GET  /health
"""

import io
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

import psycopg2
import psycopg2.extras
import pytesseract
import structlog
import uvicorn
from confluent_kafka import Producer
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pdf2image import convert_from_bytes
from pydantic import BaseModel
from pypdf import PdfReader

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
DATABASE_URL = (
    f"postgresql://{os.getenv('POSTGRES_USER', 'rag')}:"
    f"{os.getenv('POSTGRES_PASSWORD', 'ragpass')}@"
    f"{os.getenv('POSTGRES_HOST', 'postgres')}:"
    f"{os.getenv('POSTGRES_PORT', '5432')}/"
    f"{os.getenv('POSTGRES_DB', 'rag_pipeline')}"
)
ALLOWED_TYPES = {".txt", ".md", ".markdown", ".pdf"}
MAX_SIZE_MB = 10

# ── Kafka Producer ────────────────────────────────────────────────────────────

producer: Producer | None = None


def get_producer() -> Producer:
    global producer
    if producer is None:
        producer = Producer({"bootstrap.servers": KAFKA_SERVERS})
    return producer


def delivery_report(err, msg):
    if err:
        log.error("kafka_delivery_failed", error=str(err))
    else:
        log.info("kafka_delivered", topic=msg.topic(), partition=msg.partition())


# ── Database ──────────────────────────────────────────────────────────────────


def get_db():
    return psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("upload_service_starting", kafka=KAFKA_SERVERS)
    get_producer()  # warm up connection
    yield
    if producer:
        producer.flush()
    log.info("upload_service_stopped")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Upload Service", lifespan=lifespan)


class DocumentStatus(BaseModel):
    id: str
    filename: str
    file_type: str
    size_bytes: int
    status: str
    chunk_count: int
    created_at: str


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a TXT or Markdown file for processing."""
    import os as _os

    ext = _os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {ALLOWED_TYPES}")

    content = await file.read()
    size = len(content)

    if size > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large. Max {MAX_SIZE_MB}MB.")
    if size == 0:
        raise HTTPException(400, "File is empty.")

    doc_id = str(uuid.uuid4())

    # Save metadata to Postgres
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (id, filename, file_type, size_bytes, status)
                VALUES (%s, %s, %s, %s, 'pending')
                """,
                (doc_id, file.filename, ext, size),
            )
        conn.commit()
    finally:
        conn.close()

    # Extract text (PDF or plain text)
    if ext == ".pdf":
        try:
            # Step 1: try pypdf (fast, works for text-based PDFs)
            reader = PdfReader(io.BytesIO(content))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()

            # Step 2: if no text extracted, fall back to OCR
            if not text:
                log.info(
                    "pdf_ocr_fallback", filename=file.filename, reason="no embedded text found"
                )
                images = convert_from_bytes(content, dpi=200)
                text = "\n\n".join(
                    pytesseract.image_to_string(img, lang="eng") for img in images
                ).strip()

            if not text:
                raise HTTPException(400, "Could not extract text from PDF (empty document).")

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f"Failed to read PDF: {e}")
    else:
        text = content.decode("utf-8", errors="replace")

    # Publish to Kafka
    event = {
        "document_id": doc_id,
        "filename": file.filename,
        "file_type": ext,
        "content": text,
        "size_bytes": size,
        "timestamp": datetime.utcnow().isoformat(),
    }
    get_producer().produce(
        KAFKA_TOPIC,
        key=doc_id.encode(),
        value=json.dumps(event).encode(),
        callback=delivery_report,
    )
    get_producer().poll(0)

    log.info("document_uploaded", doc_id=doc_id, filename=file.filename, size=size)

    return JSONResponse(
        status_code=202,
        content={
            "document_id": doc_id,
            "filename": file.filename,
            "size_bytes": size,
            "status": "pending",
            "message": "Document queued for processing",
        },
    )


@app.post("/upload/batch")
async def upload_batch(files: list[UploadFile] = File(...)):
    """
    Upload multiple files at once. Each file is published to Kafka
    as a separate event, allowing the embedding worker to process
    them concurrently — this is where Kafka shines.
    """
    if not files:
        raise HTTPException(400, "No files provided.")
    if len(files) > 20:
        raise HTTPException(400, "Max 20 files per batch.")

    results = []
    for file in files:
        import os as _os

        ext = _os.path.splitext(file.filename or "")[1].lower()

        # Validate each file
        if ext not in ALLOWED_TYPES:
            results.append(
                {
                    "filename": file.filename,
                    "status": "rejected",
                    "reason": f"Unsupported type '{ext}'",
                }
            )
            continue

        content = await file.read()
        size = len(content)

        if size > MAX_SIZE_MB * 1024 * 1024:
            results.append(
                {
                    "filename": file.filename,
                    "status": "rejected",
                    "reason": f"Too large (max {MAX_SIZE_MB}MB)",
                }
            )
            continue

        if size == 0:
            results.append(
                {
                    "filename": file.filename,
                    "status": "rejected",
                    "reason": "Empty file",
                }
            )
            continue

        # Extract text
        try:
            if ext == ".pdf":
                reader = PdfReader(io.BytesIO(content))
                text = "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
                if not text:
                    images = convert_from_bytes(content, dpi=200)
                    text = "\n\n".join(
                        pytesseract.image_to_string(img, lang="eng") for img in images
                    ).strip()
                if not text:
                    results.append(
                        {
                            "filename": file.filename,
                            "status": "rejected",
                            "reason": "No extractable text",
                        }
                    )
                    continue
            else:
                text = content.decode("utf-8", errors="replace")
        except Exception as e:
            results.append({"filename": file.filename, "status": "rejected", "reason": str(e)})
            continue

        doc_id = str(uuid.uuid4())

        # Save to Postgres
        conn = get_db()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO documents (id, filename, file_type, size_bytes, status) "
                    "VALUES (%s, %s, %s, %s, 'pending')",
                    (doc_id, file.filename, ext, size),
                )
            conn.commit()
        finally:
            conn.close()

        # Publish to Kafka
        event = {
            "document_id": doc_id,
            "filename": file.filename,
            "file_type": ext,
            "content": text,
            "size_bytes": size,
            "timestamp": datetime.utcnow().isoformat(),
        }
        get_producer().produce(
            KAFKA_TOPIC,
            key=doc_id.encode(),
            value=json.dumps(event).encode(),
            callback=delivery_report,
        )

        results.append(
            {
                "document_id": doc_id,
                "filename": file.filename,
                "size_bytes": size,
                "status": "pending",
            }
        )
        log.info("batch_file_queued", doc_id=doc_id, filename=file.filename)

    # Flush all Kafka messages at once
    get_producer().flush()

    accepted = [r for r in results if r.get("status") == "pending"]
    rejected = [r for r in results if r.get("status") == "rejected"]
    log.info("batch_upload_complete", accepted=len(accepted), rejected=len(rejected))

    return JSONResponse(
        status_code=202,
        content={
            "accepted": len(accepted),
            "rejected": len(rejected),
            "results": results,
            "message": f"{len(accepted)} files queued for processing",
        },
    )


@app.get("/documents")
def list_documents(limit: int = 20, offset: int = 0):
    """List all uploaded documents."""
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM documents ORDER BY created_at DESC LIMIT %s OFFSET %s",
                (limit, offset),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return {"documents": [dict(r) for r in rows], "total": len(rows)}


@app.get("/documents/{doc_id}")
def get_document(doc_id: str):
    """Get status of a specific document."""
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM documents WHERE id = %s", (doc_id,))
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        raise HTTPException(404, f"Document {doc_id} not found")
    return dict(row)


@app.get("/health")
def health():
    try:
        conn = get_db()
        conn.close()
        db_ok = True
    except Exception:
        db_ok = False

    kafka_ok = producer is not None
    return {
        "status": "ok" if db_ok and kafka_ok else "degraded",
        "db": db_ok,
        "kafka": kafka_ok,
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=False)


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    """
    Delete a document and all its chunks from PostgreSQL.
    Publishes a delete event to Kafka so the embedding worker
    removes the vectors from ChromaDB.
    """
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, filename FROM documents WHERE id = %s", (doc_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, f"Document {doc_id} not found")

            # Get chroma_ids before deleting chunks
            cur.execute("SELECT chroma_id FROM chunks WHERE document_id = %s", (doc_id,))
            chroma_ids = [r["chroma_id"] for r in cur.fetchall() if r["chroma_id"]]

            # Delete chunks + document (CASCADE handles chunks)
            cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        conn.commit()
    finally:
        conn.close()

    # Publish delete event so embedding worker cleans up ChromaDB
    event = {
        "event_type": "delete",
        "document_id": doc_id,
        "chroma_ids": chroma_ids,
        "timestamp": datetime.utcnow().isoformat(),
    }
    get_producer().produce(
        KAFKA_TOPIC,
        key=doc_id.encode(),
        value=json.dumps(event).encode(),
        callback=delivery_report,
    )
    get_producer().poll(0)

    log.info("document_deleted", doc_id=doc_id, vectors_removed=len(chroma_ids))
    return {"document_id": doc_id, "deleted": True, "vectors_removed": len(chroma_ids)}


@app.post("/documents/{doc_id}/reingest")
def reingest_document(doc_id: str):
    """
    Re-process an existing document — useful after chunking/embedding
    config changes. Deletes old chunks/vectors and re-queues the document.
    Requires the original content to be stored; otherwise upload again.
    """
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM documents WHERE id = %s", (doc_id,))
            doc = cur.fetchone()
            if not doc:
                raise HTTPException(404, f"Document {doc_id} not found")
            if doc["status"] == "processing":
                raise HTTPException(409, "Document is already being processed")

            # Get old chroma_ids to clean up
            cur.execute("SELECT chroma_id FROM chunks WHERE document_id = %s", (doc_id,))
            chroma_ids = [r["chroma_id"] for r in cur.fetchall() if r["chroma_id"]]

            # Delete old chunks
            cur.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))

            # Reset document status
            cur.execute(
                "UPDATE documents SET status='pending', "
                "chunk_count=0, updated_at=NOW() WHERE id=%s",
                (doc_id,),
            )
        conn.commit()
    finally:
        conn.close()

    # Publish reingest event
    event = {
        "event_type": "reingest",
        "document_id": doc_id,
        "filename": doc["filename"],
        "chroma_ids": chroma_ids,  # old vectors to delete first
        "timestamp": datetime.utcnow().isoformat(),
    }
    get_producer().produce(
        KAFKA_TOPIC,
        key=doc_id.encode(),
        value=json.dumps(event).encode(),
        callback=delivery_report,
    )
    get_producer().poll(0)

    log.info("document_reingest_queued", doc_id=doc_id)
    return {
        "document_id": doc_id,
        "status": "pending",
        "message": "Document queued for re-ingestion",
    }
