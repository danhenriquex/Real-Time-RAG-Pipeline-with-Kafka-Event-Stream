# 🔍 Real-Time RAG Pipeline

An event-driven Retrieval-Augmented Generation (RAG) system built with Kafka, ChromaDB, PostgreSQL, and OpenAI. Upload documents (TXT, Markdown, PDF), have them automatically chunked and embedded in real time, then query them with an AI agent that returns cited answers.

---

## Architecture

```text
┌─────────────────────────────────────────────────────┐
│                  Gradio UI (:7860)                   │
│         Upload files · Query · Manage docs           │
└────────────────┬──────────────────┬─────────────────┘
                 │                  │
                 ▼                  ▼
      ┌──────────────────┐  ┌───────────────┐
      │  Upload Service  │  │   RAG API     │
      │    (:8001)       │  │   (:8000)     │
      └────────┬─────────┘  └──────┬────────┘
               │                   │ hybrid search
               ▼                   ▼
      ┌─────────────────┐  ┌───────────────┐  ┌──────────────┐
      │      Kafka      │  │   ChromaDB    │  │  PostgreSQL  │
      │  topic:docs     │  │  (vectors)    │  │ (metadata +  │
      └────────┬────────┘  └───────────────┘  │    FTS)      │
               │                              └──────────────┘
               ▼
      ┌─────────────────┐
      │ Embedding Worker│──► ChromaDB + PostgreSQL
      │  chunk·embed    │
      └─────────────────┘
```

### Services

| Service | Port | Description |
| --------- | ------ | ------------- |
| `gradio-ui` | 7860 | Upload + query interface |
| `upload-service` | 8001 | FastAPI — validates files, publishes to Kafka |
| `rag-api` | 8000 | FastAPI — hybrid search + GPT-4o-mini answers |
| `embedding-worker` | — | Kafka consumer — chunks, embeds, stores |
| `chromadb` | 8010 | Vector store |
| `postgres` | 5435 | Metadata + full-text search |
| `kafka` | 29092 | Event stream |
| `zookeeper` | — | Kafka dependency |

---

## Features

- **Real-time ingestion** — files published as Kafka events, processed asynchronously
- **Batch upload** — upload up to 20 files at once, all processed concurrently
- **Multi-format support** — TXT, Markdown, PDF (text-based + scanned via Tesseract OCR)
- **Hybrid search** — combines vector similarity (ChromaDB) + full-text search (PostgreSQL)
- **Cited answers** — GPT-4o-mini answers with `[filename]` citations
- **Document management** — delete documents, re-ingest after config changes
- **Observability** — structured JSON logs via `structlog`

---

## Quickstart

### Prerequisites

- Docker + Docker Compose
- OpenAI API key

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/your-user/rag-pipeline.git
cd rag-pipeline

# 2. Create .env
make init

# 3. Add your OpenAI API key to .env
echo "OPENAI_API_KEY=sk-..." >> .env

# 4. Build and start all services
make build
make up

# 5. Open the UI
open http://localhost:7860
```

### Upload and query

```bash
# Upload a single file
make upload FILE=path/to/document.txt

# Upload multiple files at once
make upload-batch FILES="doc1.txt doc2.pdf doc3.md"

# Watch the embedding worker process them
make logs-worker

# Query once processing is complete
make query Q="What is the main topic of the documents?"
```

---

## Project Structure

```text
rag-pipeline/
├── src/
│   ├── upload_service/       # FastAPI upload endpoints
│   │   ├── app.py
│   │   └── Dockerfile
│   ├── embedding_worker/     # Kafka consumer + embedding logic
│   │   ├── worker.py
│   │   └── Dockerfile
│   ├── rag_api/              # Query endpoint + RAG logic
│   │   ├── app.py
│   │   └── Dockerfile
│   ├── gradio_ui/            # Gradio 4-tab interface
│   │   ├── app.py
│   │   └── Dockerfile
│   └── shared/               # Shared pydantic-settings config
│       └── config.py
├── tests/
│   ├── unit/                 # Fast tests, no Docker needed
│   │   ├── test_chunking.py
│   │   ├── test_retrieval.py
│   │   ├── test_worker.py
│   │   └── test_rag_api.py
│   ├── integration/          # Full pipeline tests
│   │   └── test_pipeline.py
│   └── conftest.py
├── configs/
│   └── init.sql              # PostgreSQL schema
├── .github/workflows/
│   └── ci.yml                # lint → unit tests → build → integration tests
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── pytest.ini
```

---

## Data Flow

```text
1. User uploads file via Gradio UI
2. Upload Service validates (type, size) → saves metadata to PostgreSQL
3. Upload Service publishes event to Kafka topic: documents
4. Embedding Worker consumes event → chunks text (512 tokens, 64 overlap)
5. Worker generates embeddings via OpenAI text-embedding-3-small
6. Vectors stored in ChromaDB · chunks + metadata stored in PostgreSQL
7. User queries via Gradio UI → RAG API embeds query
8. RAG API runs hybrid search: vector (ChromaDB) + keyword (PostgreSQL FTS)
9. Results deduplicated · top-K sent to GPT-4o-mini with citations
10. Answer returned with source filenames and confidence scores
```

---

## API Reference

### Upload Service `:8001`

| Method | Endpoint | Description |
| -------- | ---------- | ------------- |
| `POST` | `/upload` | Upload a single file |
| `POST` | `/upload/batch` | Upload up to 20 files |
| `GET` | `/documents` | List all documents |
| `GET` | `/documents/{id}` | Get document status |
| `DELETE` | `/documents/{id}` | Delete document + vectors |
| `POST` | `/documents/{id}/reingest` | Re-process a document |
| `GET` | `/health` | Health check |

### RAG API `:8000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/query` | Query documents |
| `GET` | `/health` | Health check |

**Query request:**

```json
{
  "query": "What is machine learning?",
  "search_mode": "hybrid",
  "top_k": 5
}
```

**Query response:**

```json
{
  "query": "What is machine learning?",
  "answer": "Machine learning is... [ml_guide.txt]",
  "sources": [{"filename": "ml_guide.txt", "score": 0.92, "source": "vector"}],
  "latency_ms": 1823,
  "num_chunks": 5
}
```

---

## Testing

```bash
# Unit tests — no Docker needed, runs in seconds
make test

# Unit tests with coverage report
make test-cov
# → open htmlcov/index.html

# Integration tests — requires make up first
make test-integration

# All tests
make test-all
```

### CI Pipeline

Every push and PR triggers 4 jobs in order:

```text
lint → unit-tests → build (4 images in parallel) → integration-tests
```

Integration tests spin up the full Docker stack on GitHub Actions, upload a document with a unique token, wait for the embedding worker to process it, then assert the RAG API can retrieve it.

---

## Make Commands

```bash
make help           # show all commands
make build          # build all Docker images
make up             # start all services
make down           # stop all services
make logs           # tail all logs
make logs-worker    # tail embedding worker logs
make health         # check all service health endpoints

make upload FILE=x  # upload a single file
make upload-batch FILES="a.txt b.pdf"  # batch upload
make query Q="..."  # query the RAG agent
make docs           # list all documents
make delete-doc ID=<id>     # delete a document
make reingest-doc ID=<id>   # re-ingest a document

make kafka-topics   # list Kafka topics
make kafka-messages # tail documents topic
make db             # connect to PostgreSQL

make test           # unit tests
make test-cov       # unit tests + coverage
make test-integration  # integration tests
make lint           # ruff check + format
```

---

## Configuration

All configuration is via environment variables in `.env`:

```bash
OPENAI_API_KEY=sk-...         # required

# PostgreSQL
POSTGRES_USER=rag
POSTGRES_PASSWORD=ragpass
POSTGRES_DB=rag_pipeline

# Kafka
KAFKA_TOPIC_DOCUMENTS=documents
KAFKA_CONSUMER_GROUP=embedding-workers

# Chunking
CHUNK_SIZE=512                # tokens per chunk
CHUNK_OVERLAP=64              # overlap between chunks

# Retrieval
RETRIEVAL_TOP_K=5             # chunks retrieved per query

LOG_LEVEL=INFO
```

---

## Tech Stack

| Component | Technology |
| ----------- | ----------- |
| Event streaming | Apache Kafka (Confluent) |
| Vector store | ChromaDB 1.x |
| Relational DB | PostgreSQL 16 + pgvector |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-4o-mini` |
| PDF extraction | pypdf + Tesseract OCR |
| API framework | FastAPI + uvicorn |
| UI | Gradio 6 |
| Config | pydantic-settings |
| Logging | structlog |
| Testing | pytest + pytest-cov |
| Linting | ruff |
| CI/CD | GitHub Actions |
| Containers | Docker Compose |
