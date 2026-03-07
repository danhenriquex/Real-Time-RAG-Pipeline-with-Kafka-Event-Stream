.PHONY: help build up down logs health query upload

PROJECT  = rag-pipeline
COMPOSE  = docker compose -p $(PROJECT) -f docker-compose.yml --env-file .env

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Docker ────────────────────────────────────────────────────────────────────

build: ## Build all Docker images
	$(COMPOSE) build

up: ## Start all services
	$(COMPOSE) up -d
	@echo "\n✓ Services started"
	@echo "  Gradio UI:      http://localhost:7860"
	@echo "  RAG API:        http://localhost:8000/docs"
	@echo "  Upload Service: http://localhost:8001/docs"
	@echo "  ChromaDB:       http://localhost:8010"

down: ## Stop all services
	$(COMPOSE) down

logs: ## Tail all logs
	$(COMPOSE) logs -f

logs-worker: ## Tail embedding worker logs
	$(COMPOSE) logs -f embedding-worker

logs-api: ## Tail rag-api logs
	$(COMPOSE) logs -f rag-api

# ── Health ────────────────────────────────────────────────────────────────────

health: ## Check all service health
	@echo "\n=== Service Health ==="
	@curl -sf http://localhost:8001/health | python3 -m json.tool && echo "✓ upload-service" || echo "✗ upload-service"
	@curl -sf http://localhost:8000/health | python3 -m json.tool && echo "✓ rag-api" || echo "✗ rag-api"
	@curl -sf http://localhost:8010/api/v2/heartbeat && echo "\n✓ chromadb" || echo "✗ chromadb"
	@echo ""

# ── Dev commands ─────────────────────────────────────────────────────────────

upload: ## Upload a file: make upload FILE=path/to/file.txt
	@curl -X POST http://localhost:8001/upload \
		-F "file=@$(FILE)" | python3 -m json.tool

upload-batch: ## Upload multiple files: make upload-batch FILES="a.txt b.pdf c.md"
	@curl -X POST http://localhost:8001/upload/batch \
		$(foreach f,$(FILES),-F "files=@$(f)") | python3 -m json.tool

query: ## Query the RAG agent: make query Q="your question"
	@curl -X POST http://localhost:8000/query \
		-H "Content-Type: application/json" \
		-d '{"query": "$(Q)"}' | python3 -m json.tool

docs: ## List all ingested documents
	@curl -sf http://localhost:8001/documents | python3 -m json.tool

delete-doc: ## Delete a document: make delete-doc ID=<document_id>
	@curl -X DELETE http://localhost:8001/documents/$(ID) | python3 -m json.tool

reingest-doc: ## Re-ingest a document: make reingest-doc ID=<document_id>
	@curl -X POST http://localhost:8001/documents/$(ID)/reingest | python3 -m json.tool

kafka-topics: ## List Kafka topics
	$(COMPOSE) exec kafka kafka-topics --list --bootstrap-server localhost:9092

kafka-messages: ## Tail Kafka documents topic
	$(COMPOSE) exec kafka kafka-console-consumer \
		--bootstrap-server localhost:9092 \
		--topic documents \
		--from-beginning

db: ## Connect to PostgreSQL
	$(COMPOSE) exec postgres psql -U rag -d rag_pipeline

# ── Tests ─────────────────────────────────────────────────────────────────────

test: ## Run unit tests (no services needed)
	uv run pytest tests/unit/ -m unit -v

test-integration: ## Run integration tests (requires make up first)
	uv run pytest tests/integration/ -m integration -v

test-all: ## Run all tests
	uv run pytest -v

test-cov: ## Run tests with coverage report
	uv run pytest tests/unit/ --cov=src --cov-report=term-missing --cov-report=html:htmlcov
	@echo "\n✓ Coverage report: htmlcov/index.html"

# ── CI ────────────────────────────────────────────────────────────────────────

lint: ## Run ruff linter
	uv tool run ruff check .
	uv tool run ruff format --check .

init: ## First-time setup
	cp .env.example .env
	@echo "✓ Created .env — add your OPENAI_API_KEY"
