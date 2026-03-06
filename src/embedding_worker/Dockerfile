FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml .
RUN uv pip install --system --no-cache \
    openai tiktoken \
    confluent-kafka \
    chromadb \
    psycopg2-binary \
    pydantic pydantic-settings structlog tenacity
COPY src/embedding_worker/ ./src/embedding_worker/
COPY src/shared/ ./src/shared/
ENV PYTHONPATH=/app/src
CMD ["python", "src/embedding_worker/worker.py"]
