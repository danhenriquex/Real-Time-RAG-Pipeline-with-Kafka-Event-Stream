FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml .
RUN uv pip install --system --no-cache \
    fastapi "uvicorn[standard]" \
    openai tiktoken \
    chromadb \
    psycopg2-binary \
    pydantic pydantic-settings structlog tenacity
COPY src/rag_api/ ./src/rag_api/
COPY src/shared/ ./src/shared/
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["python", "src/rag_api/app.py"]
