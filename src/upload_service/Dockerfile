FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    curl \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml .
RUN uv pip install --system --no-cache \
    fastapi "uvicorn[standard]" python-multipart \
    confluent-kafka psycopg2-binary pypdf pytesseract pdf2image \
    pydantic pydantic-settings structlog
COPY src/upload_service/ ./src/upload_service/
COPY src/shared/ ./src/shared/
ENV PYTHONPATH=/app/src
EXPOSE 8001
CMD ["python", "src/upload_service/app.py"]
