FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml .
RUN uv pip install --system --no-cache \
    gradio httpx structlog pydantic
COPY src/gradio_ui/ ./src/gradio_ui/
ENV PYTHONPATH=/app/src
EXPOSE 7860
CMD ["python", "src/gradio_ui/app.py"]
