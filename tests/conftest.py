"""
Shared pytest fixtures for unit and integration tests.
"""

import os

import pytest

# Set env vars before any imports so pydantic-settings picks them up
os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake-key")
os.environ.setdefault("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
os.environ.setdefault("CHROMA_HOST", "localhost")
os.environ.setdefault("CHROMA_PORT", "8010")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PASSWORD", os.getenv("POSTGRES_PASSWORD", "dev-only-password"))


# ── Sample data fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def sample_text():
    return """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems
to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

There are three main types of machine learning:

1. **Supervised Learning**: The algorithm learns from labeled training data.
   Examples include classification and regression tasks.

2. **Unsupervised Learning**: The algorithm finds patterns in unlabeled data.
   Examples include clustering and dimensionality reduction.

3. **Reinforcement Learning**: An agent learns by interacting with an environment
   and receiving rewards or penalties for its actions.

## Applications

Machine learning is used in many domains including:
- Natural language processing
- Computer vision
- Recommendation systems
- Fraud detection
- Medical diagnosis
"""


@pytest.fixture
def short_text():
    return "This is a short document about Python programming."


@pytest.fixture
def empty_text():
    return ""


@pytest.fixture
def sample_chunks():
    return [
        {
            "content": "Machine learning is a subset of artificial intelligence.",
            "filename": "ml_intro.txt",
            "doc_id": "doc-001",
            "chunk_idx": 0,
            "score": 0.92,
            "source": "vector",
        },
        {
            "content": "There are three main types of machine learning.",
            "filename": "ml_intro.txt",
            "doc_id": "doc-001",
            "chunk_idx": 1,
            "score": 0.85,
            "source": "vector",
        },
        {
            "content": "Supervised learning uses labeled training data.",
            "filename": "ml_intro.txt",
            "doc_id": "doc-001",
            "chunk_idx": 2,
            "score": 0.78,
            "source": "keyword",
        },
    ]


@pytest.fixture
def duplicate_chunks():
    """Same chunk appearing in both vector and keyword results."""
    return [
        {
            "content": "Machine learning is a subset of AI.",
            "filename": "ml.txt",
            "doc_id": "doc-001",
            "chunk_idx": 0,
            "score": 0.90,
            "source": "vector",
        },
        {
            "content": "Machine learning is a subset of AI.",
            "filename": "ml.txt",
            "doc_id": "doc-001",
            "chunk_idx": 0,  # same doc + chunk index = duplicate
            "score": 0.75,  # lower score — should be dropped
            "source": "keyword",
        },
        {
            "content": "Deep learning is a subset of machine learning.",
            "filename": "ml.txt",
            "doc_id": "doc-001",
            "chunk_idx": 1,
            "score": 0.82,
            "source": "vector",
        },
    ]
