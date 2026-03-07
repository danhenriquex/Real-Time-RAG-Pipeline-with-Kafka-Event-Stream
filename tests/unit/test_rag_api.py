"""
Unit tests for rag_api — all external deps mocked.
Covers: vector_search, keyword_search, log_query, health endpoint.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))


@pytest.mark.unit
class TestVectorSearch:
    @patch("rag_api.app.get_collection")
    def test_returns_correctly_shaped_results(self, mock_get_col):
        from rag_api.app import vector_search

        mock_col = MagicMock()
        mock_col.query.return_value = {
            "documents": [["chunk text 1", "chunk text 2"]],
            "metadatas": [
                [
                    {"document_id": "doc-1", "filename": "a.txt", "chunk_index": 0},
                    {"document_id": "doc-1", "filename": "a.txt", "chunk_index": 1},
                ]
            ],
            "distances": [[0.1, 0.3]],
        }
        mock_get_col.return_value = mock_col

        results = vector_search([0.1] * 1536, k=2)

        assert len(results) == 2
        assert results[0]["source"] == "vector"
        assert results[0]["content"] == "chunk text 1"
        assert results[0]["score"] == round(1 - 0.1, 4)
        assert results[0]["filename"] == "a.txt"

    @patch("rag_api.app.get_collection")
    def test_score_is_cosine_similarity(self, mock_get_col):
        """Score = 1 - cosine distance."""
        from rag_api.app import vector_search

        mock_col = MagicMock()
        mock_col.query.return_value = {
            "documents": [["text"]],
            "metadatas": [[{"document_id": "d", "filename": "f.txt", "chunk_index": 0}]],
            "distances": [[0.25]],
        }
        mock_get_col.return_value = mock_col

        results = vector_search([0.0] * 1536, k=1)
        assert results[0]["score"] == 0.75  # 1 - 0.25

    @patch("rag_api.app.get_collection")
    def test_requests_correct_k(self, mock_get_col):
        from rag_api.app import vector_search

        mock_col = MagicMock()
        mock_col.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        mock_get_col.return_value = mock_col

        vector_search([0.0] * 1536, k=7)
        call_kwargs = mock_col.query.call_args.kwargs
        assert call_kwargs["n_results"] == 7


@pytest.mark.unit
class TestKeywordSearch:
    @patch("rag_api.app.get_db")
    def test_returns_keyword_source_tag(self, mock_get_db):
        from rag_api.app import keyword_search

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {
                "content": "result text",
                "filename": "b.txt",
                "doc_id": "doc-2",
                "chunk_idx": 0,
                "score": 0.8,
            }
        ]
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_db.return_value = mock_conn

        results = keyword_search("machine learning", k=5)

        assert len(results) == 1
        assert results[0]["source"] == "keyword"

    @patch("rag_api.app.get_db")
    def test_empty_results_returns_empty_list(self, mock_get_db):
        from rag_api.app import keyword_search

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_db.return_value = mock_conn

        results = keyword_search("nonexistent query xyz", k=5)
        assert results == []

    @patch("rag_api.app.get_db")
    def test_closes_db_connection(self, mock_get_db):
        from rag_api.app import keyword_search

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_db.return_value = mock_conn

        keyword_search("test", k=3)
        mock_conn.close.assert_called_once()


@pytest.mark.unit
class TestLogQuery:
    @patch("rag_api.app.get_db")
    def test_inserts_query_log(self, mock_get_db):
        from rag_api.app import log_query

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_db.return_value = mock_conn

        chunks = [
            {
                "doc_id": "doc-1",
                "score": 0.9,
                "content": "text",
                "filename": "f.txt",
                "chunk_idx": 0,
                "source": "vector",
            }
        ]
        log_query("What is AI?", chunks, "AI is...", 1500)

        mock_cursor.execute.assert_called_once()
        sql = mock_cursor.execute.call_args[0][0]
        params = mock_cursor.execute.call_args[0][1]
        assert "query_logs" in sql
        assert "What is AI?" in params
        assert 1500 in params

    @patch("rag_api.app.get_db")
    def test_commits_after_insert(self, mock_get_db):
        from rag_api.app import log_query

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_db.return_value = mock_conn

        log_query("query", [], "answer", 100)
        mock_conn.commit.assert_called_once()


@pytest.mark.unit
class TestEmbedQuery:
    @patch("rag_api.app.client")
    def test_returns_embedding_vector(self, mock_client):
        from rag_api.app import embed_query

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.5] * 1536)]
        mock_client.embeddings.create.return_value = mock_response

        result = embed_query("What is machine learning?")

        assert len(result) == 1536
        assert result[0] == 0.5

    @patch("rag_api.app.client")
    def test_passes_query_as_list(self, mock_client):
        from rag_api.app import embed_query

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.0] * 1536)]
        mock_client.embeddings.create.return_value = mock_response

        embed_query("test query")

        call_kwargs = mock_client.embeddings.create.call_args.kwargs
        assert call_kwargs["input"] == ["test query"]


@pytest.mark.unit
class TestHealthEndpoint:
    @patch("rag_api.app.get_db")
    @patch("rag_api.app.get_collection")
    @patch("rag_api.app.get_chroma_client")
    def test_health_ok_when_all_services_up(self, mock_get_chroma, mock_get_col, mock_get_db):
        from rag_api.app import health

        mock_get_chroma.return_value = MagicMock()

        mock_col = MagicMock()
        mock_col.count.return_value = 0
        mock_get_col.return_value = mock_col

        mock_conn = MagicMock()
        mock_get_db.return_value = mock_conn

        result = health()
        assert result["status"] == "ok"
        assert result["chromadb"] is True
        assert result["db"] is True

    @patch("rag_api.app.get_db")
    @patch("rag_api.app.get_collection")
    @patch("rag_api.app.get_chroma_client")
    def test_health_degraded_when_chroma_down(self, mock_get_chroma, mock_get_col, mock_get_db):
        from rag_api.app import health

        mock_get_chroma.side_effect = Exception("ChromaDB down")
        mock_get_col.return_value = MagicMock()

        mock_conn = MagicMock()
        mock_get_db.return_value = mock_conn

        result = health()
        assert result["status"] == "degraded"
        assert result["chromadb"] is False
        assert result["db"] is True

    @patch("rag_api.app.get_db")
    @patch("rag_api.app.get_collection")
    @patch("rag_api.app.get_chroma_client")
    def test_health_degraded_when_db_down(self, mock_get_chroma, mock_get_col, mock_get_db):
        from rag_api.app import health

        mock_get_chroma.return_value = MagicMock()

        mock_col = MagicMock()
        mock_col.count.return_value = 0
        mock_get_col.return_value = mock_col

        mock_get_db.side_effect = Exception("Postgres down")

        result = health()
        assert result["status"] == "degraded"
        assert result["db"] is False
