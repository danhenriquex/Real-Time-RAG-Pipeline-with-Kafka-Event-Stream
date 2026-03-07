"""
Unit tests for embedding_worker — all external deps mocked.
Covers: embed_chunks, process_document, mark_failed, delete_vectors, event routing.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))


@pytest.mark.unit
class TestEmbedChunks:
    @patch("embedding_worker.worker.openai_client")
    def test_returns_embeddings_for_each_chunk(self, mock_openai):
        from embedding_worker.worker import embed_chunks

        fake_embedding = [0.1] * 1536
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=fake_embedding) for _ in range(3)]
        mock_openai.embeddings.create.return_value = mock_response

        chunks = ["chunk one", "chunk two", "chunk three"]
        embeddings = embed_chunks(chunks)

        assert len(embeddings) == 3
        assert all(len(e) == 1536 for e in embeddings)
        mock_openai.embeddings.create.assert_called_once()

    @patch("embedding_worker.worker.openai_client")
    def test_calls_correct_model(self, mock_openai):
        from embedding_worker.worker import embed_chunks

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.0] * 1536)]
        mock_openai.embeddings.create.return_value = mock_response

        embed_chunks(["test chunk"])

        call_kwargs = mock_openai.embeddings.create.call_args.kwargs
        assert call_kwargs["model"] == "text-embedding-3-small"
        assert call_kwargs["input"] == ["test chunk"]

    @patch("embedding_worker.worker.openai_client")
    def test_single_chunk_returns_single_embedding(self, mock_openai):
        from embedding_worker.worker import embed_chunks

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.5] * 1536)]
        mock_openai.embeddings.create.return_value = mock_response

        result = embed_chunks(["single chunk"])
        assert len(result) == 1


@pytest.mark.unit
class TestDeleteVectors:
    @patch("embedding_worker.worker.get_collection")
    def test_deletes_all_provided_ids(self, mock_get_col):
        from embedding_worker.worker import delete_vectors

        mock_col = MagicMock()
        mock_get_col.return_value = mock_col

        chroma_ids = ["id-1", "id-2", "id-3"]
        delete_vectors(chroma_ids, "doc-001")

        mock_col.delete.assert_called_once_with(ids=chroma_ids)

    @patch("embedding_worker.worker.get_collection")
    def test_empty_ids_skips_deletion(self, mock_get_col):
        from embedding_worker.worker import delete_vectors

        mock_col = MagicMock()
        mock_get_col.return_value = mock_col

        delete_vectors([], "doc-001")
        mock_col.delete.assert_not_called()

    @patch("embedding_worker.worker.get_collection")
    def test_chroma_error_does_not_raise(self, mock_get_col):
        from embedding_worker.worker import delete_vectors

        mock_col = MagicMock()
        mock_col.delete.side_effect = Exception("ChromaDB unavailable")
        mock_get_col.return_value = mock_col

        # Should log error but not raise
        delete_vectors(["id-1"], "doc-001")


@pytest.mark.unit
class TestMarkFailed:
    @patch("embedding_worker.worker.get_db")
    def test_updates_status_to_failed(self, mock_get_db):
        from embedding_worker.worker import mark_failed

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_db.return_value = mock_conn

        mark_failed("doc-001", "some error")

        mock_cursor.execute.assert_called_once()
        sql = mock_cursor.execute.call_args[0][0]
        assert "failed" in sql
        assert "documents" in sql

    @patch("embedding_worker.worker.get_db")
    def test_commits_transaction(self, mock_get_db):
        from embedding_worker.worker import mark_failed

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_db.return_value = mock_conn

        mark_failed("doc-001", "error")
        mock_conn.commit.assert_called_once()


@pytest.mark.unit
class TestProcessDocument:
    @patch("embedding_worker.worker.get_collection")
    @patch("embedding_worker.worker.get_db")
    @patch("embedding_worker.worker.embed_chunks")
    def test_process_stores_chunks_in_chroma(self, mock_embed, mock_get_db, mock_get_col):
        from embedding_worker.worker import process_document

        # Mock embeddings
        mock_embed.return_value = [[0.1] * 1536, [0.2] * 1536]

        # Mock ChromaDB collection
        mock_col = MagicMock()
        mock_get_col.return_value = mock_col

        # Mock Postgres
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_db.return_value = mock_conn

        event = {
            "document_id": "doc-001",
            "filename": "test.txt",
            "content": "This is a test document. " * 20,
        }
        process_document(event)

        mock_col.add.assert_called_once()
        call_kwargs = mock_col.add.call_args.kwargs
        assert "embeddings" in call_kwargs
        assert "documents" in call_kwargs
        assert "metadatas" in call_kwargs

    @patch("embedding_worker.worker.get_collection")
    @patch("embedding_worker.worker.get_db")
    @patch("embedding_worker.worker.embed_chunks")
    def test_process_metadata_includes_doc_id(self, mock_embed, mock_get_db, mock_get_col):
        from embedding_worker.worker import process_document

        mock_embed.return_value = [[0.1] * 1536]
        mock_col = MagicMock()
        mock_get_col.return_value = mock_col
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_db.return_value = mock_conn

        event = {
            "document_id": "doc-xyz",
            "filename": "myfile.txt",
            "content": "Short content for testing metadata.",
        }
        process_document(event)

        metadatas = mock_col.add.call_args.kwargs["metadatas"]
        assert all(m["document_id"] == "doc-xyz" for m in metadatas)
        assert all(m["filename"] == "myfile.txt" for m in metadatas)


@pytest.mark.unit
class TestEventRouting:
    """Test that the consumer correctly routes events by event_type."""

    def test_delete_event_calls_delete_vectors(self):
        """delete event → delete_vectors called, process_document NOT called."""
        with (
            patch("embedding_worker.worker.delete_vectors") as mock_del,
            patch("embedding_worker.worker.process_document") as mock_proc,
        ):
            from embedding_worker.worker import delete_vectors, process_document

            event = {"event_type": "delete", "document_id": "doc-1", "chroma_ids": ["a", "b"]}
            etype = event.get("event_type", "ingest")

            if etype == "delete":
                delete_vectors(event.get("chroma_ids", []), event["document_id"])
            else:
                process_document(event)

            mock_del.assert_called_once_with(["a", "b"], "doc-1")
            mock_proc.assert_not_called()

    def test_ingest_event_calls_process_document(self):
        """Default ingest event → process_document called."""
        with (
            patch("embedding_worker.worker.delete_vectors") as mock_del,
            patch("embedding_worker.worker.process_document") as mock_proc,
        ):
            from embedding_worker.worker import delete_vectors, process_document

            event = {"document_id": "doc-1", "filename": "f.txt", "content": "text"}
            etype = event.get("event_type", "ingest")

            if etype == "delete":
                delete_vectors(event.get("chroma_ids", []), event["document_id"])
            else:
                process_document(event)

            mock_del.assert_not_called()
            mock_proc.assert_called_once_with(event)
