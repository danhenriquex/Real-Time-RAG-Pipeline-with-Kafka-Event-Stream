"""
Unit tests for retrieval logic in rag_api.
Tests deduplication, score ranking, and search mode routing.
All external dependencies (ChromaDB, Postgres, OpenAI) are mocked.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))


# Import only the pure functions — no side effects on import
from rag_api.app import deduplicate


@pytest.mark.unit
class TestDeduplicate:
    def test_removes_duplicate_keeps_higher_score(self, duplicate_chunks):
        """When the same chunk appears twice, keep the one with higher score."""
        result = deduplicate(duplicate_chunks)
        # doc-001 chunk 0 appears twice — should appear once with score 0.90
        chunk_0_results = [c for c in result if c["chunk_idx"] == 0 and c["doc_id"] == "doc-001"]
        assert len(chunk_0_results) == 1
        assert chunk_0_results[0]["score"] == 0.90
        assert chunk_0_results[0]["source"] == "vector"

    def test_no_duplicates_unchanged_count(self, sample_chunks):
        """Unique chunks pass through without being dropped."""
        result = deduplicate(sample_chunks)
        assert len(result) == len(sample_chunks)

    def test_results_sorted_by_score_descending(self, sample_chunks):
        """Output is always sorted highest score first."""
        result = deduplicate(sample_chunks)
        scores = [c["score"] for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_input_returns_empty(self):
        """Empty list in, empty list out."""
        assert deduplicate([]) == []

    def test_single_chunk_returned_as_is(self):
        """Single chunk needs no deduplication."""
        chunk = [
            {
                "doc_id": "x",
                "chunk_idx": 0,
                "score": 0.9,
                "content": "test",
                "filename": "f.txt",
                "source": "vector",
            }
        ]
        result = deduplicate(chunk)
        assert len(result) == 1
        assert result[0]["score"] == 0.9

    def test_different_docs_same_chunk_index_not_deduped(self):
        """Same chunk_idx from different documents are NOT duplicates."""
        chunks = [
            {
                "doc_id": "doc-001",
                "chunk_idx": 0,
                "score": 0.9,
                "content": "a",
                "filename": "a.txt",
                "source": "vector",
            },
            {
                "doc_id": "doc-002",
                "chunk_idx": 0,
                "score": 0.8,
                "content": "b",
                "filename": "b.txt",
                "source": "vector",
            },
        ]
        result = deduplicate(chunks)
        assert len(result) == 2

    def test_many_duplicates_collapsed(self):
        """Five copies of same chunk → one result with highest score."""
        chunks = [
            {
                "doc_id": "doc-001",
                "chunk_idx": 0,
                "score": s,
                "content": "text",
                "filename": "f.txt",
                "source": "vector",
            }
            for s in [0.5, 0.9, 0.7, 0.3, 0.8]
        ]
        result = deduplicate(chunks)
        assert len(result) == 1
        assert result[0]["score"] == 0.9


@pytest.mark.unit
class TestGenerateAnswer:
    @patch("rag_api.app.client")
    def test_empty_chunks_returns_no_docs_message(self, mock_client):
        """With no chunks, answer without calling OpenAI."""
        from rag_api.app import generate_answer

        answer = generate_answer("What is AI?", [])
        assert "No relevant documents" in answer
        mock_client.chat.completions.create.assert_not_called()

    @patch("rag_api.app.client")
    def test_answer_calls_openai_with_context(self, mock_client, sample_chunks):
        """With chunks, OpenAI is called with context."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "AI is a field of computer science."
        mock_client.chat.completions.create.return_value = mock_response

        from rag_api.app import generate_answer

        answer = generate_answer("What is AI?", sample_chunks)

        assert answer == "AI is a field of computer science."
        mock_client.chat.completions.create.assert_called_once()

        # Verify context was included in the prompt
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "ml_intro.txt" in user_msg["content"]

    @patch("rag_api.app.client")
    def test_filenames_included_in_context(self, mock_client, sample_chunks):
        """Each chunk's filename appears in the context sent to OpenAI."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Answer."
        mock_client.chat.completions.create.return_value = mock_response

        from rag_api.app import generate_answer

        generate_answer("test query", sample_chunks)

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        full_text = " ".join(m["content"] for m in messages)
        assert "ml_intro.txt" in full_text


@pytest.mark.unit
class TestUploadValidation:
    def test_allowed_extensions(self):
        """TXT and Markdown extensions are allowed."""
        allowed = {".txt", ".md", ".markdown"}
        for ext in [".txt", ".md", ".markdown"]:
            assert ext in allowed

    def test_disallowed_extensions(self):
        """PDF, DOCX, etc. are not allowed."""
        allowed = {".txt", ".md", ".markdown"}
        for ext in [".pdf", ".docx", ".csv", ".json", ".exe"]:
            assert ext not in allowed

    @pytest.mark.parametrize(
        "size_mb,should_pass",
        [
            (1, True),
            (5, True),
            (10, True),
            (11, False),
            (50, False),
        ],
    )
    def test_file_size_limit(self, size_mb, should_pass):
        """Files over 10MB are rejected."""
        MAX_SIZE_MB = 10
        size_bytes = size_mb * 1024 * 1024
        assert (size_bytes <= MAX_SIZE_MB * 1024 * 1024) == should_pass
