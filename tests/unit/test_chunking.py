"""
Unit tests for the text chunking logic in embedding_worker.
Tests chunk size, overlap, boundary conditions, and edge cases.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from embedding_worker.worker import chunk_text


@pytest.mark.unit
class TestChunkText:
    def test_short_text_single_chunk(self, short_text):
        """Text shorter than chunk_size produces exactly one chunk."""
        chunks = chunk_text(short_text, chunk_size=512, overlap=64)
        assert len(chunks) == 1
        assert chunks[0] == short_text

    def test_long_text_multiple_chunks(self, sample_text):
        """Long text is split into multiple chunks."""
        chunks = chunk_text(sample_text, chunk_size=100, overlap=10)
        assert len(chunks) > 1

    def test_chunk_size_respected(self, sample_text):
        """Each chunk stays within the token budget (with small tolerance)."""
        import tiktoken

        enc = tiktoken.encoding_for_model("text-embedding-3-small")
        chunk_size = 100
        chunks = chunk_text(sample_text, chunk_size=chunk_size, overlap=10)
        for chunk in chunks:
            tokens = len(enc.encode(chunk))
            assert tokens <= chunk_size + 5, f"Chunk too large: {tokens} tokens"

    def test_overlap_creates_continuity(self, sample_text):
        """Overlapping chunks share tokens at boundaries."""
        import tiktoken

        enc = tiktoken.encoding_for_model("text-embedding-3-small")
        overlap = 20
        chunks = chunk_text(sample_text, chunk_size=80, overlap=overlap)

        if len(chunks) < 2:
            pytest.skip("Need at least 2 chunks to test overlap")

        # Last tokens of chunk N should appear in beginning of chunk N+1
        for i in range(len(chunks) - 1):
            tokens_a = enc.encode(chunks[i])
            tokens_b = enc.encode(chunks[i + 1])
            # The tail of chunk A should be present at the head of chunk B
            tail = tokens_a[-overlap:]
            head = tokens_b[:overlap]
            overlap_count = len(set(tail) & set(head))
            assert overlap_count > 0, f"No overlap found between chunk {i} and {i + 1}"

    def test_empty_text_returns_empty_list(self, empty_text):
        """Empty text produces no chunks."""
        chunks = chunk_text(empty_text, chunk_size=512, overlap=64)
        assert chunks == []

    def test_all_text_preserved(self, sample_text):
        """All tokens from the original text appear across chunks."""
        import tiktoken

        enc = tiktoken.encoding_for_model("text-embedding-3-small")
        chunks = chunk_text(sample_text, chunk_size=100, overlap=20)
        original_tokens = set(enc.encode(sample_text))
        chunk_tokens = set()
        for chunk in chunks:
            chunk_tokens.update(enc.encode(chunk))
        # Every token in the original must appear in at least one chunk
        assert original_tokens.issubset(chunk_tokens)

    def test_single_word_text(self):
        """Single word produces one chunk."""
        chunks = chunk_text("hello", chunk_size=512, overlap=64)
        assert len(chunks) == 1
        assert "hello" in chunks[0]

    def test_no_overlap_no_duplication(self, sample_text):
        """With zero overlap, chunk boundaries don't duplicate tokens."""
        import tiktoken

        enc = tiktoken.encoding_for_model("text-embedding-3-small")
        chunks = chunk_text(sample_text, chunk_size=100, overlap=0)
        total_chunk_tokens = sum(len(enc.encode(c)) for c in chunks)
        original_token_count = len(enc.encode(sample_text))
        assert total_chunk_tokens == original_token_count

    @pytest.mark.parametrize(
        "chunk_size,overlap",
        [
            (64, 8),
            (128, 16),
            (256, 32),
            (512, 64),
        ],
    )
    def test_various_chunk_sizes(self, sample_text, chunk_size, overlap):
        """Chunking works correctly across common configurations."""
        chunks = chunk_text(sample_text, chunk_size=chunk_size, overlap=overlap)
        assert isinstance(chunks, list)
        assert all(isinstance(c, str) for c in chunks)
        assert all(len(c) > 0 for c in chunks)
