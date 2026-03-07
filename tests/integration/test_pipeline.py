"""
Integration tests for the full RAG pipeline.
Requires all services running: make up

Run with:
  make test-integration
"""

import time

import httpx
import pytest

UPLOAD_URL = "http://localhost:8001"
RAG_URL = "http://localhost:8000"
TIMEOUT = 60


def services_available() -> bool:
    try:
        r1 = httpx.get(f"{UPLOAD_URL}/health", timeout=3)
        r2 = httpx.get(f"{RAG_URL}/health", timeout=3)
        return r1.status_code == 200 and r2.status_code == 200
    except Exception:
        return False


skip_if_no_services = pytest.mark.skipif(
    not services_available(),
    reason="Services not running — start with 'make up'",
)


# ── Upload Service ─────────────────────────────────────────────────────────────


@pytest.mark.integration
@skip_if_no_services
class TestUploadService:
    def test_health_returns_ok(self):
        resp = httpx.get(f"{UPLOAD_URL}/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_upload_txt_file(self):
        content = b"# Test\nThis document is about quantum computing."
        resp = httpx.post(
            f"{UPLOAD_URL}/upload",
            files={"file": ("test.txt", content, "text/plain")},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "document_id" in data
        assert data["status"] == "pending"

    def test_upload_markdown_file(self):
        content = b"# ML Guide\nMachine learning is fascinating."
        resp = httpx.post(
            f"{UPLOAD_URL}/upload",
            files={"file": ("guide.md", content, "text/markdown")},
        )
        assert resp.status_code == 202

    def test_upload_unsupported_type_rejected(self):
        resp = httpx.post(
            f"{UPLOAD_URL}/upload",
            files={"file": ("doc.exe", b"binary", "application/octet-stream")},
        )
        assert resp.status_code == 400

    def test_upload_empty_file_rejected(self):
        resp = httpx.post(
            f"{UPLOAD_URL}/upload",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert resp.status_code == 400

    def test_list_documents(self):
        resp = httpx.get(f"{UPLOAD_URL}/documents")
        assert resp.status_code == 200
        assert "documents" in resp.json()

    def test_get_document_status(self):
        content = b"Status check document."
        upload = httpx.post(
            f"{UPLOAD_URL}/upload",
            files={"file": ("status_test.txt", content, "text/plain")},
        )
        doc_id = upload.json()["document_id"]
        resp = httpx.get(f"{UPLOAD_URL}/documents/{doc_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == doc_id

    def test_get_nonexistent_document_404(self):
        resp = httpx.get(f"{UPLOAD_URL}/documents/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404


# ── Batch Upload ───────────────────────────────────────────────────────────────


@pytest.mark.integration
@skip_if_no_services
class TestBatchUpload:
    def test_batch_upload_multiple_files(self):
        """Multiple files are all accepted and queued."""
        files = [
            ("files", ("batch1.txt", b"Document about neural networks.", "text/plain")),
            ("files", ("batch2.txt", b"Document about transformers.", "text/plain")),
            (
                "files",
                ("batch3.txt", b"Document about reinforcement learning.", "text/plain"),
            ),
        ]
        resp = httpx.post(f"{UPLOAD_URL}/upload/batch", files=files, timeout=30)
        assert resp.status_code == 202
        data = resp.json()
        assert data["accepted"] == 3
        assert data["rejected"] == 0
        assert len(data["results"]) == 3
        for r in data["results"]:
            assert r["status"] == "pending"
            assert "document_id" in r

    def test_batch_upload_mixed_valid_invalid(self):
        """Valid files accepted, invalid files rejected in same batch."""
        files = [
            ("files", ("good.txt", b"Valid content here.", "text/plain")),
            ("files", ("bad.exe", b"binary content", "application/octet-stream")),
            ("files", ("empty.txt", b"", "text/plain")),
        ]
        resp = httpx.post(f"{UPLOAD_URL}/upload/batch", files=files, timeout=30)
        assert resp.status_code == 202
        data = resp.json()
        assert data["accepted"] == 1
        assert data["rejected"] == 2

    def test_batch_upload_exceeds_limit(self):
        """More than 20 files returns 400."""
        files = [("files", (f"file{i}.txt", b"content", "text/plain")) for i in range(21)]
        resp = httpx.post(f"{UPLOAD_URL}/upload/batch", files=files, timeout=30)
        assert resp.status_code == 400

    def test_batch_each_file_gets_unique_id(self):
        """Each file in a batch gets a distinct document_id."""
        files = [
            ("files", ("a.txt", b"Content A about Python.", "text/plain")),
            ("files", ("b.txt", b"Content B about Java.", "text/plain")),
        ]
        resp = httpx.post(f"{UPLOAD_URL}/upload/batch", files=files, timeout=30)
        data = resp.json()
        ids = [r["document_id"] for r in data["results"]]
        assert len(ids) == len(set(ids))  # all unique

    def test_batch_all_appear_in_documents_list(self):
        """All accepted files appear in /documents after batch upload."""
        files = [
            ("files", ("list_test_a.txt", b"List test content A.", "text/plain")),
            ("files", ("list_test_b.txt", b"List test content B.", "text/plain")),
        ]
        resp = httpx.post(f"{UPLOAD_URL}/upload/batch", files=files, timeout=30)
        doc_ids = {r["document_id"] for r in resp.json()["results"]}

        docs_resp = httpx.get(f"{UPLOAD_URL}/documents?limit=50")
        listed_ids = {d["id"] for d in docs_resp.json()["documents"]}
        assert doc_ids.issubset(listed_ids)


# ── Delete + Re-ingest ─────────────────────────────────────────────────────────


@pytest.mark.integration
@skip_if_no_services
class TestDeleteAndReingest:
    def _upload(self, content: bytes, filename: str) -> str:
        resp = httpx.post(
            f"{UPLOAD_URL}/upload",
            files={"file": (filename, content, "text/plain")},
        )
        assert resp.status_code == 202
        return resp.json()["document_id"]

    def test_delete_document(self):
        doc_id = self._upload(b"Document to delete.", "delete_me.txt")
        resp = httpx.delete(f"{UPLOAD_URL}/documents/{doc_id}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

    def test_deleted_document_not_found(self):
        doc_id = self._upload(b"Delete and verify gone.", "gone.txt")
        httpx.delete(f"{UPLOAD_URL}/documents/{doc_id}")
        resp = httpx.get(f"{UPLOAD_URL}/documents/{doc_id}")
        assert resp.status_code == 404

    def test_delete_nonexistent_returns_404(self):
        resp = httpx.delete(f"{UPLOAD_URL}/documents/00000000-0000-0000-0000-000000000001")
        assert resp.status_code == 404

    def test_reingest_resets_status(self):
        doc_id = self._upload(b"Document to reingest.", "reingest_me.txt")
        resp = httpx.post(f"{UPLOAD_URL}/documents/{doc_id}/reingest")
        assert resp.status_code == 200
        assert resp.json()["status"] == "pending"

    def test_reingest_nonexistent_returns_404(self):
        resp = httpx.post(f"{UPLOAD_URL}/documents/00000000-0000-0000-0000-000000000001/reingest")
        assert resp.status_code == 404


# ── RAG API ────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@skip_if_no_services
class TestRAGApi:
    def test_health_returns_ok(self):
        resp = httpx.get(f"{RAG_URL}/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid"])
    def test_search_modes(self, mode):
        resp = httpx.post(
            f"{RAG_URL}/query",
            json={"query": "machine learning", "search_mode": mode, "top_k": 3},
            timeout=30,
        )
        assert resp.status_code == 200

    def test_query_response_schema(self):
        resp = httpx.post(f"{RAG_URL}/query", json={"query": "What is AI?"}, timeout=30)
        data = resp.json()
        assert isinstance(data["query"], str)
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["latency_ms"], int)
        assert isinstance(data["num_chunks"], int)


# ── Full Pipeline ──────────────────────────────────────────────────────────────


@pytest.mark.integration
@skip_if_no_services
class TestFullPipeline:
    UNIQUE_CONTENT = (
        "XYZZY42 is a special token used only in pipeline integration tests. "
        "The answer to the pipeline test question is forty-two."
    )

    def _upload_and_wait(self, content: str, filename: str, max_wait: int = TIMEOUT) -> str:
        resp = httpx.post(
            f"{UPLOAD_URL}/upload",
            files={"file": (filename, content.encode(), "text/plain")},
        )
        assert resp.status_code == 202
        doc_id = resp.json()["document_id"]
        deadline = time.time() + max_wait
        while time.time() < deadline:
            status = httpx.get(f"{UPLOAD_URL}/documents/{doc_id}").json()["status"]
            if status == "complete":
                return doc_id
            if status == "failed":
                pytest.fail(f"Document {doc_id} failed processing")
            time.sleep(2)
        pytest.fail(f"Document not processed within {max_wait}s")

    def test_single_upload_then_query(self):
        self._upload_and_wait(self.UNIQUE_CONTENT, "pipeline_test.txt")
        resp = httpx.post(
            f"{RAG_URL}/query",
            json={"query": "What is XYZZY42?", "search_mode": "hybrid", "top_k": 5},
            timeout=30,
        )
        data = resp.json()
        assert data["num_chunks"] > 0
        assert any("pipeline_test.txt" in s["filename"] for s in data["sources"])

    def test_batch_upload_then_query_finds_all(self):
        """Batch upload 3 docs — all should be queryable after processing."""
        docs = [
            ("batch_a.txt", "ALPHA42 is the first unique batch token."),
            ("batch_b.txt", "BETA42 is the second unique batch token."),
            ("batch_c.txt", "GAMMA42 is the third unique batch token."),
        ]
        files = [("files", (name, content.encode(), "text/plain")) for name, content in docs]
        resp = httpx.post(f"{UPLOAD_URL}/upload/batch", files=files, timeout=30)
        doc_ids = [r["document_id"] for r in resp.json()["results"]]

        # Wait for all to complete
        deadline = time.time() + TIMEOUT
        while time.time() < deadline:
            statuses = [
                httpx.get(f"{UPLOAD_URL}/documents/{did}").json()["status"] for did in doc_ids
            ]
            if all(s == "complete" for s in statuses):
                break
            if any(s == "failed" for s in statuses):
                pytest.fail("One or more batch docs failed processing")
            time.sleep(3)
        else:
            pytest.fail("Batch docs not all processed within timeout")

        # Query for each unique token
        for token in ["ALPHA42", "BETA42", "GAMMA42"]:
            resp = httpx.post(
                f"{RAG_URL}/query",
                json={
                    "query": f"What is {token}?",
                    "search_mode": "hybrid",
                    "top_k": 5,
                },
                timeout=30,
            )
            assert resp.json()["num_chunks"] > 0, f"{token} not found after batch upload"
