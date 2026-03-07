"""
Gradio UI
---------
Unified interface for:
  - Uploading TXT/Markdown files
  - Querying the RAG agent
  - Viewing document list and status
"""

import os

import gradio as gr
import httpx
import structlog

log = structlog.get_logger()

UPLOAD_URL = os.getenv("UPLOAD_SERVICE_URL", "http://upload-service:8001")
RAG_URL = os.getenv("RAG_API_URL", "http://rag-api:8000")


# ── API Calls ─────────────────────────────────────────────────────────────────


def upload_files(files) -> str:
    if not files:
        return "⚠️ No files selected."

    # Single file — use /upload
    if not isinstance(files, list):
        files = [files]

    if len(files) == 1:
        file = files[0]
        try:
            filename = os.path.basename(file.name)
            ext = os.path.splitext(filename)[1].lower()
            mime = "application/pdf" if ext == ".pdf" else "text/plain"
            with open(file.name, "rb") as f:
                resp = httpx.post(
                    f"{UPLOAD_URL}/upload",
                    files={"file": (filename, f, mime)},
                    timeout=60,
                )
            if resp.status_code == 202:
                data = resp.json()
                return (
                    f"✅ **Uploaded successfully!**\n\n"
                    f"- **Document ID:** `{data['document_id']}`\n"
                    f"- **Filename:** {data['filename']}\n"
                    f"- **Size:** {data['size_bytes']:,} bytes\n"
                    f"- **Status:** {data['status']}\n\n"
                    f"Check the Documents tab for processing status."
                )
            return f"❌ Upload failed: {resp.text}"
        except Exception as e:
            return f"❌ Error: {e}"

    # Multiple files — use /upload/batch
    try:
        file_tuples = []
        for file in files:
            filename = os.path.basename(file.name)
            ext = os.path.splitext(filename)[1].lower()
            mime = "application/pdf" if ext == ".pdf" else "text/plain"
            file_tuples.append(("files", (filename, open(file.name, "rb"), mime)))

        resp = httpx.post(f"{UPLOAD_URL}/upload/batch", files=file_tuples, timeout=120)

        if resp.status_code == 202:
            data = resp.json()
            accepted = data["accepted"]
            rejected = data["rejected"]
            lines = [f"✅ **{accepted} file(s) queued for processing**"]
            if rejected:
                lines.append(f"⚠️ {rejected} file(s) rejected:")
            for r in data["results"]:
                if r.get("status") == "rejected":
                    lines.append(f"  - ❌ {r['filename']}: {r['reason']}")
                else:
                    lines.append(f"  - ✅ {r['filename']} (`{r['document_id'][:8]}...`)")
            lines.append("\nCheck the Documents tab for processing status.")
            return "\n".join(lines)
        return f"❌ Batch upload failed: {resp.text}"
    except Exception as e:
        return f"❌ Error: {e}"


def query_rag(question: str, search_mode: str, top_k: int) -> tuple[str, str]:
    if not question.strip():
        return "⚠️ Please enter a question.", ""
    try:
        resp = httpx.post(
            f"{RAG_URL}/query",
            json={"query": question, "search_mode": search_mode.lower(), "top_k": top_k},
            timeout=60,
        )
        if resp.status_code == 200:
            data = resp.json()
            answer = data["answer"]
            latency = data["latency_ms"]
            sources = data["sources"]

            sources_md = "\n".join(
                f"- **{s['filename']}** — score: {s['score']:.3f} ({s['source']})" for s in sources
            )
            stats = f"⏱️ {latency}ms · {data['num_chunks']} chunks retrieved"

            return answer, f"{stats}\n\n**Sources:**\n{sources_md}"
        else:
            return f"❌ Query failed: {resp.text}", ""
    except Exception as e:
        return f"❌ Error: {e}", ""


def list_documents() -> str:
    try:
        resp = httpx.get(f"{UPLOAD_URL}/documents", timeout=10)
        if resp.status_code == 200:
            docs = resp.json()["documents"]
            if not docs:
                return "No documents uploaded yet."
            rows = []
            for d in docs:
                status_icon = {
                    "complete": "✅",
                    "processing": "⏳",
                    "pending": "🕐",
                    "failed": "❌",
                }.get(d["status"], "❓")
                rows.append(
                    f"{status_icon} **{d['filename']}** — "
                    f"{d['chunk_count']} chunks · "
                    f"{d['size_bytes']:,} bytes · "
                    f"`{d['id'][:8]}...`"
                )
            return "\n\n".join(rows)
        return f"❌ Error: {resp.text}"
    except Exception as e:
        return f"❌ Error: {e}"


def delete_document(doc_id: str) -> str:
    if not doc_id.strip():
        return "⚠️ Enter a document ID."
    try:
        resp = httpx.delete(f"{UPLOAD_URL}/documents/{doc_id.strip()}", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            vectors = data["vectors_removed"]
            return f"✅ Deleted `{doc_id[:8]}...` — {vectors} vectors removed."
        return f"❌ {resp.json().get('detail', resp.text)}"
    except Exception as e:
        return f"❌ Error: {e}"


def reingest_document(doc_id: str) -> str:
    if not doc_id.strip():
        return "⚠️ Enter a document ID."
    try:
        resp = httpx.post(f"{UPLOAD_URL}/documents/{doc_id.strip()}/reingest", timeout=10)
        if resp.status_code == 200:
            return f"✅ Re-ingestion queued for `{doc_id[:8]}...`"
        return f"❌ {resp.json().get('detail', resp.text)}"
    except Exception as e:
        return f"❌ Error: {e}"


def check_health() -> str:
    results = []
    for name, url in [("Upload Service", UPLOAD_URL), ("RAG API", RAG_URL)]:
        try:
            resp = httpx.get(f"{url}/health", timeout=5)
            data = resp.json()
            icon = "✅" if data.get("status") == "ok" else "⚠️"
            results.append(f"{icon} **{name}**: {data}")
        except Exception as e:
            results.append(f"❌ **{name}**: {e}")
    return "\n\n".join(results)


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="RAG Pipeline") as demo:
    gr.Markdown("# 📚 Real-Time RAG Pipeline\nUpload documents and query them with AI.")

    with gr.Tabs():
        # ── Upload Tab ──────────────────────────────────────────────────────
        with gr.TabItem("📤 Upload"):
            gr.Markdown("Upload TXT or Markdown files to be indexed.")
            with gr.Row():
                upload_input = gr.File(
                    label="Select files (hold Ctrl/Cmd for multiple)",
                    file_types=[".txt", ".md", ".markdown", ".pdf"],
                    file_count="multiple",
                )
                upload_button = gr.Button("Upload", variant="primary")
            upload_output = gr.Markdown()
            upload_button.click(fn=upload_files, inputs=upload_input, outputs=upload_output)

        # ── Query Tab ───────────────────────────────────────────────────────
        with gr.TabItem("🔍 Query"):
            gr.Markdown("Ask questions about your uploaded documents.")
            with gr.Row():
                query_input = gr.Textbox(
                    label="Your question",
                    placeholder="What is the main topic of the documents?",
                    lines=2,
                    scale=4,
                )
                with gr.Column(scale=1):
                    search_mode = gr.Radio(
                        choices=["Hybrid", "Vector", "Keyword"],
                        value="Hybrid",
                        label="Search mode",
                    )
                    top_k = gr.Slider(1, 10, value=5, step=1, label="Top K chunks")

            query_button = gr.Button("Ask", variant="primary")
            answer_output = gr.Markdown(label="Answer")
            sources_output = gr.Markdown(label="Sources & Stats")

            query_button.click(
                fn=query_rag,
                inputs=[query_input, search_mode, top_k],
                outputs=[answer_output, sources_output],
            )

        # ── Documents Tab ───────────────────────────────────────────────────
        with gr.TabItem("📋 Documents"):
            gr.Markdown("All uploaded documents and their processing status.")
            refresh_button = gr.Button("🔄 Refresh")
            docs_output = gr.Markdown()
            refresh_button.click(fn=list_documents, outputs=docs_output)
            demo.load(fn=list_documents, outputs=docs_output)

        # ── Health Tab ──────────────────────────────────────────────────────
        with gr.TabItem("❤️ Health"):
            health_button = gr.Button("Check Services")
            health_output = gr.Markdown()
            health_button.click(fn=check_health, outputs=health_output)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
