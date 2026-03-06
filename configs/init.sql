-- Documents metadata table
CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename    TEXT NOT NULL,
    file_type   TEXT NOT NULL,
    size_bytes  INTEGER,
    status      TEXT NOT NULL DEFAULT 'pending',  -- pending | processing | complete | failed
    chunk_count INTEGER DEFAULT 0,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Chunks table with full-text search
CREATE TABLE IF NOT EXISTS chunks (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content     TEXT NOT NULL,
    token_count INTEGER,
    chroma_id   TEXT,                             -- reference to ChromaDB vector
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    -- full-text search vector
    content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

-- Query logs for monitoring
CREATE TABLE IF NOT EXISTS query_logs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query           TEXT NOT NULL,
    retrieved_ids   TEXT[],
    answer          TEXT,
    latency_ms      INTEGER,
    num_chunks      INTEGER,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_documents_status   ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_created  ON documents(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chunks_document    ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_fts         ON chunks USING GIN(content_tsv);
CREATE INDEX IF NOT EXISTS idx_query_logs_created ON query_logs(created_at DESC);
