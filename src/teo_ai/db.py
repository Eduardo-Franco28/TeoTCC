import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from teo_ai.config import settings

_pool: ConnectionPool | None = None


def create_schema() -> None:
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("CREATE SCHEMA IF NOT EXISTS teo")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS teo.documents (
                id BIGSERIAL PRIMARY KEY,
                file TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                link TEXT,
                status TEXT NOT NULL DEFAULT 'ativo',
                content_hash TEXT NOT NULL,
                ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS teo.chunks (
                id BIGSERIAL PRIMARY KEY,
                document_id BIGINT NOT NULL REFERENCES teo.documents(id) ON DELETE CASCADE,
                position INT NOT NULL,
                page INT,
                content TEXT NOT NULL,
                embedding halfvec({settings.teo_embedding_dim}) NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_idx
            ON teo.chunks USING hnsw (embedding halfvec_cosine_ops)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS teo.conversations (
                id UUID PRIMARY KEY,
                user_email TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS teo.messages (
                id BIGSERIAL PRIMARY KEY,
                conversation_id UUID NOT NULL REFERENCES teo.conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS messages_conversation_idx
            ON teo.messages (conversation_id, id)
        """)
        conn.execute("ALTER TABLE teo.conversations ADD COLUMN IF NOT EXISTS title TEXT")
        conn.execute("""
            CREATE INDEX IF NOT EXISTS conversations_user_idx
            ON teo.conversations (user_email, created_at DESC)
        """)


def open_pool() -> None:
    global _pool
    _pool = ConnectionPool(settings.database_url, min_size=1, max_size=5, configure=register_vector, open=True)


def close_pool() -> None:
    if _pool is not None:
        _pool.close()


def pool() -> ConnectionPool:
    if _pool is None:
        open_pool()
    return _pool
