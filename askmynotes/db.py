"""Postgres + pgvector storage layer."""
from __future__ import annotations

import argparse
from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector

from . import config
from .chunk import Chunk

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id              BIGSERIAL PRIMARY KEY,
    filename        TEXT        NOT NULL,
    content_hash    TEXT        NOT NULL UNIQUE,
    page_count      INT         NOT NULL,
    token_count     INT         NOT NULL,
    chunk_count     INT         NOT NULL,
    embedding_model TEXT        NOT NULL,
    embedding_dim   INT         NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT    NOT NULL,
    text        TEXT   NOT NULL,
    token_count INT    NOT NULL,
    page_start  INT    NOT NULL,
    page_end    INT    NOT NULL,
    embedding   VECTOR({dim}) NOT NULL,
    UNIQUE (document_id, chunk_index)
);

-- Cosine distance, because that is what the query side will search with.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
"""


def _to_vector_literal(vec: list[float]) -> str:
    """pgvector's text input format. Used with an explicit ::vector cast so we do
    not depend on adapter registration having succeeded."""
    return "[" + ",".join(repr(float(v)) for v in vec) + "]"


@contextmanager
def connect(dsn: str | None = None):
    with psycopg.connect(dsn or config.DATABASE_URL) as conn:
        try:
            register_vector(conn)
        except Exception:
            # The vector type does not exist until init_schema() has run. Reads
            # in later phases need the adapter; writes here use an explicit cast.
            pass
        yield conn


def init_schema(conn, dim: int = config.EMBEDDING_DIM) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA.format(dim=dim))
    conn.commit()


def find_document(conn, content_hash: str) -> tuple[int, str] | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, filename FROM documents WHERE content_hash = %s", (content_hash,)
        )
        return cur.fetchone()


def delete_document(conn, document_id: int) -> None:
    """Chunks cascade. Does NOT commit -- the caller decides the transaction
    boundary, so a replace can delete and re-insert atomically."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))


def insert_document(
    conn,
    *,
    filename: str,
    content_hash: str,
    page_count: int,
    token_count: int,
    chunk_count: int,
    embedding_model: str,
    embedding_dim: int,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (filename, content_hash, page_count, token_count,
                                   chunk_count, embedding_model, embedding_dim)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                filename,
                content_hash,
                page_count,
                token_count,
                chunk_count,
                embedding_model,
                embedding_dim,
            ),
        )
        return cur.fetchone()[0]


def insert_chunks(
    conn, document_id: int, chunks: list[Chunk], embeddings: list[list[float]]
) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("chunk/embedding count mismatch")
    rows = [
        (document_id, c.index, c.text, c.token_count, c.page_start, c.page_end,
         _to_vector_literal(e))
        for c, e in zip(chunks, embeddings)
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks (document_id, chunk_index, text, token_count,
                                page_start, page_end, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
            """,
            rows,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the AskMyNotes schema.")
    parser.add_argument("--dim", type=int, default=config.EMBEDDING_DIM)
    args = parser.parse_args()
    with connect() as conn:
        init_schema(conn, dim=args.dim)
    print(f"Schema ready at {config.DATABASE_URL} (vector dim {args.dim}).")


if __name__ == "__main__":
    main()
