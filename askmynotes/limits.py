"""Ingestion limits.

Two caps, enforced at two different points in the pipeline:

1. File size, BEFORE we open the PDF. This is a server-safety guard: it stops a
   400 MB upload from exhausting memory or disk before we have parsed anything.
   It says nothing about how much *text* is inside.

2. Token count, AFTER extraction. This is the cap that actually matters. A 2 MB
   PDF of dense text can hold far more text than a 9 MB PDF of screenshots, and
   embedding cost, index size, and retrieval quality all scale with token count,
   not bytes. Bytes are about the transport; tokens are about the work.
"""
from __future__ import annotations

from pathlib import Path


class LimitExceeded(Exception):
    """Raised when a document violates an ingestion limit."""


def format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1_048_576:.1f} MB"


def check_file_size(path: str | Path, max_bytes: int) -> int:
    size = Path(path).stat().st_size
    if size > max_bytes:
        raise LimitExceeded(
            f"File is {format_bytes(size)}, limit is {format_bytes(max_bytes)}."
        )
    return size


def check_token_budget(token_count: int, max_tokens: int) -> int:
    if token_count > max_tokens:
        raise LimitExceeded(
            f"Extracted text is {token_count:,} tokens, limit is {max_tokens:,}. "
            "Split the document and ingest it in parts."
        )
    return token_count
