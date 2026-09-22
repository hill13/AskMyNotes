"""Chunking: split a document into retrieval-sized pieces.

Why chunk at all? Retrieval returns whole units. A whole 40-page deck is one
unit, so retrieving it tells you nothing about *where* the answer is and blows
past the generation context. A single sentence is too small to carry the
surrounding meaning that makes it findable. A few hundred tokens is the sweet
spot: big enough to be self-contained, small enough that a top-k retrieval
returns mostly relevant text.

Why overlap? A fixed window will eventually cut through the middle of the one
explanation you needed. Overlapping the windows means any given sentence
appears near the *middle* of at least one chunk, with its context intact.

The window slides over tokens, not characters, so chunk size is expressed in
the same unit the embedding model's own limit is expressed in.
"""
from __future__ import annotations

from dataclasses import dataclass

from .extract import Page
from .tokenizer import decode, encode


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    token_count: int
    page_start: int
    page_end: int


def chunk_pages(
    pages: list[Page], chunk_tokens: int = 800, overlap_tokens: int = 100
) -> list[Chunk]:
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if not 0 <= overlap_tokens < chunk_tokens:
        raise ValueError("overlap_tokens must be >= 0 and < chunk_tokens")

    # Flatten the document into one token stream, remembering which page each
    # token came from. That way the sliding window is simple, and page
    # attribution falls out of it for free.
    tokens: list[int] = []
    token_page: list[int] = []
    for page in pages:
        if not page.text:
            continue
        page_tokens = encode(page.text + "\n\n")
        tokens.extend(page_tokens)
        token_page.extend([page.number] * len(page_tokens))

    if not tokens:
        return []

    stride = chunk_tokens - overlap_tokens
    chunks: list[Chunk] = []
    start = 0
    n = len(tokens)

    while start < n:
        end = min(start + chunk_tokens, n)
        window = tokens[start:end]
        text = decode(window).strip()
        if text:
            chunks.append(
                Chunk(
                    index=len(chunks),
                    text=text,
                    token_count=len(window),
                    page_start=token_page[start],
                    page_end=token_page[end - 1],
                )
            )
        if end == n:
            break
        start += stride

    return chunks


def total_tokens(pages: list[Page]) -> int:
    return sum(len(encode(p.text)) for p in pages if p.text)
