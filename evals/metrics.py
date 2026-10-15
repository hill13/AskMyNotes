"""Retrieval metrics. Pure functions -- no database, no embedder -- so the
scoring logic is unit-tested on its own before any number from it is trusted.

A retrieved chunk is a hit when it comes from the gold document AND its page
range overlaps a gold page (decisions.md #14: labels are by page, not chunk,
so they survive re-chunking).
"""
from __future__ import annotations

from math import comb

from askmynotes.db import Hit


def is_hit(hit: Hit, gold_document: str, gold_pages: list[int]) -> bool:
    # Document first: page 11 of some other file is not the answer.
    if hit.filename != gold_document:
        return False
    return any(hit.page_start <= p <= hit.page_end for p in gold_pages)


def first_hit_rank(hits: list[Hit], gold_document: str, gold_pages: list[int]) -> int | None:
    """1-based rank of the first relevant chunk, or None if none was retrieved.

    recall@k = share of questions where this is not None.
    MRR      = mean of 1/rank, counting None as 0.
    """
    for rank, hit in enumerate(hits, start=1):
        if is_hit(hit, gold_document, gold_pages):
            return rank
    return None


def random_baseline(n: int, g: int, k: int) -> float:
    """Chance that k chunks drawn at random from n include at least one of the
    g relevant ones: 1 - C(n-g, k) / C(n, k).

    C(n-g, k) counts the draws that miss every relevant chunk; C(n, k) counts
    all draws. A retriever is only doing something if it beats this.
    """
    if n < 1 or not 0 <= g <= n or k < 1:
        raise ValueError(f"need n >= 1, 0 <= g <= n, k >= 1 (got n={n}, g={g}, k={k})")
    # Retrieval can't return more chunks than exist, so asking for k > n
    # really means taking all n.
    k = min(k, n)
    return 1 - comb(n - g, k) / comb(n, k)
