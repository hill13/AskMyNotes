"""Retrieval metric math. No database or embedder needed."""
from __future__ import annotations

import pytest

from askmynotes.db import Hit
from evals.metrics import first_hit_rank, is_hit, random_baseline

DECK = "lecture.pdf"
OTHER = "other.pdf"


def hit(page_start, page_end, filename=DECK):
    return Hit(chunk_id=0, document_id=1, filename=filename, chunk_index=0,
               text="", page_start=page_start, page_end=page_end, distance=0.0)


def gold(*pages, document=DECK):
    return [(document, list(pages))]


# --- is_hit -----------------------------------------------------------------

def test_range_containing_gold_page_is_hit():
    assert is_hit(hit(18, 20), gold(19))


def test_range_edges_count():
    assert is_hit(hit(19, 22), gold(19))
    assert is_hit(hit(15, 19), gold(19))


def test_range_missing_gold_page_is_not_hit():
    assert not is_hit(hit(20, 21), gold(19))


def test_any_of_several_gold_pages_is_enough():
    assert is_hit(hit(21, 23), gold(19, 22))


def test_right_pages_wrong_document_is_not_hit():
    assert not is_hit(hit(18, 20, filename=OTHER), gold(19))


def test_empty_gold_never_hits():
    assert not is_hit(hit(1, 50), [])


def test_either_gold_document_counts():
    # Reused slides: the same content lives in two decks; citing either is right.
    both = [(DECK, [5]), (OTHER, [41])]
    assert is_hit(hit(4, 6), both)
    assert is_hit(hit(40, 42, filename=OTHER), both)


def test_pages_belong_to_their_own_document():
    # Page 41 is gold only in OTHER -- page 41 of DECK must not count.
    both = [(DECK, [5]), (OTHER, [41])]
    assert not is_hit(hit(40, 42), both)
    assert not is_hit(hit(4, 6, filename=OTHER), both)


# --- first_hit_rank ---------------------------------------------------------

def test_rank_is_one_based():
    hits = [hit(1, 3), hit(18, 20), hit(19, 19)]
    assert first_hit_rank(hits, gold(19)) == 2


def test_rank_none_when_nothing_hits():
    assert first_hit_rank([hit(1, 3), hit(4, 6)], gold(19)) is None


def test_rank_none_on_empty_results():
    assert first_hit_rank([], gold(19)) is None


# --- random_baseline --------------------------------------------------------

def test_baseline_one_relevant_chunk_is_k_over_n():
    assert random_baseline(7, 1, 5) == pytest.approx(5 / 7)


def test_baseline_two_relevant_chunks():
    # 1 - C(5,5)/C(7,5) = 1 - 1/21
    assert random_baseline(7, 2, 5) == pytest.approx(20 / 21)


def test_baseline_no_relevant_chunks_is_zero():
    assert random_baseline(7, 0, 5) == 0


def test_baseline_k_covering_corpus_is_certain():
    assert random_baseline(7, 1, 7) == 1
    assert random_baseline(7, 1, 10) == 1


def test_baseline_rejects_impossible_inputs():
    with pytest.raises(ValueError):
        random_baseline(7, 8, 5)
    with pytest.raises(ValueError):
        random_baseline(0, 0, 5)
