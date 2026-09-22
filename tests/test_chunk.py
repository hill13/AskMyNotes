import pytest

from askmynotes.chunk import chunk_pages, total_tokens
from askmynotes.extract import Page
from askmynotes.tokenizer import count_tokens

WORDS = " ".join(f"word{i}" for i in range(600))


def make_pages(n=3):
    return [Page(number=i, text=f"Page {i}. {WORDS}") for i in range(1, n + 1)]


def test_chunks_respect_size_and_are_ordered():
    chunks = chunk_pages(make_pages(), chunk_tokens=200, overlap_tokens=50)
    assert chunks
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.token_count <= 200 for c in chunks)


def test_overlap_repeats_content_between_neighbours():
    chunks = chunk_pages(make_pages(1), chunk_tokens=200, overlap_tokens=50)
    assert len(chunks) > 1
    tail = chunks[0].text.split()[-10:]
    assert any(w in chunks[1].text for w in tail)


def test_no_overlap_still_works():
    chunks = chunk_pages(make_pages(1), chunk_tokens=200, overlap_tokens=0)
    assert len(chunks) > 1


def test_pages_are_attributed_and_monotonic():
    chunks = chunk_pages(make_pages(4), chunk_tokens=300, overlap_tokens=30)
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 4
    for c in chunks:
        assert c.page_start <= c.page_end
    for a, b in zip(chunks, chunks[1:]):
        assert a.page_start <= b.page_start


def test_empty_pages_are_skipped():
    assert chunk_pages([Page(1, ""), Page(2, "")], 100, 10) == []


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_pages(make_pages(1), chunk_tokens=100, overlap_tokens=100)


def test_total_tokens_matches_tokenizer():
    pages = make_pages(2)
    assert total_tokens(pages) == sum(count_tokens(p.text) for p in pages)
