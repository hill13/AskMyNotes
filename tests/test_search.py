"""Retrieval tests. Skipped automatically when no database is up.

These test the STRUCTURE of search (shape, ordering, scoping, limits), not
relevance -- relevance is meaningless under the fake embedder and belongs in the
Phase 4 eval harness, not a unit test.
"""
from __future__ import annotations

import textwrap

import pytest

from askmynotes import db
from askmynotes.embed import FakeEmbedder
from askmynotes.ingest import ingest_pdf
from askmynotes.query import retrieve


@pytest.fixture(scope="module")
def conn():
    try:
        with db.connect() as c:
            db.init_schema(c)
            yield c
    except Exception as exc:
        pytest.skip(f"database not available: {exc}")


def _pdf(tmp_path, name, topics):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    for t in topics:
        page = doc.new_page(); y = 60
        for line in textwrap.wrap(t * 10, 80):
            page.insert_text((50, y), line, fontsize=9); y += 12
    path = tmp_path / name
    doc.save(str(path)); doc.close()
    return path


@pytest.fixture
def corpus(conn, tmp_path):
    with conn.cursor() as cur:
        cur.execute("TRUNCATE documents CASCADE")
    conn.commit()
    a = ingest_pdf(_pdf(tmp_path, "a.pdf", ["Gradient descent and learning rates. "] * 3),
                   conn=conn, embedder=FakeEmbedder(), chunk_tokens=200, overlap_tokens=20)
    b = ingest_pdf(_pdf(tmp_path, "b.pdf", ["Normal distributions and variance. "] * 2),
                   conn=conn, embedder=FakeEmbedder(), chunk_tokens=200, overlap_tokens=20)
    return a, b


def test_returns_at_most_k(conn, corpus):
    assert len(retrieve("anything", conn=conn, embedder=FakeEmbedder(), k=2)) == 2


def test_k_larger_than_corpus_returns_everything(conn, corpus):
    a, b = corpus
    hits = retrieve("anything", conn=conn, embedder=FakeEmbedder(), k=999)
    assert len(hits) == a.chunk_count + b.chunk_count


def test_results_are_ordered_by_ascending_distance(conn, corpus):
    hits = retrieve("anything", conn=conn, embedder=FakeEmbedder(), k=5)
    distances = [h.distance for h in hits]
    assert distances == sorted(distances)


def test_document_filter_scopes_results(conn, corpus):
    a, _ = corpus
    hits = retrieve("anything", conn=conn, embedder=FakeEmbedder(), k=99,
                    document_id=a.document_id)
    assert hits
    assert {h.document_id for h in hits} == {a.document_id}


def test_hit_carries_citation_fields(conn, corpus):
    hit = retrieve("anything", conn=conn, embedder=FakeEmbedder(), k=1)[0]
    assert hit.filename.endswith(".pdf")
    assert hit.page_start <= hit.page_end
    assert hit.text
    assert hit.citation.startswith(hit.filename)


def test_exact_chunk_text_retrieves_itself_first(conn, corpus):
    """Sanity check on the mechanism: a chunk's own text must rank itself top."""
    with conn.cursor() as cur:
        cur.execute("SELECT chunk_index, text FROM chunks ORDER BY id LIMIT 1")
        idx, text = cur.fetchone()
    hit = retrieve(text, conn=conn, embedder=FakeEmbedder(), k=1)[0]
    assert hit.chunk_index == idx
    assert hit.distance == pytest.approx(0.0, abs=1e-6)


def test_similarity_is_one_minus_distance(conn, corpus):
    hit = retrieve("anything", conn=conn, embedder=FakeEmbedder(), k=1)[0]
    assert hit.similarity == pytest.approx(1.0 - hit.distance)


def test_invalid_k_rejected(conn, corpus):
    with pytest.raises(ValueError):
        retrieve("anything", conn=conn, embedder=FakeEmbedder(), k=0)
