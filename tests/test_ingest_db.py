"""End-to-end ingestion tests. Skipped automatically when no database is up."""
from __future__ import annotations

import textwrap

import pytest

from askmynotes import db
from askmynotes.embed import FakeEmbedder
from askmynotes.ingest import ingest_pdf
from askmynotes.limits import LimitExceeded


@pytest.fixture(scope="module")
def conn():
    try:
        with db.connect() as c:
            db.init_schema(c)
            yield c
    except Exception as exc:  # no DB reachable
        pytest.skip(f"database not available: {exc}")


@pytest.fixture
def notes_pdf(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    body = ("Gradient descent minimizes a loss function by stepping downhill. "
            "The learning rate controls the step size. ") * 12
    doc = pymupdf.open()
    for i in range(1, 4):
        page = doc.new_page()
        page.insert_text((60, 60), f"Lecture {i}", fontsize=16)
        y = 100
        for line in textwrap.wrap(body, 80):
            page.insert_text((60, y), line, fontsize=10)
            y += 14
    path = tmp_path / "notes.pdf"
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture(autouse=True)
def clean(conn):
    with conn.cursor() as cur:
        cur.execute("TRUNCATE documents CASCADE")
    conn.commit()


def test_round_trip(conn, notes_pdf):
    result = ingest_pdf(notes_pdf, conn=conn, embedder=FakeEmbedder())
    assert result.chunk_count > 0
    assert result.page_count == 3
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks WHERE document_id = %s", (result.document_id,))
        assert cur.fetchone()[0] == result.chunk_count


def test_identical_file_is_skipped(conn, notes_pdf):
    first = ingest_pdf(notes_pdf, conn=conn, embedder=FakeEmbedder())
    second = ingest_pdf(notes_pdf, conn=conn, embedder=FakeEmbedder())
    assert second.skipped
    assert second.document_id == first.document_id
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM documents")
        assert cur.fetchone()[0] == 1


def test_force_replaces_without_duplicating(conn, notes_pdf):
    ingest_pdf(notes_pdf, conn=conn, embedder=FakeEmbedder(), chunk_tokens=400, overlap_tokens=50)
    second = ingest_pdf(
        notes_pdf, conn=conn, embedder=FakeEmbedder(), chunk_tokens=200, overlap_tokens=20, force=True
    )
    assert not second.skipped
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM documents")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT count(*) FROM chunks")
        assert cur.fetchone()[0] == second.chunk_count


def test_failed_force_does_not_destroy_existing(conn, notes_pdf):
    """Regression: validation must happen before the old version is deleted."""
    first = ingest_pdf(notes_pdf, conn=conn, embedder=FakeEmbedder())

    with pytest.raises(LimitExceeded):
        ingest_pdf(notes_pdf, conn=conn, embedder=FakeEmbedder(), max_tokens=10, force=True)
    conn.rollback()

    with conn.cursor() as cur:
        cur.execute("SELECT id, chunk_count FROM documents")
        rows = cur.fetchall()
    assert rows == [(first.document_id, first.chunk_count)]


def test_chunks_are_cosine_searchable(conn, notes_pdf):
    result = ingest_pdf(notes_pdf, conn=conn, embedder=FakeEmbedder())
    probe = FakeEmbedder().embed(["gradient descent"])[0]
    literal = db._to_vector_literal(probe)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT chunk_index, embedding <=> %s::vector FROM chunks "
            "WHERE document_id = %s ORDER BY 2 LIMIT 3",
            (literal, result.document_id),
        )
        hits = cur.fetchall()
    assert len(hits) == min(3, result.chunk_count)
    assert all(0.0 <= d <= 2.0 for _, d in hits)
