"""The ingestion pipeline: PDF -> text -> limits -> chunks -> embeddings -> pgvector.

`ingest_pdf` is deliberately a plain function with no CLI or HTTP in it, so the
Phase 2 FastAPI endpoint can call exactly the same code path.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

from . import config, db
from .chunk import chunk_pages, total_tokens
from .embed import Embedder, get_embedder
from .extract import ExtractionError, extract_pages
from .limits import LimitExceeded, check_file_size, check_token_budget, format_bytes


@dataclass
class IngestResult:
    document_id: int
    filename: str
    page_count: int
    token_count: int
    chunk_count: int
    embedding_model: str
    skipped: bool = False


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ingest_pdf(
    path: str | Path,
    *,
    conn,
    embedder: Embedder | None = None,
    chunk_tokens: int = config.CHUNK_TOKENS,
    overlap_tokens: int = config.CHUNK_OVERLAP_TOKENS,
    max_file_bytes: int = config.MAX_FILE_BYTES,
    max_tokens: int = config.MAX_DOCUMENT_TOKENS,
    force: bool = False,
    verbose: bool = False,
) -> IngestResult:
    path = Path(path)
    embedder = embedder or get_embedder()

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    # 1. Cheap guard first: reject oversized files before parsing anything.
    size = check_file_size(path, max_file_bytes)
    log(f"file       {path.name} ({format_bytes(size)})")

    content_hash = file_hash(path)
    existing = db.find_document(conn, content_hash)
    if existing and not force:
        doc_id, name = existing
        log(f"skip       identical content already ingested as document {doc_id} ({name})")
        return IngestResult(doc_id, name, 0, 0, 0, embedder.model, skipped=True)

    # Release the read snapshot before the slow work below; embedding can take
    # a while and there is no reason to hold a transaction open across it.
    conn.rollback()

    # 2. Extract, then apply the limit that actually matters.
    pages = extract_pages(path)
    tokens = total_tokens(pages)
    check_token_budget(tokens, max_tokens)
    log(f"extract    {len(pages)} pages, {tokens:,} tokens")

    # 3. Chunk.
    chunks = chunk_pages(pages, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
    if not chunks:
        raise ExtractionError(f"{path.name} produced no chunks.")
    log(
        f"chunk      {len(chunks)} chunks "
        f"(size {chunk_tokens}, overlap {overlap_tokens} tokens)"
    )

    # 4. Embed.
    log(f"embed      {len(chunks)} chunks with {embedder.model}")
    vectors = embedder.embed([c.text for c in chunks])
    if any(len(v) != embedder.dim for v in vectors):
        raise RuntimeError("Embedder returned a vector of unexpected dimension.")

    # 5. Store. Everything above can fail; only now is it safe to drop the
    # previous version of this document. Delete + insert share one transaction,
    # so a failure here leaves the old document intact.
    if existing:
        log(f"replace    superseding previous document {existing[0]}")
        db.delete_document(conn, existing[0])

    doc_id = db.insert_document(
        conn,
        filename=path.name,
        content_hash=content_hash,
        page_count=len(pages),
        token_count=tokens,
        chunk_count=len(chunks),
        embedding_model=embedder.model,
        embedding_dim=embedder.dim,
    )
    db.insert_chunks(conn, doc_id, chunks, vectors)
    conn.commit()
    log(f"store      document {doc_id}, {len(chunks)} chunks written")

    return IngestResult(
        document_id=doc_id,
        filename=path.name,
        page_count=len(pages),
        token_count=tokens,
        chunk_count=len(chunks),
        embedding_model=embedder.model,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m askmynotes.ingest",
        description="Ingest a PDF of course notes into pgvector.",
    )
    parser.add_argument("pdf", nargs="+", help="one or more PDF paths")
    parser.add_argument("--chunk-tokens", type=int, default=config.CHUNK_TOKENS)
    parser.add_argument("--overlap-tokens", type=int, default=config.CHUNK_OVERLAP_TOKENS)
    parser.add_argument(
        "--provider",
        default=config.EMBEDDING_PROVIDER,
        choices=["fake", "openai"],
        help="embedding provider (default from EMBEDDING_PROVIDER)",
    )
    parser.add_argument("--force", action="store_true", help="re-ingest even if unchanged")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    embedder = get_embedder(args.provider)
    failures = 0

    with db.connect() as conn:
        db.init_schema(conn, dim=embedder.dim)
        for pdf in args.pdf:
            if not args.quiet:
                print(f"\n=== {pdf} ===")
            try:
                result = ingest_pdf(
                    pdf,
                    conn=conn,
                    embedder=embedder,
                    chunk_tokens=args.chunk_tokens,
                    overlap_tokens=args.overlap_tokens,
                    force=args.force,
                    verbose=not args.quiet,
                )
            except (LimitExceeded, ExtractionError, FileNotFoundError) as exc:
                conn.rollback()
                sys.stdout.flush()
                print(f"FAILED     {exc}", file=sys.stderr, flush=True)
                failures += 1
                continue
            if not args.quiet and not result.skipped:
                print(
                    f"OK         document {result.document_id}: "
                    f"{result.chunk_count} chunks from {result.page_count} pages",
                    flush=True,
                )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
