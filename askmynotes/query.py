"""Retrieval: question in, ranked chunks out.

This is the read half of the pipeline. Phase 2's generation step will sit on top
of `retrieve()`; keeping them separate means I can judge retrieval quality on its
own, before any LLM is involved. If an answer is wrong later, this is how I tell
"retrieval found the wrong chunks" apart from "the prompt was bad".
"""
from __future__ import annotations

import argparse

from . import config, db
from .embed import Embedder, get_embedder


def retrieve(
    question: str,
    *,
    conn,
    embedder: Embedder | None = None,
    k: int = 5,
    document_id: int | None = None,
) -> list[db.Hit]:
    """Embed the question and return the k nearest chunks.

    Only the QUESTION is embedded here -- chunk vectors were computed once at
    ingestion and have been sitting in Postgres ever since. That asymmetry is the
    point of precomputing: one embedding call per question, not thousands.
    """
    embedder = embedder or get_embedder()
    question_vector = embedder.embed([question])[0]
    return db.search(conn, question_vector, k=k, document_id=document_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m askmynotes.query",
        description="Retrieve the chunks most similar to a question (no LLM involved).",
    )
    parser.add_argument("question", help="the question to search for")
    parser.add_argument("-k", type=int, default=5, help="how many chunks to return")
    parser.add_argument("--document", type=int, help="restrict to one document id")
    parser.add_argument("--provider", default=config.EMBEDDING_PROVIDER,
                        choices=["fake", "openai"])
    parser.add_argument("--full", action="store_true", help="print whole chunks")
    args = parser.parse_args(argv)

    embedder = get_embedder(args.provider)
    with db.connect() as conn:
        hits = retrieve(args.question, conn=conn, embedder=embedder,
                        k=args.k, document_id=args.document)

    if not hits:
        print("No chunks found. Is anything ingested? Try: python -m askmynotes.stats")
        return 1

    if embedder.model.startswith("fake-"):
        print("WARNING: fake embedder -- this ranking is noise, not relevance.\n")

    print(f'Question: {args.question}\n')
    for rank, hit in enumerate(hits, start=1):
        body = hit.text if args.full else " ".join(hit.text.split())[:220] + "..."
        print(f"--- #{rank}  similarity {hit.similarity:.3f}  {hit.citation} "
              f"(chunk {hit.chunk_index})")
        print(f"{body}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
