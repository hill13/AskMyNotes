"""Inspect what is actually in the database. Useful for sanity-checking a run."""
from __future__ import annotations

import argparse

from . import db


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m askmynotes.stats")
    parser.add_argument("--document", type=int, help="show chunk previews for this document id")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args(argv)

    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, filename, page_count, token_count, chunk_count,
                   embedding_model, created_at
            FROM documents ORDER BY id
            """
        )
        rows = cur.fetchall()
        if not rows:
            print("No documents ingested yet.")
            return 0
        print(f"{'id':>4}  {'pages':>5}  {'tokens':>7}  {'chunks':>6}  model                 file")
        for r in rows:
            print(f"{r[0]:>4}  {r[2]:>5}  {r[3]:>7,}  {r[4]:>6}  {r[5]:<20}  {r[1]}")

        if args.document:
            cur.execute(
                """
                SELECT chunk_index, token_count, page_start, page_end, text
                FROM chunks WHERE document_id = %s
                ORDER BY chunk_index LIMIT %s
                """,
                (args.document, args.limit),
            )
            for idx, ntok, p0, p1, text in cur.fetchall():
                pages = f"p{p0}" if p0 == p1 else f"p{p0}-{p1}"
                preview = " ".join(text.split())[:200]
                print(f"\n--- chunk {idx}  {ntok} tokens  {pages} ---\n{preview}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
