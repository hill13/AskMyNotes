"""Retrieval eval: did retrieve() put a gold chunk in the top k?

Runs every question in evals/questions.yaml through query.retrieve() against
the REAL database (everything ingested, not scoped to the gold document -- the
way the app is actually used), and reports recall@k and MRR next to the random
baseline for the same corpus. No LLM involved.

    python -m evals.retrieval_eval -k 5

Sanity check: with -k equal to the number of chunks, every chunk is retrieved,
so recall must be exactly 1.0. If it isn't, the harness is miscounting.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from askmynotes import config, db
from askmynotes.embed import get_embedder
from askmynotes.query import retrieve
from evals.metrics import first_hit_rank, is_hit, random_baseline

QUESTIONS = Path(__file__).parent / "questions.yaml"


def all_chunks(conn) -> list[db.Hit]:
    """Every chunk's location, as Hits, so g is counted with the SAME is_hit
    used for scoring -- recall and baseline can't disagree on 'relevant'."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT c.id, c.document_id, d.filename, c.chunk_index, c.page_start, c.page_end "
            "FROM chunks c JOIN documents d ON d.id = c.document_id"
        )
        return [
            db.Hit(chunk_id=r[0], document_id=r[1], filename=r[2], chunk_index=r[3],
                   text="", page_start=r[4], page_end=r[5], distance=0.0)
            for r in cur.fetchall()
        ]


def row(qid, category, rank, g, base, retrieved) -> str:
    return f"{qid:<5}  {category:<14}  {rank!s:>4}  {g!s:>3}  {base!s:>5}    {retrieved}"


def stored_models(conn) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT embedding_model FROM documents")
        return {r[0] for r in cur.fetchall()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.retrieval_eval",
                                     description="Score retrieval against the gold set.")
    parser.add_argument("-k", type=int, default=5, help="chunks retrieved per question")
    parser.add_argument("--provider", default=config.EMBEDDING_PROVIDER,
                        choices=["fake", "openai"])
    args = parser.parse_args(argv)

    questions = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))
    embedder = get_embedder(args.provider)

    with db.connect() as conn:
        corpus = all_chunks(conn)
        models = stored_models(conn)
        if not corpus:
            print("No chunks in the database. Ingest a document first.")
            return 1
        n = len(corpus)
        # One letter per deck (A, B, C... by filename), with a legend at the end.
        # With several decks, "p20-28" alone doesn't say which deck it came from.
        decks = sorted({c.filename for c in corpus})
        label = {name: chr(ord("A") + i) for i, name in enumerate(decks)}

        scored, ranks, baselines, warnings = [], [], [], []
        print(row("id", "category", "rank", "g", "base",
                  "retrieved: deck:pages  similarity  (* = gold)"))
        print()

        for q in questions:
            hits = retrieve(q["question"], conn=conn, embedder=embedder, k=args.k)
            gold = [(span["document"], span["pages"]) for span in q["gold"]]
            # Fixed-width cells so the same rank lines up down the column.
            shown = "   ".join(
                f"{f'{label[h.filename]}:p{h.page_start}-{h.page_end}':<9} {h.similarity:5.2f}"
                + ("*" if is_hit(h, gold) else " ")
                for h in hits
            )

            if not gold:
                # Out of scope: recall is undefined (nothing to find). Its top
                # similarity is still useful -- it's what the relevance gate
                # threshold will be chosen against.
                top = hits[0].similarity if hits else 0.0
                print(row(q["id"], q["category"], "-", "-", "-",
                          f"top similarity {top:.2f}   (not scored)"))
                continue

            # Checked per span: with two gold documents, a typo in one would
            # still leave g > 0 and quietly shrink the answer key.
            for document, pages in gold:
                if not any(is_hit(c, [(document, pages)]) for c in corpus):
                    # Usually a filename mismatch between the YAML and the DB.
                    # Without this it would look like a retrieval miss.
                    warnings.append(f"{q['id']}: no chunk in the DB matches "
                                    f"'{document}' pages {pages} -- check the filename / ingest")
            g = sum(is_hit(c, gold) for c in corpus)
            rank = first_hit_rank(hits, gold)
            base = random_baseline(n, g, args.k)

            scored.append(q["id"])
            ranks.append(rank)
            baselines.append(base)
            print(row(q["id"], q["category"], rank or "-", g, f"{base:.2f}", shown))

    recall = sum(r is not None for r in ranks) / len(ranks)
    mrr = sum(1 / r for r in ranks if r is not None) / len(ranks)
    mean_base = sum(baselines) / len(baselines)

    print()
    print("-" * 100)
    print(f"recall@{args.k}  {recall:.2f}     MRR  {mrr:.2f}     random baseline  {mean_base:.2f}     "
          f"scored  {len(scored)}     N = {n} chunks")
    print("decks:   " + "   ".join(f"{label[d]} = {d}" for d in decks))
    print(f"embedder: {embedder.model}")
    if embedder.model.startswith("fake-"):
        print("WARNING: fake embedder -- these numbers measure the harness, not retrieval.")
    if models != {embedder.model}:
        print(f"WARNING: stored vectors were made by {sorted(models)}, but the questions "
              f"were embedded by {embedder.model}. Re-ingest with --force.")
    for w in warnings:
        print(f"WARNING: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
