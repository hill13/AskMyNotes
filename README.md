# AskMyNotes

A RAG system that answers questions about your own course notes, grounded only in
those notes, with an eval harness that measures whether it actually works.

**Stack:** FastAPI · PostgreSQL + pgvector · OpenAI · minimal frontend

## Status

- [x] **Phase 1 — Ingestion.** PDF extraction → limit enforcement → chunking →
      embedding → storage in pgvector.
- [ ] Phase 2 — Query: embed question, retrieve top-k, generate grounded answer.
- [ ] Phase 3 — Evals.
- [ ] Phase 4 — API + frontend.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

cp .env.example .env            # defaults work as-is

docker compose up -d            # Postgres 17 + pgvector on localhost:5434
python -m askmynotes.db         # create the schema
```

Port **5434** is used on purpose so this does not collide with a local Postgres
install. If something already listens there, change the host-side port in both
`docker-compose.yml` and `DATABASE_URL`. On Windows, Docker will happily bind a
port that another process already holds and the *other* process keeps winning the
connection, which shows up as a confusing `password authentication failed` error
rather than a bind failure — so check with `netstat -ano | findstr 543` first.

## Ingesting notes

```bash
python -m askmynotes.ingest data/lecture-notes.pdf
python -m askmynotes.ingest data/*.pdf --chunk-tokens 600 --overlap-tokens 80
python -m askmynotes.ingest data/notes.pdf --force      # re-ingest after a change
python -m askmynotes.stats --document 1                 # inspect what landed
```

Re-ingesting the same file is a no-op: documents are keyed by a SHA-256 of their
bytes, so an unchanged file is skipped unless you pass `--force`.

### Embedding providers

`EMBEDDING_PROVIDER` selects the embedder:

- `fake` (default) — deterministic hash-seeded vectors. Free, offline, and makes
  the whole pipeline testable without an API key. **Retrieval quality under it is
  meaningless** — it has no semantics. It exists to test plumbing, not answers.
- `openai` — `text-embedding-3-small`, 1536 dimensions. Set `OPENAI_API_KEY`.

Both produce 1536-dim vectors, so the schema does not change when you switch.
Vectors from different models are not comparable, so **re-ingest everything with
`--force` after switching providers.** Every document records the model that
produced its vectors.

## Tests

```bash
python -m pytest
```

The tests build their PDFs on the fly with PyMuPDF and use the fake embedder, so
they need no corpus and never call an API. The unit tests (extraction, chunking,
limits, embedding) need nothing else; the end-to-end tests in
`tests/test_ingest_db.py` use the database and **skip themselves** if it is not
running, so `docker compose up -d` first to exercise the full path.

---

## The three things this phase is about

### What is a chunk, and why chunk at all?

A chunk is one retrievable unit of the document — here, a window of ~800 tokens
of text with the page numbers it came from.

You chunk because retrieval returns whole units, so the size of a unit decides
what retrieval can even express. Store a 40-page deck as one row and the only
answer to "what's in my notes about gradient descent?" is "this whole deck" —
useless as a citation, and far too much to feed the generator. Store one sentence
per row and each row is too thin to carry the context that makes it findable; a
sentence saying "it converges faster in practice" is meaningless without knowing
what "it" is.

A few hundred tokens is the working range: large enough to be self-contained,
small enough that a top-5 retrieval is mostly signal.

**Why overlap?** A fixed window will eventually slice through the middle of the
one explanation you needed, leaving the setup in chunk *N* and the conclusion in
chunk *N+1*, with neither one answering the question. Overlapping the windows by
~100 tokens means any given sentence sits near the middle of at least one chunk,
with its surrounding context intact. The cost is roughly 12% more rows and
embedding spend at these settings — cheap insurance.

Chunking happens over the **token stream**, not the character string, so chunk
size is expressed in the same unit as the embedding model's own input limit.
Each token carries the page it came from, so page attribution falls out of the
sliding window for free (`chunk.py`).

### What is an embedding?

An embedding is a fixed-length vector of floats — 1536 of them here — that places
a piece of text at a point in a semantic space. The model is trained so that
texts meaning similar things land near each other, **even when they share no
words at all**.

That is the entire retrieval mechanism. The question "what is backpropagation?"
can find a chunk that reads "the chain rule is applied backwards through the
network" because those two land close together, while keyword search finds
nothing — the word "backpropagation" never appears. Conversely, "bank account"
and "river bank" land far apart despite sharing a word.

Concretely: you embed each chunk once at ingestion time, embed the question at
query time, and ask the database for the chunks whose vectors are nearest. With
normalized vectors, cosine similarity is just the dot product, and "nearest"
means "most likely to be about the same thing."

Two constraints follow:
- Vectors are only comparable within the same model. Mixing models silently
  produces garbage rankings, which is why the model name is stored per document.
- The comparison must be the one the index was built for. The index here is
  `hnsw (embedding vector_cosine_ops)`, so the query side must search with
  cosine distance (`<=>`) or the index will not be used.

### Why limit by tokens, not file size?

There are two caps, enforced at two different moments, for two different reasons.

**File size (10 MB), checked *before* opening the PDF.** This is a server-safety
guard. It stops a 400 MB upload from exhausting memory or disk before anything
has been parsed. It is a cheap check on an untrusted input, and it tells you
nothing about the document's content.

**Token count (50k), checked *after* extraction.** This is the cap that actually
matters, because tokens are the unit everything downstream is denominated in:
embedding cost is per token, the model's input limit is in tokens, index size
grows with tokens.

File size is a terrible proxy for it. A 9 MB deck that is mostly screenshots may
extract to 3,000 tokens; a 400 KB text-only PDF can extract to 80,000. The two
numbers are only loosely related, so a byte limit either lets through documents
that are too expensive to process or blocks documents that are perfectly cheap.

The ordering is the point: **the cheap guard runs on the untrusted input, the
meaningful limit runs on the measured content.** You cannot count tokens without
extracting first, and you should not extract without a size guard first.
