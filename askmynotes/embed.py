"""Embedding providers behind one interface.

An embedding is a fixed-length list of floats that positions a piece of text in
a semantic space: texts that mean similar things land near each other, even
when they share no words. That is the whole trick behind retrieval here --
"what is backpropagation?" can find a chunk that says "the chain rule is
applied backwards through the network" because the two sit close together in
that space, which keyword search would miss.

The vectors are only comparable to each other if they come from the *same*
model, so the model name and dimension are recorded on every document.

Two implementations:
  FakeEmbedder   - deterministic, offline, free. Same text -> same vector, so
                   the DB layer and chunking are fully testable without a key.
                   It has no real semantics: unrelated texts get unrelated
                   vectors, so retrieval quality under it is meaningless.
  OpenAIEmbedder - the real thing.
"""
from __future__ import annotations

import hashlib
import math
import random
from typing import Protocol

from . import config


class Embedder(Protocol):
    model: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


class FakeEmbedder:
    """Deterministic stand-in. Seeded by the hash of the text, then normalized."""

    def __init__(self, dim: int = config.EMBEDDING_DIM):
        self.dim = dim
        self.model = f"fake-{dim}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
            rng = random.Random(seed)
            out.append(_normalize([rng.gauss(0.0, 1.0) for _ in range(self.dim)]))
        return out


class OpenAIEmbedder:
    BATCH_SIZE = 100

    def __init__(self, model: str = config.OPENAI_EMBEDDING_MODEL, dim: int = config.EMBEDDING_DIM):
        from openai import OpenAI

        if not config.OPENAI_API_KEY:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is not set. "
                "Add it to .env, or use EMBEDDING_PROVIDER=fake."
            )
        self.model = model
        self.dim = dim
        self._client = OpenAI(api_key=config.OPENAI_API_KEY)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            response = self._client.embeddings.create(
                model=self.model, input=batch, dimensions=self.dim
            )
            # The API does not guarantee ordering; index is authoritative.
            out.extend(d.embedding for d in sorted(response.data, key=lambda d: d.index))
        return out


def get_embedder(provider: str | None = None) -> Embedder:
    provider = (provider or config.EMBEDDING_PROVIDER).strip().lower()
    if provider == "fake":
        return FakeEmbedder()
    if provider == "openai":
        return OpenAIEmbedder()
    raise ValueError(f"Unknown EMBEDDING_PROVIDER {provider!r}; expected 'fake' or 'openai'.")
