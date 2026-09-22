"""Central configuration, read once from the environment (.env supported)."""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    # tolerate inline comments in .env files: "10485760  # 10 MB"
    return int(raw.split("#", 1)[0].strip())


DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://askmynotes:askmynotes@localhost:5434/askmynotes"
)

# Two different limits, enforced at two different moments. See limits.py.
MAX_FILE_BYTES = _int("MAX_FILE_BYTES", 10 * 1024 * 1024)
MAX_DOCUMENT_TOKENS = _int("MAX_DOCUMENT_TOKENS", 50_000)

CHUNK_TOKENS = _int("CHUNK_TOKENS", 800)
CHUNK_OVERLAP_TOKENS = _int("CHUNK_OVERLAP_TOKENS", 100)

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "fake").strip().lower()
EMBEDDING_DIM = _int("EMBEDDING_DIM", 1536)
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# cl100k_base is the tokenizer used by text-embedding-3-* and gpt-4o-class models.
TOKENIZER_ENCODING = os.getenv("TOKENIZER_ENCODING", "cl100k_base")
