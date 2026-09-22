"""Token counting.

Everything downstream is measured in *tokens*, not characters or pages, because
tokens are what the embedding and generation APIs actually bill and truncate on.
"""
from __future__ import annotations

import functools

from .config import TOKENIZER_ENCODING


@functools.lru_cache(maxsize=4)
def get_encoding(name: str = TOKENIZER_ENCODING):
    import tiktoken

    try:
        return tiktoken.get_encoding(name)
    except Exception as exc:  # pragma: no cover - network/first-run path
        raise RuntimeError(
            f"Could not load the '{name}' tokenizer. tiktoken downloads its BPE table "
            "on first use, so this needs network access once. After that it is cached "
            "(set TIKTOKEN_CACHE_DIR to control where)."
        ) from exc


def encode(text: str) -> list[int]:
    return get_encoding().encode(text)


def decode(tokens: list[int]) -> str:
    return get_encoding().decode(tokens)


def count_tokens(text: str) -> int:
    return len(encode(text))
