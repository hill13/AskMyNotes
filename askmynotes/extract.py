"""PDF text extraction with PyMuPDF.

Text is kept per-page so every chunk can carry the page range it came from.
That page range is what turns a retrieved chunk into a citation the reader can
go verify in the original deck.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


class ExtractionError(Exception):
    """Raised when a PDF yields no usable text."""


@dataclass(frozen=True)
class Page:
    number: int  # 1-based, matches what the PDF viewer shows
    text: str


_MULTI_BLANK = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+\n")


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = _TRAILING_WS.sub("\n", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def extract_pages(path: str | Path) -> list[Page]:
    """Return one Page per page of the PDF. Pages with no text are kept as empty."""
    import pymupdf  # the `fitz` name is deprecated as of PyMuPDF 1.24.3

    path = Path(path)
    pages: list[Page] = []
    try:
        with pymupdf.open(path) as doc:
            for i, page in enumerate(doc, start=1):
                # "text" mode reads the page in natural reading order, which is
                # what we want for slides built out of separate text boxes.
                pages.append(Page(number=i, text=_clean(page.get_text("text"))))
    except pymupdf.FileDataError as exc:
        # Translate the library's error into ours, so callers never need to know
        # which PDF library we use -- and so one bad file cannot kill a batch.
        raise ExtractionError(
            f"{path.name} is not a readable PDF (corrupt, truncated, or not a PDF at all)."
        ) from exc

    if not any(p.text for p in pages):
        raise ExtractionError(
            f"No text found in {path.name}. It is probably a scanned/image-only PDF, "
            "which would need OCR before it can be ingested."
        )
    return pages
