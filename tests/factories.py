"""Helpers to build synthetic parsed documents for tests (no real PDFs needed)."""

from __future__ import annotations

from app.services.processing import Page, ParsedDocument
from app.models import PaperMetadata


def make_doc(sections: dict[str, str], *, start_page: int = 1) -> ParsedDocument:
    """Build a ParsedDocument where each (heading -> body) becomes page text.

    Each section is placed on its own page with the heading on its own line.
    """
    pages = []
    page_no = start_page
    for heading, body in sections.items():
        text = f"{heading}\n\n{body}"
        pages.append(Page(number=page_no, text=text))
        page_no += 1
    return ParsedDocument(pages=pages, metadata=PaperMetadata())


LOREM = (
    "Neural retrieval systems encode passages into dense vectors and compare "
    "them with a query embedding using cosine similarity. "
)


def long_body(times: int) -> str:
    return (LOREM * times).strip()
