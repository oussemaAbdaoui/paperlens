"""M1 (US-02) — Import a paper by arXiv ID or DOI.

arXiv: fetch metadata via the arXiv Atom API and download the PDF.
DOI:   resolve metadata via Crossref; download the PDF only if an open-access
       link is available. If no PDF is reachable, raise a clear error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import requests

from ..models import PaperMetadata

ARXIV_ID_RE = re.compile(r"^(?:arxiv:)?(\d{4}\.\d{4,5})(v\d+)?$", re.IGNORECASE)
DOI_ID_RE = re.compile(r"^(?:doi:)?(10\.\d{4,9}/\S+)$", re.IGNORECASE)

_TIMEOUT = 5  # US-02: invalid/inaccessible IDs error within 5 seconds


class ImportError_(Exception):
    """User-readable import failure."""


@dataclass
class ImportedPaper:
    pdf_bytes: bytes
    filename: str
    metadata: PaperMetadata


def import_identifier(identifier: str) -> ImportedPaper:
    identifier = identifier.strip()
    if ARXIV_ID_RE.match(identifier):
        return _import_arxiv(ARXIV_ID_RE.match(identifier).group(1))
    if DOI_ID_RE.match(identifier):
        return _import_doi(DOI_ID_RE.match(identifier).group(1))
    raise ImportError_(
        f"'{identifier}' is not a recognized arXiv ID (e.g. 2301.07041) or DOI (e.g. 10.1000/xyz)."
    )


def _import_arxiv(arxiv_id: str) -> ImportedPaper:
    import feedparser

    api = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        resp = requests.get(api, timeout=_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise ImportError_(f"Could not reach arXiv for '{arxiv_id}': {e}") from e

    feed = feedparser.parse(resp.content)
    if not feed.entries:
        raise ImportError_(f"No arXiv paper found for ID '{arxiv_id}'.")
    entry = feed.entries[0]

    meta = PaperMetadata(
        title=re.sub(r"\s+", " ", entry.get("title", "")).strip() or None,
        authors=[a.name for a in entry.get("authors", [])],
        year=_year_from_published(entry.get("published")),
        venue="arXiv",
        abstract=re.sub(r"\s+", " ", entry.get("summary", "")).strip() or None,
        doi=f"arXiv:{arxiv_id}",
        language="en",
    )

    pdf_url = next((l.href for l in entry.get("links", []) if l.get("type") == "application/pdf"), None)
    if not pdf_url:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    pdf_bytes = _download_pdf(pdf_url)
    return ImportedPaper(pdf_bytes=pdf_bytes, filename=f"arXiv_{arxiv_id}.pdf", metadata=meta)


def _import_doi(doi: str) -> ImportedPaper:
    doi = doi.rstrip(".")
    try:
        resp = requests.get(
            f"https://api.crossref.org/works/{doi}",
            timeout=_TIMEOUT,
            headers={"User-Agent": "PaperLens/1.0 (mailto:research@paperlens.local)"},
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise ImportError_(f"Could not resolve DOI '{doi}': {e}") from e

    msg = resp.json().get("message", {})
    title = (msg.get("title") or [None])[0]
    authors = [
        " ".join(p for p in (a.get("given"), a.get("family")) if p)
        for a in msg.get("author", [])
    ]
    year = None
    for key in ("published-print", "published-online", "issued"):
        parts = msg.get(key, {}).get("date-parts", [[None]])
        if parts and parts[0] and parts[0][0]:
            year = int(parts[0][0])
            break

    meta = PaperMetadata(
        title=title,
        authors=[a for a in authors if a],
        year=year,
        venue=(msg.get("container-title") or [None])[0],
        abstract=re.sub(r"<[^>]+>", "", msg.get("abstract", "")).strip() or None,
        doi=doi,
        language=msg.get("language"),
    )

    # Look for an open-access PDF link.
    pdf_url = next(
        (
            l.get("URL")
            for l in msg.get("link", [])
            if l.get("content-type") == "application/pdf"
        ),
        None,
    )
    if not pdf_url:
        raise ImportError_(
            f"Metadata for DOI '{doi}' was found, but no open-access PDF is available. "
            "Upload the PDF manually instead."
        )
    pdf_bytes = _download_pdf(pdf_url)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", doi)
    return ImportedPaper(pdf_bytes=pdf_bytes, filename=f"{safe}.pdf", metadata=meta)


def _download_pdf(url: str) -> bytes:
    try:
        resp = requests.get(url, timeout=max(_TIMEOUT, 30), headers={"User-Agent": "PaperLens/1.0"})
        resp.raise_for_status()
    except requests.RequestException as e:
        raise ImportError_(f"Failed to download PDF from {url}: {e}") from e
    if b"%PDF" not in resp.content[:1024]:
        raise ImportError_(f"The resource at {url} does not appear to be a PDF.")
    return resp.content


def _year_from_published(published: str | None) -> int | None:
    if not published:
        return None
    m = re.match(r"(\d{4})", published)
    return int(m.group(1)) if m else None
