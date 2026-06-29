"""M2 — Document processing. Locked parser: PyMuPDF.

Extracts per-page text and best-effort metadata (title, authors, year, venue,
abstract, DOI, language). Per US-04, fields that cannot be confidently
extracted are left blank — never hallucinated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..models import PaperMetadata

DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
ARXIV_RE = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)
VENUE_RE = re.compile(
    r"\b(?:Proceedings of|In Proceedings|Conference on|Journal of|"
    r"NeurIPS|ICML|ICLR|ACL|EMNLP|NAACL|CVPR|ICCV|ECCV|AAAI|IJCAI|KDD|SIGIR)\b[^\n]*",
    re.IGNORECASE,
)


@dataclass
class Page:
    number: int  # 1-based
    text: str


@dataclass
class ParsedDocument:
    pages: list[Page] = field(default_factory=list)
    metadata: PaperMetadata = field(default_factory=PaperMetadata)

    @property
    def num_pages(self) -> int:
        return len(self.pages)

    @property
    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages)


def parse_pdf(path: str) -> ParsedDocument:
    """Parse a PDF into pages + extracted metadata using PyMuPDF."""
    import fitz  # PyMuPDF

    doc = fitz.open(path)
    pages = [Page(number=i + 1, text=doc.load_page(i).get_text("text")) for i in range(doc.page_count)]
    embedded_title = (doc.metadata or {}).get("title") or None
    doc.close()

    parsed = ParsedDocument(pages=pages)
    parsed.metadata = extract_metadata(pages, embedded_title=embedded_title)
    return parsed


def detect_language(text: str) -> str | None:
    """Lightweight language detection. Returns an ISO-639-1 code or None.

    Uses langdetect when available; otherwise a tiny English stop-word heuristic.
    Never guesses on too little text — returns None instead."""
    sample = text.strip()
    if len(sample) < 40:
        return None
    try:  # pragma: no cover - optional dependency
        from langdetect import detect

        return detect(sample)
    except Exception:
        common = {"the", "and", "of", "to", "in", "is", "we", "this", "for", "are"}
        words = re.findall(r"[a-zA-Z]+", sample.lower())
        if not words:
            return None
        ratio = sum(w in common for w in words) / len(words)
        return "en" if ratio > 0.08 else None


def extract_metadata(pages: list[Page], embedded_title: str | None = None) -> PaperMetadata:
    meta = PaperMetadata()
    if not pages:
        return meta

    first = pages[0].text
    head = "\n".join(p.text for p in pages[:2])

    # --- DOI / arXiv ---
    m = DOI_RE.search(head)
    if m:
        meta.doi = m.group(0).rstrip(".")
    else:
        am = ARXIV_RE.search(head)
        if am:
            meta.doi = f"arXiv:{am.group(1)}"

    # --- Title: prefer the first substantial line of the first page ---
    meta.title = _guess_title(first, embedded_title)

    # --- Authors: line(s) just below the title ---
    meta.authors = _guess_authors(first, meta.title)

    # --- Year ---
    years = [int(y.group(0)) for y in YEAR_RE.finditer(head)]
    years = [y for y in years if 1950 <= y <= 2100]
    if years:
        meta.year = max(years)  # publication year tends to be the latest mentioned

    # --- Venue ---
    vm = VENUE_RE.search(head)
    if vm:
        meta.venue = vm.group(0).strip()[:200]

    # --- Abstract ---
    meta.abstract = _extract_abstract(head)

    # --- Language ---
    meta.language = detect_language(meta.abstract or head)
    return meta


def _guess_title(first_page: str, embedded_title: str | None) -> str | None:
    if embedded_title and len(embedded_title.strip()) > 5:
        return embedded_title.strip()
    for line in (ln.strip() for ln in first_page.splitlines()):
        if len(line) < 8:
            continue
        if DOI_RE.search(line) or ARXIV_RE.search(line):
            continue
        if line.lower().startswith(("abstract", "keywords")):
            break
        # A title line is typically wordy and not an email/affiliation.
        if "@" in line or line.count(",") > 4:
            continue
        if len(line.split()) >= 3:
            return line[:300]
    return None


def _guess_authors(first_page: str, title: str | None) -> list[str]:
    lines = [ln.strip() for ln in first_page.splitlines() if ln.strip()]
    if title:
        for i, ln in enumerate(lines):
            if ln.startswith(title[:40]):
                lines = lines[i + 1 :]
                break
    for ln in lines[:6]:
        if ln.lower().startswith(("abstract", "keywords")) or "@" in ln:
            break
        # Author lines look like "Name, Name and Name".
        if re.match(r"^[A-Z][A-Za-z.\-]+(?:\s+[A-Z][A-Za-z.\-]+)+", ln) and (
            "," in ln or " and " in ln.lower() or len(ln.split()) <= 4
        ):
            parts = re.split(r",| and |&", ln)
            authors = [p.strip() for p in parts if len(p.strip()) > 2 and " " in p.strip()]
            if authors:
                return authors[:20]
    return []


def _extract_abstract(text: str) -> str | None:
    m = re.search(r"\bAbstract\b[:.\s]*", text, re.IGNORECASE)
    if not m:
        return None
    rest = text[m.end():]
    stop = re.search(r"\n\s*(?:1\.?\s+)?Introduction\b|\bKeywords\b", rest, re.IGNORECASE)
    abstract = rest[: stop.start()] if stop else rest[:1500]
    abstract = re.sub(r"\s+", " ", abstract).strip()
    return abstract or None
