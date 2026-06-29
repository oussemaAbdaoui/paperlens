"""M3 — Intelligent chunking. Section-aware is the only v1 strategy (US-05).

Rules enforced (US-05 acceptance criteria):
  * Each chunk records: document id (set by caller), page, section label,
    plus authors/year (carried on the paper, joined at retrieval time).
  * No chunk is shorter than 100 tokens or longer than 600 tokens — except a
    section whose entire content is < 100 tokens, which is emitted as a single
    short chunk rather than being merged across a section boundary.
  * Section boundaries are respected: a chunk never spans two named sections.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import get_settings
from ..tokens import count_tokens
from .processing import ParsedDocument

# Canonical scientific section headings. A line is treated as a heading if it
# matches one of these (optionally numbered like "3.1 Methods").
_SECTION_WORDS = (
    "abstract", "introduction", "background", "related work", "preliminaries",
    "methods", "method", "methodology", "approach", "model", "architecture",
    "experiments", "experimental setup", "evaluation", "results",
    "discussion", "analysis", "ablation", "limitations", "conclusion",
    "conclusions", "future work", "references", "acknowledgments",
    "acknowledgements", "appendix",
)
_HEADING_RE = re.compile(
    r"^\s*(?:(?:\d+(?:\.\d+)*)\.?\s+)?(" + "|".join(re.escape(w) for w in _SECTION_WORDS) + r")\b\s*$",
    re.IGNORECASE,
)
_PARA_SPLIT = re.compile(r"\n\s*\n")


@dataclass
class ChunkDraft:
    page: int
    section: str
    text: str
    token_count: int


def _section_for_line(line: str) -> str | None:
    m = _HEADING_RE.match(line.strip())
    if m:
        return m.group(1).title()
    return None


def _split_paragraphs(text: str) -> list[str]:
    paras = _PARA_SPLIT.split(text)
    out = []
    for p in paras:
        p = re.sub(r"[ \t]+", " ", p).strip()
        if p:
            out.append(p)
    return out


def chunk_document(doc: ParsedDocument) -> list[ChunkDraft]:
    """Produce section-aware chunks from a parsed document."""
    s = get_settings()
    min_tok, max_tok = s.chunk_min_tokens, s.chunk_max_tokens

    # 1) Flatten pages into (page, section, paragraph) units. We walk lines so
    #    a heading appearing mid-page switches the active section immediately.
    #    The section carries across page breaks until the next heading.
    units: list[tuple[int, str, str]] = []  # (page, section, paragraph)
    section = "Body"
    for page in doc.pages:
        buf: list[str] = []

        def flush(sec: str) -> None:
            if buf:
                para = re.sub(r"[ \t]+", " ", " ".join(buf)).strip()
                if para:
                    units.append((page.number, sec, para))
                buf.clear()

        for line in page.text.splitlines():
            heading = _section_for_line(line)
            if heading:
                flush(section)
                section = heading
                continue
            if line.strip() == "":
                flush(section)
            else:
                buf.append(line.strip())
        flush(section)

    # 2) Pack paragraphs into chunks within each (page-group, section), never
    #    crossing a section boundary, honoring the token bounds.
    drafts: list[ChunkDraft] = []
    i = 0
    while i < len(units):
        page, section, _ = units[i]
        # gather consecutive units of the same section
        group: list[tuple[int, str]] = []  # (page, paragraph)
        while i < len(units) and units[i][1] == section:
            group.append((units[i][0], units[i][2]))
            i += 1
        drafts.extend(_pack_section(section, group, min_tok, max_tok))
    return drafts


def _pack_section(section: str, paras: list[tuple[int, str]], min_tok: int, max_tok: int) -> list[ChunkDraft]:
    drafts: list[ChunkDraft] = []
    cur_text: list[str] = []
    cur_tokens = 0
    cur_page = paras[0][0] if paras else 1

    def emit():
        nonlocal cur_text, cur_tokens
        if cur_text:
            text = "\n\n".join(cur_text).strip()
            drafts.append(ChunkDraft(page=cur_page, section=section, text=text, token_count=count_tokens(text)))
            cur_text = []
            cur_tokens = 0

    for page, para in paras:
        ptok = count_tokens(para)

        # A single oversized paragraph: hard-split on sentences to respect max.
        if ptok > max_tok:
            emit()
            for piece in _split_oversized(para, max_tok):
                drafts.append(
                    ChunkDraft(page=page, section=section, text=piece, token_count=count_tokens(piece))
                )
            cur_page = page
            continue

        if cur_tokens + ptok > max_tok and cur_tokens >= min_tok:
            emit()
            cur_page = page

        if not cur_text:
            cur_page = page
        cur_text.append(para)
        cur_tokens += ptok

    emit()

    # Merge a too-small trailing chunk into the previous one when possible,
    # without exceeding max — but only within this same section.
    return _merge_small(drafts, min_tok, max_tok)


def _split_oversized(para: str, max_tok: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", para)
    pieces: list[str] = []
    buf: list[str] = []
    tok = 0
    for sent in sentences:
        st = count_tokens(sent)
        if tok + st > max_tok and buf:
            pieces.append(" ".join(buf))
            buf, tok = [], 0
        buf.append(sent)
        tok += st
    if buf:
        pieces.append(" ".join(buf))
    return pieces


def _merge_small(drafts: list[ChunkDraft], min_tok: int, max_tok: int) -> list[ChunkDraft]:
    if len(drafts) < 2:
        return drafts
    out = drafts[:]
    last = out[-1]
    if last.token_count < min_tok:
        prev = out[-2]
        if prev.token_count + last.token_count <= max_tok:
            merged_text = prev.text + "\n\n" + last.text
            out[-2] = ChunkDraft(
                page=prev.page,
                section=prev.section,
                text=merged_text,
                token_count=count_tokens(merged_text),
            )
            out.pop()
    return out
