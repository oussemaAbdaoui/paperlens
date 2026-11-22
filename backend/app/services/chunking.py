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


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class _Atom:
    """Smallest unit the packer moves: a sentence, or a window of a run-on one."""

    page: int
    text: str
    joiner: str  # separator placed before this atom when it follows another one


def _join(atoms: list[_Atom]) -> str:
    if not atoms:
        return ""
    return atoms[0].text + "".join(a.joiner + a.text for a in atoms[1:])


def _tokens(atoms: list[_Atom]) -> int:
    return count_tokens(_join(atoms))


def _pack_section(section: str, paras: list[tuple[int, str]], min_tok: int, max_tok: int) -> list[ChunkDraft]:
    """Pack one section's paragraphs into chunks of ``min_tok``..``max_tok`` tokens.

    Paragraph boundaries are preferred: a paragraph is split (at sentence
    boundaries, or at word boundaries for run-on text with no punctuation) only
    when it cannot be placed whole. Atoms are capped below ``max_tok - min_tok``
    tokens, so whenever the next atom does not fit, the current chunk already
    holds at least ``min_tok`` tokens; only the last chunk of a section can come
    out short, and it is then merged into or rebalanced with its predecessor.
    """
    atom_cap = max(max_tok - min_tok, max_tok // 2, 1)
    chunks: list[list[_Atom]] = []
    cur: list[_Atom] = []

    for page, para in paras:
        para_atoms = _atomize(page, para, atom_cap)
        # Start a fresh chunk at this paragraph boundary when the paragraph
        # fits in a chunk of its own but not in the current one.
        if (
            cur
            and _tokens(cur + para_atoms) > max_tok
            and _tokens(para_atoms) <= max_tok
            and _tokens(cur) >= min_tok
        ):
            chunks.append(cur)
            cur = []
        for atom in para_atoms:
            if cur and _tokens(cur + [atom]) > max_tok:
                chunks.append(cur)
                cur = []
            cur.append(atom)
    if cur:
        chunks.append(cur)

    # A short trailing chunk: merge it back if that stays within max, otherwise
    # move the split point so both chunks satisfy the bounds. Never crosses the
    # section, since everything here belongs to ``section``.
    if len(chunks) >= 2 and _tokens(chunks[-1]) < min_tok:
        prev, last = chunks[-2], chunks[-1]
        if _tokens(prev + last) <= max_tok:
            chunks[-2:] = [prev + last]
        else:
            split = _balanced_split(prev + last, min_tok, max_tok)
            if split is not None:
                chunks[-2:] = split

    drafts = []
    for atoms in chunks:
        text = _join(atoms)
        drafts.append(ChunkDraft(page=atoms[0].page, section=section, text=text, token_count=count_tokens(text)))
    return drafts


def _atomize(page: int, para: str, cap: int) -> list[_Atom]:
    atoms: list[_Atom] = []
    for i, sent in enumerate(s for s in _SENTENCE_SPLIT.split(para) if s):
        joiner = "\n\n" if i == 0 else " "
        pieces = [sent] if count_tokens(sent) <= cap else _word_windows(sent, cap)
        for j, piece in enumerate(pieces):
            atoms.append(_Atom(page=page, text=piece, joiner=joiner if j == 0 else " "))
    return atoms


def _word_windows(text: str, cap: int) -> list[str]:
    """Hard-split text with no usable sentence boundary into <= cap-token windows."""
    words: list[str] = []
    for w in text.split():
        if count_tokens(w) <= cap:
            words.append(w)
        else:  # a single "word" over the cap (e.g. a long URL): split by characters
            words.extend(w[k : k + cap] for k in range(0, len(w), cap))
    windows: list[str] = []
    buf: list[str] = []
    for w in words:
        if buf and count_tokens(" ".join(buf + [w])) > cap:
            windows.append(" ".join(buf))
            buf = []
        buf.append(w)
    if buf:
        windows.append(" ".join(buf))
    return windows


def _balanced_split(atoms: list[_Atom], min_tok: int, max_tok: int) -> list[list[_Atom]] | None:
    """Split ``atoms`` into two chunks that both satisfy the bounds, as evenly as
    possible. Returns None if no atom boundary works."""
    best = None
    for k in range(1, len(atoms)):
        left, right = _tokens(atoms[:k]), _tokens(atoms[k:])
        if min_tok <= left <= max_tok and min_tok <= right <= max_tok:
            gap = abs(left - right)
            if best is None or gap < best[0]:
                best = (gap, k)
    if best is None:
        return None
    return [atoms[: best[1]], atoms[best[1] :]]
