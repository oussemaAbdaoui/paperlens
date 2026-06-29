"""M8 — Question answering. Grounded, cited generation. Default model: GPT-4o.

Hard guarantees (US-08 + NFR §6 hallucination guard):
  * If retrieval returns zero chunks, we DO NOT call the LLM. We return the
    fixed refusal string and ``grounded=False``. This is the hallucination
    guard and is covered by an automated test.
  * The prompt forbids using outside knowledge and requires inline citations in
    the format ``(Author et al., YEAR — Paper Title, p. N)``.
  * Citations are also returned structurally (chunk_id + paper_id) so the UI can
    open the exact source chunk in context.
"""

from __future__ import annotations

from ..config import get_settings
from ..models import Citation, QAResponse, RetrievedChunk

NO_PASSAGES_MESSAGE = "No relevant passages found in your corpus for this query."

SYSTEM_PROMPT = (
    "You are PaperLens, a research assistant that answers ONLY from the provided "
    "paper excerpts. You must never use outside or general knowledge.\n"
    "Rules:\n"
    "1. Every claim must be supported by the excerpts below.\n"
    "2. Add an inline citation after each claim in EXACTLY this format: "
    "(Author et al., YEAR — Paper Title, p. N).\n"
    "3. If the excerpts do not contain the answer, reply exactly: "
    f"'{NO_PASSAGES_MESSAGE}'\n"
    "4. If your answer is longer than three paragraphs, cite at least two "
    "distinct sources.\n"
    "5. Do not invent citations, page numbers, authors, or titles."
)


def _author_label(chunk: RetrievedChunk) -> str:
    if not chunk.authors:
        return "Unknown"
    first = chunk.authors[0].split()[-1] if chunk.authors[0] else "Unknown"
    return f"{first} et al." if len(chunk.authors) > 1 else first


def citation_label(chunk: RetrievedChunk) -> str:
    """Build the canonical inline citation label (US-08 format)."""
    year = chunk.year if chunk.year else "n.d."
    title = chunk.title or "Untitled"
    return f"({_author_label(chunk)}, {year} — {title}, p. {chunk.page})"


def _build_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(
            f"[Excerpt {i}] {citation_label(c)}\n"
            f"Section: {c.section}\n"
            f"{c.text}"
        )
    return "\n\n".join(blocks)


def answer_question(question: str, chunks: list[RetrievedChunk], latency_ms: float = 0.0) -> QAResponse:
    """Generate a grounded answer. ``chunks`` are the reranked top-k from M6/M7."""
    # --- Hallucination guard: refuse with zero retrieved support. ---
    if not chunks:
        return QAResponse(
            answer=NO_PASSAGES_MESSAGE,
            citations=[],
            sources=[],
            grounded=False,
            latency_ms=latency_ms,
        )

    s = get_settings()
    context = _build_context(chunks)
    user_prompt = (
        f"Question: {question}\n\n"
        f"Paper excerpts you may use:\n\n{context}\n\n"
        "Answer the question using only these excerpts, with inline citations."
    )

    answer_text = _call_llm(s, SYSTEM_PROMPT, user_prompt)

    citations = [
        Citation(chunk_id=c.chunk_id, paper_id=c.paper_id, label=citation_label(c))
        for c in chunks
    ]
    return QAResponse(
        answer=answer_text,
        citations=citations,
        sources=chunks,
        grounded=True,
        latency_ms=latency_ms,
    )


def _call_llm(settings, system: str, user: str) -> str:
    """Call OpenAI chat completions. Isolated so tests can monkeypatch it."""
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        timeout=settings.generation_timeout_s,
    )
    return resp.choices[0].message.content.strip()
