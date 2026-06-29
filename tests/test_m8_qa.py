"""M8 — US-08 QA guarantees: hallucination guard, citation format, grounding."""

from __future__ import annotations

from app.models import RetrievedChunk
from app.services import qa


def _chunk(**kw) -> RetrievedChunk:
    base = dict(
        chunk_id=1, paper_id=1, title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"], year=2017,
        page=3, section="Methods", text="The Transformer uses self-attention.", score=0.9,
    )
    base.update(kw)
    return RetrievedChunk(**base)


def test_hallucination_guard_refuses_without_chunks(settings_env, monkeypatch):
    # If the LLM were called, this would explode — proving no call is made.
    monkeypatch.setattr(qa, "_call_llm", lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM called!")))
    resp = qa.answer_question("What is X?", [], latency_ms=1.0)
    assert resp.grounded is False
    assert resp.answer == qa.NO_PASSAGES_MESSAGE
    assert resp.citations == []


def test_citation_label_format(settings_env):
    label = qa.citation_label(_chunk())
    assert label == "(Vaswani et al., 2017 — Attention Is All You Need, p. 3)"


def test_single_author_label(settings_env):
    label = qa.citation_label(_chunk(authors=["Jane Doe"]))
    assert label == "(Doe, 2017 — Attention Is All You Need, p. 3)"


def test_grounded_answer_has_citations(settings_env, monkeypatch):
    monkeypatch.setattr(
        qa, "_call_llm",
        lambda settings, system, user: "The Transformer relies on self-attention "
        "(Vaswani et al., 2017 — Attention Is All You Need, p. 3).",
    )
    resp = qa.answer_question("How does the Transformer work?", [_chunk()], latency_ms=2.0)
    assert resp.grounded is True
    assert len(resp.citations) == 1
    assert resp.citations[0].chunk_id == 1
    assert "p. 3" in resp.citations[0].label


def test_context_includes_all_chunks(settings_env):
    chunks = [_chunk(chunk_id=1, page=3), _chunk(chunk_id=2, page=7, section="Results")]
    ctx = qa._build_context(chunks)
    assert "Excerpt 1" in ctx and "Excerpt 2" in ctx
    assert "p. 3" in ctx and "p. 7" in ctx
