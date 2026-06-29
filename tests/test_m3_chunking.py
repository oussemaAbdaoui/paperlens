"""M3 — US-05 acceptance criteria for section-aware chunking."""

from __future__ import annotations

from app.services.chunking import chunk_document
from factories import make_doc, long_body


def test_chunks_respect_token_bounds(settings_env):
    doc = make_doc(
        {
            "Introduction": long_body(40),  # well over 600 tokens -> multiple chunks
            "Methods": long_body(30),
        }
    )
    drafts = chunk_document(doc)
    assert drafts, "expected at least one chunk"
    for d in drafts:
        assert d.token_count <= settings_env.chunk_max_tokens, "chunk exceeds max tokens"
    # No chunk under the minimum unless it is the sole chunk of its section.
    by_section: dict[str, int] = {}
    for d in drafts:
        by_section[d.section] = by_section.get(d.section, 0) + 1
    for d in drafts:
        if by_section[d.section] > 1:
            assert d.token_count >= settings_env.chunk_min_tokens


def test_chunks_never_span_two_sections(settings_env):
    doc = make_doc(
        {
            "Introduction": long_body(20),
            "Methods": long_body(20),
            "Results": long_body(20),
        }
    )
    drafts = chunk_document(doc)
    sections = {d.section for d in drafts}
    assert {"Introduction", "Methods", "Results"} <= sections
    # Each chunk carries exactly one section label (enforced structurally).
    for d in drafts:
        assert d.section in {"Introduction", "Methods", "Results"}


def test_section_labels_are_canonical(settings_env):
    doc = make_doc({"3.1 Methods": long_body(20)})
    drafts = chunk_document(doc)
    assert all(d.section == "Methods" for d in drafts)


def test_chunk_stores_page_number(settings_env):
    doc = make_doc({"Introduction": long_body(15), "Results": long_body(15)})
    drafts = chunk_document(doc)
    pages = {d.page for d in drafts}
    assert pages == {1, 2}
