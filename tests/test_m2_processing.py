"""M2 — US-04 metadata extraction. Unextractable fields stay blank."""

from __future__ import annotations

from app.services.processing import Page, extract_metadata


def test_extracts_core_fields(settings_env):
    first = (
        "Attention Is All You Need\n"
        "Ashish Vaswani, Noam Shazeer and Niki Parmar\n"
        "Google Brain\n"
        "Abstract\n"
        "The dominant sequence transduction models are based on recurrent networks. "
        "We propose the Transformer. \n"
        "1 Introduction\n"
        "Recurrent models...\n"
        "arXiv:1706.03762\n"
        "Published 2017\n"
    )
    meta = extract_metadata([Page(number=1, text=first)])
    assert meta.title and "Attention" in meta.title
    assert any("Vaswani" in a for a in meta.authors)
    assert meta.year == 2017
    assert meta.abstract and "Transformer" in meta.abstract
    assert meta.doi and "1706.03762" in meta.doi


def test_missing_fields_are_none_not_hallucinated(settings_env):
    meta = extract_metadata([Page(number=1, text="x")])
    # Too little text -> nothing confidently extractable, nothing invented.
    assert meta.year is None
    assert meta.doi is None
    assert meta.authors == []
