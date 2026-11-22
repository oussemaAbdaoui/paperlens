"""M3 — invariants of section-aware chunking under adversarial layouts.

``test_m3_chunking.py`` covers the happy path with uniform paragraphs. These
tests build layouts that stress the packer (a short paragraph followed by a
near-max one, a run-on paragraph with no sentence punctuation, an oversized
paragraph whose split leaves a small remainder, headings mid-page, sections
spanning page breaks) and assert the US-05 invariants on every chunk:

  * no chunk exceeds ``chunk_max_tokens``;
  * no chunk is below ``chunk_min_tokens`` unless its whole section is;
  * a chunk never mixes text from two sections;
  * no text is lost or duplicated.
"""

from __future__ import annotations

import random

import pytest

from app.models import PaperMetadata
from app.services.chunking import chunk_document
from app.services.processing import Page, ParsedDocument
from app.tokens import count_tokens
from factories import LOREM


def _doc(*pages: str) -> ParsedDocument:
    return ParsedDocument(
        pages=[Page(number=i + 1, text=t) for i, t in enumerate(pages)],
        metadata=PaperMetadata(),
    )


def _words(text: str) -> list[str]:
    return text.split()


def _assert_invariants(drafts, doc, settings, section_bodies: dict[str, str]):
    lo, hi = settings.chunk_min_tokens, settings.chunk_max_tokens
    assert drafts, "expected at least one chunk"

    for d in drafts:
        assert d.token_count == count_tokens(d.text), "stored token_count is stale"
        assert d.token_count <= hi, f"{d.section} chunk has {d.token_count} > {hi} tokens"
        section_total = count_tokens(section_bodies[d.section])
        if section_total >= lo:
            assert d.token_count >= lo, (
                f"{d.section} chunk has {d.token_count} < {lo} tokens although the "
                f"section has {section_total}"
            )

    # Text is preserved per section, in order, with nothing lost or duplicated.
    for section, body in section_bodies.items():
        got = [w for d in drafts if d.section == section for w in _words(d.text)]
        assert got == _words(body), f"text of section {section!r} was altered"


def test_short_paragraph_then_near_max_paragraph_stays_within_max(settings_env):
    # 50-token paragraph followed by a 589-token one: packing both together
    # would give ~640 tokens, which must not happen.
    short = " ".join(["alpha"] * 49) + "."
    near_max = (LOREM * 31).strip()
    assert count_tokens(short) < settings_env.chunk_min_tokens
    assert count_tokens(near_max) <= settings_env.chunk_max_tokens
    body = f"{short}\n\n{near_max}"
    doc = _doc(f"Introduction\n\n{body}")
    _assert_invariants(chunk_document(doc), doc, settings_env, {"Introduction": body})


def test_run_on_paragraph_without_punctuation_is_hard_split(settings_env):
    # e.g. a table or reference list extracted as one long line, no periods.
    body = " ".join(f"tok{i}" for i in range(1500))
    doc = _doc(f"Methods\n\n{body}")
    drafts = chunk_document(doc)
    assert len(drafts) >= 3
    _assert_invariants(drafts, doc, settings_env, {"Methods": body})


def test_oversized_paragraph_remainder_is_not_left_tiny(settings_env):
    # 627-token paragraph: a naive sentence split leaves a ~38-token piece in
    # the middle of the section.
    big = (LOREM * 33).strip()
    tail = (LOREM * 20).strip()
    body = f"{big}\n\n{tail}"
    doc = _doc(f"Results\n\n{body}")
    _assert_invariants(chunk_document(doc), doc, settings_env, {"Results": body})


def test_small_trailing_remainder_is_rebalanced(settings_env):
    # 31 + 4 sentences: the tail cannot be merged into a ~589-token chunk
    # without exceeding max, so it must be rebalanced instead of left short.
    body = "\n\n".join((LOREM * n).strip() for n in (31, 4))
    doc = _doc(f"Discussion\n\n{body}")
    _assert_invariants(chunk_document(doc), doc, settings_env, {"Discussion": body})


def test_short_section_is_emitted_alone_not_merged_across_boundary(settings_env):
    intro = (LOREM * 3).strip()  # < 100 tokens: the whole section is short
    methods = (LOREM * 12).strip()
    doc = _doc(f"Introduction\n\n{intro}\n\nMethods\n\n{methods}")
    drafts = chunk_document(doc)
    assert [d.section for d in drafts] == ["Introduction", "Methods"]
    assert drafts[0].text == intro
    _assert_invariants(drafts, doc, settings_env, {"Introduction": intro, "Methods": methods})


def test_heading_mid_page_switches_section_and_section_spans_pages(settings_env):
    # Page 1: end of Introduction, then the Methods heading mid-page.
    # Page 2: Methods continues with no new heading.
    intro = (LOREM * 8).strip()
    m1 = (LOREM * 10).strip()
    m2 = (LOREM * 10).strip()
    doc = _doc(f"1 Introduction\n{intro}\n\n2. Methods\n{m1}", m2)
    drafts = chunk_document(doc)
    for d in drafts:
        if d.section == "Introduction":
            assert "Methods" not in d.text
    methods = [d for d in drafts if d.section == "Methods"]
    assert methods and methods[0].page == 1
    _assert_invariants(drafts, doc, settings_env, {"Introduction": intro, "Methods": f"{m1}\n\n{m2}"})


@pytest.mark.parametrize("seed", range(25))
def test_random_layouts_respect_all_invariants(settings_env, seed):
    rng = random.Random(seed)
    names = ["Abstract", "Introduction", "Related Work", "Methods", "Results", "Conclusion"]
    bodies: dict[str, str] = {}
    pages: list[str] = []
    for name in rng.sample(names, k=rng.randint(1, len(names))):
        paras = []
        for _ in range(rng.randint(1, 8)):
            kind = rng.random()
            if kind < 0.1:  # run-on, no punctuation
                paras.append(" ".join(f"w{rng.randint(0, 999)}" for _ in range(rng.randint(50, 900))))
            else:
                paras.append((LOREM * rng.randint(1, 40)).strip())
        bodies[name] = "\n\n".join(paras)
        # Spread each section's paragraphs over one or two pages.
        cut = rng.randint(1, len(paras))
        pages.append(f"{name}\n\n" + "\n\n".join(paras[:cut]))
        if paras[cut:]:
            pages.append("\n\n".join(paras[cut:]))
    doc = _doc(*pages)
    _assert_invariants(chunk_document(doc), doc, settings_env, bodies)


def test_chunk_page_is_the_page_where_the_chunk_starts(settings_env):
    other = LOREM.replace("Neural", "Sparse")
    doc = _doc(f"Introduction\n\n{(LOREM * 20).strip()}", (other * 20).strip())
    drafts = chunk_document(doc)
    assert len(drafts) >= 2
    for d in drafts:
        expected = 2 if d.text.startswith("Sparse") else 1
        assert d.page == expected
