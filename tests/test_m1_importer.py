"""M1 — US-02 identifier parsing and clear errors for bad input."""

from __future__ import annotations

import pytest

from app.services import importer


def test_rejects_garbage_identifier(settings_env):
    with pytest.raises(importer.ImportError_) as e:
        importer.import_identifier("not-an-id")
    assert "not a recognized" in str(e.value)


def test_arxiv_id_recognized(settings_env, monkeypatch):
    captured = {}

    def fake_arxiv(arxiv_id):
        captured["id"] = arxiv_id
        return importer.ImportedPaper(pdf_bytes=b"%PDF", filename="x.pdf", metadata=importer.PaperMetadata())

    monkeypatch.setattr(importer, "_import_arxiv", fake_arxiv)
    importer.import_identifier("arXiv:2301.07041")
    assert captured["id"] == "2301.07041"


def test_doi_recognized(settings_env, monkeypatch):
    captured = {}

    def fake_doi(doi):
        captured["doi"] = doi
        return importer.ImportedPaper(pdf_bytes=b"%PDF", filename="x.pdf", metadata=importer.PaperMetadata())

    monkeypatch.setattr(importer, "_import_doi", fake_doi)
    importer.import_identifier("10.1145/3292500.3330701")
    assert captured["doi"].startswith("10.1145")
