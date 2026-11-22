"""M1 (US-01/US-02) — duplicate detection at the DB and API level.

Network import is stubbed: ``importer.import_identifier`` returns a canned
``ImportedPaper`` and background indexing is disabled, so nothing leaves the
process.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.models import PaperMetadata
from app.services import importer, ingest

PDF = b"%PDF-1.4\n%%EOF\n"


@pytest.fixture
def client(settings_env, monkeypatch):
    monkeypatch.setattr(ingest, "index_paper", lambda *a, **k: None)
    from app.main import create_app

    return TestClient(create_app())


def _stub_import(monkeypatch, *, doi, filename="imported.pdf"):
    monkeypatch.setattr(
        importer,
        "import_identifier",
        lambda ident: importer.ImportedPaper(
            pdf_bytes=PDF, filename=filename, metadata=PaperMetadata(title="T", doi=doi)
        ),
    )


def test_find_duplicate_prefers_doi_over_filename(db):
    a = db.insert_paper(filename="a.pdf", path="/a", collection_id=None)
    db.update_metadata(a, {"doi": "10.1000/xyz"})
    b = db.insert_paper(filename="b.pdf", path="/b", collection_id=None)

    assert db.find_duplicate(filename="b.pdf", doi="10.1000/xyz")["id"] == a
    assert db.find_duplicate(filename="b.pdf", doi="10.9999/other")["id"] == b
    assert db.find_duplicate(filename="c.pdf", doi=None) is None


def test_import_same_doi_twice_is_skipped(client, monkeypatch):
    _stub_import(monkeypatch, doi="10.1000/xyz", filename="first.pdf")
    r1 = client.post("/papers/import", json={"identifier": "10.1000/xyz"}).json()
    assert r1["skipped"] is False and r1["paper"]["metadata"]["doi"] == "10.1000/xyz"

    # Different filename, same DOI -> still a duplicate.
    _stub_import(monkeypatch, doi="10.1000/xyz", filename="second.pdf")
    r2 = client.post("/papers/import", json={"identifier": "10.1000/xyz"}).json()
    assert r2["skipped"] is True
    assert r2["duplicate_of"] == r1["paper"]["id"]
    assert len(client.get("/papers").json()) == 1


def test_upload_overwrite_replaces_paper_and_its_vectors(client, settings_env):
    import numpy as np

    from app.state import get_db, get_vectorstore

    r1 = client.post("/papers/upload", files={"file": ("p.pdf", PDF, "application/pdf")}).json()
    old_id = r1["paper"]["id"]
    db, store = get_db(), get_vectorstore()
    cid = db.insert_chunk(paper_id=old_id, page=1, section="Body", text="t", token_count=1)
    (vid,) = store.add(np.ones((1, settings_env.embedding_dim), dtype="float32"))
    db.set_chunk_vector_idx(cid, vid)

    r2 = client.post(
        "/papers/upload",
        files={"file": ("p.pdf", PDF, "application/pdf")},
        data={"on_duplicate": "overwrite"},
    ).json()

    assert r2["skipped"] is False and r2["paper"]["id"] != old_id
    assert db.get_paper(old_id) is None
    assert db.get_chunk(cid) is None
    assert store.size == 0
    assert [p["id"] for p in client.get("/papers").json()] == [r2["paper"]["id"]]


def test_fake_pdf_with_pdf_extension_rejected(client):
    r = client.post("/papers/upload", files={"file": ("x.pdf", b"not a pdf", "application/pdf")})
    assert r.status_code == 415


def test_doi_match_is_case_insensitive(db):
    # DOIs are case-insensitive; the PDF text and Crossref/user input often
    # differ in case for the same paper.
    a = db.insert_paper(filename="a.pdf", path="/a", collection_id=None)
    db.update_metadata(a, {"doi": "10.1145/ABC.Def-12"})
    assert db.find_duplicate(filename="other.pdf", doi="10.1145/abc.def-12")["id"] == a
