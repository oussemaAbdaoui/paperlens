"""API-level tests (M1 collections + dedup, M8 endpoint guard) via TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(settings_env):
    from app.main import create_app

    return TestClient(create_app())


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["embedding_backend"] == "hash"


def test_collection_crud(client):
    r = client.post("/collections", json={"name": "NLP"})
    assert r.status_code == 201
    cid = r.json()["id"]

    # duplicate name rejected
    assert client.post("/collections", json={"name": "NLP"}).status_code == 409

    r = client.patch(f"/collections/{cid}", json={"name": "NLP Papers"})
    assert r.status_code == 200 and r.json()["name"] == "NLP Papers"

    assert any(c["name"] == "NLP Papers" for c in client.get("/collections").json())

    assert client.delete(f"/collections/{cid}?mode=move").status_code == 200


def _fake_pdf() -> bytes:
    return b"%PDF-1.4\n%%EOF\n"


def test_non_pdf_rejected(client):
    r = client.post(
        "/papers/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 415


def test_duplicate_upload_skipped(client, monkeypatch):
    # Avoid real parsing in the background task.
    from app.services import ingest

    monkeypatch.setattr(ingest, "index_paper", lambda *a, **k: None)

    files = {"file": ("paper.pdf", _fake_pdf(), "application/pdf")}
    r1 = client.post("/papers/upload", files=files, data={"on_duplicate": "skip"})
    assert r1.status_code == 200 and r1.json()["paper"] is not None

    files = {"file": ("paper.pdf", _fake_pdf(), "application/pdf")}
    r2 = client.post("/papers/upload", files=files, data={"on_duplicate": "skip"})
    assert r2.json()["skipped"] is True


def test_qa_endpoint_refuses_on_empty_corpus(client):
    r = client.post("/qa", json={"question": "What is attention?"})
    assert r.status_code == 200
    body = r.json()
    assert body["grounded"] is False
    assert "No relevant passages" in body["answer"]
