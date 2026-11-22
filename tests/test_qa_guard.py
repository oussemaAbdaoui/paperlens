"""M8 — what the LLM is given and how its output is turned into a response.

The OpenAI call (``qa._call_llm``) is replaced by a recording stub, so these
tests check the real prompt construction, citation wiring and guard logic
without a key or network.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.models import RetrievedChunk
from app.services import ingest, qa
from factories import long_body, make_doc


def _chunk(i: int, **kw) -> RetrievedChunk:
    base = dict(
        chunk_id=i, paper_id=10 + i, title=f"Paper {i}", authors=["Ada Lovelace", "Alan Turing"],
        year=2020 + i, page=i, section="Results", text=f"Finding number {i}.", score=1.0,
    )
    base.update(kw)
    return RetrievedChunk(**base)


class RecordingLLM:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def __call__(self, settings, system, user):
        self.calls.append((system, user))
        return self.reply


def test_prompt_contains_question_and_every_excerpt_with_its_label(settings_env, monkeypatch):
    llm = RecordingLLM("ok")
    monkeypatch.setattr(qa, "_call_llm", llm)
    chunks = [_chunk(1), _chunk(2, authors=["Grace Hopper"])]

    qa.answer_question("What was found?", chunks)

    (system, user), = llm.calls
    assert "never use outside" in system and qa.NO_PASSAGES_MESSAGE in system
    assert "Question: What was found?" in user
    for c in chunks:
        assert qa.citation_label(c) in user
        assert c.text in user
    assert "(Lovelace et al., 2021 — Paper 1, p. 1)" in user
    assert "(Hopper, 2022 — Paper 2, p. 2)" in user


def test_citations_and_sources_map_one_to_one_to_retrieved_chunks(settings_env, monkeypatch):
    monkeypatch.setattr(qa, "_call_llm", RecordingLLM("Answer (Lovelace et al., 2021 — Paper 1, p. 1)."))
    chunks = [_chunk(1), _chunk(2), _chunk(3)]

    resp = qa.answer_question("q", chunks)

    assert resp.grounded is True
    assert [(c.chunk_id, c.paper_id) for c in resp.citations] == [(1, 11), (2, 12), (3, 13)]
    assert [c.label for c in resp.citations] == [qa.citation_label(c) for c in chunks]
    assert [s.chunk_id for s in resp.sources] == [1, 2, 3]


def test_missing_metadata_never_invents_citation_fields(settings_env):
    label = qa.citation_label(_chunk(1, authors=[], year=None, title=None, page=4))
    assert label == "(Unknown, n.d. — Untitled, p. 4)"


@pytest.fixture
def client(settings_env):
    from app.main import create_app

    return TestClient(create_app())


def test_qa_endpoint_end_to_end_citations_resolve_to_chunks(client, monkeypatch):
    from app.state import get_db, get_vectorstore

    db, store = get_db(), get_vectorstore()
    pid = db.insert_paper(filename="p.pdf", path="/fake/p.pdf", collection_id=None)
    monkeypatch.setattr(ingest, "parse_pdf", lambda path: make_doc({"Introduction": long_body(20)}))
    ingest.index_paper(pid, db, store, persist=False)
    db.update_metadata(pid, {"title": "Dense Retrieval", "authors": ["Jane Doe"], "year": 2021})

    llm = RecordingLLM("Dense retrieval compares vectors (Doe, 2021 — Dense Retrieval, p. 1).")
    monkeypatch.setattr(qa, "_call_llm", llm)

    body = client.post("/qa", json={"question": "How does dense retrieval work?"}).json()

    assert body["grounded"] is True
    assert body["answer"] == llm.reply
    assert len(llm.calls) == 1
    assert body["citations"] and len(body["citations"]) == len(body["sources"])
    for cit in body["citations"]:
        chunk = client.get(f"/chunks/{cit['chunk_id']}").json()
        assert chunk["paper_id"] == pid == cit["paper_id"]
        assert chunk["text"] in llm.calls[0][1]  # the cited text was in the prompt
        assert cit["label"] == "(Doe, 2021 — Dense Retrieval, p. 1)"


def test_qa_endpoint_with_filter_matching_nothing_skips_llm(client, monkeypatch):
    from app.state import get_db, get_vectorstore

    db, store = get_db(), get_vectorstore()
    pid = db.insert_paper(filename="p.pdf", path="/fake/p.pdf", collection_id=None)
    monkeypatch.setattr(ingest, "parse_pdf", lambda path: make_doc({"Introduction": long_body(20)}))
    ingest.index_paper(pid, db, store, persist=False)
    llm = RecordingLLM("should not be called")
    monkeypatch.setattr(qa, "_call_llm", llm)

    body = client.post("/qa", json={"question": "dense retrieval", "filters": {"year_min": 2090}}).json()

    assert body["grounded"] is False and body["citations"] == [] and body["sources"] == []
    assert llm.calls == []
