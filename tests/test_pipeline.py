"""End-to-end ingestion + retrieval (M3→M7) using the offline hash backend.

Indexes synthetic papers (parse_pdf monkeypatched) and verifies US-06/US-07:
indexing populates chunks + FAISS, retrieval returns reranked top-k, and the
metadata filters (collection, year, language) are applied before reranking.
"""

from __future__ import annotations

import pytest

from app.services import ingest
from app.models import SearchFilters
from app.services.retrieval import retrieve
from factories import make_doc, long_body


def _index(db, store, monkeypatch, *, filename, doc, collection_id=None, year=None, language="en"):
    pid = db.insert_paper(filename=filename, path=f"/fake/{filename}", collection_id=collection_id)
    monkeypatch.setattr(ingest, "parse_pdf", lambda path: doc)
    ingest.index_paper(pid, db, store, persist=False)
    db.update_metadata(pid, {"year": year, "language": language, "title": filename})
    return pid


def test_ingest_populates_chunks_and_index(db, store, monkeypatch):
    doc = make_doc({"Introduction": long_body(20), "Methods": long_body(20)})
    pid = _index(db, store, monkeypatch, filename="p1.pdf", doc=doc)
    paper = db.get_paper(pid)
    assert paper["status"] == "Indexed"
    assert paper["num_chunks"] > 0
    assert store.size == paper["num_chunks"]


def test_retrieval_returns_reranked_topk(db, store, monkeypatch):
    doc = make_doc(
        {
            "Introduction": "Dense retrieval uses embeddings and cosine similarity for search.",
            "Methods": "We fine-tune a cross encoder reranker on labeled query passage pairs.",
        }
    )
    _index(db, store, monkeypatch, filename="p.pdf", doc=doc)
    results = retrieve("cross encoder reranker", SearchFilters(), db, store)
    assert results
    assert len(results) <= 5  # rerank_top_k


def test_year_filter_excludes_out_of_range(db, store, monkeypatch):
    doc = make_doc({"Introduction": long_body(10)})
    _index(db, store, monkeypatch, filename="old.pdf", doc=doc, year=2010)
    _index(db, store, monkeypatch, filename="new.pdf", doc=make_doc({"Introduction": long_body(10)}), year=2023)

    res = retrieve("retrieval embeddings cosine", SearchFilters(year_min=2020), db, store)
    assert res
    assert all(r.year is None or r.year >= 2020 for r in res)
    assert all(r.year != 2010 for r in res)


def test_collection_scope(db, store, monkeypatch):
    c1 = db.create_collection("NLP")
    c2 = db.create_collection("CV")
    _index(db, store, monkeypatch, filename="nlp.pdf", doc=make_doc({"Introduction": long_body(10)}), collection_id=c1)
    _index(db, store, monkeypatch, filename="cv.pdf", doc=make_doc({"Introduction": long_body(10)}), collection_id=c2)

    res = retrieve("retrieval embeddings", SearchFilters(collection_id=c1), db, store)
    assert res
    for r in res:
        assert db.get_paper(r.paper_id)["collection_id"] == c1


def test_empty_corpus_returns_nothing(db, store):
    assert retrieve("anything", SearchFilters(), db, store) == []
