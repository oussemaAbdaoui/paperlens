"""M6 + M7 — retrieval filtering and reranker wiring with a stubbed cross-encoder.

Chunks are written straight into SQLite + FAISS (bypassing PDF parsing and
chunking) so each test controls exactly what is in the index. The embedder is
the offline hash backend; the reranker is replaced by a stub whose scores are a
known function of the passage text, so the expected final order is exact.
"""

from __future__ import annotations

import pytest

from app.models import SearchFilters
from app.services import reranking
from app.services.embeddings import embed_texts
from app.services.retrieval import retrieve


def add_paper(db, store, texts, *, collection_id=None, year=None, language="en", title="P"):
    pid = db.insert_paper(filename=f"{title}.pdf", path=f"/fake/{title}.pdf", collection_id=collection_id)
    db.update_metadata(pid, {"title": title, "year": year, "language": language, "authors": ["A B"]})
    chunk_ids = [
        db.insert_chunk(paper_id=pid, page=i + 1, section="Body", text=t, token_count=len(t.split()))
        for i, t in enumerate(texts)
    ]
    for cid, vid in zip(chunk_ids, store.add(embed_texts(texts))):
        db.set_chunk_vector_idx(cid, vid)
    return pid, chunk_ids


class KeywordReranker:
    """Scores a passage by how often it contains ``keyword``; records its input."""

    def __init__(self, keyword: str):
        self.keyword = keyword
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query, passages):
        self.calls.append((query, list(passages)))
        return [float(p.split().count(self.keyword)) for p in passages]


@pytest.fixture
def stub_reranker(monkeypatch):
    def install(keyword):
        rr = KeywordReranker(keyword)
        monkeypatch.setattr(reranking, "get_reranker", lambda: rr)
        return rr

    return install


def test_reranker_reorders_and_truncates_to_rerank_top_k(db, store, stub_reranker, settings_env):
    # 8 passages that all match the query; the k-th contains "gold" k times.
    texts = [("dense retrieval query " + "gold " * k).strip() for k in range(8)]
    _, chunk_ids = add_paper(db, store, texts)
    rr = stub_reranker("gold")

    res = retrieve("dense retrieval query", SearchFilters(), db, store)

    assert len(res) == settings_env.rerank_top_k == 5
    # Final order is the reranker's, not the cosine order.
    assert [r.chunk_id for r in res] == [chunk_ids[k] for k in (7, 6, 5, 4, 3)]
    # Returned scores are the reranker's scores.
    assert [r.score for r in res] == [7.0, 6.0, 5.0, 4.0, 3.0]
    # The reranker saw the query and every candidate exactly once.
    assert len(rr.calls) == 1
    assert rr.calls[0][0] == "dense retrieval query"
    assert sorted(rr.calls[0][1]) == sorted(texts)


def test_reranker_only_sees_top_k_cosine_candidates(db, store, stub_reranker, settings_env):
    # 30 passages: 25 share the query terms, 5 are unrelated. Only the 20 best
    # by cosine may reach the cross-encoder.
    related = [f"graph neural networks message passing variant{i}" for i in range(25)]
    unrelated = [f"sourdough bread baking hydration{i}" for i in range(5)]
    add_paper(db, store, related + unrelated)
    rr = stub_reranker("passing")

    retrieve("graph neural networks message passing", SearchFilters(), db, store)

    seen = rr.calls[0][1]
    assert len(seen) == settings_env.retrieve_top_k == 20
    assert not any("sourdough" in p for p in seen)


def test_filters_are_applied_before_reranking(db, store, stub_reranker):
    # The off-filter paper is the reranker's favourite; it must never reach it.
    add_paper(db, store, ["attention heads attention heads attention"], year=2015, title="Old")
    _, new_ids = add_paper(db, store, ["attention heads analysis"], year=2022, title="New")
    rr = stub_reranker("attention")

    res = retrieve("attention heads", SearchFilters(year_min=2020), db, store)

    assert [r.chunk_id for r in res] == new_ids
    assert all("attention heads attention heads" not in p for p in rr.calls[0][1])


def test_language_filter_is_case_insensitive_and_excludes_unknown(db, store):
    _, en = add_paper(db, store, ["saliency maps for vision transformers"], language="EN", title="En")
    add_paper(db, store, ["saliency maps for vision transformers"], language="fr", title="Fr")
    add_paper(db, store, ["saliency maps for vision transformers"], language=None, title="Unk")

    res = retrieve("saliency maps", SearchFilters(language="en"), db, store)
    assert [r.chunk_id for r in res] == en


def test_year_range_excludes_papers_without_year(db, store):
    _, ok = add_paper(db, store, ["counterfactual explanations"], year=2021, title="Y")
    add_paper(db, store, ["counterfactual explanations"], year=None, title="NoYear")
    add_paper(db, store, ["counterfactual explanations"], year=2024, title="Late")

    res = retrieve("counterfactual explanations", SearchFilters(year_min=2020, year_max=2022), db, store)
    assert [r.chunk_id for r in res] == ok


def test_deleted_paper_is_never_retrieved(db, store):
    gone, _ = add_paper(db, store, ["shapley values feature attribution"], title="Gone")
    _, kept = add_paper(db, store, ["shapley values feature attribution"], title="Kept")
    store.remove(db.delete_paper(gone))

    res = retrieve("shapley values", SearchFilters(), db, store)
    assert [r.chunk_id for r in res] == kept
    assert store.size == 1


def test_retrieved_chunk_carries_paper_metadata(db, store):
    pid, ids = add_paper(db, store, ["probing classifiers"], year=2020, title="Probing")
    (r,) = retrieve("probing classifiers", SearchFilters(), db, store)
    assert (r.chunk_id, r.paper_id, r.title, r.year, r.authors, r.page) == (ids[0], pid, "Probing", 2020, ["A B"], 1)


def test_rerank_handles_empty_and_small_pools(stub_reranker):
    stub_reranker("x")
    assert reranking.rerank("q", [], lambda c: c, top_k=5) == []
    out = reranking.rerank("q", ["a", "x x", "x"], lambda c: c, top_k=5)
    assert out == [("x x", 2.0), ("x", 1.0), ("a", 0.0)]


def test_filtered_scope_is_not_starved_by_a_larger_unfiltered_pool(db, store):
    # 250 near-identical chunks in collection A outrank everything for this
    # query; collection B holds 3 weaker matches. A query scoped to B must still
    # find them, even though none is in the global top-200 cosine pool.
    a = db.create_collection("A")
    b = db.create_collection("B")
    add_paper(db, store, ["transformer attention heads"] * 250, collection_id=a, title="A")
    _, b_ids = add_paper(
        db, store,
        [f"transformer attention variant{i} unrelated words here" for i in range(3)],
        collection_id=b, title="B",
    )

    res = retrieve("transformer attention heads", SearchFilters(collection_id=b), db, store)
    assert sorted(r.chunk_id for r in res) == sorted(b_ids)
