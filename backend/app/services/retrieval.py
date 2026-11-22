"""M6 + M7 — Retrieval and reranking.

Pipeline (US-07):
  1. Embed the query with BGE-M3.
  2. FAISS cosine search for a candidate pool.
  3. Apply metadata filters BEFORE reranking: year range, collection, language.
  4. Keep the top-20 by cosine, then rerank with BGE-Reranker to top-5.

Filters are applied to the candidate pool rather than the index because the v1
FAISS flat index has no native metadata filtering. We over-fetch so filtering
still leaves a healthy pool for the 500-paper target scale.
"""

from __future__ import annotations

import json

from ..config import get_settings
from ..db import Database
from ..models import RetrievedChunk, SearchFilters
from ..services.vectorstore import VectorStore
from . import reranking
from .embeddings import embed_query

# Over-fetch factor so metadata filtering doesn't starve the top-20 pool.
_CANDIDATE_MULTIPLIER = 10
_MIN_CANDIDATES = 200


def _passes_filters(row, filters: SearchFilters) -> bool:
    if filters.collection_id is not None and row["collection_id"] != filters.collection_id:
        return False
    if filters.language and (row["language"] or "").lower() != filters.language.lower():
        return False
    year = row["year"]
    if filters.year_min is not None and (year is None or year < filters.year_min):
        return False
    if filters.year_max is not None and (year is None or year > filters.year_max):
        return False
    return True


def _to_retrieved(row, score: float) -> RetrievedChunk:
    authors = row["authors"]
    try:
        authors = json.loads(authors) if authors else []
    except (TypeError, json.JSONDecodeError):
        authors = []
    return RetrievedChunk(
        chunk_id=row["id"],
        paper_id=row["paper_id"],
        title=row["title"],
        authors=authors,
        year=row["year"],
        page=row["page"],
        section=row["section"],
        text=row["text"],
        score=float(score),
    )


def retrieve(
    query: str,
    filters: SearchFilters,
    db: Database,
    store: VectorStore,
) -> list[RetrievedChunk]:
    """Run the full M6+M7 pipeline, returning up to ``rerank_top_k`` chunks."""
    s = get_settings()
    if store.size == 0:
        return []

    # 1 + 2: embed and search a candidate pool.
    qvec = embed_query(query)
    pool_size = max(_MIN_CANDIDATES, s.retrieve_top_k * _CANDIDATE_MULTIPLIER)
    while True:
        hits = store.search(qvec, k=pool_size)  # [(vector_idx, score)]
        if not hits:
            return []

        rows = db.chunks_by_vector_idx([vid for vid, _ in hits])

        # 3: filter, then keep the top-20 by cosine (hits are already score-sorted).
        candidates: list[RetrievedChunk] = []
        for vid, score in hits:
            row = rows.get(vid)
            if row is None or not _passes_filters(row, filters):
                continue
            candidates.append(_to_retrieved(row, score))
            if len(candidates) >= s.retrieve_top_k:
                break

        # A selective filter can reject the whole pool even though matching
        # chunks exist further down; widen the search until the pool is full
        # or the index is exhausted.
        if len(candidates) >= s.retrieve_top_k or pool_size >= store.size:
            break
        pool_size = min(store.size, pool_size * 4)

    if not candidates:
        return []

    # 4: rerank top-20 -> top-5 with the cross-encoder.
    ranked = reranking.rerank(query, candidates, lambda c: c.text, s.rerank_top_k)
    results = []
    for chunk, rscore in ranked:
        chunk.score = float(rscore)
        results.append(chunk)
    return results
