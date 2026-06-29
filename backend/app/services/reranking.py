"""M7 — Reranking. Locked cross-encoder: BAAI/bge-reranker-v2-m3.

Reranks the top-20 retrieved candidates down to the top-5 (US-07). Like the
embedder, it supports an offline fallback so the pipeline runs without model
downloads in tests/CI — there it falls back to preserving the retrieval order.
"""

from __future__ import annotations

from functools import lru_cache

from ..config import get_settings


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        pairs = [[query, p] for p in passages]
        return [float(s) for s in self.model.predict(pairs)]


class IdentityReranker:
    """Offline fallback: keeps the retrieval ranking (descending synthetic
    scores) so ``rerank`` is order-preserving without a model."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        n = len(passages)
        return [float(n - i) for i in range(n)]


@lru_cache
def get_reranker():
    s = get_settings()
    if s.embedding_backend == "hash":  # offline mode mirrors the embedder
        return IdentityReranker()
    return CrossEncoderReranker(s.reranker_model)


def rerank(query: str, candidates: list, text_getter, top_k: int) -> list:
    """Reorder ``candidates`` by cross-encoder relevance and return the top_k.

    ``text_getter(candidate) -> str`` extracts the passage text. The candidate
    objects themselves are returned untouched (their ``.score`` may be updated
    by the caller if desired)."""
    if not candidates:
        return []
    scores = get_reranker().score(query, [text_getter(c) for c in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [(c, s) for c, s in ranked[:top_k]]
