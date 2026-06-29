"""M4 — Embedding generation. Locked model: BAAI/bge-m3 (1024-dim, multilingual).

Two backends:
  * "bge"  — loads the real SentenceTransformer model (default, production).
  * "hash" — a deterministic, seeded hashing embedder that needs no downloads
             or GPU. Used for tests/CI so the whole pipeline is exercisable
             offline (NFR §6: modularity / reproducibility).

Both return L2-normalized float32 vectors so cosine similarity == inner product
in FAISS (M5).
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np

from ..config import get_settings


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype("float32")


class HashEmbedder:
    """Deterministic offline embedder. Maps tokens into a fixed-dim space via a
    seeded hash. Not semantically meaningful, but stable and fast — enough to
    validate indexing/retrieval plumbing without the real model."""

    def __init__(self, dim: int, seed: int):
        self.dim = dim
        self.seed = seed

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for tok in text.lower().split():
            h = hashlib.md5(f"{self.seed}:{tok}".encode()).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if h[4] % 2 == 0 else -1.0
            v[idx] += sign
        return v

    def encode(self, texts: list[str]) -> np.ndarray:
        mat = np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), "float32")
        return _normalize(mat)


class BGEEmbedder:
    """Real BGE-M3 via sentence-transformers."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), "float32")
        emb = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return emb.astype("float32")


@lru_cache
def get_embedder():
    s = get_settings()
    if s.embedding_backend == "hash":
        return HashEmbedder(dim=s.embedding_dim, seed=s.seed)
    return BGEEmbedder(s.embedding_model)


def embed_texts(texts: list[str]) -> np.ndarray:
    return get_embedder().encode(texts)


def embed_query(query: str) -> np.ndarray:
    return get_embedder().encode([query])[0]
