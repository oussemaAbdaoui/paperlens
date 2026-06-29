"""M5 — Vector indexing with FAISS (local, no external service).

Uses an ``IndexIDMap`` over ``IndexFlatIP``. Vectors are L2-normalized upstream,
so inner product == cosine similarity. Each vector carries an explicit integer
id (the FAISS ``vector_idx`` stored in SQLite), which lets us delete a paper's
vectors without rebuilding the whole index.

Flat index is exact and easily handles the 500-paper / ~250k-chunk target
(NFR §6). Approximate indexes (IVF/HNSW) are a later optimization.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np


class VectorStore:
    def __init__(self, dim: int, index_path: Path | str):
        import faiss

        self._faiss = faiss
        self.dim = dim
        self.index_path = Path(index_path)
        self._lock = threading.RLock()
        self._next_id = 0

        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            # Recover the id high-water mark so new ids never collide.
            self._next_id = self._max_id() + 1
        else:
            base = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIDMap2(base)

    def _max_id(self) -> int:
        try:
            ids = self._faiss.vector_to_array(self.index.id_map)
            return int(ids.max()) if ids.size else -1
        except Exception:
            return self.index.ntotal - 1

    def add(self, vectors: np.ndarray) -> list[int]:
        """Add vectors, returning the assigned ids (in order)."""
        with self._lock:
            n = vectors.shape[0]
            ids = np.arange(self._next_id, self._next_id + n, dtype="int64")
            self.index.add_with_ids(vectors.astype("float32"), ids)
            self._next_id += n
            return ids.tolist()

    def remove(self, ids: list[int]) -> None:
        if not ids:
            return
        with self._lock:
            sel = self._faiss.IDSelectorBatch(np.array(ids, dtype="int64"))
            self.index.remove_ids(sel)

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return up to k (vector_idx, score) pairs, best first."""
        with self._lock:
            if self.index.ntotal == 0:
                return []
            q = query.reshape(1, -1).astype("float32")
            k = min(k, self.index.ntotal)
            scores, ids = self.index.search(q, k)
            out: list[tuple[int, float]] = []
            for vid, score in zip(ids[0], scores[0]):
                if vid != -1:
                    out.append((int(vid), float(score)))
            return out

    def persist(self) -> None:
        with self._lock:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self._faiss.write_index(self.index, str(self.index_path))

    @property
    def size(self) -> int:
        return self.index.ntotal
