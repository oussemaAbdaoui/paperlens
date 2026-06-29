"""Application state — process-wide singletons (DB + FAISS index).

Kept tiny and explicit (no DI framework). The ingestion and retrieval services
receive these as arguments, which keeps each module independently testable with
mocked dependencies (NFR §6: modularity).
"""

from __future__ import annotations

from functools import lru_cache

from .config import get_settings
from .db import Database
from .services.vectorstore import VectorStore


@lru_cache
def get_db() -> Database:
    s = get_settings()
    s.ensure_dirs()
    return Database(s.db_path)


@lru_cache
def get_vectorstore() -> VectorStore:
    s = get_settings()
    s.ensure_dirs()
    return VectorStore(dim=s.embedding_dim, index_path=s.index_dir / "faiss.index")
