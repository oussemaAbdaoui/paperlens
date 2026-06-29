"""Shared test fixtures.

All tests run with the offline "hash" embedding backend and a temporary data
directory, so the full pipeline (chunk → embed → FAISS → retrieve → rerank) is
exercised without model downloads, a GPU, or network access (NFR §6).
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


@pytest.fixture
def settings_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERLENS_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("PAPERLENS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PAPERLENS_SEED", "42")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    # Reset cached singletons that captured a previous settings object.
    from app import config, state
    from app.services import embeddings, reranking

    config.get_settings.cache_clear()
    state.get_db.cache_clear()
    state.get_vectorstore.cache_clear()
    embeddings.get_embedder.cache_clear()
    reranking.get_reranker.cache_clear()
    return config.get_settings()


@pytest.fixture
def db(settings_env):
    from app.state import get_db

    return get_db()


@pytest.fixture
def store(settings_env):
    from app.state import get_vectorstore

    return get_vectorstore()
