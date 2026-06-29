"""Central configuration. All v1 technology choices are locked (see spec §7).

Nothing here builds a swappability abstraction — per the PO decision, components
are configured by value and replaced directly if needed.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PAPERLENS_",
        env_file=".env",
        extra="ignore",
    )

    # --- Generation (M8). Default GPT-4o, configurable per spec US-08. ---
    # OPENAI_API_KEY is read directly from the environment (no prefix).
    llm_model: str = "gpt-4o"
    openai_api_key: str = ""

    # --- Locked v1 models (M4 embeddings, M7 reranker). ---
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    embedding_dim: int = 1024  # BGE-M3 output dimension
    # "bge" loads the real model; "hash" is a deterministic offline embedder
    # used for tests/CI so the pipeline is exercisable without GPU/downloads.
    embedding_backend: str = "bge"

    # --- Reproducibility (NFR §6). ---
    seed: int = 42

    # --- Storage. ---
    data_dir: Path = Path("./data")

    # --- Retrieval / chunking knobs (spec-locked defaults). ---
    retrieve_top_k: int = 20  # M6: top-20 by cosine
    rerank_top_k: int = 5     # M7: rerank to top-5
    chunk_min_tokens: int = 100
    chunk_max_tokens: int = 600

    # --- Generation latency guard (NFR §6). ---
    generation_timeout_s: int = 20

    @property
    def db_path(self) -> Path:
        return self.data_dir / "paperlens.db"

    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "pdfs"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.pdf_dir, self.index_dir):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    import os

    s = Settings()
    # OpenAI key has no PAPERLENS_ prefix; pull it from the standard env var.
    s.openai_api_key = os.getenv("OPENAI_API_KEY", s.openai_api_key)
    return s
