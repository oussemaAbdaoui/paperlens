"""PaperLens FastAPI application (MVP — modules M1–M8)."""

from __future__ import annotations

import logging
import random

import numpy as np
from fastapi import FastAPI

from . import __version__
from .config import get_settings
from .routers import collections, corpus, qa, search

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app() -> FastAPI:
    settings = get_settings()
    settings.ensure_dirs()

    # Reproducibility (NFR §6): fix seeds at startup.
    random.seed(settings.seed)
    np.random.seed(settings.seed)

    app = FastAPI(
        title="PaperLens API",
        version=__version__,
        description="AI-powered research assistant — grounded, cited QA over a paper corpus.",
    )

    app.include_router(corpus.router)
    app.include_router(collections.router)
    app.include_router(search.router)
    app.include_router(qa.router)

    @app.get("/health", tags=["meta"])
    def health():
        from .state import get_vectorstore

        return {
            "status": "ok",
            "version": __version__,
            "embedding_backend": settings.embedding_backend,
            "llm_model": settings.llm_model,
            "indexed_vectors": get_vectorstore().size,
        }

    return app


app = create_app()
