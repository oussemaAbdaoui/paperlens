"""M6 / M7 — Semantic search endpoint (US-07)."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from ..db import Database
from ..models import SearchRequest, SearchResponse
from ..services import retrieval
from ..state import get_db, get_vectorstore

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search(body: SearchRequest, db: Database = Depends(get_db)):
    start = time.perf_counter()
    results = retrieval.retrieve(body.query, body.filters, db, get_vectorstore())
    latency_ms = (time.perf_counter() - start) * 1000
    return SearchResponse(results=results, latency_ms=latency_ms)
