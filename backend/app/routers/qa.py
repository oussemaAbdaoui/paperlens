"""M8 — Question answering endpoint (US-08)."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException

from ..db import Database
from ..models import Chunk, QARequest, QAResponse
from ..services import qa, retrieval
from ..state import get_db, get_vectorstore

router = APIRouter(tags=["qa"])


@router.post("/qa", response_model=QAResponse)
def ask(body: QARequest, db: Database = Depends(get_db)):
    start = time.perf_counter()
    chunks = retrieval.retrieve(body.question, body.filters, db, get_vectorstore())
    latency_ms = (time.perf_counter() - start) * 1000
    # Hallucination guard lives in qa.answer_question: zero chunks -> refusal,
    # no LLM call.
    return qa.answer_question(body.question, chunks, latency_ms=latency_ms)


@router.get("/chunks/{chunk_id}", response_model=Chunk)
def get_chunk(chunk_id: int, db: Database = Depends(get_db)):
    """Fetch a chunk by id so the UI can show a citation's source in context
    (US-08: clickable citations)."""
    row = db.get_chunk(chunk_id)
    if row is None:
        raise HTTPException(404, "Chunk not found")
    return Chunk(
        id=row["id"],
        paper_id=row["paper_id"],
        page=row["page"],
        section=row["section"],
        text=row["text"],
        token_count=row["token_count"],
    )
