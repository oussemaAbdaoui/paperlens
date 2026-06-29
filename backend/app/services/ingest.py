"""Ingestion orchestrator — wires M2 → M3 → M4 → M5 for one paper.

US-06: indexing runs automatically after upload, shows progress via the paper's
status, and surfaces a user-readable error with retry on failure.

Flow:
  Processing → parse PDF (M2) → extract metadata (M2) → chunk (M3) →
  embed (M4) → add to FAISS (M5) → Indexed.

On any exception the paper is marked Failed with the error message; the row is
retained so the user can retry (re-running ``index_paper``).
"""

from __future__ import annotations

import logging

from ..db import Database
from ..services.vectorstore import VectorStore
from .chunking import chunk_document
from .embeddings import embed_texts
from .processing import parse_pdf

log = logging.getLogger("paperlens.ingest")


def index_paper(paper_id: int, db: Database, store: VectorStore, *, persist: bool = True) -> None:
    """Parse, chunk, embed and index a single paper. Idempotent on retry:
    existing chunks for the paper are cleared first."""
    paper = db.get_paper(paper_id)
    if paper is None:
        raise ValueError(f"Paper {paper_id} not found")

    db.set_paper_status(paper_id, "Processing")
    try:
        # Clean up any partial chunks/vectors from a previous failed attempt.
        stale_vecs = [
            int(c["vector_idx"])
            for c in db.get_chunks_for_paper(paper_id)
            if c["vector_idx"] is not None
        ]
        if stale_vecs:
            store.remove(stale_vecs)
        db._exec("DELETE FROM chunks WHERE paper_id = ?", (paper_id,))  # noqa: SLF001

        # M2 — parse + metadata.
        doc = parse_pdf(paper["path"])
        db.set_paper_pages(paper_id, doc.num_pages)
        _apply_extracted_metadata(db, paper, doc.metadata)

        # M3 — section-aware chunking.
        drafts = chunk_document(doc)
        if not drafts:
            raise RuntimeError("No extractable text — the PDF may be scanned or empty.")

        chunk_ids = [
            db.insert_chunk(
                paper_id=paper_id,
                page=d.page,
                section=d.section,
                text=d.text,
                token_count=d.token_count,
            )
            for d in drafts
        ]

        # M4 — embeddings (BGE-M3).
        vectors = embed_texts([d.text for d in drafts])

        # M5 — index in FAISS and link vector ids back to chunks.
        vector_ids = store.add(vectors)
        for chunk_id, vec_id in zip(chunk_ids, vector_ids):
            db.set_chunk_vector_idx(chunk_id, vec_id)

        if persist:
            store.persist()
        db.set_paper_status(paper_id, "Indexed")
        log.info("Indexed paper %s (%d chunks)", paper_id, len(drafts))
    except Exception as exc:  # noqa: BLE001 - surface to the user, allow retry
        db.set_paper_status(paper_id, "Failed", error=str(exc))
        log.exception("Indexing failed for paper %s", paper_id)
        raise


def _apply_extracted_metadata(db: Database, paper, meta) -> None:
    """Fill only fields the user hasn't already set; never overwrite edits and
    never write hallucinated (None) values over existing data."""
    updates: dict = {}
    mapping = {
        "title": meta.title,
        "authors": meta.authors or None,
        "year": meta.year,
        "venue": meta.venue,
        "abstract": meta.abstract,
        "doi": meta.doi,
        "language": meta.language,
    }
    for field, value in mapping.items():
        if value and not paper[field]:
            updates[field] = value
    if updates:
        db.update_metadata(paper["id"], updates)
