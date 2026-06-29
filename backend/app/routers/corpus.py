"""M1 / M2 — Corpus management: upload, import, list, metadata edit, delete.

Indexing (M4/M5) runs in a background task immediately after a paper row is
created (US-06), so the upload request returns fast while the paper shows
status 'Processing'.
"""

from __future__ import annotations

import logging

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from ..config import get_settings
from ..db import Database
from ..models import ImportRequest, MetadataUpdate, Paper, UploadResult
from ..serialize import row_to_paper
from ..services import importer, ingest
from ..state import get_db, get_vectorstore

log = logging.getLogger("paperlens.corpus")
router = APIRouter(prefix="/papers", tags=["corpus"])


def _index_in_background(paper_id: int) -> None:
    """Background indexing job. Errors are recorded on the paper row (US-06)."""
    try:
        ingest.index_paper(paper_id, get_db(), get_vectorstore())
    except Exception:  # noqa: BLE001 - already persisted as Failed status
        log.exception("Background indexing failed for paper %s", paper_id)


def _save_pdf(content: bytes, filename: str) -> str:
    s = get_settings()
    s.ensure_dirs()
    # Avoid collisions on identical filenames in different uploads.
    dest = s.pdf_dir / filename
    n = 1
    while dest.exists():
        dest = s.pdf_dir / f"{dest.stem}_{n}{dest.suffix}"
        n += 1
    dest.write_bytes(content)
    return str(dest)


@router.get("", response_model=list[Paper])
def list_papers(collection_id: int | None = None, db: Database = Depends(get_db)):
    return [row_to_paper(r) for r in db.list_papers(collection_id)]


@router.get("/{paper_id}", response_model=Paper)
def get_paper(paper_id: int, db: Database = Depends(get_db)):
    row = db.get_paper(paper_id)
    if row is None:
        raise HTTPException(404, "Paper not found")
    return row_to_paper(row)


@router.post("/upload", response_model=UploadResult)
async def upload_pdf(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    collection_id: int | None = Form(None),
    on_duplicate: str = Form("skip"),  # "skip" | "overwrite"
    db: Database = Depends(get_db),
):
    """Upload a single PDF (US-01). Non-PDFs are rejected with a clear error;
    duplicates (by filename/DOI) are skipped or overwritten per ``on_duplicate``."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(415, f"'{file.filename}' is not a PDF. Skipped.")

    content = await file.read()
    if b"%PDF" not in content[:1024]:
        raise HTTPException(415, f"'{file.filename}' is not a valid PDF.")

    dup = db.find_duplicate(filename=file.filename, doi=None)
    if dup is not None:
        if on_duplicate == "skip":
            return UploadResult(
                skipped=True,
                duplicate_of=dup["id"],
                message=f"Duplicate of paper #{dup['id']} ('{dup['filename']}'). Skipped.",
            )
        # overwrite: drop the old paper + vectors first.
        store = get_vectorstore()
        store.remove(db.delete_paper(dup["id"]))
        store.persist()

    path = _save_pdf(content, file.filename)
    paper_id = db.insert_paper(filename=file.filename, path=path, collection_id=collection_id)
    background.add_task(_index_in_background, paper_id)
    return UploadResult(paper=row_to_paper(db.get_paper(paper_id)), message="Upload accepted; indexing started.")


@router.post("/import", response_model=UploadResult)
def import_paper(body: ImportRequest, background: BackgroundTasks, db: Database = Depends(get_db)):
    """Import by arXiv ID or DOI (US-02)."""
    try:
        imported = importer.import_identifier(body.identifier)
    except importer.ImportError_ as e:
        raise HTTPException(400, str(e)) from e

    dup = db.find_duplicate(filename=imported.filename, doi=imported.metadata.doi)
    if dup is not None:
        return UploadResult(
            skipped=True,
            duplicate_of=dup["id"],
            message=f"'{body.identifier}' is already in your corpus as paper #{dup['id']}.",
        )

    path = _save_pdf(imported.pdf_bytes, imported.filename)
    paper_id = db.insert_paper(
        filename=imported.filename, path=path, collection_id=body.collection_id
    )
    # Seed the fetched metadata before indexing fills any gaps.
    db.update_metadata(paper_id, imported.metadata.model_dump(exclude_none=True))
    background.add_task(_index_in_background, paper_id)
    return UploadResult(paper=row_to_paper(db.get_paper(paper_id)), message="Imported; indexing started.")


@router.post("/{paper_id}/reindex", response_model=Paper)
def reindex_paper(paper_id: int, background: BackgroundTasks, db: Database = Depends(get_db)):
    """Retry indexing for a failed paper (US-06)."""
    if db.get_paper(paper_id) is None:
        raise HTTPException(404, "Paper not found")
    background.add_task(_index_in_background, paper_id)
    db.set_paper_status(paper_id, "Processing")
    return row_to_paper(db.get_paper(paper_id))


@router.patch("/{paper_id}", response_model=Paper)
def update_metadata(paper_id: int, body: MetadataUpdate, db: Database = Depends(get_db)):
    """Edit extracted metadata (US-04: editable at any time)."""
    if db.get_paper(paper_id) is None:
        raise HTTPException(404, "Paper not found")
    db.update_metadata(paper_id, body.model_dump(exclude_unset=True))
    return row_to_paper(db.get_paper(paper_id))


@router.delete("/{paper_id}")
def delete_paper(paper_id: int, db: Database = Depends(get_db)):
    if db.get_paper(paper_id) is None:
        raise HTTPException(404, "Paper not found")
    store = get_vectorstore()
    store.remove(db.delete_paper(paper_id))
    store.persist()
    return {"deleted": True}
