"""M1 / US-03 — Collection management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from ..db import Database
from ..models import Collection, CollectionCreate, CollectionRename
from ..serialize import row_to_collection
from ..state import get_db, get_vectorstore

router = APIRouter(prefix="/collections", tags=["collections"])


@router.get("", response_model=list[Collection])
def list_collections(db: Database = Depends(get_db)):
    return [row_to_collection(r) for r in db.list_collections()]


@router.post("", response_model=Collection, status_code=201)
def create_collection(body: CollectionCreate, db: Database = Depends(get_db)):
    try:
        cid = db.create_collection(body.name.strip())
    except Exception as e:  # UNIQUE violation -> duplicate name
        raise HTTPException(409, f"A collection named '{body.name}' already exists.") from e
    return row_to_collection(_with_count(db, cid))


def _with_count(db: Database, cid: int):
    for r in db.list_collections():
        if r["id"] == cid:
            return r
    raise HTTPException(404, "Collection not found")


@router.patch("/{cid}", response_model=Collection)
def rename_collection(cid: int, body: CollectionRename, db: Database = Depends(get_db)):
    if db.get_collection(cid) is None:
        raise HTTPException(404, "Collection not found")
    db.rename_collection(cid, body.name.strip())
    return row_to_collection(_with_count(db, cid))


@router.delete("/{cid}")
def delete_collection(
    cid: int,
    mode: str = Query("move", regex="^(move|delete)$"),
    db: Database = Depends(get_db),
):
    """Delete a collection. ``mode=move`` reassigns papers to Uncategorized
    (collection_id NULL); ``mode=delete`` removes the papers and their vectors
    too (US-03: confirmation handled client-side)."""
    if db.get_collection(cid) is None:
        raise HTTPException(404, "Collection not found")

    if mode == "delete":
        store = get_vectorstore()
        for paper in db.list_papers(collection_id=cid):
            vecs = db.delete_paper(paper["id"])
            store.remove(vecs)
        store.persist()
        db.delete_collection(cid, move_to_uncategorized=False)
        return {"deleted": True, "papers": "deleted"}

    db.delete_collection(cid, move_to_uncategorized=True)
    return {"deleted": True, "papers": "moved to Uncategorized"}
