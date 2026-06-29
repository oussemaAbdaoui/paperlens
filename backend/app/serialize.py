"""Helpers to turn SQLite rows into API models."""

from __future__ import annotations

import json

from .models import Collection, Paper, PaperMetadata


def row_to_paper(row) -> Paper:
    authors = row["authors"]
    try:
        authors = json.loads(authors) if authors else []
    except (TypeError, json.JSONDecodeError):
        authors = []
    return Paper(
        id=row["id"],
        collection_id=row["collection_id"],
        collection_name=row["collection_name"] if "collection_name" in row.keys() else None,
        filename=row["filename"],
        status=row["status"],
        num_pages=row["num_pages"],
        num_chunks=row["num_chunks"] if "num_chunks" in row.keys() else 0,
        error=row["error"],
        metadata=PaperMetadata(
            title=row["title"],
            authors=authors,
            year=row["year"],
            venue=row["venue"],
            abstract=row["abstract"],
            doi=row["doi"],
            language=row["language"],
        ),
    )


def row_to_collection(row) -> Collection:
    return Collection(
        id=row["id"],
        name=row["name"],
        paper_count=row["paper_count"] if "paper_count" in row.keys() else 0,
    )
