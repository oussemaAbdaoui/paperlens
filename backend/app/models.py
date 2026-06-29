"""Pydantic models — API request/response shapes and internal records."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

PaperStatus = Literal["Processing", "Indexed", "Failed"]


# --------------------------------------------------------------------------- #
# Collections (M1, US-03)
# --------------------------------------------------------------------------- #
class Collection(BaseModel):
    id: int
    name: str
    paper_count: int = 0


class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class CollectionRename(BaseModel):
    name: str = Field(min_length=1, max_length=200)


# --------------------------------------------------------------------------- #
# Papers (M1/M2)
# --------------------------------------------------------------------------- #
class PaperMetadata(BaseModel):
    """Extracted metadata (M2, US-04). Unextractable fields stay None — never
    hallucinated."""

    title: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    abstract: Optional[str] = None
    doi: Optional[str] = None
    language: Optional[str] = None


class Paper(BaseModel):
    id: int
    collection_id: Optional[int] = None
    collection_name: Optional[str] = None
    filename: str
    status: PaperStatus
    num_pages: Optional[int] = None
    num_chunks: int = 0
    error: Optional[str] = None
    metadata: PaperMetadata = Field(default_factory=PaperMetadata)


class MetadataUpdate(BaseModel):
    """Partial update — only provided fields are changed (US-04: editable)."""

    title: Optional[str] = None
    authors: Optional[list[str]] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    abstract: Optional[str] = None
    doi: Optional[str] = None
    language: Optional[str] = None


class ImportRequest(BaseModel):
    """arXiv ID or DOI import (US-02)."""

    identifier: str = Field(min_length=3)
    collection_id: Optional[int] = None


class UploadResult(BaseModel):
    paper: Optional[Paper] = None
    skipped: bool = False
    duplicate_of: Optional[int] = None
    message: str = ""


# --------------------------------------------------------------------------- #
# Chunks (M3)
# --------------------------------------------------------------------------- #
class Chunk(BaseModel):
    id: int
    paper_id: int
    page: int
    section: str
    text: str
    token_count: int


# --------------------------------------------------------------------------- #
# Retrieval / QA (M6, M7, M8)
# --------------------------------------------------------------------------- #
class RetrievedChunk(BaseModel):
    chunk_id: int
    paper_id: int
    title: Optional[str]
    authors: list[str]
    year: Optional[int]
    page: int
    section: str
    text: str
    score: float


class SearchFilters(BaseModel):
    collection_id: Optional[int] = None
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    language: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    filters: SearchFilters = Field(default_factory=SearchFilters)


class SearchResponse(BaseModel):
    results: list[RetrievedChunk]
    latency_ms: float


class Citation(BaseModel):
    chunk_id: int
    paper_id: int
    label: str  # e.g. "(Vaswani et al., 2017 — Attention Is All You Need, p. 3)"


class QARequest(BaseModel):
    question: str = Field(min_length=1)
    filters: SearchFilters = Field(default_factory=SearchFilters)


class QAResponse(BaseModel):
    answer: str
    citations: list[Citation]
    sources: list[RetrievedChunk]
    grounded: bool
    latency_ms: float
