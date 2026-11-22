"""SQLite metadata store.

Holds collections, papers, and chunks. Vector data lives in FAISS (see
vectorstore.py); a chunk's ``vector_idx`` links the two. SQLite is sufficient
for the 500-paper / ~250k-chunk target scale (NFR §6).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS collections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS papers (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_id INTEGER REFERENCES collections(id) ON DELETE SET NULL,
    filename      TEXT NOT NULL,
    path          TEXT,
    status        TEXT NOT NULL DEFAULT 'Processing',
    num_pages     INTEGER,
    error         TEXT,
    title         TEXT,
    authors       TEXT,            -- JSON list
    year          INTEGER,
    venue         TEXT,
    abstract      TEXT,
    doi           TEXT,
    language      TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id    INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    page        INTEGER NOT NULL,
    section     TEXT NOT NULL,
    text        TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    vector_idx  INTEGER             -- row in the FAISS index
);

CREATE INDEX IF NOT EXISTS idx_chunks_paper   ON chunks(paper_id);
CREATE INDEX IF NOT EXISTS idx_chunks_vecidx  ON chunks(vector_idx);
CREATE INDEX IF NOT EXISTS idx_papers_coll    ON papers(collection_id);
CREATE INDEX IF NOT EXISTS idx_papers_doi     ON papers(doi);
"""


class Database:
    def __init__(self, path: Path | str):
        self.path = str(path)
        # check_same_thread=False + a lock: FastAPI may hit this from threads.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.RLock()
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- low-level helpers --------------------------------------------------
    def _exec(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            self._conn.commit()
            return cur

    def _query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchall()

    def _one(self, sql: str, params: Iterable[Any] = ()) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchone()

    # -- collections (M1, US-03) -------------------------------------------
    def create_collection(self, name: str) -> int:
        cur = self._exec("INSERT INTO collections(name) VALUES (?)", (name,))
        return int(cur.lastrowid)

    def rename_collection(self, cid: int, name: str) -> None:
        self._exec("UPDATE collections SET name = ? WHERE id = ?", (name, cid))

    def delete_collection(self, cid: int, *, move_to_uncategorized: bool) -> None:
        if move_to_uncategorized:
            self._exec("UPDATE papers SET collection_id = NULL WHERE collection_id = ?", (cid,))
        # else: papers cascade is handled by caller deleting papers first.
        self._exec("DELETE FROM collections WHERE id = ?", (cid,))

    def get_collection(self, cid: int) -> Optional[sqlite3.Row]:
        return self._one("SELECT * FROM collections WHERE id = ?", (cid,))

    def list_collections(self) -> list[sqlite3.Row]:
        return self._query(
            """
            SELECT c.*, COUNT(p.id) AS paper_count
            FROM collections c
            LEFT JOIN papers p ON p.collection_id = c.id
            GROUP BY c.id ORDER BY c.name
            """
        )

    # -- papers (M1/M2) -----------------------------------------------------
    def insert_paper(self, *, filename: str, path: str, collection_id: Optional[int]) -> int:
        cur = self._exec(
            "INSERT INTO papers(filename, path, collection_id, status) VALUES (?,?,?,'Processing')",
            (filename, path, collection_id),
        )
        return int(cur.lastrowid)

    def set_paper_status(self, pid: int, status: str, error: Optional[str] = None) -> None:
        self._exec("UPDATE papers SET status = ?, error = ? WHERE id = ?", (status, error, pid))

    def set_paper_pages(self, pid: int, num_pages: int) -> None:
        self._exec("UPDATE papers SET num_pages = ? WHERE id = ?", (num_pages, pid))

    def update_metadata(self, pid: int, fields: dict[str, Any]) -> None:
        if not fields:
            return
        if "authors" in fields and fields["authors"] is not None:
            fields = {**fields, "authors": json.dumps(fields["authors"])}
        cols = ", ".join(f"{k} = ?" for k in fields)
        self._exec(f"UPDATE papers SET {cols} WHERE id = ?", (*fields.values(), pid))

    def get_paper(self, pid: int) -> Optional[sqlite3.Row]:
        return self._one(
            """
            SELECT p.*, c.name AS collection_name,
                   (SELECT COUNT(*) FROM chunks ch WHERE ch.paper_id = p.id) AS num_chunks
            FROM papers p LEFT JOIN collections c ON c.id = p.collection_id
            WHERE p.id = ?
            """,
            (pid,),
        )

    def list_papers(self, collection_id: Optional[int] = None) -> list[sqlite3.Row]:
        if collection_id is None:
            return self._query(
                """
                SELECT p.*, c.name AS collection_name,
                       (SELECT COUNT(*) FROM chunks ch WHERE ch.paper_id = p.id) AS num_chunks
                FROM papers p LEFT JOIN collections c ON c.id = p.collection_id
                ORDER BY p.created_at DESC
                """
            )
        return self._query(
            """
            SELECT p.*, c.name AS collection_name,
                   (SELECT COUNT(*) FROM chunks ch WHERE ch.paper_id = p.id) AS num_chunks
            FROM papers p LEFT JOIN collections c ON c.id = p.collection_id
            WHERE p.collection_id = ?
            ORDER BY p.created_at DESC
            """,
            (collection_id,),
        )

    def find_duplicate(self, *, filename: str, doi: Optional[str]) -> Optional[sqlite3.Row]:
        """Duplicate detection by DOI first, then filename (US-01)."""
        if doi:
            row = self._one("SELECT * FROM papers WHERE doi = ? COLLATE NOCASE LIMIT 1", (doi,))
            if row:
                return row
        return self._one("SELECT * FROM papers WHERE filename = ? LIMIT 1", (filename,))

    def delete_paper(self, pid: int) -> list[int]:
        """Delete a paper; returns the FAISS vector_idx values to purge."""
        rows = self._query("SELECT vector_idx FROM chunks WHERE paper_id = ? AND vector_idx IS NOT NULL", (pid,))
        self._exec("DELETE FROM papers WHERE id = ?", (pid,))
        return [int(r["vector_idx"]) for r in rows]

    # -- chunks (M3/M5) -----------------------------------------------------
    def insert_chunk(self, *, paper_id: int, page: int, section: str, text: str, token_count: int) -> int:
        cur = self._exec(
            "INSERT INTO chunks(paper_id, page, section, text, token_count) VALUES (?,?,?,?,?)",
            (paper_id, page, section, text, token_count),
        )
        return int(cur.lastrowid)

    def set_chunk_vector_idx(self, chunk_id: int, vector_idx: int) -> None:
        self._exec("UPDATE chunks SET vector_idx = ? WHERE id = ?", (vector_idx, chunk_id))

    def get_chunks_for_paper(self, pid: int) -> list[sqlite3.Row]:
        return self._query("SELECT * FROM chunks WHERE paper_id = ? ORDER BY id", (pid,))

    def get_chunk(self, chunk_id: int) -> Optional[sqlite3.Row]:
        return self._one("SELECT * FROM chunks WHERE id = ?", (chunk_id,))

    def chunks_by_vector_idx(self, vector_idxs: list[int]) -> dict[int, sqlite3.Row]:
        """Bulk-load chunks (joined with paper metadata) by FAISS row id."""
        if not vector_idxs:
            return {}
        placeholders = ",".join("?" for _ in vector_idxs)
        rows = self._query(
            f"""
            SELECT ch.*, p.title, p.authors, p.year, p.language, p.collection_id
            FROM chunks ch JOIN papers p ON p.id = ch.paper_id
            WHERE ch.vector_idx IN ({placeholders})
            """,
            vector_idxs,
        )
        return {int(r["vector_idx"]): r for r in rows}
