"""Fetch and index the benchmark papers (spec §8.2) into a fresh corpus.

Reads eval/benchmark/papers.jsonl ({"arxiv_id", "domain", "title"}) and imports
each via the arXiv importer, then indexes it. Run before Milestone 3.

Usage:
    python scripts/build_benchmark_corpus.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services import importer, ingest  # noqa: E402
from app.state import get_db, get_vectorstore  # noqa: E402
from app.config import get_settings  # noqa: E402


def main() -> None:
    papers_file = ROOT / "eval/benchmark/papers.jsonl"
    if not papers_file.exists():
        print(f"Missing {papers_file}. See eval/benchmark/README.md for the schema.")
        sys.exit(1)

    settings = get_settings()
    settings.ensure_dirs()
    db, store = get_db(), get_vectorstore()

    collection_id = db.create_collection("Benchmark")
    items = [json.loads(l) for l in papers_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Importing {len(items)} benchmark papers...")

    for item in items:
        arxiv_id = item["arxiv_id"]
        try:
            imported = importer.import_identifier(arxiv_id)
            path = settings.pdf_dir / imported.filename
            path.write_bytes(imported.pdf_bytes)
            pid = db.insert_paper(filename=imported.filename, path=str(path), collection_id=collection_id)
            db.update_metadata(pid, imported.metadata.model_dump(exclude_none=True))
            ingest.index_paper(pid, db, store)
            print(f"  [OK] {arxiv_id} -> paper #{pid}")
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {arxiv_id}: {e}")

    print(f"Done. {store.size} chunks indexed.")


if __name__ == "__main__":
    main()
