# PaperLens — AI-Powered Research Assistant (v1 / MVP)

PaperLens lets a researcher ask questions across their **entire paper corpus** and
get **grounded, cited answers in seconds**. Every answer is traceable to a
specific paper and page — it is a research accelerator, not a document chatbot.

This repository implements the **MVP (Milestones 1–3, modules M1–M8)** defined in
`PaperLens-Specification-Book.docx`.

---

## What's implemented (MVP scope)

| Module | Feature | Where |
|--------|---------|-------|
| **M1** | Corpus management: PDF/folder upload, arXiv/DOI import, collections, dedup | `routers/corpus.py`, `routers/collections.py`, `services/importer.py` |
| **M2** | Document processing: PyMuPDF parsing + metadata extraction | `services/processing.py` |
| **M3** | Section-aware chunking (100–600 tokens, never crossing sections) | `services/chunking.py` |
| **M4** | Embeddings — **BGE-M3** | `services/embeddings.py` |
| **M5** | Vector indexing — **FAISS** (cosine via inner product) | `services/vectorstore.py` |
| **M6** | Retrieval — top-20 by cosine, metadata-filtered | `services/retrieval.py` |
| **M7** | Reranking — **BGE-Reranker-v2-m3**, top-20 → top-5 | `services/reranking.py` |
| **M8** | Question answering — grounded, cited, **GPT-4o** (configurable), with hallucination guard | `services/qa.py` |

v2 modules (cross-paper comparison, literature review, gap detection) and all
"Cut" modules are **out of scope** and not built (spec §1.4, §3).

## Locked v1 technology (spec §7)

FastAPI · Streamlit · GPT-4o (env-configurable) · BGE-M3 · FAISS · BGE-Reranker-v2-m3 ·
PyMuPDF · RAGAS · Docker/compose · pytest. No swappability abstraction layers — by design.

---

## Quick start (Docker — reproducible, single command)

```bash
cp .env.example .env          # then set OPENAI_API_KEY
docker-compose up --build
```

- API:      http://localhost:8000  (OpenAPI docs at `/docs`)
- Frontend: http://localhost:8501

The first start downloads the BGE-M3 and BGE-Reranker models into a cached
volume (a few minutes). Data (SQLite + PDFs + FAISS index) persists in the
`paperlens_data` volume.

## Local development (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...                          # Windows: set OPENAI_API_KEY=...

# Terminal 1 — API
uvicorn app.main:app --app-dir backend --reload

# Terminal 2 — UI
streamlit run frontend/streamlit_app.py
```

### Running without model downloads / GPU (CI, offline)

Set `PAPERLENS_EMBEDDING_BACKEND=hash` to use a deterministic, seeded offline
embedder and an order-preserving reranker. The full pipeline (chunk → embed →
FAISS → retrieve → rerank → guard) runs end-to-end without downloads. This is how
the test suite runs.

---

## Testing

```bash
pip install -r requirements.txt
pytest                      # all unit/integration tests (offline backend)
pytest --cov=backend/app    # with coverage (target: 80% on M1–M8)
```

Test coverage maps to the user stories:

- `test_m2_processing.py` — metadata extraction, no hallucinated fields (US-04)
- `test_m3_chunking.py` — token bounds + section boundaries (US-05)
- `test_pipeline.py` — ingest → retrieve, metadata filters, collection scope (US-06/07)
- `test_m8_qa.py` — citation format, grounding, **hallucination guard** (US-08)
- `test_hallucination_guard.py` — 20 out-of-scope queries all refuse (checklist §9)
- `test_m1_importer.py` — arXiv/DOI parsing + clear errors (US-02)
- `test_api.py` — collections CRUD, dedup, QA endpoint refusal (US-01/03/08)
- `test_eval_metrics.py` — Recall@k / Precision@k / MRR (§8.3)

---

## Evaluation (spec §8)

Build the benchmark corpus (50 arXiv papers across NLP/CV/XAI — see
`eval/benchmark/README.md` for the required schema), then run the suite:

```bash
python scripts/build_benchmark_corpus.py            # fetch + index benchmark papers
python -m eval.run_eval --questions eval/benchmark/questions.jsonl
python -m eval.run_eval --questions ... --generation   # add RAGAS generation metrics
```

The runner exits non-zero if any metric falls below its §8.3/§8.4 threshold, so
it can gate a release in CI (checklist §9). Retrieval metrics are dependency-free;
generation metrics use RAGAS and require an LLM key.

> ⚠️ Per spec §8.2, the **full 100-question benchmark dataset must be constructed
> and committed before Milestone 3**. The repo ships only a tiny illustrative
> sample (`sample_questions.jsonl`).

---

## Configuration

All settings are environment variables (see `.env.example`). Key ones:

| Variable | Default | Meaning |
|----------|---------|---------|
| `OPENAI_API_KEY` | — | Required for M8 generation |
| `PAPERLENS_LLM_MODEL` | `gpt-4o` | Generation model (US-08: configurable) |
| `PAPERLENS_EMBEDDING_BACKEND` | `bge` | `bge` (real) or `hash` (offline) |
| `PAPERLENS_RETRIEVE_TOP_K` | `20` | M6 candidate count |
| `PAPERLENS_RERANK_TOP_K` | `5` | M7 final count |
| `PAPERLENS_CHUNK_MIN_TOKENS` / `MAX_TOKENS` | `100` / `600` | M3 bounds |
| `PAPERLENS_SEED` | `42` | Reproducibility |

---

## Architecture (one screen)

```
Streamlit UI ──HTTP──> FastAPI
                          │
        ┌─────────────────┼───────────────────────────┐
        │ Ingestion (background task per upload)        │
        │  PyMuPDF ─> metadata ─> section chunker ─>    │
        │  BGE-M3 embeddings ─> FAISS index             │
        └───────────────────────────────────────────────┘
                          │
        ┌─────────────────┼───────────────────────────┐
        │ Query path                                    │
        │  BGE-M3 query embed ─> FAISS top-N ─>         │
        │  metadata filter ─> top-20 ─> BGE-Reranker ─> │
        │  top-5 ─> GPT-4o grounded answer + citations  │
        └───────────────────────────────────────────────┘

State: SQLite (collections, papers, chunks) + FAISS (vectors), in /data.
```

Internal design details (DB schema, class diagrams) belong in the TDD, not here
(spec §10).

---

## Spec traceability

This codebase is the implementation of `PaperLens-Specification-Book.docx` v1.0.
The MVP acceptance checklist (§9) is tracked in `ACCEPTANCE.md`.
