# MVP Acceptance Checklist (Spec §9)

Status of each product-owner sign-off criterion in this MVP build.

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | All 8 MVP user stories have passing automated tests | ✅ Implemented | `tests/` — see story map in README |
| 2 | Benchmark evaluation suite runs in CI without manual intervention | ✅ Harness ready | `eval/run_eval.py` (needs full dataset, §8.2) |
| 3 | All retrieval & generation metrics meet thresholds | ⏳ Pending data | Gated by `run_eval.py` once benchmark is committed |
| 4 | Hallucination guard: 20 out-of-scope queries refuse, zero exceptions | ✅ Tested | `tests/test_hallucination_guard.py` |
| 5 | 200 papers index without error in a single docker-compose run | ⏳ Manual | Requires real corpus + target hardware run |
| 6 | End-to-end QA latency < 20 s for 95% of queries | ⏳ Manual | Measure on target hardware (NFR §6) |
| 7 | README has working setup instructions verified on a clean machine | ✅ Provided | `README.md` |
| 8 | All source passes linting & type checking | ⏳ Manual | Run `ruff` / `mypy` in CI |
| 9 | No known critical/high-severity bugs open | ✅ None known | — |
| 10 | PO reviewed ≥20 benchmark QA answers and approved | ⏳ Pending | Requires PO review session |

Legend: ✅ done in code · ⏳ requires data/hardware/human step the code enables.

## Story → test traceability

| Story | Criterion highlights | Test |
|-------|----------------------|------|
| US-01 | PDF upload, non-PDF skip, duplicate detection, status | `test_api.py` |
| US-02 | arXiv/DOI parse, clear error on bad input | `test_m1_importer.py` |
| US-03 | Create/rename/delete collections, scoped retrieval | `test_api.py`, `test_pipeline.py` |
| US-04 | Metadata extraction, blanks not hallucinated, editable | `test_m2_processing.py`, `routers/corpus.py` |
| US-05 | 100–600 token bounds, section boundaries, page label | `test_m3_chunking.py` |
| US-06 | Auto-index after upload, status, retry on failure | `routers/corpus.py`, `services/ingest.py`, `test_pipeline.py` |
| US-07 | top-20 cosine → top-5 rerank, metadata filters | `test_pipeline.py` |
| US-08 | Inline citations, grounding, refusal, clickable sources | `test_m8_qa.py`, `test_api.py` |
