"""Evaluation runner (spec §8).

Runs the retrieval benchmark against a live index and prints a report. Retrieval
metrics (§8.3) are always computed. Generation metrics (§8.4) via RAGAS are run
only with --generation (they require an LLM key and the ragas package).

Exit code is non-zero if any computed metric is below its threshold, so this can
gate a release in CI (MVP checklist §9).

Usage:
    python -m eval.run_eval --questions eval/benchmark/sample_questions.jsonl
    python -m eval.run_eval --questions ... --generation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from eval.metrics import aggregate  # noqa: E402


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_retrieval_eval(questions: list[dict]) -> None:
    from app.models import SearchFilters
    from app.services.retrieval import retrieve
    from app.state import get_db, get_vectorstore

    db, store = get_db(), get_vectorstore()
    per_query = []
    for q in questions:
        results = retrieve(q["question"], SearchFilters(), db, store)
        retrieved_papers = [r.paper_id for r in results]
        per_query.append(
            {"retrieved": retrieved_papers, "relevant": set(q.get("relevant_paper_ids", []))}
        )

    report = aggregate(per_query)
    print("\n=== Retrieval metrics (§8.3) ===")
    for name, value in report.as_dict().items():
        key = {"Recall@5": "recall_at_5", "Recall@10": "recall_at_10",
               "Precision@5": "precision_at_5", "MRR": "mrr"}[name]
        threshold = report.THRESHOLDS[key]
        status = "PASS" if value >= threshold else "FAIL"
        print(f"  {name:<12} {value:<8} (min {threshold})  [{status}]")

    if not report.passes():
        print("\nRetrieval metrics below threshold — release rejected.")
        sys.exit(1)
    print("\nAll retrieval metrics meet thresholds.")


def run_generation_eval(questions: list[dict]) -> None:
    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from datasets import Dataset
    except ImportError:
        print("ragas/datasets not installed — skipping generation metrics.")
        return

    from app.models import SearchFilters
    from app.services import qa as qa_service
    from app.services.retrieval import retrieve
    from app.state import get_db, get_vectorstore

    db, store = get_db(), get_vectorstore()
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for q in questions:
        chunks = retrieve(q["question"], SearchFilters(), db, store)
        resp = qa_service.answer_question(q["question"], chunks)
        rows["question"].append(q["question"])
        rows["answer"].append(resp.answer)
        rows["contexts"].append([c.text for c in chunks])
        rows["ground_truth"].append(q.get("ground_truth", ""))

    dataset = Dataset.from_dict(rows)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    print("\n=== Generation metrics (§8.4, RAGAS) ===")
    print(result)

    thresholds = {
        "faithfulness": 0.80,
        "answer_relevancy": 0.75,
        "context_precision": 0.70,
        "context_recall": 0.70,
    }
    failed = [m for m, t in thresholds.items() if result.get(m, 0) < t]
    if failed:
        print(f"\nGeneration metrics below threshold: {failed} — release rejected.")
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="PaperLens evaluation runner")
    ap.add_argument("--questions", type=Path, default=ROOT / "eval/benchmark/sample_questions.jsonl")
    ap.add_argument("--generation", action="store_true", help="also run RAGAS generation metrics")
    args = ap.parse_args()

    questions = load_questions(args.questions)
    print(f"Loaded {len(questions)} benchmark questions from {args.questions}")
    run_retrieval_eval(questions)
    if args.generation:
        run_generation_eval(questions)


if __name__ == "__main__":
    main()
