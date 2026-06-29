"""Custom retrieval metrics (spec §8.3).

These are dependency-free so the retrieval benchmark can run in CI without
RAGAS. Generation metrics (faithfulness, etc.) live in run_eval.py behind the
optional RAGAS import.

A benchmark item provides ``relevant_paper_ids`` (or ``relevant_chunk_ids``) —
the ground-truth sources. ``retrieved`` is the ordered list of ids the system
returned for that query.
"""

from __future__ import annotations

from dataclasses import dataclass


def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return len(top & relevant) / len(relevant)


def precision_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    if k == 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r in relevant) / len(top)


def reciprocal_rank(retrieved: list[int], relevant: set[int]) -> float:
    for i, r in enumerate(retrieved, 1):
        if r in relevant:
            return 1.0 / i
    return 0.0


@dataclass
class RetrievalReport:
    recall_at_5: float
    recall_at_10: float
    precision_at_5: float
    mrr: float

    THRESHOLDS = {  # spec §8.3 minimum thresholds
        "recall_at_5": 0.70,
        "recall_at_10": 0.80,
        "precision_at_5": 0.60,
        "mrr": 0.65,
    }

    def passes(self) -> bool:
        return (
            self.recall_at_5 >= self.THRESHOLDS["recall_at_5"]
            and self.recall_at_10 >= self.THRESHOLDS["recall_at_10"]
            and self.precision_at_5 >= self.THRESHOLDS["precision_at_5"]
            and self.mrr >= self.THRESHOLDS["mrr"]
        )

    def as_dict(self) -> dict:
        return {
            "Recall@5": round(self.recall_at_5, 4),
            "Recall@10": round(self.recall_at_10, 4),
            "Precision@5": round(self.precision_at_5, 4),
            "MRR": round(self.mrr, 4),
        }


def aggregate(per_query: list[dict]) -> RetrievalReport:
    """per_query items: {'retrieved': [...], 'relevant': set(...)}"""
    n = len(per_query) or 1
    return RetrievalReport(
        recall_at_5=sum(recall_at_k(q["retrieved"], q["relevant"], 5) for q in per_query) / n,
        recall_at_10=sum(recall_at_k(q["retrieved"], q["relevant"], 10) for q in per_query) / n,
        precision_at_5=sum(precision_at_k(q["retrieved"], q["relevant"], 5) for q in per_query) / n,
        mrr=sum(reciprocal_rank(q["retrieved"], q["relevant"]) for q in per_query) / n,
    )
