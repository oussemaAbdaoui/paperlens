"""Unit tests for the evaluation metrics (spec §8.3)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import aggregate, precision_at_k, recall_at_k, reciprocal_rank


def test_recall_and_precision():
    retrieved = [1, 2, 3, 4, 5]
    relevant = {3, 9}
    assert recall_at_k(retrieved, relevant, 5) == 0.5
    assert precision_at_k(retrieved, relevant, 5) == 0.2


def test_reciprocal_rank():
    assert reciprocal_rank([7, 4, 2], {2}) == 1 / 3
    assert reciprocal_rank([1, 2], {9}) == 0.0


def test_aggregate_and_thresholds():
    perfect = [{"retrieved": [1], "relevant": {1}} for _ in range(3)]
    report = aggregate(perfect)
    assert report.recall_at_5 == 1.0
    assert report.mrr == 1.0
    assert report.passes()
