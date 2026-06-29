# Benchmark dataset

Per spec §8.2, the benchmark corpus must be **constructed and committed before
Milestone 3** begins. It consists of:

- **50 open-access papers** from arXiv across 3 domains: NLP, Computer Vision,
  Explainable AI.
- **100 question–answer pairs** with ground-truth source citations, produced by
  a human annotator.

## Files

| File | Purpose |
|------|---------|
| `papers.jsonl`   | One line per benchmark paper: `{"arxiv_id", "domain", "title"}`. Used by `scripts/build_benchmark_corpus.py` to fetch and index. |
| `questions.jsonl`| One line per QA pair (schema below). |

## `questions.jsonl` schema

```json
{
  "id": "q001",
  "question": "What loss function does the Transformer use for training?",
  "ground_truth": "Cross-entropy with label smoothing of 0.1.",
  "relevant_paper_ids": [12],
  "relevant_chunk_keys": ["1706.03762::Methods::p5"],
  "domain": "NLP"
}
```

- `relevant_paper_ids` / `relevant_chunk_keys` are the gold sources used to score
  Recall@k, Precision@k, and MRR (§8.3).
- `ground_truth` is the reference answer used by RAGAS for context recall and
  (optionally) answer correctness (§8.4).

`sample_questions.jsonl` in this folder is a tiny illustrative example — replace
it with the full 100-item annotated set before Milestone 3.
