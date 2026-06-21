"""Retrieval-quality metrics for the PaiSmart RAG pipeline.

All functions here are pure (no I/O, no network) so they can be unit-tested in
CI without any running infrastructure. The orchestrator in ``evaluate.py`` feeds
them the ranked lists produced either by the live ``/api/v1/search/hybrid``
endpoint or by recorded fixtures.

Definitions
-----------
A *retrieved item* is identified by its source document id (we evaluate
relevance at document granularity, since a question is "answered" if any chunk
of the right document is retrieved). ``relevant`` is the set of document ids
that genuinely answer the question, taken from the gold set.

- Recall@k : did we retrieve at least one relevant doc within the top-k?
              (a.k.a. hit-rate@k — the metric that matters for RAG, because the
              LLM only needs one good chunk to ground its answer)
- Precision@k : fraction of the top-k that are relevant.
- MRR       : mean reciprocal rank of the first relevant doc (0 if none found).
- nDCG@k    : rank-discounted gain, rewards putting relevant docs higher.
"""

from __future__ import annotations

import math
from statistics import median
from typing import Iterable, Sequence


def _first_relevant_rank(ranked_ids: Sequence[str], relevant: set[str]) -> int | None:
    """1-based rank of the first relevant id, or None if absent."""
    for idx, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            return idx
    return None


def recall_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant doc appears in the top-k, else 0.0.

    For a single query this is binary; averaged over the gold set it becomes the
    familiar "Recall@k / hit-rate" percentage.
    """
    if not relevant:
        return 0.0
    return 1.0 if any(doc_id in relevant for doc_id in ranked_ids[:k]) else 0.0


def precision_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = ranked_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for doc_id in top if doc_id in relevant)
    return hits / len(top)


def reciprocal_rank(ranked_ids: Sequence[str], relevant: set[str]) -> float:
    rank = _first_relevant_rank(ranked_ids, relevant)
    return 1.0 / rank if rank else 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary-gain nDCG@k. IDCG assumes all relevant docs could sit at the top."""
    if not relevant:
        return 0.0
    dcg = 0.0
    for idx, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile (p in [0, 100]). Empty -> 0.0."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    rank = (p / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[int(rank)])
    frac = rank - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * frac)


def latency_summary(latencies_ms: Sequence[float]) -> dict[str, float]:
    """p50 / p95 / p99 / mean / max over a list of per-query latencies (ms)."""
    if not latencies_ms:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "p50": round(median(latencies_ms), 1),
        "p95": round(percentile(latencies_ms, 95), 1),
        "p99": round(percentile(latencies_ms, 99), 1),
        "mean": round(sum(latencies_ms) / len(latencies_ms), 1),
        "max": round(max(latencies_ms), 1),
    }


def aggregate_retrieval(per_query: Iterable[dict], ks: Sequence[int]) -> dict:
    """Average per-query metric dicts into corpus-level numbers.

    Each ``per_query`` item is expected to carry ``recall@{k}``, ``ndcg@{k}`` and
    ``rr`` keys (as produced by :func:`score_query`).
    """
    rows = list(per_query)
    n = len(rows)
    if n == 0:
        return {"n": 0}
    out: dict[str, float] = {"n": n}
    for k in ks:
        out[f"recall@{k}"] = round(sum(r[f"recall@{k}"] for r in rows) / n, 4)
        out[f"ndcg@{k}"] = round(sum(r[f"ndcg@{k}"] for r in rows) / n, 4)
    out["mrr"] = round(sum(r["rr"] for r in rows) / n, 4)
    return out


def score_query(ranked_ids: Sequence[str], relevant: set[str], ks: Sequence[int]) -> dict:
    """Compute every retrieval metric for one query."""
    row: dict[str, float] = {"rr": reciprocal_rank(ranked_ids, relevant)}
    for k in ks:
        row[f"recall@{k}"] = recall_at_k(ranked_ids, relevant, k)
        row[f"ndcg@{k}"] = ndcg_at_k(ranked_ids, relevant, k)
    return row
