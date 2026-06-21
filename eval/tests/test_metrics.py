"""Unit tests for the retrieval metric math.

These run in CI with no infrastructure (no ES, no API keys) and guard against
regressions in how we compute Recall@k / MRR / nDCG / latency percentiles — the
numbers that end up on dashboards and in reports.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import math

import metrics  # noqa: E402


def test_recall_at_k_hit_and_miss():
    assert metrics.recall_at_k(["a", "b", "c"], {"c"}, 3) == 1.0
    assert metrics.recall_at_k(["a", "b", "c"], {"c"}, 2) == 0.0
    assert metrics.recall_at_k(["a", "b"], {"z"}, 5) == 0.0
    # no relevant docs defined -> undefined, treated as 0
    assert metrics.recall_at_k(["a"], set(), 1) == 0.0


def test_precision_at_k():
    assert metrics.precision_at_k(["a", "b", "c", "d"], {"a", "c"}, 4) == 0.5
    assert metrics.precision_at_k(["a", "b"], {"a"}, 1) == 1.0
    assert metrics.precision_at_k([], {"a"}, 3) == 0.0


def test_reciprocal_rank():
    assert metrics.reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert metrics.reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert metrics.reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


def test_ndcg_perfect_vs_worse():
    # relevant doc at rank 1 -> ndcg 1.0
    assert metrics.ndcg_at_k(["a", "b", "c"], {"a"}, 3) == 1.0
    # same relevant doc at rank 3 -> discounted
    worse = metrics.ndcg_at_k(["x", "y", "a"], {"a"}, 3)
    assert 0.0 < worse < 1.0
    assert math.isclose(worse, (1.0 / math.log2(4)) / 1.0, rel_tol=1e-9)


def test_ndcg_two_relevant():
    # both relevant docs in top-2 in ideal order -> 1.0
    assert math.isclose(metrics.ndcg_at_k(["a", "b", "c"], {"a", "b"}, 3), 1.0, rel_tol=1e-9)


def test_percentile_interpolation():
    data = [10, 20, 30, 40]
    assert metrics.percentile(data, 0) == 10
    assert metrics.percentile(data, 100) == 40
    assert math.isclose(metrics.percentile(data, 50), 25.0, rel_tol=1e-9)


def test_latency_summary_shape():
    s = metrics.latency_summary([100, 200, 300, 400, 500])
    assert s["p50"] == 300
    assert s["max"] == 500
    assert set(s) == {"p50", "p95", "p99", "mean", "max"}


def test_score_and_aggregate():
    ks = [1, 3, 5]
    q1 = metrics.score_query(["a", "b", "c"], {"a"}, ks)  # perfect
    q2 = metrics.score_query(["x", "y", "c"], {"c"}, ks)  # relevant at rank 3
    agg = metrics.aggregate_retrieval([q1, q2], ks)
    assert agg["n"] == 2
    assert agg["recall@5"] == 1.0          # both found within 5
    assert agg["recall@1"] == 0.5          # only q1 found at rank 1
    assert 0.0 < agg["mrr"] <= 1.0
