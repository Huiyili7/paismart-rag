"""Render evaluation results to JSON + a human-readable Markdown report."""

from __future__ import annotations

import json
import os
from typing import Any


def write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def render_markdown(payload: dict[str, Any]) -> str:
    r = payload["retrieval"]
    lat = payload["latency_ms"]
    g = payload["generation"]
    gates = payload.get("gates", {})

    lines: list[str] = []
    lines.append("# PaiSmart RAG 评测报告")
    lines.append("")
    lines.append(f"- 运行模式: `{payload['mode']}`")
    lines.append(f"- 数据集: `{payload['dataset']}`  (问题数: {payload['n_questions']})")
    lines.append(f"- top_k: {payload['top_k']}")
    if payload.get("note"):
        lines.append(f"- 说明: {payload['note']}")
    lines.append("")

    lines.append("## 检索质量")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("| --- | --- |")
    for k in payload["ks"]:
        lines.append(f"| Recall@{k} | {_pct(r[f'recall@{k}'])} |")
    for k in payload["ks"]:
        lines.append(f"| nDCG@{k} | {r[f'ndcg@{k}']:.3f} |")
    lines.append(f"| MRR | {r['mrr']:.3f} |")
    lines.append("")

    lines.append("## 延迟 (检索接口, ms)")
    lines.append("")
    lines.append("| p50 | p95 | p99 | mean | max |")
    lines.append("| --- | --- | --- | --- | --- |")
    lines.append(f"| {lat['p50']} | {lat['p95']} | {lat['p99']} | {lat['mean']} | {lat['max']} |")
    lines.append("")

    if g.get("judged"):
        lines.append("## 生成质量 (groundedness / 幻觉)")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("| --- | --- |")
        lines.append(f"| 幻觉率 | {_pct(g['hallucination_rate'])} |")
        lines.append(f"| 拒答正确率 (不可答问题) | {_pct(g['refusal_accuracy'])} |")
        lines.append(f"| 平均 groundedness | {g['avg_grounded_score']:.3f} |")
        lines.append("")

    if gates:
        lines.append("## 回归门禁 (shift-left)")
        lines.append("")
        lines.append("| 门禁 | 阈值 | 实测 | 结果 |")
        lines.append("| --- | --- | --- | --- |")
        for name, gate in gates.items():
            status = "✅ PASS" if gate["passed"] else "❌ FAIL"
            lines.append(f"| {name} | {gate['threshold']} | {gate['actual']} | {status} |")
        lines.append("")

    # per-question failures, to support root-cause analysis
    fails = [q for q in payload["per_question"] if not q["recall_hit"] or q.get("hallucinated")]
    if fails:
        lines.append("## 失败用例 (供根因分析)")
        lines.append("")
        lines.append("| id | 类型 | 召回命中 | 幻觉 | 问题 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for q in fails:
            lines.append(
                f"| {q['id']} | {q['type']} | {'是' if q['recall_hit'] else '否'} "
                f"| {'是' if q.get('hallucinated') else '否'} | {q['question'][:40]} |"
            )
        lines.append("")

    return "\n".join(lines)


def write_markdown(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_markdown(payload))
