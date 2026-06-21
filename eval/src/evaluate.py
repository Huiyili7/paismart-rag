"""End-to-end RAG evaluation orchestrator for PaiSmart.

Two modes
---------
offline : read recorded retrieval results from a fixtures file and compute all
          metrics deterministically. No infrastructure or API keys required —
          this is what runs in CI to guard the metric pipeline and the gates.

live    : log in to the running backend, hit /api/v1/search/hybrid for every
          gold question, optionally generate an answer with the production
          prompt and judge its groundedness. This is what you run locally
          against seeded Elasticsearch to produce the real headline numbers.

Usage
-----
  python src/evaluate.py --mode offline \
      --dataset datasets/education_qa.jsonl \
      --fixtures fixtures/retrieval_results.jsonl \
      --config config.example.yaml --out reports

  python src/evaluate.py --mode live \
      --dataset datasets/education_qa.jsonl \
      --config config.yaml --generate --judge --out reports

Exit code is non-zero if any configured gate fails (shift-left: a regression
breaks the build).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))

import metrics  # noqa: E402
import prompts  # noqa: E402
from judge import detect_refusal, heuristic_verdict, judge_with_llm  # noqa: E402

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _load_yaml(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML not installed but a --config was given")
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    # expand ${ENV_VAR} references
    raw = re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), raw)
    return yaml.safe_load(raw) or {}


def load_dataset(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_fixtures(path: str) -> dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                out[row["id"]] = row
    return out


def evaluate(args) -> dict:
    cfg = _load_yaml(args.config)
    ks = cfg.get("retrieval", {}).get("ks", [1, 3, 5])
    top_k = args.top_k or cfg.get("retrieval", {}).get("top_k", 5)
    dataset = load_dataset(args.dataset)

    backend = llm = None
    if args.mode == "live":
        from clients import BackendClient, LLMClient

        b = cfg.get("backend", {})
        backend = BackendClient(b.get("base_url", "http://localhost:8080"),
                                b.get("username", "admin"), b.get("password", "admin123"))
        backend.login()
        if args.generate or args.judge:
            lc = cfg.get("llm", {})
            llm = LLMClient(lc.get("base_url", "https://api.deepseek.com"),
                            lc.get("api_key", ""), lc.get("model", "deepseek-chat"))
        fixtures = {}
    else:
        fixtures = load_fixtures(args.fixtures)

    per_query_metrics: list[dict] = []
    latencies: list[float] = []
    per_question: list[dict] = []
    grounded_scores: list[float] = []
    hallucinations: list[bool] = []
    refusal_correct: list[bool] = []
    judged = False

    for item in dataset:
        qid = item["id"]
        relevant = set(item.get("relevant_docs", []))
        qtype = item.get("type", "answerable")

        # --- retrieval ---
        if args.mode == "live":
            chunks, latency = backend.search(item["question"], top_k)
            ranked_ids = [c.doc_id for c in chunks]
        else:
            fx = fixtures.get(qid, {})
            ranked_ids = fx.get("retrieved", [])
            latency = float(fx.get("latency_ms", 0.0))
            chunks = None
        latencies.append(latency)

        # Retrieval metrics are only defined for answerable questions (those with
        # a non-empty gold relevant set). Unanswerable questions are scored on
        # refusal/hallucination instead, not recall — so keep them out of the
        # Recall/MRR/nDCG denominator.
        is_answerable = bool(relevant)
        row = metrics.score_query(ranked_ids, relevant, ks)
        if is_answerable:
            per_query_metrics.append(row)
        recall_hit = bool(row.get(f"recall@{top_k}", row.get(f"recall@{max(ks)}", 0.0))) if is_answerable else True

        # --- generation + judging (optional) ---
        answer = None
        verdict = None
        if args.mode == "live" and args.generate and llm is not None:
            context = prompts.build_context(chunks or [])
            messages = prompts.build_messages(item["question"], context)
            answer = llm.chat(messages)
        elif args.mode == "offline":
            answer = fixtures.get(qid, {}).get("answer")

        if answer is not None:
            retrieved_relevant = recall_hit
            if args.judge and llm is not None and args.mode == "live":
                context = prompts.build_context(chunks or [])
                verdict = judge_with_llm(llm, item["question"], context, answer)
                judged = True
            else:
                verdict = heuristic_verdict(qtype, answer, retrieved_relevant)
                judged = True
            grounded_scores.append(verdict["grounded_score"])
            hallucinations.append(verdict["hallucinated"])
            if qtype == "unanswerable":
                refusal_correct.append(verdict["refused"])

        per_question.append({
            "id": qid,
            "type": qtype,
            "question": item["question"],
            "ranked": ranked_ids[:top_k],
            "recall_hit": recall_hit,
            "rr": row["rr"],
            "latency_ms": round(latency, 1),
            "answer": answer,
            "hallucinated": verdict["hallucinated"] if verdict else None,
            "refused": verdict["refused"] if verdict else None,
        })

    retrieval_agg = metrics.aggregate_retrieval(per_query_metrics, ks)
    latency_agg = metrics.latency_summary(latencies)
    generation = {"judged": judged}
    if judged:
        generation.update({
            "hallucination_rate": round(sum(hallucinations) / len(hallucinations), 4) if hallucinations else 0.0,
            "refusal_accuracy": round(sum(refusal_correct) / len(refusal_correct), 4) if refusal_correct else 0.0,
            "avg_grounded_score": round(sum(grounded_scores) / len(grounded_scores), 4) if grounded_scores else 0.0,
        })

    # --- gates ---
    gate_cfg = cfg.get("gates", {})
    gates = {}
    all_passed = True
    for name, threshold in gate_cfg.items():
        if name == "hallucination_rate":
            actual = generation.get("hallucination_rate", 0.0)
            passed = actual <= threshold
        elif name in retrieval_agg:
            actual = retrieval_agg[name]
            passed = actual >= threshold
        else:
            continue
        gates[name] = {"threshold": threshold, "actual": actual, "passed": passed}
        all_passed = all_passed and passed

    return {
        "mode": args.mode,
        "dataset": os.path.basename(args.dataset),
        "n_questions": len(dataset),
        "top_k": top_k,
        "ks": ks,
        "note": args.note,
        "retrieval": retrieval_agg,
        "latency_ms": latency_agg,
        "generation": generation,
        "gates": gates,
        "all_gates_passed": all_passed,
        "per_question": per_question,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="PaiSmart RAG evaluation")
    p.add_argument("--mode", choices=["offline", "live"], default="offline")
    p.add_argument("--dataset", default="datasets/education_qa.jsonl")
    p.add_argument("--fixtures", default="fixtures/retrieval_results.jsonl")
    p.add_argument("--config", default="config.example.yaml")
    p.add_argument("--out", default="reports")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--generate", action="store_true", help="(live) generate answers with the production prompt")
    p.add_argument("--judge", action="store_true", help="(live) run LLM-as-judge for groundedness")
    p.add_argument("--note", default=None)
    args = p.parse_args()

    # Windows consoles default to GBK and choke on report symbols (✅/✗/…).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    from report import write_json, write_markdown, render_markdown

    payload = evaluate(args)
    os.makedirs(args.out, exist_ok=True)
    write_json(os.path.join(args.out, "latest.json"), payload)
    write_markdown(os.path.join(args.out, "latest.md"), payload)
    print(render_markdown(payload))

    if not payload["all_gates_passed"]:
        print("\n[GATE] one or more quality gates FAILED", file=sys.stderr)
        return 1
    print("\n[GATE] all quality gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
