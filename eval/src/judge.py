"""Groundedness / hallucination judging.

Two layers:
  1. A cheap deterministic refusal detector (string heuristics) — runs anywhere,
     used to score whether the system correctly says "暂无相关信息" on
     out-of-knowledge-base questions.
  2. An optional LLM-as-judge that reads (question, retrieved context, answer)
     and decides whether every claim in the answer is supported by the context.

Hallucination, as scored here, is the union of two failure modes:
  - answerable question, but the answer contains claims not supported by the
    retrieved context; or
  - unanswerable question (nothing relevant in the KB), but the system answered
    anyway instead of refusing.
"""

from __future__ import annotations

import json
import re

REFUSAL_MARKERS = ["暂无相关信息", "没有相关信息", "无法回答", "未找到相关", "不知道"]

JUDGE_SYSTEM = (
    "你是一个严格的 RAG 答案评审。你会收到【问题】【检索到的参考信息】和【模型回答】。"
    "你的任务是判断回答是否完全基于参考信息（grounded），即回答中的每一个事实性论断"
    "是否都能在参考信息中找到依据。\n"
    "评判规则：\n"
    "- 如果回答中存在参考信息无法支撑的事实性论断，判为 hallucinated=true。\n"
    "- 如果参考信息为空或与问题无关，而回答仍给出了具体事实，判为 hallucinated=true。\n"
    "- 如果回答是明确的拒答（如\"暂无相关信息\"），判为 refused=true, hallucinated=false。\n"
    "- groundedness 打分范围 0~1，1 表示完全有依据，0 表示完全编造。\n"
    "只返回 JSON，不要任何多余文字，格式："
    '{"grounded_score": <0~1 float>, "hallucinated": <bool>, "refused": <bool>, "reason": "<简短中文理由>"}'
)


def detect_refusal(answer: str) -> bool:
    return any(marker in answer for marker in REFUSAL_MARKERS)


def _extract_json(text: str) -> dict:
    """Best-effort JSON extraction from an LLM response."""
    text = text.strip()
    # strip ```json fences if present
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def judge_with_llm(llm, question: str, context: str, answer: str) -> dict:
    """Run the LLM-as-judge. ``llm`` is an LLMClient. Returns the parsed verdict."""
    user = (
        f"【问题】\n{question}\n\n"
        f"【检索到的参考信息】\n{context if context.strip() else '(无)'}\n\n"
        f"【模型回答】\n{answer}"
    )
    raw = llm.chat(
        [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=400,
    )
    verdict = _extract_json(raw)
    # normalise
    verdict["grounded_score"] = float(verdict.get("grounded_score", 0.0))
    verdict["hallucinated"] = bool(verdict.get("hallucinated", False))
    verdict["refused"] = bool(verdict.get("refused", detect_refusal(answer)))
    return verdict


def heuristic_verdict(question_type: str, answer: str, retrieved_relevant: bool) -> dict:
    """Deterministic fallback verdict when no judge LLM is configured.

    - unanswerable question: correct iff the system refused.
    - answerable question: we cannot verify grounding without an LLM, so we only
      flag the obvious failure of refusing when relevant context *was* retrieved.
    """
    refused = detect_refusal(answer)
    if question_type == "unanswerable":
        return {
            "grounded_score": 1.0 if refused else 0.0,
            "hallucinated": not refused,
            "refused": refused,
            "reason": "应拒答" + ("，已正确拒答" if refused else "，却给出了具体回答（疑似幻觉）"),
        }
    # answerable
    return {
        "grounded_score": 0.0 if (refused and retrieved_relevant) else 1.0,
        "hallucinated": False,
        "refused": refused,
        "reason": "启发式判定（未启用 LLM 评审，无法逐句核验依据）",
    }
