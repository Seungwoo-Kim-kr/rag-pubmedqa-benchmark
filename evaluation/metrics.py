"""
evaluation/metrics.py
=====================
자동 평가 지표 계산.
- token_overlap: gold answer와 predicted answer의 토큰 겹침 (F1)
- answer_length: 답변 길이
- retrieval_hit: gold evidence가 retrieved context에 포함되는지 여부
"""

import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def token_f1(gold: str, pred: str) -> float:
    """token overlap F1 (SQuAD 스타일)."""
    gold_tokens = Counter(_tokenize(gold))
    pred_tokens = Counter(_tokenize(pred))

    common = sum((gold_tokens & pred_tokens).values())
    if common == 0:
        return 0.0

    precision = common / sum(pred_tokens.values())
    recall = common / sum(gold_tokens.values())
    return 2 * precision * recall / (precision + recall)


def exact_match(gold: str, pred: str) -> bool:
    return _tokenize(gold) == _tokenize(pred)


def retrieval_hit(gold_evidence: list[str], retrieved_chunks: list) -> bool:
    """gold evidence 중 하나라도 retrieved context에 포함되는지 확인."""
    if not gold_evidence or not retrieved_chunks:
        return False

    retrieved_texts = []
    for c in retrieved_chunks:
        if isinstance(c, dict):
            retrieved_texts.append(c.get("text", "").lower())
        elif isinstance(c, str):
            retrieved_texts.append(c.lower())

    combined_retrieved = " ".join(retrieved_texts)

    for evidence in gold_evidence:
        if not isinstance(evidence, str):
            evidence = " ".join(evidence) if isinstance(evidence, list) else str(evidence)
        ev_tokens = set(_tokenize(evidence))
        if not ev_tokens:
            continue
        hit_count = sum(1 for t in ev_tokens if t in combined_retrieved)
        if hit_count / len(ev_tokens) >= 0.5:
            return True

    return False


def score_result(result: dict, question: dict) -> dict:
    """단일 결과에 대해 자동 평가 지표를 계산합니다."""
    gold = question.get("gold_answer", "")
    pred = result.get("answer", "")
    gold_evidence = question.get("gold_evidence", [])
    retrieved = result.get("retrieved_context", [])

    return {
        "question_id": result["question_id"],
        "baseline": result["baseline"],
        "question_type": result["question_type"],
        "token_f1": round(token_f1(gold, pred), 4),
        "exact_match": exact_match(gold, pred),
        "retrieval_hit": retrieval_hit(gold_evidence, retrieved),
        "answer_length": len(pred.split()),
        "runtime_sec": result.get("runtime_sec", 0),
    }
