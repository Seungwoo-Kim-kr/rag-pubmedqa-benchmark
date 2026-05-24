"""
evaluation/llm_judge.py
=======================
LLM-as-judge: GPT가 각 (질문, gold 답변, 예측 답변)을 평가합니다.

점수:
  correct  (1.0) - 예측이 gold를 의미적으로 포함하거나 동등
  partial  (0.5) - 부분적으로 맞음 (핵심 정보 일부 포함)
  incorrect(0.0) - 완전히 틀리거나 gold와 무관

주의:
  rate limit 방지를 위해 호출 사이에 짧은 딜레이를 둡니다.
"""

import time
from openai import OpenAI

JUDGE_PROMPT = """\
You are an expert evaluator for question answering systems.

Given a question, a gold (reference) answer, and a predicted answer, \
score the predicted answer on a 3-point scale:

- correct   (score=1.0): The predicted answer is semantically equivalent to \
or contains all the key information from the gold answer.
- partial   (score=0.5): The predicted answer contains some relevant information \
from the gold answer but is missing key details or adds significant irrelevant content.
- incorrect (score=0.0): The predicted answer is factually wrong, contradicts the \
gold answer, or contains none of the key information.

Respond with a JSON object in this exact format (nothing else):
{{"label": "correct"|"partial"|"incorrect", "score": 1.0|0.5|0.0, "reason": "<one sentence>"}}

Question: {question}
Gold answer: {gold}
Predicted answer: {pred}
"""

SCORE_MAP = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def judge_one(
    client: OpenAI,
    question: str,
    gold: str,
    pred: str,
    model: str = "gpt-4o-mini",
    retries: int = 3,
    delay: float = 0.5,
) -> dict:
    """단일 QA쌍을 GPT로 평가. 실패 시 retries번 재시도."""
    prompt = JUDGE_PROMPT.format(
        question=question.strip(),
        gold=gold.strip(),
        pred=pred.strip(),
    )

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=200,
            )
            raw = resp.choices[0].message.content
            import json
            parsed = json.loads(raw)
            label = parsed.get("label", "incorrect")
            score = float(parsed.get("score", SCORE_MAP.get(label, 0.0)))
            reason = parsed.get("reason", "")
            return {
                "judge_label": label,
                "judge_score": score,
                "judge_reason": reason,
            }
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {
                    "judge_label": "error",
                    "judge_score": 0.0,
                    "judge_reason": f"judge error: {e}",
                }
    # should not reach
    return {"judge_label": "error", "judge_score": 0.0, "judge_reason": "unknown"}


def run_judge(
    results: list[dict],
    question_map: dict[str, dict],
    model: str = "gpt-4o-mini",
    delay: float = 0.3,
) -> list[dict]:
    """
    results: raw 또는 scored 결과 리스트
    question_map: question_id → question dict (gold_answer 포함)
    반환: judge_label / judge_score / judge_reason 필드가 추가된 리스트
    """
    client = OpenAI()
    judged = []

    for i, r in enumerate(results):
        q = question_map.get(r["question_id"])
        if q is None:
            r = dict(r, judge_label="skip", judge_score=None, judge_reason="question not found")
            judged.append(r)
            continue

        verdict = judge_one(
            client=client,
            question=q["question"],
            gold=q["gold_answer"],
            pred=r.get("answer", ""),
            model=model,
        )
        r = dict(r, **verdict)
        judged.append(r)

        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(results)}] 완료")

        time.sleep(delay)

    return judged
