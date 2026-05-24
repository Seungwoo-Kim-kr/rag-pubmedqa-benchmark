"""
generate_questions.py
=====================
GPT를 사용해 각 문서에서 질문 타입별로 균등한 질문셋을 생성합니다.
목표: local_factual 15개 / global_synthesis 15개 / terminology_sensitive 15개 = 총 45개

실행:
    python generate_questions.py
"""

import json
import os
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

QUESTION_PROMPTS = {
    "local_factual": """\
Read the following scientific passage and generate ONE specific factual question that:
- Has a clear, concise answer found in the passage
- Asks about a specific detail, finding, number, method, or result
- Can be answered with 1-2 sentences from the passage

Passage:
{context}

Respond in JSON format:
{{
  "question": "...",
  "gold_answer": "...",
  "gold_evidence": "the exact sentence from the passage that contains the answer"
}}""",

    "global_synthesis": """\
Read the following scientific passage and generate ONE synthesis question that:
- Requires understanding the OVERALL message or conclusion of the passage
- Cannot be answered by a single sentence — requires integrating multiple parts
- Asks about main findings, overall conclusions, or the relationship between multiple concepts

Passage:
{context}

Respond in JSON format:
{{
  "question": "...",
  "gold_answer": "...",
  "gold_evidence": "the key sentences from the passage that together support the answer"
}}""",

    "terminology_sensitive": """\
Read the following scientific passage and generate ONE terminology question that:
- Asks about the meaning, definition, or precise usage of a specific technical term or concept in this passage
- The answer depends on how the term is used specifically in this passage (not general knowledge)
- Examples: "What does X refer to in this study?", "How is Y defined here?", "What is meant by Z in this context?"

Passage:
{context}

Respond in JSON format:
{{
  "question": "...",
  "gold_answer": "...",
  "gold_evidence": "the exact sentence that defines or explains the term"
}}""",
}


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def build_context(doc: dict, max_chars: int = 3000) -> str:
    parts = []
    if doc.get("abstract"):
        parts.append(doc["abstract"])
    for section in doc.get("sections", []):
        parts.append(f"[{section['section_title']}]\n{section['text']}")
    text = "\n\n".join(parts)
    return text[:max_chars]


def generate_question(client: OpenAI, context: str, q_type: str):
    prompt = QUESTION_PROMPTS[q_type].format(context=context)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=400,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        data = json.loads(response.choices[0].message.content)
        if not data.get("question") or not data.get("gold_answer"):
            return None
        return data
    except Exception as e:
        print(f"    생성 오류: {e}")
        return None


def main():
    load_dotenv()
    config = load_config()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    docs_path = Path(config["paths"]["data_processed"]) / "documents.json"
    with open(docs_path, encoding="utf-8") as f:
        docs = json.load(f)

    target_per_type = 15
    q_types = ["local_factual", "global_synthesis", "terminology_sensitive"]

    questions = []
    counts = {t: 0 for t in q_types}
    q_idx = 0

    print(f"목표: 각 타입별 {target_per_type}개 (총 {target_per_type * len(q_types)}개)\n")

    # 문서를 순환하면서 부족한 타입 채우기
    max_rounds = 3
    for round_num in range(max_rounds):
        if all(counts[t] >= target_per_type for t in q_types):
            break

        print(f"Round {round_num + 1} — 현재: {counts}")
        for doc in tqdm(docs, desc=f"  문서 처리"):
            if all(counts[t] >= target_per_type for t in q_types):
                break

            context = build_context(doc)
            if len(context) < 200:
                continue

            for q_type in q_types:
                if counts[q_type] >= target_per_type:
                    continue

                result = generate_question(client, context, q_type)
                if result is None:
                    continue

                evidence = result.get("gold_evidence", "")
                questions.append({
                    "question_id": f"q_{q_idx:03d}",
                    "doc_id": doc["doc_id"],
                    "question_type": q_type,
                    "question": result["question"],
                    "gold_answer": result["gold_answer"],
                    "gold_evidence": [evidence] if evidence else [],
                })
                counts[q_type] += 1
                q_idx += 1
                time.sleep(0.2)

    # 저장
    qa_dir = Path(config["paths"]["qa_sets"])
    qa_dir.mkdir(parents=True, exist_ok=True)
    out_path = qa_dir / "questions_balanced.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"\n완료!")
    print(f"  총 질문: {len(questions)}개")
    for t in q_types:
        print(f"  {t}: {counts[t]}개")
    print(f"  저장: {out_path}")


if __name__ == "__main__":
    main()
