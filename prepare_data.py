"""
prepare_data.py
===============
QASPER 데이터셋을 HuggingFace API로 로드해서 실험용 포맷으로 저장합니다.
다운로드 없이 API 호출로 동작합니다.

실행:
    python prepare_data.py
"""

import json
import os
import re
from pathlib import Path

import yaml
from datasets import load_dataset
from tqdm import tqdm


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_sections(paper: dict) -> list[dict]:
    """QASPER 논문에서 section 단위 텍스트를 추출합니다."""
    sections = []
    full_text = paper.get("full_text", {})

    section_titles = full_text.get("section_name", []) or []
    paragraphs_list = full_text.get("paragraphs", []) or []

    for title, paragraphs in zip(section_titles, paragraphs_list):
        if not paragraphs:
            continue
        combined = " ".join(clean_text(p) for p in paragraphs if p)
        if len(combined) < 50:
            continue
        sections.append({
            "section_title": title or "Unknown",
            "text": combined,
        })

    return sections


def classify_question_type(question: str) -> str:
    """질문 텍스트를 보고 타입을 간단히 분류합니다."""
    q = question.lower()
    if any(w in q for w in ["what is", "define", "definition", "term", "mean"]):
        return "terminology_sensitive"
    if any(w in q for w in ["how many", "how much", "what number", "percentage", "size"]):
        return "local_factual"
    if any(w in q for w in ["overall", "main", "general", "broadly", "summarize", "across"]):
        return "global_synthesis"
    return "local_factual"


def extract_gold_answer(answer_data: dict) -> tuple[str, list[str]]:
    """QASPER answer 포맷에서 gold answer와 evidence를 추출합니다."""
    answers = answer_data.get("answers", [])
    if not answers:
        return "", []

    first = answers[0]
    answer_type = first.get("answer_type", "")
    free_form = first.get("free_form_answer", "") or ""
    yes_no = first.get("yes_no_answer", "") or ""
    evidence_list = first.get("evidence", []) or []

    if free_form:
        gold = clean_text(free_form)
    elif yes_no:
        gold = yes_no
    elif answer_type == "extractive":
        gold = " ".join(evidence_list[:1])
    else:
        gold = ""

    evidence = [clean_text(e) for e in evidence_list if e]
    return gold, evidence


def process_dataset(config: dict) -> tuple[list[dict], list[dict]]:
    """PubMedQA를 로드해서 문서 리스트와 질문셋을 반환합니다."""
    print("PubMedQA 데이터셋 로드 중 (HuggingFace API)...")
    # pqa_labeled: 전문가 라벨링된 1,000개 QA (gold answer + long answer context 포함)
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

    num_docs = config["dataset"]["num_docs"]
    num_questions = config["dataset"]["num_questions"]

    docs = []
    questions = []
    q_count = 0

    for idx, item in enumerate(tqdm(dataset, desc="문서 처리")):
        if idx >= num_docs:
            break
        if q_count >= num_questions:
            break

        doc_id = f"doc_{idx:03d}"

        # PubMedQA context: {"contexts": [str, str, ...]}
        contexts = item.get("context", {})
        paragraphs = contexts.get("contexts", []) or []

        sections = []
        for i, para in enumerate(paragraphs):
            text = clean_text(str(para))
            if len(text) < 50:
                continue
            sections.append({
                "section_title": f"Context {i+1}",
                "text": text,
            })

        if not sections:
            continue

        doc = {
            "doc_id": doc_id,
            "title": item.get("question", "")[:80],  # 제목 대용
            "abstract": "",
            "sections": sections,
        }
        docs.append(doc)

        # 질문 추출
        q_text = item.get("question", "")
        gold_answer = item.get("long_answer", "") or item.get("final_decision", "")
        gold_answer = clean_text(str(gold_answer))
        gold_evidence = paragraphs[:3]

        if not q_text or not gold_answer:
            continue

        questions.append({
            "question_id": f"q_{q_count:03d}",
            "doc_id": doc_id,
            "question_type": classify_question_type(q_text),
            "question": q_text,
            "gold_answer": gold_answer,
            "gold_evidence": [clean_text(e) for e in gold_evidence if e],
        })
        q_count += 1

    return docs, questions


def save_data(docs: list[dict], questions: list[dict], config: dict):
    proc_dir = Path(config["paths"]["data_processed"])
    qa_dir = Path(config["paths"]["qa_sets"])
    proc_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    docs_path = proc_dir / "documents.json"
    qa_path = qa_dir / "questions.json"

    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료:")
    print(f"  문서: {docs_path}  ({len(docs)}개)")
    print(f"  질문: {qa_path}  ({len(questions)}개)")

    # 질문 타입 분포 출력
    from collections import Counter
    type_counts = Counter(q["question_type"] for q in questions)
    print("\n질문 타입 분포:")
    for t, c in type_counts.items():
        print(f"  {t}: {c}개")


def main():
    config = load_config()
    docs, questions = process_dataset(config)
    save_data(docs, questions, config)


if __name__ == "__main__":
    main()
