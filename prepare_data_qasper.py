"""
prepare_data_qasper.py
======================
QASPER 데이터셋을 raw JSON에서 파싱해서 실험용 포맷으로 저장합니다.
- 질문: human-annotated (사람이 직접 작성)
- gold answer: human-annotated (사람이 직접 작성)
- 질문 타입: 키워드 기반 자동 분류

실행:
    python prepare_data_qasper.py
"""

import json
import re
from collections import Counter
from pathlib import Path

import yaml
from tqdm import tqdm


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


# ── 텍스트 정리 ──────────────────────────────────────────────
REF_PATTERN = re.compile(r'\b(BIBREF|TABREF|FIGREF|SECREF|FOOTREF|EQREF|APPREF)\d*\b')

def clean_text(text: str) -> str:
    text = REF_PATTERN.sub("", text)       # 참조 태그 제거
    text = re.sub(r"\s+", " ", text)       # 공백 정규화
    return text.strip()


# ── 섹션 추출 ─────────────────────────────────────────────────
def extract_sections(paper: dict) -> list:
    sections = []
    for section in paper.get("full_text", []):
        title = section.get("section_name", "") or "Unknown"
        paragraphs = section.get("paragraphs", []) or []
        text = " ".join(clean_text(p) for p in paragraphs if p)
        if len(text) < 50:
            continue
        sections.append({"section_title": title, "text": text})
    return sections


# ── gold answer 추출 ──────────────────────────────────────────
def extract_gold(qa: dict):
    """
    QASPER QA에서 gold answer, gold evidence, answer_type을 추출합니다.
    답변 우선순위: free_form > extractive_spans > yes_no
    unanswerable 질문은 건너뜁니다.
    """
    answers = qa.get("answers", [])
    if not answers:
        return None, [], None

    ans = answers[0].get("answer", {})

    if ans.get("unanswerable"):
        return None, [], None

    free_form  = clean_text(ans.get("free_form_answer", "") or "")
    extractive = [clean_text(s) for s in (ans.get("extractive_spans", []) or []) if s]
    yes_no     = ans.get("yes_no", None)
    evidence   = [clean_text(e) for e in (ans.get("evidence", []) or []) if e and len(e) > 20]

    if free_form:
        gold = free_form
        atype = "free_form"
    elif extractive:
        gold = " ".join(extractive[:2])
        atype = "extractive"
    elif yes_no is not None:
        gold = "Yes" if yes_no else "No"
        atype = "yes_no"
    else:
        return None, [], None

    return gold, evidence[:3], atype


# ── 질문 타입 분류 ────────────────────────────────────────────
def classify_question_type(question: str, answer_type: str) -> str:
    q = question.lower()

    if answer_type == "yes_no":
        return "local_factual"

    # terminology: 용어/약어/개념 정의를 묻는 질문
    if any(w in q for w in [
        "what does", "what is meant", "define", "definition",
        "refer to", "stand for", "abbreviat", "denote",
        "what are the", "what is the term", "what is a ",
        "how is", "how do the authors define", "how do the authors use",
    ]):
        return "terminology_sensitive"

    # global_synthesis: 전체 논문 이해 필요
    if any(w in q for w in [
        "overall", "main contribution", "broadly", "in general",
        "summarize", "key finding", "main idea", "conclude",
        "approach", "propose", "novel", "main advantage",
        "why did", "what is the purpose", "what is the goal",
        "what is the motivation", "how does the proposed",
        "what method", "what technique", "how does the model",
        "what architecture", "describe the",
    ]):
        return "global_synthesis"

    # free_form이고 질문이 길면 synthesis 가능성 높음
    if answer_type == "free_form" and len(question.split()) > 10:
        return "global_synthesis"

    return "local_factual"


# ── 메인 파싱 ─────────────────────────────────────────────────
def process_qasper(json_path: str, num_docs: int, target_per_type: int) -> tuple:
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)

    docs = []
    questions = []
    type_counts = Counter()
    q_idx = 0
    target_types = {"local_factual", "global_synthesis", "terminology_sensitive"}

    for paper_id, paper in tqdm(raw.items(), desc="QASPER 파싱"):
        if len(docs) >= num_docs:
            break
        if all(type_counts[t] >= target_per_type for t in target_types):
            break

        sections = extract_sections(paper)
        if not sections:
            continue

        doc_id = f"doc_{len(docs):03d}"
        doc = {
            "doc_id": doc_id,
            "paper_id": paper_id,
            "title": paper.get("title", ""),
            "abstract": clean_text(paper.get("abstract", "") or ""),
            "sections": sections,
        }
        docs.append(doc)

        for qa in paper.get("qas", []):
            if all(type_counts[t] >= target_per_type for t in target_types):
                break

            question_text = qa.get("question", "").strip()
            if not question_text:
                continue

            gold, evidence, atype = extract_gold(qa)
            if not gold:
                continue

            q_type = classify_question_type(question_text, atype)
            if type_counts[q_type] >= target_per_type:
                continue

            questions.append({
                "question_id": f"q_{q_idx:03d}",
                "doc_id": doc_id,
                "paper_id": paper_id,
                "question_type": q_type,
                "answer_type": atype,
                "question": question_text,
                "gold_answer": gold,
                "gold_evidence": evidence,
                "human_annotated": True,
            })
            type_counts[q_type] += 1
            q_idx += 1

    return docs, questions


def save(docs, questions, config):
    proc_dir = Path(config["paths"]["data_processed"])
    qa_dir   = Path(config["paths"]["qa_sets"])
    proc_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    docs_path = proc_dir  / "documents_qasper.json"
    qa_path   = qa_dir    / "questions_qasper.json"

    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료:")
    print(f"  문서: {docs_path}  ({len(docs)}개)")
    print(f"  질문: {qa_path}  ({len(questions)}개)")

    type_dist = Counter(q["question_type"] for q in questions)
    print("\n질문 타입 분포 (모두 human-annotated):")
    for t, c in type_dist.items():
        print(f"  {t}: {c}개")

    answer_type_dist = Counter(q["answer_type"] for q in questions)
    print("\nanswer type 분포:")
    for t, c in answer_type_dist.items():
        print(f"  {t}: {c}개")


def main():
    config = load_config()

    # train set 사용 (888개 논문) — dev(281개)보다 넓게 스캔
    json_path = "data/raw/qasper-train-v0.3.json"
    num_docs = 200         # 넓게 스캔
    target_per_type = 15   # 각 타입 15개 → 총 45개

    print(f"QASPER dev set 파싱 중: {json_path}")
    print(f"목표: {num_docs}개 문서, 타입별 {target_per_type}개 질문\n")

    docs, questions = process_qasper(json_path, num_docs, target_per_type)
    save(docs, questions, config)


if __name__ == "__main__":
    main()
