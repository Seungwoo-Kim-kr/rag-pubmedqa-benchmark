"""
extract_failure_cases.py
========================
논문 table용 대표 failure case를 추출합니다.
각 failure_type × baseline 조합에서 가장 극명한 예시를 선별합니다.

실행:
    python extract_failure_cases.py
"""

import json
from collections import defaultdict
from pathlib import Path


def load_data():
    labeled_path = Path("results/scored_outputs/scored_results_20260503_153414_labeled.json")
    raw_path     = Path("results/raw_outputs/results_20260503_153414.json")
    qa_path      = Path("data/qa_sets/questions_balanced.json")

    with open(labeled_path, encoding="utf-8") as f:
        scored = json.load(f)
    with open(raw_path, encoding="utf-8") as f:
        raw = json.load(f)
    with open(qa_path, encoding="utf-8") as f:
        questions = json.load(f)

    raw_map = {(r["question_id"], r["baseline"]): r for r in raw}
    q_map   = {q["question_id"]: q for q in questions}
    return scored, raw_map, q_map


def truncate(text: str, n: int = 120) -> str:
    text = text.replace("\n", " ").strip()
    return text[:n] + "..." if len(text) > n else text


def extract_cases(scored, raw_map, q_map):
    # failure_type × baseline 조합별로 그룹화
    groups = defaultdict(list)
    for s in scored:
        key = (s["failure_type"], s["baseline"])
        groups[key].append(s)

    # 논문에 실을 핵심 케이스 조합
    targets = [
        # (failure_type, baseline, 선택기준)
        ("correct",                 "summary_mediated_qa", "max_f1"),
        ("correct",                 "direct_qa",           "max_f1"),
        ("global_synthesis_failure","standard_rag",        "min_f1"),
        ("terminology_mismatch",    "standard_rag",        "min_f1"),
        ("grounding_failure",       "direct_qa",           "min_f1"),
        ("low_overlap",             "standard_rag",        "min_f1"),
        ("retrieval_miss",          "direct_qa",           "min_f1"),
    ]

    cases = []
    for (ftype, baseline, criterion) in targets:
        items = groups.get((ftype, baseline), [])
        if not items:
            continue
        if criterion == "max_f1":
            item = max(items, key=lambda x: x["token_f1"])
        else:
            item = min(items, key=lambda x: x["token_f1"])

        qid = item["question_id"]
        raw = raw_map.get((qid, baseline), {})
        q   = q_map.get(qid, {})

        cases.append({
            "failure_type": ftype,
            "baseline": baseline,
            "question_type": item["question_type"],
            "token_f1": item["token_f1"],
            "retrieval_hit": item["retrieval_hit"],
            "question": q.get("question", ""),
            "gold_answer": q.get("gold_answer", ""),
            "predicted_answer": raw.get("answer", ""),
            "gold_evidence": q.get("gold_evidence", []),
        })

    return cases


def print_cases(cases):
    SEP = "─" * 90

    print("\n" + "=" * 90)
    print("  FAILURE CASE EXAMPLES  (논문 Table용)")
    print("=" * 90)

    for i, c in enumerate(cases, 1):
        tag = "✅ CORRECT" if c["failure_type"] == "correct" else f"❌ {c['failure_type'].upper()}"
        print(f"\n[Case {i}]  {tag}")
        print(f"  Baseline      : {c['baseline']}")
        print(f"  Question Type : {c['question_type']}")
        print(f"  Token F1      : {c['token_f1']:.3f}   Retrieval Hit: {c['retrieval_hit']}")
        print(SEP)
        print(f"  Q  : {truncate(c['question'], 110)}")
        print(f"  A✓ : {truncate(c['gold_answer'], 110)}")
        print(f"  A✗ : {truncate(c['predicted_answer'], 110)}")
        if c["gold_evidence"]:
            ev = c["gold_evidence"][0] if isinstance(c["gold_evidence"][0], str) else " ".join(c["gold_evidence"][0])
            print(f"  Ev : {truncate(ev, 110)}")
        print()

    print("=" * 90)


def save_cases(cases):
    out_path = Path("results/analysis/failure_cases_paper.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print(f"저장: {out_path}")


def main():
    scored, raw_map, q_map = load_data()
    cases = extract_cases(scored, raw_map, q_map)
    print_cases(cases)
    save_cases(cases)


if __name__ == "__main__":
    main()
