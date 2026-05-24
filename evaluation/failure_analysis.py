"""
evaluation/failure_analysis.py
================================
각 결과에 대해 failure_type을 자동 라벨링합니다.

Failure type 정의:
- correct              : Token F1 >= 0.5 (정답으로 간주)
- grounding_failure    : retrieval_hit=False이고 F1이 낮음 (근거 없는 답변)
- retrieval_miss       : retrieval_hit=False이지만 어느 정도 답은 맞음 (검색 실패)
- low_overlap          : retrieval_hit=True지만 F1이 낮음 (검색은 됐으나 답 생성 실패)
- global_synthesis_failure : global_synthesis 타입에서 F1이 낮음
- terminology_mismatch : terminology_sensitive 타입에서 F1이 낮음
"""

CORRECT_THRESHOLD = 0.5
LOW_THRESHOLD = 0.25


def label_failure(scored: dict) -> str:
    f1 = scored["token_f1"]
    hit = scored["retrieval_hit"]
    q_type = scored["question_type"]

    if f1 >= CORRECT_THRESHOLD:
        return "correct"

    if not hit:
        if f1 < LOW_THRESHOLD:
            return "grounding_failure"
        else:
            return "retrieval_miss"

    # hit=True 이지만 F1 낮음
    if q_type == "global_synthesis":
        return "global_synthesis_failure"
    if q_type == "terminology_sensitive":
        return "terminology_mismatch"

    return "low_overlap"


def add_failure_labels(scored_list: list[dict]) -> list[dict]:
    for item in scored_list:
        item["failure_type"] = label_failure(item)
    return scored_list


def print_failure_summary(scored_list: list[dict]):
    from collections import defaultdict, Counter

    print("\n" + "=" * 60)
    print("Failure Analysis Summary")
    print("=" * 60)

    by_baseline = defaultdict(list)
    for s in scored_list:
        by_baseline[s["baseline"]].append(s)

    for baseline, items in sorted(by_baseline.items()):
        counts = Counter(i["failure_type"] for i in items)
        total = len(items)
        correct = counts.get("correct", 0)
        print(f"\n[{baseline}]  정답률: {correct}/{total} ({correct/total*100:.1f}%)")
        for ftype, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            bar = "█" * cnt
            print(f"  {ftype:<30} {bar} ({cnt})")

    print("\n" + "=" * 60)
    print("Failure Type × Question Type 교차표")
    print("=" * 60)

    q_types = ["local_factual", "global_synthesis", "terminology_sensitive"]
    f_types = ["correct", "low_overlap", "global_synthesis_failure",
               "terminology_mismatch", "grounding_failure", "retrieval_miss"]

    header = f"{'Failure Type':<30}" + "".join(f"{qt[:12]:>14}" for qt in q_types)
    print(header)
    print("-" * len(header))

    from collections import Counter
    cross = Counter((s["failure_type"], s["question_type"]) for s in scored_list)
    for ft in f_types:
        row = f"{ft:<30}"
        for qt in q_types:
            row += f"{cross.get((ft, qt), 0):>14}"
        print(row)
