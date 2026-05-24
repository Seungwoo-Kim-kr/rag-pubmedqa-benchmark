"""
analyze_results.py
==================
실험 결과를 로드해서 baseline별 / 질문타입별 성능을 출력합니다.

실행:
    python analyze_results.py --input results/raw_outputs/results_XXXXXX.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

from evaluation.metrics import score_result


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def load_results(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_questions(config: dict, qa_file: str = None) -> dict[str, dict]:
    qa_filename = qa_file if qa_file else "questions.json"
    qa_path = Path(config["paths"]["qa_sets"]) / qa_filename
    with open(qa_path, encoding="utf-8") as f:
        questions = json.load(f)
    return {q["question_id"]: q for q in questions}


def compute_scores(results: list[dict], question_map: dict) -> list[dict]:
    scored = []
    for r in results:
        q = question_map.get(r["question_id"])
        if q:
            scored.append(score_result(r, q))
    return scored


def print_table(scored: list[dict]):
    # baseline × question_type 집계
    agg = defaultdict(lambda: defaultdict(list))
    for s in scored:
        agg[s["baseline"]][s["question_type"]].append(s)
        agg[s["baseline"]]["ALL"].append(s)

    baselines = sorted(agg.keys())
    qtypes = ["local_factual", "global_synthesis", "terminology_sensitive", "ALL"]

    header = f"{'Baseline':<25}" + "".join(f"{'  ' + qt:<28}" for qt in qtypes)
    print("\n" + "=" * len(header))
    print("Token F1 by Baseline × Question Type")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for baseline in baselines:
        row = f"{baseline:<25}"
        for qt in qtypes:
            items = agg[baseline].get(qt, [])
            if items:
                avg_f1 = sum(i["token_f1"] for i in items) / len(items)
                avg_hit = sum(1 for i in items if i["retrieval_hit"]) / len(items)
                row += f"  F1={avg_f1:.3f} Hit={avg_hit:.2f} (n={len(items):<3})"
            else:
                row += f"  {'N/A':<25}"
        print(row)

    print("=" * len(header))

    # runtime 요약
    print("\nAverage Runtime (sec):")
    for baseline in baselines:
        items = agg[baseline]["ALL"]
        avg = sum(i["runtime_sec"] for i in items) / len(items)
        print(f"  {baseline:<25} {avg:.2f}s")


def save_scored(scored: list[dict], input_path: str, config: dict):
    out_dir = Path(config["paths"]["results_scored"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem
    out_path = out_dir / f"scored_{stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scored, f, ensure_ascii=False, indent=2)
    print(f"\n채점 결과 저장: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="raw results JSON 경로")
    parser.add_argument("--questions", type=str, default=None,
                        help="질문셋 파일명 (예: questions_balanced.json)")
    args = parser.parse_args()

    config = load_config()
    results = load_results(args.input)
    question_map = load_questions(config, qa_file=args.questions)
    scored = compute_scores(results, question_map)
    print_table(scored)
    save_scored(scored, args.input, config)


if __name__ == "__main__":
    main()
