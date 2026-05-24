"""
run_judge.py
============
LLM-as-judge 평가 실행 스크립트.
기존 raw results JSON을 입력으로 받아 GPT judge 점수를 추가합니다.

실행:
    python run_judge.py --input results/raw_outputs/results_XXXXXX.json --questions questions_qasper.json
    python run_judge.py --input results/all4/results_all4_balanced.json --questions questions_balanced.json

출력:
    results/judged/judged_STEM.json          — judge 점수 포함 전체 결과
    results/judged/summary_STEM.txt          — 텍스트 요약
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from evaluation.metrics import token_f1, retrieval_hit
from evaluation.llm_judge import run_judge


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def load_results(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_question_map(config: dict, qa_file: str) -> dict[str, dict]:
    qa_filename = qa_file if qa_file else "questions.json"
    qa_path = Path(config["paths"]["qa_sets"]) / qa_filename
    with open(qa_path, encoding="utf-8") as f:
        questions = json.load(f)
    return {q["question_id"]: q for q in questions}


def add_token_f1(results: list[dict], question_map: dict) -> list[dict]:
    """raw results에 token_f1 / retrieval_hit 추가 (아직 없는 경우)."""
    enriched = []
    for r in results:
        q = question_map.get(r["question_id"])
        if q and "token_f1" not in r:
            gold = q.get("gold_answer", "")
            pred = r.get("answer", "")
            gold_ev = q.get("gold_evidence", [])
            retrieved = r.get("retrieved_context", [])
            r = dict(r,
                     token_f1=round(token_f1(gold, pred), 4),
                     retrieval_hit=retrieval_hit(gold_ev, retrieved))
        enriched.append(r)
    return enriched


def print_summary(judged: list[dict]):
    agg = defaultdict(lambda: defaultdict(list))
    for r in judged:
        agg[r["baseline"]][r["question_type"]].append(r)
        agg[r["baseline"]]["ALL"].append(r)

    baselines = sorted(agg.keys())
    qtypes = ["local_factual", "global_synthesis", "terminology_sensitive", "ALL"]

    # ── Judge Score 테이블
    header = f"\n{'Baseline':<28}" + "".join(f"{'  ' + qt:<26}" for qt in qtypes)
    print("\n" + "=" * (len(header) - 1))
    print("Judge Score (GPT-as-judge) by Baseline × Question Type")
    print("=" * (len(header) - 1))
    print(header)
    print("-" * (len(header) - 1))

    for baseline in baselines:
        row = f"{baseline:<28}"
        for qt in qtypes:
            items = [x for x in agg[baseline].get(qt, []) if x.get("judge_score") is not None]
            if items:
                avg_j = sum(x["judge_score"] for x in items) / len(items)
                correct_n = sum(1 for x in items if x.get("judge_label") == "correct")
                row += f"  J={avg_j:.3f} C={correct_n}/{len(items):<6}"
            else:
                row += f"  {'N/A':<24}"
        print(row)

    print("=" * (len(header) - 1))

    # ── Token F1 vs Judge 비교 (ALL만)
    print("\nToken F1 vs Judge Score (ALL questions):")
    print(f"  {'Baseline':<28} {'Token F1':>10} {'Judge Score':>12}")
    print(f"  {'-'*28} {'-'*10} {'-'*12}")
    for baseline in baselines:
        items = agg[baseline].get("ALL", [])
        f1s = [x["token_f1"] for x in items if "token_f1" in x]
        js = [x["judge_score"] for x in items if x.get("judge_score") is not None]
        avg_f1 = sum(f1s) / len(f1s) if f1s else 0
        avg_j = sum(js) / len(js) if js else 0
        print(f"  {baseline:<28} {avg_f1:>10.3f} {avg_j:>12.3f}")

    # ── Label 분포
    print("\nJudge Label Distribution:")
    print(f"  {'Baseline':<28} {'correct':>8} {'partial':>8} {'incorrect':>10} {'error':>7}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*10} {'-'*7}")
    for baseline in baselines:
        items = agg[baseline].get("ALL", [])
        c = sum(1 for x in items if x.get("judge_label") == "correct")
        p = sum(1 for x in items if x.get("judge_label") == "partial")
        i = sum(1 for x in items if x.get("judge_label") == "incorrect")
        e = sum(1 for x in items if x.get("judge_label") == "error")
        print(f"  {baseline:<28} {c:>8} {p:>8} {i:>10} {e:>7}")


def save_judged(judged: list[dict], input_path: str):
    out_dir = Path("results/judged")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem
    out_path = out_dir / f"judged_{stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(judged, f, ensure_ascii=False, indent=2)
    print(f"\n저장 완료: {out_path}  ({len(judged)}개)")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="raw results JSON 경로")
    parser.add_argument("--questions", type=str, default=None,
                        help="질문셋 파일명 (예: questions_qasper.json)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="judge에 사용할 모델 (default: gpt-4o-mini)")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="API 호출 간격 초 (default: 0.3)")
    args = parser.parse_args()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("오류: OPENAI_API_KEY가 없습니다.")
        sys.exit(1)

    config = load_config()
    results = load_results(args.input)
    question_map = load_question_map(config, args.questions)

    print(f"입력: {args.input}  ({len(results)}개)")
    print(f"질문셋: {args.questions or 'questions.json'}  ({len(question_map)}개)")
    print(f"Judge 모델: {args.model}\n")

    # token_f1 / retrieval_hit 추가
    results = add_token_f1(results, question_map)

    # LLM judge 실행
    print("Judge 평가 시작...")
    judged = run_judge(results, question_map, model=args.model, delay=args.delay)

    print_summary(judged)
    out_path = save_judged(judged, args.input)
    print(f"\n시각화: python visualize_judge.py --input {out_path}")


if __name__ == "__main__":
    main()
