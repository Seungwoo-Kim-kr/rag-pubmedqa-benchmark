"""
visualize_results.py
====================
실험 결과를 시각화합니다. 생성 파일:
  1. f1_heatmap.png        — baseline × question type Token F1 히트맵
  2. f1_bar.png            — baseline별 질문 타입 grouped bar chart
  3. failure_bar.png       — baseline별 failure type 분포
  4. runtime_bar.png       — baseline별 평균 응답 시간

실행:
    python visualize_results.py --input results/scored_outputs/scored_XXXX.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless 환경용
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from evaluation.failure_analysis import add_failure_labels, print_failure_summary

BASELINES = ["direct_qa", "standard_rag", "summary_mediated_qa", "lightrag_hybrid"]
Q_TYPES   = ["local_factual", "global_synthesis", "terminology_sensitive"]
BASELINE_LABELS = {
    "direct_qa": "Direct QA",
    "standard_rag": "Standard RAG",
    "summary_mediated_qa": "Summary-\nMediated QA",
    "lightrag_hybrid": "LightRAG\n(hybrid)",
}
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#8172B2"]


def load_scored(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── 1. F1 Heatmap ────────────────────────────────────────────
def plot_f1_heatmap(scored: list[dict], out_dir: Path):
    matrix = np.zeros((len(BASELINES), len(Q_TYPES)))
    counts = np.zeros((len(BASELINES), len(Q_TYPES)), dtype=int)

    for s in scored:
        if s["baseline"] not in BASELINES:
            continue
        bi = BASELINES.index(s["baseline"])
        if s["question_type"] not in Q_TYPES:
            continue
        qi = Q_TYPES.index(s["question_type"])
        matrix[bi][qi] += s["token_f1"]
        counts[bi][qi] += 1

    avg = np.where(counts > 0, matrix / counts, 0)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(avg, cmap="YlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Token F1")

    ax.set_xticks(range(len(Q_TYPES)))
    ax.set_yticks(range(len(BASELINES)))
    ax.set_xticklabels([qt.replace("_", "\n") for qt in Q_TYPES], fontsize=10)
    ax.set_yticklabels([BASELINE_LABELS[b] for b in BASELINES], fontsize=10)

    for bi in range(len(BASELINES)):
        for qi in range(len(Q_TYPES)):
            val = avg[bi][qi]
            color = "black" if val < 0.6 else "white"
            ax.text(qi, bi, f"{val:.3f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    ax.set_title("Token F1 by Baseline × Question Type", fontsize=13, pad=12)
    plt.tight_layout()
    path = out_dir / "f1_heatmap.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  저장: {path}")


# ── 2. F1 Grouped Bar ─────────────────────────────────────────
def plot_f1_bar(scored: list[dict], out_dir: Path):
    agg = defaultdict(lambda: defaultdict(list))
    for s in scored:
        if s["baseline"] in BASELINES and s["question_type"] in Q_TYPES:
            agg[s["baseline"]][s["question_type"]].append(s["token_f1"])

    x = np.arange(len(Q_TYPES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (baseline, color) in enumerate(zip(BASELINES, COLORS)):
        vals = [np.mean(agg[baseline][qt]) if agg[baseline][qt] else 0 for qt in Q_TYPES]
        bars = ax.bar(x + i * width, vals, width, label=BASELINE_LABELS[baseline],
                      color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([qt.replace("_", "\n") for qt in Q_TYPES], fontsize=10)
    ax.set_ylabel("Token F1", fontsize=11)
    ax.set_ylim(0, 0.85)
    ax.set_title("Token F1 by Baseline and Question Type", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / "f1_bar.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  저장: {path}")


# ── 3. Failure Type Stacked Bar ───────────────────────────────
def plot_failure_bar(scored: list[dict], out_dir: Path):
    scored = add_failure_labels(scored)

    F_TYPES = ["correct", "low_overlap", "global_synthesis_failure",
               "terminology_mismatch", "grounding_failure", "retrieval_miss"]
    F_COLORS = ["#2ca02c", "#98df8a", "#ff7f0e", "#ffbb78", "#d62728", "#ff9896"]

    counts = {b: defaultdict(int) for b in BASELINES}
    for s in scored:
        if s["baseline"] in BASELINES:
            counts[s["baseline"]][s["failure_type"]] += 1

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(BASELINES))
    bottoms = np.zeros(len(BASELINES))

    for ft, fc in zip(F_TYPES, F_COLORS):
        vals = [counts[b][ft] for b in BASELINES]
        ax.bar(x, vals, bottom=bottoms, color=fc, label=ft, edgecolor="white", alpha=0.9)
        for xi, (v, bot) in enumerate(zip(vals, bottoms)):
            if v > 0:
                ax.text(xi, bot + v / 2, str(v), ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")
        bottoms += np.array(vals, dtype=float)

    ax.set_xticks(x)
    ax.set_xticklabels([BASELINE_LABELS[b] for b in BASELINES], fontsize=10)
    ax.set_ylabel("Number of Questions", fontsize=11)
    ax.set_title("Failure Type Distribution by Baseline", fontsize=13)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / "failure_bar.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  저장: {path}")
    return scored


# ── 4. Runtime Bar ────────────────────────────────────────────
def plot_runtime_bar(scored: list[dict], out_dir: Path):
    agg = defaultdict(list)
    for s in scored:
        if s["baseline"] in BASELINES:
            agg[s["baseline"]].append(s["runtime_sec"])

    avgs = [np.mean(agg[b]) for b in BASELINES]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([BASELINE_LABELS[b] for b in BASELINES], avgs,
                  color=COLORS, edgecolor="white", alpha=0.85)
    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Average Runtime (sec)", fontsize=11)
    ax.set_title("Average Response Time by Baseline", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / "runtime_bar.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  저장: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="scored results JSON 경로")
    args = parser.parse_args()

    out_dir = Path("results/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    scored = load_scored(args.input)
    print(f"\n총 {len(scored)}개 결과 시각화 시작...\n")

    plot_f1_heatmap(scored, out_dir)
    plot_f1_bar(scored, out_dir)
    scored = plot_failure_bar(scored, out_dir)
    plot_runtime_bar(scored, out_dir)

    print_failure_summary(scored)

    # failure label 포함 결과 저장
    labeled_path = Path(args.input).parent / (Path(args.input).stem + "_labeled.json")
    with open(labeled_path, "w", encoding="utf-8") as f:
        json.dump(scored, f, ensure_ascii=False, indent=2)
    print(f"\nFailure 라벨 포함 결과: {labeled_path}")
    print(f"시각화 파일: {out_dir}/")


if __name__ == "__main__":
    main()
