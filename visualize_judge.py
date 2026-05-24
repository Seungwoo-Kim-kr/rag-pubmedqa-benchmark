"""
visualize_judge.py
==================
LLM-as-judge 결과를 시각화합니다. 생성 파일:
  1. judge_vs_f1.png      — Token F1 vs Judge Score 비교 bar chart
  2. judge_heatmap.png    — baseline × question type Judge Score 히트맵
  3. judge_label_dist.png — correct/partial/incorrect 분포 stacked bar

실행:
    python visualize_judge.py --input results/judged/judged_STEM.json [--title "QASPER"]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASELINES = ["direct_qa", "standard_rag", "summary_mediated_qa", "lightrag_hybrid"]
Q_TYPES   = ["local_factual", "global_synthesis", "terminology_sensitive"]
BASELINE_LABELS = {
    "direct_qa":            "Direct QA",
    "standard_rag":         "Standard RAG",
    "summary_mediated_qa":  "Summary-\nMediated QA",
    "lightrag_hybrid":      "LightRAG\n(hybrid)",
}
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#8172B2"]


def load_judged(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_present_baselines(judged: list[dict]) -> list[str]:
    present = {r["baseline"] for r in judged}
    return [b for b in BASELINES if b in present] or sorted(present)


# ── 1. Token F1 vs Judge Score 비교 ──────────────────────────
def plot_judge_vs_f1(judged: list[dict], out_dir: Path, title_prefix: str):
    baselines = get_present_baselines(judged)

    agg = defaultdict(lambda: {"f1": [], "judge": []})
    for r in judged:
        if r["baseline"] in baselines:
            if "token_f1" in r:
                agg[r["baseline"]]["f1"].append(r["token_f1"])
            if r.get("judge_score") is not None:
                agg[r["baseline"]]["judge"].append(r["judge_score"])

    x = np.arange(len(baselines))
    width = 0.35
    f1_vals    = [np.mean(agg[b]["f1"])    if agg[b]["f1"]    else 0 for b in baselines]
    judge_vals = [np.mean(agg[b]["judge"]) if agg[b]["judge"] else 0 for b in baselines]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, f1_vals,    width, label="Token F1",    color="#4C72B0", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width/2, judge_vals, width, label="Judge Score", color="#DD8452", alpha=0.85, edgecolor="white")

    for bar, val in zip(bars1, f1_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, judge_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    labels = [BASELINE_LABELS.get(b, b).replace("\n", " ") for b in baselines]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{title_prefix} — Token F1 vs Judge Score by Baseline", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / "judge_vs_f1.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  저장: {path}")


# ── 2. Judge Score Heatmap ───────────────────────────────────
def plot_judge_heatmap(judged: list[dict], out_dir: Path, title_prefix: str):
    baselines = get_present_baselines(judged)
    matrix = np.zeros((len(baselines), len(Q_TYPES)))
    counts = np.zeros((len(baselines), len(Q_TYPES)), dtype=int)

    for r in judged:
        if r["baseline"] not in baselines:
            continue
        if r["question_type"] not in Q_TYPES:
            continue
        if r.get("judge_score") is None:
            continue
        bi = baselines.index(r["baseline"])
        qi = Q_TYPES.index(r["question_type"])
        matrix[bi][qi] += r["judge_score"]
        counts[bi][qi] += 1

    avg = np.where(counts > 0, matrix / counts, 0)

    fig, ax = plt.subplots(figsize=(8, max(3, len(baselines))))
    im = ax.imshow(avg, cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Judge Score")

    ax.set_xticks(range(len(Q_TYPES)))
    ax.set_yticks(range(len(baselines)))
    ax.set_xticklabels([qt.replace("_", "\n") for qt in Q_TYPES], fontsize=10)
    ax.set_yticklabels([BASELINE_LABELS.get(b, b) for b in baselines], fontsize=10)

    for bi in range(len(baselines)):
        for qi in range(len(Q_TYPES)):
            val = avg[bi][qi]
            color = "black" if val < 0.6 else "white"
            ax.text(qi, bi, f"{val:.3f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    ax.set_title(f"{title_prefix} — Judge Score by Baseline × Question Type", fontsize=13, pad=12)
    plt.tight_layout()
    path = out_dir / "judge_heatmap.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  저장: {path}")


# ── 3. Judge Label 분포 ─────────────────────────────────────
def plot_label_dist(judged: list[dict], out_dir: Path, title_prefix: str):
    baselines = get_present_baselines(judged)
    LABELS   = ["correct", "partial", "incorrect"]
    L_COLORS = ["#2ca02c", "#ffbb78", "#d62728"]

    counts = {b: defaultdict(int) for b in baselines}
    for r in judged:
        if r["baseline"] in baselines:
            label = r.get("judge_label", "incorrect")
            if label in LABELS:
                counts[r["baseline"]][label] += 1

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(baselines))
    bottoms = np.zeros(len(baselines))

    for lbl, lc in zip(LABELS, L_COLORS):
        vals = [counts[b][lbl] for b in baselines]
        bars = ax.bar(x, vals, bottom=bottoms, color=lc, label=lbl, edgecolor="white", alpha=0.9)
        for xi, (v, bot) in enumerate(zip(vals, bottoms)):
            if v > 0:
                ax.text(xi, bot + v/2, str(v), ha="center", va="center",
                        fontsize=10, fontweight="bold", color="white")
        bottoms += np.array(vals, dtype=float)

    xlabels = [BASELINE_LABELS.get(b, b).replace("\n", " ") for b in baselines]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_ylabel("Number of Questions", fontsize=11)
    ax.set_title(f"{title_prefix} — Judge Label Distribution by Baseline", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / "judge_label_dist.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  저장: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="judged results JSON 경로")
    parser.add_argument("--title", type=str, default="RAG Experiment",
                        help="차트 제목 접두어 (예: 'QASPER')")
    parser.add_argument("--outdir", type=str, default=None,
                        help="출력 디렉토리 (기본: results/analysis/<stem>)")
    args = parser.parse_args()

    stem = Path(args.input).stem
    out_dir = Path(args.outdir) if args.outdir else Path("results/analysis") / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    judged = load_judged(args.input)
    print(f"\n총 {len(judged)}개 judge 결과 시각화 시작...\n")

    plot_judge_vs_f1(judged, out_dir, args.title)
    plot_judge_heatmap(judged, out_dir, args.title)
    plot_label_dist(judged, out_dir, args.title)

    print(f"\n시각화 파일 저장 위치: {out_dir}/")


if __name__ == "__main__":
    main()
