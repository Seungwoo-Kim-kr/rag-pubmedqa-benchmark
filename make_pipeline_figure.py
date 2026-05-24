"""Generate a pipeline comparison diagram for the README."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

COLORS = {
    "input":  "#1f6feb",
    "index":  "#388bfd",
    "llm":    "#3fb950",
    "output": "#58a6ff",
    "arrow":  "#8b949e",
    "label":  "#e6edf3",
    "dim":    "#8b949e",
    "bg":     "#161b22",
    "figbg":  "#0d1117",
}

PIPELINES = [
    {
        "name": "1. Direct QA",
        "nodes": [
            ("Question",    COLORS["input"]),
            ("GPT-4o-mini", COLORS["llm"]),
            ("Answer",      COLORS["output"]),
        ],
    },
    {
        "name": "2. Standard RAG",
        "nodes": [
            ("Question",      COLORS["input"]),
            ("FAISS Retrieval", COLORS["index"]),
            ("Top-k Chunks",  COLORS["index"]),
            ("GPT-4o-mini",   COLORS["llm"]),
            ("Answer",        COLORS["output"]),
        ],
    },
    {
        "name": "3. Summary-Mediated QA",
        "nodes": [
            ("Question",          COLORS["input"]),
            ("FAISS Retrieval",   COLORS["index"]),
            ("Top-k Chunks",      COLORS["index"]),
            ("Summariser",        COLORS["llm"]),
            ("Summary",           COLORS["index"]),
            ("GPT-4o-mini",       COLORS["llm"]),
            ("Answer",            COLORS["output"]),
        ],
    },
    {
        "name": "4. LightRAG",
        "nodes": [
            ("Question",    COLORS["input"]),
            ("Graph Index", COLORS["index"]),
            ("GPT-4o-mini", COLORS["llm"]),
            ("Answer",      COLORS["output"]),
        ],
    },
]

ROW_H = 1.8
LABEL_X = 0.18   # right edge of the pipeline label column
START_X = 0.22   # left edge of first node centre
END_X   = 0.91   # right edge of last node centre
BOX_H   = 0.9
Y_C     = ROW_H / 2

fig_h = ROW_H * len(PIPELINES) + 0.6
fig, ax = plt.subplots(figsize=(13, fig_h))
fig.patch.set_facecolor(COLORS["figbg"])
ax.set_facecolor(COLORS["figbg"])
ax.set_xlim(0, 1)
ax.set_ylim(0, fig_h)
ax.axis("off")

ax.text(0.5, fig_h - 0.15, "Pipeline Architectures",
        ha="center", va="top", fontsize=14, fontweight="bold",
        color=COLORS["label"])

for row_i, pipe in enumerate(PIPELINES):
    y_base = (len(PIPELINES) - 1 - row_i) * ROW_H
    y_c = y_base + Y_C

    # Pipeline label
    ax.text(LABEL_X - 0.01, y_c, pipe["name"],
            va="center", ha="right", fontsize=10.5, fontweight="bold",
            color=COLORS["label"])

    nodes = pipe["nodes"]
    n = len(nodes)
    box_w = min(0.10, (END_X - START_X) / n * 0.72)
    gap = (END_X - START_X) / (n - 1) if n > 1 else 0
    xs = [START_X + i * gap for i in range(n)]

    for i, ((label, color), x) in enumerate(zip(nodes, xs)):
        # Box
        box = FancyBboxPatch(
            (x - box_w / 2, y_c - BOX_H / 2),
            box_w, BOX_H,
            boxstyle="round,pad=0.015",
            linewidth=1.4,
            edgecolor=color,
            facecolor=color + "2a",
            zorder=3,
        )
        ax.add_patch(box)

        font_size = 8.0 if n <= 5 else 7.2
        ax.text(x, y_c, label, va="center", ha="center",
                fontsize=font_size, color=COLORS["label"],
                zorder=4, linespacing=1.35,
                wrap=True)

        # Arrow to next node
        if i < n - 1:
            x_next = xs[i + 1]
            ax.annotate(
                "",
                xy=(x_next - box_w / 2 - 0.004, y_c),
                xytext=(x + box_w / 2 + 0.004, y_c),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=COLORS["arrow"],
                    lw=1.4,
                ),
                zorder=2,
            )

    # Horizontal separator above row (except last)
    if row_i < len(PIPELINES) - 1:
        sep_y = y_base + ROW_H
        ax.axhline(sep_y, color="#30363d", lw=0.8, zorder=1)

plt.tight_layout(pad=0.3)
plt.savefig("results/analysis/pipeline.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved results/analysis/pipeline.png")
