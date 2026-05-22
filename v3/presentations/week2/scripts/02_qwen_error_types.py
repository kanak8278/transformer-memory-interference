"""
Slide 9a — Error-type breakdown across the Qwen family.

Two-panel horizontal stacked bar:
  LEFT  : aggregate across the full feasible (K, N) grid
  RIGHT : single operating point (2 keys × 10 updates) for direct comparability

Categories (stacked left-to-right, summing to 100%):
  correct · intermediate intrusion · primacy intrusion · garbage
(recency_intrusion was always ~0% across the Qwen family, so it's omitted)

Source : v3/results_vllm/analysis/error_positions_arbitrary_single.json
Outputs:
  - presentations/week2/plots/slide9a_error_types.png
  - presentations/week2/plots/slide9a_error_types.json
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
SRC  = ROOT / "results_vllm" / "analysis" / "error_positions_arbitrary_single.json"
OUT_PNG  = HERE.parents[1] / "plots" / "slide9a_error_types.png"
OUT_JSON = HERE.parents[1] / "plots" / "slide9a_error_types.json"

# Top-to-bottom in the chart, but Matplotlib draws bottom-up — we'll reverse later.
QWEN_ORDER = [
    ("Qwen2.5-0.5B-Instruct", "Qwen 0.5B Inst"),
    ("Qwen2.5-1.5B-Instruct", "Qwen 1.5B Inst"),
    ("Qwen2.5-3B-Instruct",   "Qwen 3B Inst"),
    ("Qwen2.5-3B",            "Qwen 3B Base"),
]

CATEGORIES = [
    ("correct",                "Correct",                 "#2E7D32",
        "exact match to the expected last value"),
    ("intermediate_intrusion", "Intermediate intrusion",  "#FB8C00",
        "returned a value from the middle of the stream"),
    ("primacy_intrusion",      "Primacy intrusion",       "#7B1FA2",
        "returned the first value when asked for the last"),
    ("garbage",                "Garbage",                 "#9E9E9E",
        "output not present anywhere in the stream"),
]

FOCUS_CELL = "2_10"

# ── Load ─────────────────────────────────────────────────────────────────────
with open(SRC) as f:
    data = json.load(f)


def aggregate(model_key):
    cells = data[model_key]["per_cell"]
    totals = {cat: 0 for cat, _, _, _ in CATEGORIES}
    n_total = 0
    for _, info in cells.items():
        n_total += info["n_total"]
        for cat, _, _, _ in CATEGORIES:
            totals[cat] += info["error_types"].get(cat, 0)
    return totals, n_total, len(cells)


def cell(model_key, cell_key):
    info = data[model_key]["per_cell"].get(cell_key)
    if info is None:
        return None, 0
    return info["error_types"], info["n_total"]


def to_pct(counts, total):
    return {k: (counts[k] / total * 100 if total else 0.0) for k in counts}


# Build per-panel records
panels = {"aggregate": {}, "cell": {}}
for mkey, mlabel in QWEN_ORDER:
    a_counts, a_total, a_ncells = aggregate(mkey)
    panels["aggregate"][mlabel] = {
        "pct":     to_pct(a_counts, a_total),
        "counts":  a_counts,
        "n_total": a_total,
        "n_cells": a_ncells,
    }
    c_counts, c_total = cell(mkey, FOCUS_CELL)
    if c_counts is not None:
        panels["cell"][mlabel] = {
            "pct":     to_pct(c_counts, c_total),
            "counts":  c_counts,
            "n_total": c_total,
        }
    else:
        panels["cell"][mlabel] = None

# ── Plot ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 6.8))
fig.patch.set_facecolor("white")

# Make room: top band for title; main area split into two panels.
# Bottom is generous because the legend now spans 2 rows (label + description per item).
gs = fig.add_gridspec(1, 2, left=0.10, right=0.97, top=0.80, bottom=0.26, wspace=0.45)
axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

panel_specs = [
    (axes[0], "aggregate",
     "Aggregate across full feasible grid",
     "K ∈ [2..30] × N ∈ [5..100], all cells averaged"),
    (axes[1], "cell",
     "Single operating point: 2 keys × 10 updates",
     "direct cross-model comparison at a fixed cell"),
]


def draw_panel(ax, panel_key, title, subtitle):
    labels = [m[1] for m in QWEN_ORDER][::-1]
    y = np.arange(len(labels))

    M = np.zeros((len(labels), len(CATEGORIES)))
    for i, lbl in enumerate(labels):
        rec = panels[panel_key][lbl]
        if rec is None:
            M[i, :] = np.nan
        else:
            for j, (cat, _, _, _) in enumerate(CATEGORIES):
                M[i, j] = rec["pct"].get(cat, 0)

    left = np.zeros(len(labels))
    for j, (cat, _, color, _) in enumerate(CATEGORIES):
        widths = M[:, j]
        ax.barh(y, widths, left=left, color=color,
                edgecolor="white", linewidth=1.0, height=0.62)
        for i in range(len(labels)):
            w = widths[i]
            if np.isnan(w) or w < 6:
                continue
            ax.text(left[i] + w / 2, y[i], f"{w:.0f}%",
                    ha="center", va="center",
                    fontsize=10, color="white", fontweight="bold")
        left = left + np.nan_to_num(widths)

    # Y axis: model labels only (no n inline — n goes as separate footer line)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)

    # n annotation as faint text directly above x-axis
    n_lines = []
    for lbl in labels:
        rec = panels[panel_key][lbl]
        if rec is None:
            n_lines.append("n/a")
        elif panel_key == "aggregate":
            n_lines.append(f"n={rec['n_total']:,} ({rec['n_cells']} cells)")
        else:
            n_lines.append(f"n={rec['n_total']}")
    # Place small italic n-labels just below each bar, inside the chart
    for i, n_txt in enumerate(n_lines):
        ax.text(0.5, y[i] - 0.36, n_txt,
                ha="left", va="top", fontsize=8.5, color="#888", style="italic")

    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10)
    ax.set_xlabel("Share of all PI trials", fontsize=10.5, color="#333", labelpad=4)

    # Panel title (single line) above chart
    ax.set_title(title, fontsize=12.5, fontweight="bold", color="#222", pad=22)
    # Subtitle below the panel title, anchored above the axes
    ax.text(0.0, 1.04, subtitle, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=10, color="#888", style="italic")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#CCC")
    ax.spines["bottom"].set_color("#CCC")
    ax.tick_params(left=False, bottom=False, color="#CCC")
    ax.grid(axis="x", color="#EEE", linewidth=0.7)
    ax.set_axisbelow(True)


for ax, key, t, st in panel_specs:
    draw_panel(ax, key, t, st)

# Top band: suptitle + takeaway
fig.text(0.5, 0.94,
         "How does PI fail? — Error-type breakdown across the Qwen family",
         ha="center", va="bottom", fontsize=15, fontweight="bold", color="#111")
fig.text(0.5, 0.895,
         "Green = correct (= PI accuracy itself).  Orange = the dominant error mode.",
         ha="center", va="top", fontsize=10.5, color="#555", style="italic")

# Legend at bottom — each item: bold label + one-line description on the same row
import matplotlib.patches as mpatches
legend_handles = [
    mpatches.Patch(color=c, label=f"{l}  —  {desc}")
    for _, l, c, desc in CATEGORIES
]
fig.legend(handles=legend_handles, loc="lower center",
           bbox_to_anchor=(0.5, 0.045), ncol=2, frameon=False,
           fontsize=10.5, columnspacing=2.4, handletextpad=0.8,
           handlelength=1.4, handleheight=1.0)

# Footer
fig.text(0.5, -0.005,
         "Dataset: ARBITRARY-SINGLE · 100 trials per cell · greedy decoding · exact-match scoring",
         ha="center", va="top", fontsize=9.5, color="#888", style="italic")

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUT_PNG}")

# ── Dump JSON ────────────────────────────────────────────────────────────────
json_out = {
    "title": "How does PI fail? — Error-type breakdown for the Qwen family",
    "categories": [{"key": k, "label": l, "color": c, "desc": d} for k, l, c, d in CATEGORIES],
    "models": [m[1] for m in QWEN_ORDER],
    "panels": {
        "aggregate": {
            "title": panel_specs[0][2],
            "subtitle": panel_specs[0][3],
            "data": {m: panels["aggregate"][m] for m in [x[1] for x in QWEN_ORDER]},
        },
        "cell": {
            "title": panel_specs[1][2],
            "subtitle": panel_specs[1][3],
            "cell_key": FOCUS_CELL,
            "data": {m: panels["cell"][m] for m in [x[1] for x in QWEN_ORDER]},
        },
    },
}
with open(OUT_JSON, "w") as f:
    json.dump(json_out, f, indent=2)
print(f"Saved → {OUT_JSON}")
