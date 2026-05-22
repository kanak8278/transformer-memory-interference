"""
Slide 7 — Qwen family results heatmap.

Two side-by-side panels (ARBITRARY-SINGLE | SEMANTIC-MULTI). Each panel shows
4 models × 3 columns (RI / PI / Gap), color-coded:
  RI  : white → green   (high = good)
  PI  : white → red     (low  = bad)
  Gap : blue ↔ white ↔ red  (diverging at 0)

Cell values printed on each cell. Hidden numbers (gridlines, frames) minimised.

Source : v3/results_vllm/summary/cross_model_comparison.json
Outputs:
  - presentations/week2/plots/slide7_qwen_heatmap.png
  - presentations/week2/plots/slide7_qwen_heatmap.json   (data for React)
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
SRC_ARBI = ROOT / "results_vllm" / "summary" / "all_models_arbitrary_single.json"
SRC_SEM  = ROOT / "results_vllm" / "summary" / "all_models_semantic_multi.json"
OUT_PNG  = HERE.parents[1] / "plots" / "slide7_qwen_heatmap.png"
OUT_JSON = HERE.parents[1] / "plots" / "slide7_qwen_heatmap.json"

# ── Config ───────────────────────────────────────────────────────────────────
QWEN_ORDER = [
    ("Qwen2.5-0.5B-Instruct", "Qwen 0.5B Inst"),
    ("Qwen2.5-1.5B-Instruct", "Qwen 1.5B Inst"),
    ("Qwen2.5-3B-Instruct",   "Qwen 3B Inst"),
    ("Qwen2.5-3B",            "Qwen 3B Base"),
]

DATASETS = [
    ("arbitrary_single", "ARBITRARY-SINGLE\n(single-token, no semantics)", SRC_ARBI),
    ("semantic_multi",   "SEMANTIC-MULTI\n(multi-token, real categories)",  SRC_SEM),
]

# ── Colour maps ──────────────────────────────────────────────────────────────
CMAP_RI  = LinearSegmentedColormap.from_list("ri",
            ["#FFFFFF", "#A5D6A7", "#2E7D32"])              # white→green
CMAP_PI  = LinearSegmentedColormap.from_list("pi",
            ["#B71C1C", "#FFCDD2", "#FFFFFF"])              # red→white (low PI = red)
CMAP_GAP = LinearSegmentedColormap.from_list("gap",
            ["#1565C0", "#FFFFFF", "#C62828"])              # blue↔red (diverging)

# ── Load (full-grid means, canonical numbers from results_table.txt) ────────
ds_data = {}
for ds_key, _, src in DATASETS:
    with open(src) as f:
        ds_data[ds_key] = json.load(f)

records = {}
for model_key, label in QWEN_ORDER:
    records[label] = {}
    for ds_key, _, _ in DATASETS:
        d = ds_data[ds_key].get(model_key, {})
        records[label][ds_key] = {
            "ri":  float(d.get("mean_ri",  float("nan"))),
            "pi":  float(d.get("mean_pi",  float("nan"))),
            "gap": float(d.get("mean_gap", float("nan"))),
            "n_cells": int(d.get("n_cells", 0)),
        }

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5),
                         gridspec_kw={"wspace": 0.30})
fig.patch.set_facecolor("white")

COLS = [("RI",  "ri",  CMAP_RI,  0.0, 1.0, None),
        ("PI",  "pi",  CMAP_PI,  0.0, 1.0, None),
        ("Gap", "gap", CMAP_GAP, -0.30, 0.70, 0.0)]

n_rows = len(QWEN_ORDER)
n_cols = len(COLS)

for ax_i, (ds_key, ds_title) in enumerate(DATASETS):
    ax = axes[ax_i]
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)  # row 0 at top
    ax.set_aspect("equal")
    ax.set_facecolor("white")

    # Draw cells
    for j, (cname, fkey, cmap, vmin, vmax, vcenter) in enumerate(COLS):
        if vcenter is not None:
            norm = TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
        else:
            from matplotlib.colors import Normalize
            norm = Normalize(vmin=vmin, vmax=vmax)
        for i, (_, mlabel) in enumerate(QWEN_ORDER):
            v = records[mlabel][ds_key][fkey]
            color = cmap(norm(v))
            rect = Rectangle((j - 0.45, i - 0.45), 0.9, 0.9,
                             facecolor=color, edgecolor="#FFFFFF",
                             linewidth=2.5)
            ax.add_patch(rect)
            # Cell label
            txt = f"{v:+.0%}" if fkey == "gap" else f"{v:.0%}"
            # Pick text color by perceived luminance
            r, g, b, _ = color
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            tcol = "white" if lum < 0.55 else "#222222"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=14, fontweight="bold", color=tcol)

    # Column headers
    for j, (cname, *_rest) in enumerate(COLS):
        ax.text(j, -0.95, cname, ha="center", va="bottom",
                fontsize=12, fontweight="bold", color="#333")

    # Row labels (model names)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([m[1] for m in QWEN_ORDER],
                       fontsize=11, color="#333")
    ax.set_xticks([])

    # Frame off
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False)

    ax.set_title(ds_title, fontsize=12, fontweight="bold", pad=22, color="#222")

# Suptitle
fig.suptitle(
    "Qwen 2.5 Family — Position-Dependent Retrieval Across Two Datasets\n"
    "values averaged across 6 operating points (K∈{2,5,10} × N∈{5,10,20}), 100 trials/cell",
    fontsize=13, fontweight="bold", y=1.02, color="#111"
)

# Footer key
key_txt = ("RI = recall first value · PI = recall last value · "
           "Gap = RI − PI  (positive = primacy bias, negative = recency wins)")
fig.text(0.5, -0.02, key_txt, ha="center", va="top",
         fontsize=10, color="#555", style="italic")

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUT_PNG}")

# ── Dump JSON for React deck ──────────────────────────────────────────────────
json_out = {
    "title": "Qwen 2.5 Family — Position-Dependent Retrieval",
    "subtitle": "averaged across 6 operating points · 100 trials/cell",
    "models": [m[1] for m in QWEN_ORDER],
    "datasets": [{"key": k, "title": t.replace("\n", " ")} for k, t in DATASETS],
    "operating_points": SUMMARY_CELLS,
    "data": records,
    "color_scales": {
        "ri":  {"type": "sequential",   "low": "#FFFFFF", "high": "#2E7D32",
                "vmin": 0.0,  "vmax": 1.0},
        "pi":  {"type": "sequential",   "low": "#B71C1C", "high": "#FFFFFF",
                "vmin": 0.0,  "vmax": 1.0},
        "gap": {"type": "diverging",    "low": "#1565C0", "mid": "#FFFFFF",
                "high": "#C62828", "vcenter": 0.0,
                "vmin": -0.30, "vmax": 0.70},
    },
}
with open(OUT_JSON, "w") as f:
    json.dump(json_out, f, indent=2)
print(f"Saved → {OUT_JSON}")
