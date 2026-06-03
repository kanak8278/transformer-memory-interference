"""
RI−PI gap vs N for 8 models at keys=2.
Single panel: gap > 0 means RI better than PI (primacy bias).
Data: v3/results_vllm/analysis/scaling_law_fits_arbitrary_single.json
Output: v3/results_vllm/plots/pi_vs_n_all_models.png
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path

BASE = Path(__file__).parent
FITS = BASE / "results_vllm" / "analysis" / "scaling_law_fits_arbitrary_single.json"
OUT  = BASE / "results_vllm" / "plots" / "pi_vs_n_all_models.png"

# ── Model config (key, label, color, marker, linestyle, group) ────────────────
# Dropped: Gemma 270M, Pythia 410M, Mamba 1.4B
MODEL_CFG = [
    # Instruct / Chat
    ("Qwen2.5-0.5B-Instruct",    "Qwen 0.5B",      "#E53935", "o", "-",  "inst"),
    ("Qwen2.5-1.5B-Instruct",    "Qwen 1.5B",      "#1E88E5", "o", "-",  "inst"),
    ("Qwen2.5-3B-Instruct",      "Qwen 3B-Inst",   "#43A047", "o", "-",  "inst"),
    ("gemma-3-1b-it",            "Gemma 1B",       "#8E24AA", "D", "-",  "inst"),
    ("gemma-3-4b-it",            "Gemma 4B",       "#4A148C", "D", "-",  "inst"),
    ("TinyLlama-1.1B-Chat-v1.0", "TinyLlama 1.1B", "#FF7043", "^", "-",  "inst"),
    ("stablelm-2-1_6b-chat",     "StableLM 1.6B",  "#795548", "v", "-",  "inst"),
    # Base (no instruction tuning) — dashed
    ("Qwen2.5-3B",               "Qwen 3B-Base",   "#43A047", "s", "--", "base"),
]

with open(FITS) as f:
    fits = json.load(f)

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor("white")

for model_key, label, color, marker, lstyle, group in MODEL_CFG:
    if model_key not in fits:
        continue
    k2 = fits[model_key].get("keys_2")
    if not k2 or not k2.get("data"):
        continue

    data   = k2["data"]
    N_pts  = [d["N"] for d in data]
    gap    = [d["ri"] - d["pi"] for d in data]

    lw    = 2.0 if group == "inst" else 1.7
    ms    = 7
    alpha = 0.92 if group == "inst" else 0.75
    zord  = 3 if group == "inst" else 2

    ax.plot(N_pts, gap,
            color=color, marker=marker, linestyle=lstyle,
            linewidth=lw, markersize=ms, alpha=alpha,
            label=label, zorder=zord)

# ── Zero reference & styling ──────────────────────────────────────────────────

ax.axhline(0, color="#333333", linewidth=1.2, linestyle="-", alpha=0.6, zorder=1)
ax.fill_between([3, 105], 0, 1.05, color="#EBF5EB", alpha=0.35, zorder=0)  # green zone
ax.fill_between([3, 105], -0.2, 0, color="#FDECEA", alpha=0.35, zorder=0)  # red zone

ax.text(40, 0.04, "RI > PI  (primacy bias)", ha="left", va="bottom",
        fontsize=9, color="#2E7D32", style="italic")
ax.text(40, -0.08, "PI > RI  (recency wins)", ha="left", va="top",
        fontsize=9, color="#C62828", style="italic")

ax.set_xlabel("Number of updates (N)", fontsize=12)
ax.set_ylabel("RI − PI Gap", fontsize=12)
ax.set_title(
    "Primacy Bias Grows with N: RI − PI Gap Across 8 Models\n"
    "Fixed at 2 keys, 100 trials/cell  ·  ARBITRARY_SINGLE dataset",
    fontsize=13, fontweight="bold", pad=12
)
ax.set_xlim(3, 105)
ax.set_ylim(-0.25, 1.05)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:+.0%}"))
ax.grid(True, color="#E5E5E5", linewidth=0.7, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── Legend ────────────────────────────────────────────────────────────────────

inst_handle = mlines.Line2D([], [], color="grey", linestyle="-",
                             marker="o", markersize=6, label="── Instruct / Chat")
base_handle = mlines.Line2D([], [], color="grey", linestyle="--",
                             marker="s", markersize=6, label="╌╌ Base (no SFT)")

model_handles, model_labels = ax.get_legend_handles_labels()

# Single unified legend: placed outside the plot to the right
ax.legend(
    handles=model_handles + [inst_handle, base_handle],
    labels=model_labels + ["── Instruct / Chat", "╌╌ Base (no SFT)"],
    loc="upper left",
    bbox_to_anchor=(1.01, 1.0),
    fontsize=9,
    framealpha=0.93, edgecolor="#CCCCCC",
    ncol=1, borderpad=0.8, handlelength=2.2,
)

plt.tight_layout(pad=1.5)
plt.subplots_adjust(right=0.75)  # make room for the outside legend
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUT}")
