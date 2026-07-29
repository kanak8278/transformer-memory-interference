#!/usr/bin/env python3
"""
RI vs PI accuracy for a single model on narrative (Dota 2) dataset.
Usage: edit MODEL_NAME / DATA_FILE at the top, then run.
"""

import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen3.5-9B"
DATA_FILE  = "/Users/kanak.raj/workspace/hobby/research_work_ri/v3/results_vllm/narrative_dota2/Qwen3.5-9B/"

RI_COLOR = "#2166ac"
PI_COLOR = "#d6604d"

# ── Load ─────────────────────────────────────────────────────────────────────
files = [f for f in glob.glob(DATA_FILE + "*.json") if "_trials" not in f]
with open(files[0]) as f:
    d = json.load(f)

cfg           = d["config"]
results       = d["results"]
KEY_LEVELS    = sorted(cfg["key_levels"])
UPDATE_LEVELS = sorted(cfg["update_levels"])
N_TRIALS      = cfg["trials_per_cell"]

# Build arrays: ri[nk] and pi[nk] are lists over UPDATE_LEVELS
ri = {nk: [] for nk in KEY_LEVELS}
pi = {nk: [] for nk in KEY_LEVELS}
for nk in KEY_LEVELS:
    for nu in UPDATE_LEVELS:
        ck = f"{nk}k_{nu}u"
        ri[nk].append(results[ck]["stats"]["RI"]["accuracy"])
        pi[nk].append(results[ck]["stats"]["PI"]["accuracy"])

# ── Plot: 3×3 grid, one subplot per key level ─────────────────────────────
N_COLS, N_ROWS = 3, 3
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(13, 11),
                          sharex=True, sharey=True)
axes = axes.flatten()
x = np.array(UPDATE_LEVELS)

for idx, nk in enumerate(KEY_LEVELS):
    ax = axes[idx]
    ri_y = np.array(ri[nk])
    pi_y = np.array(pi[nk])

    ax.plot(x, ri_y, color=RI_COLOR, lw=2, marker="o", markersize=6,
            markerfacecolor="white", markeredgewidth=2, zorder=5)
    ax.plot(x, pi_y, color=PI_COLOR, lw=2, marker="s", markersize=6,
            markerfacecolor="white", markeredgewidth=2, zorder=5)

    # Shade gap between the two lines
    ax.fill_between(x, ri_y, pi_y, alpha=0.10,
                    color=RI_COLOR if np.mean(ri_y) >= np.mean(pi_y) else PI_COLOR)

    ax.set_title(f"keys = {nk}", fontsize=11, fontweight="bold", pad=4)
    ax.set_ylim(0.3, 1.02)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_xticks(UPDATE_LEVELS)
    ax.set_xticklabels(UPDATE_LEVELS, fontsize=8, rotation=45)
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    if idx % N_COLS == 0:
        ax.set_ylabel("Accuracy", fontsize=9)
    if idx >= N_COLS * (N_ROWS - 1):
        ax.set_xlabel("# updates", fontsize=9)

handles = [
    plt.Line2D([0], [0], color=RI_COLOR, lw=2, marker="o", markersize=6,
               markerfacecolor="white", markeredgewidth=2, label="RI — first value"),
    plt.Line2D([0], [0], color=PI_COLOR, lw=2, marker="s", markersize=6,
               markerfacecolor="white", markeredgewidth=2, label="PI — last value"),
]
fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=11,
           frameon=True, bbox_to_anchor=(0.5, 1.01))

fig.suptitle(
    f"{MODEL_NAME} · Narrative (Dota 2) · {N_TRIALS} trials/cell",
    fontsize=13, y=1.04
)
plt.tight_layout()

out = f"/Users/kanak.raj/workspace/hobby/research_work_ri/v3/plots/narrative_{MODEL_NAME}_ri_pi.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=160, bbox_inches="tight")
print(f"Saved → {out}")
