#!/usr/bin/env python3
"""
Generate FVQ/CVQ vs K and vs N figure for Gemma-3-4b.
Saves to paper/figures/gemma3_4b_fvq_cvq.pdf and .png
"""

import json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Load data ─────────────────────────────────────────────────────────────────
f = glob.glob(
    "/Users/kanak.raj/workspace/hobby/research_work_ri/v3/results_vllm/"
    "arbitrary_single/gemma-3-4b-it/stage1_sweep*.json"
)[0]
with open(f) as fp:
    d = json.load(fp)

cells = d["cells"]
grid = {}
for ck, v in cells.items():
    if "stats" not in v:
        continue
    nk, nu = int(ck.split("_")[0]), int(ck.split("_")[1])
    ri = v["stats"].get("RI", {}).get("accuracy")
    pi = v["stats"].get("PI", {}).get("accuracy")
    ri_lo = v["stats"].get("RI", {}).get("ci_lower")
    ri_hi = v["stats"].get("RI", {}).get("ci_upper")
    pi_lo = v["stats"].get("PI", {}).get("ci_lower")
    pi_hi = v["stats"].get("PI", {}).get("ci_upper")
    if ri is not None and pi is not None:
        grid[(nk, nu)] = (ri, pi, ri_lo, ri_hi, pi_lo, pi_hi)

# ── Style ──────────────────────────────────────────────────────────────────────
FVQ_COLOR = "#2166ac"   # blue
CVQ_COLOR = "#d6604d"   # red

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
})

fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))

# ── Panel A: vs K at fixed N=10 ───────────────────────────────────────────────
N_FIXED = 10
K_LEVELS = [2, 3, 5, 7, 10, 15, 20, 25, 30]

fvq_k, cvq_k = [], []
fvq_k_lo, fvq_k_hi, cvq_k_lo, cvq_k_hi = [], [], [], []
k_valid = []

for k in K_LEVELS:
    v = grid.get((k, N_FIXED))
    if v:
        fvq_k.append(v[0]); cvq_k.append(v[1])
        fvq_k_lo.append(v[2]); fvq_k_hi.append(v[3])
        cvq_k_lo.append(v[4]); cvq_k_hi.append(v[5])
        k_valid.append(k)

ax = axes[0]
k_arr = np.array(k_valid)
ax.plot(k_arr, fvq_k, color=FVQ_COLOR, lw=2, marker="o", markersize=5,
        markerfacecolor="white", markeredgewidth=1.5, zorder=5, label="FVQ")
ax.plot(k_arr, cvq_k, color=CVQ_COLOR, lw=2, marker="s", markersize=5,
        markerfacecolor="white", markeredgewidth=1.5, zorder=5, label="CVQ")
ax.fill_between(k_arr,
                [max(0, v) for v in fvq_k_lo], [min(1, v) for v in fvq_k_hi],
                alpha=0.12, color=FVQ_COLOR)
ax.fill_between(k_arr,
                [max(0, v) for v in cvq_k_lo], [min(1, v) for v in cvq_k_hi],
                alpha=0.12, color=CVQ_COLOR)

ax.set_xlabel("Number of keys ($K$)", fontsize=10)
ax.set_ylabel("Accuracy", fontsize=10)
ax.set_title(f"Varying key count ($N={N_FIXED}$ fixed)", fontsize=10, pad=6)
ax.set_ylim(0, 1.05)
ax.set_xticks(k_valid)
ax.xaxis.set_tick_params(labelsize=8)
ax.yaxis.set_tick_params(labelsize=8)
ax.legend(frameon=False, fontsize=9, loc="lower left")
ax.text(0.97, 0.97, "(a)", transform=ax.transAxes,
        ha="right", va="top", fontsize=10, fontweight="bold")

# ── Panel B: vs N at fixed K=5 ────────────────────────────────────────────────
K_FIXED = 5
N_LEVELS = [5, 7, 10, 15, 20, 30, 50, 75, 100]

fvq_n, cvq_n = [], []
fvq_n_lo, fvq_n_hi, cvq_n_lo, cvq_n_hi = [], [], [], []
n_valid = []

for n in N_LEVELS:
    v = grid.get((K_FIXED, n))
    if v:
        fvq_n.append(v[0]); cvq_n.append(v[1])
        fvq_n_lo.append(v[2]); fvq_n_hi.append(v[3])
        cvq_n_lo.append(v[4]); cvq_n_hi.append(v[5])
        n_valid.append(n)

ax = axes[1]
n_arr = np.array(n_valid)
ax.plot(n_arr, fvq_n, color=FVQ_COLOR, lw=2, marker="o", markersize=5,
        markerfacecolor="white", markeredgewidth=1.5, zorder=5, label="FVQ")
ax.plot(n_arr, cvq_n, color=CVQ_COLOR, lw=2, marker="s", markersize=5,
        markerfacecolor="white", markeredgewidth=1.5, zorder=5, label="CVQ")
ax.fill_between(n_arr,
                [max(0, v) for v in fvq_n_lo], [min(1, v) for v in fvq_n_hi],
                alpha=0.12, color=FVQ_COLOR)
ax.fill_between(n_arr,
                [max(0, v) for v in cvq_n_lo], [min(1, v) for v in cvq_n_hi],
                alpha=0.12, color=CVQ_COLOR)

ax.set_xlabel("Updates per key ($N$)", fontsize=10)
ax.set_ylabel("Accuracy", fontsize=10)
ax.set_title(f"Varying update depth ($K={K_FIXED}$ fixed)", fontsize=10, pad=6)
ax.set_ylim(0, 1.05)
ax.set_xticks(n_valid)
ax.xaxis.set_tick_params(labelsize=8, rotation=45)
ax.yaxis.set_tick_params(labelsize=8)
ax.legend(frameon=False, fontsize=9, loc="lower left")
ax.text(0.97, 0.97, "(b)", transform=ax.transAxes,
        ha="right", va="top", fontsize=10, fontweight="bold")

# ── Save ───────────────────────────────────────────────────────────────────────
fig.suptitle("Gemma-3-4b · Arbitrary-Single · 100 trials/cell",
             fontsize=10, y=1.02)
plt.tight_layout()

out_base = "/Users/kanak.raj/workspace/hobby/research_work_ri/paper/figures/gemma3_4b_fvq_cvq"
plt.savefig(out_base + ".pdf", dpi=200, bbox_inches="tight")
plt.savefig(out_base + ".png", dpi=200, bbox_inches="tight")
print(f"Saved → {out_base}.pdf / .png")
