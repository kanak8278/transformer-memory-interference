"""
PI(N) exponential scaling curves: PI(N) = a·exp(-b·N) + c
Shows smooth fitted curves (for R²>0.8 models) + raw data dots.
Highlights the asymptotic floor (c) each model settles at.
Data: v3/results_vllm/analysis/scaling_law_fits_arbitrary_single.json
Output: v3/results_vllm/plots/scaling_curves.png
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent
FITS = BASE / "results_vllm" / "analysis" / "scaling_law_fits_arbitrary_single.json"
OUT  = BASE / "results_vllm" / "plots" / "scaling_curves.png"

# Models: (key, label, color, marker, show_fit)
# show_fit = True only for R² > 0.8
MODEL_CFG = [
    ("Qwen2.5-0.5B-Instruct",    "Qwen 0.5B",      "#E53935", "o",  True),
    ("Qwen2.5-1.5B-Instruct",    "Qwen 1.5B",      "#1E88E5", "o",  True),
    ("Qwen2.5-3B-Instruct",      "Qwen 3B-Inst",   "#43A047", "o",  True),
    ("Qwen2.5-3B",               "Qwen 3B-Base",   "#A5D6A7", "s",  True),
    ("gemma-3-1b-it",            "Gemma 1B",       "#8E24AA", "D",  True),
    ("gemma-3-4b-it",            "Gemma 4B",       "#4A148C", "D",  True),
    ("TinyLlama-1.1B-Chat-v1.0", "TinyLlama 1.1B", "#FF7043", "^",  False),
    ("stablelm-2-1_6b-chat",     "StableLM 1.6B",  "#795548", "v",  False),
]

with open(FITS) as f:
    fits = json.load(f)

N_smooth = np.linspace(5, 100, 400)

fig, ax = plt.subplots(figsize=(13, 6.5))
fig.patch.set_facecolor("white")

# ── RI reference band ─────────────────────────────────────────────────────────
# Collect RI means across models to make a band
ri_means = []
for model_key, *_ in MODEL_CFG:
    k2 = fits.get(model_key, {}).get("keys_2", {})
    if k2.get("data"):
        ri_vals = [d["ri"] for d in k2["data"]]
        ri_means.append(np.mean(ri_vals))

if ri_means:
    ri_lo = min(ri_means) - 0.05
    ri_hi = max(ri_means) + 0.03
    ax.axhspan(ri_lo, min(ri_hi, 1.0), color="#E8F5E9", alpha=0.6, zorder=0)
    ax.text(102, (ri_lo + ri_hi) / 2, "RI\n(robust)", ha="left", va="center",
            fontsize=8.5, color="#2E7D32", fontweight="bold")

# ── Plot each model ───────────────────────────────────────────────────────────

for model_key, label, color, marker, show_fit in MODEL_CFG:
    if model_key not in fits:
        continue
    k2 = fits[model_key].get("keys_2")
    if not k2 or not k2.get("data"):
        continue

    data   = k2["data"]
    N_pts  = np.array([d["N"] for d in data])
    pi_pts = np.array([d["pi"] for d in data])
    fit    = k2.get("pi_fit_exp", {})
    r2     = fit.get("r2", 0) or 0
    c_floor = fit.get("c")

    leg_label = f"{label}  (R²={r2:.2f})" if show_fit else f"{label}  (no fit)"

    # Raw dots
    ax.scatter(N_pts, pi_pts,
               color=color, marker=marker, s=55, zorder=4,
               alpha=0.6 if show_fit else 0.4)

    if show_fit and fit.get("a") is not None:
        a, b, c = fit["a"], fit["b"], fit["c"]
        y_fit = a * np.exp(-b * N_smooth) + c
        y_fit = np.clip(y_fit, 0, 1)

        # Smooth fitted curve
        ax.plot(N_smooth, y_fit,
                color=color, linewidth=2.2, alpha=0.9,
                label=leg_label, zorder=3)

        # Asymptotic floor line
        ax.axhline(c, color=color, linewidth=0.8,
                   linestyle=":", alpha=0.5, zorder=2)

        # Floor annotation on right edge
        ax.annotate(f"floor={c:.0%}",
                    xy=(101, c),
                    fontsize=7, color=color, va="center", ha="left",
                    annotation_clip=False)
    else:
        # No fit — just connect dots with thin dashed line
        ax.plot(N_pts, pi_pts,
                color=color, linewidth=1.2, linestyle="--",
                alpha=0.5, label=leg_label, zorder=2)

# ── Formatting ────────────────────────────────────────────────────────────────

ax.set_xlabel("Number of updates (N)", fontsize=12)
ax.set_ylabel("PI Accuracy  (recall last value)", fontsize=12)
ax.set_title(
    "PI Accuracy Decays Exponentially:  PI(N) = a · e^{−bN} + c\n"
    "Fitted curves shown for R² > 0.80  ·  keys = 2  ·  ARBITRARY_SINGLE",
    fontsize=13, fontweight="bold", pad=12
)
ax.set_xlim(3, 105)
ax.set_ylim(-0.04, 1.05)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.axhline(0, color="#AAAAAA", linewidth=0.7, linestyle="-", alpha=0.4)
ax.axhline(0.5, color="#CCCCCC", linewidth=0.7, linestyle=":", alpha=0.5)
ax.text(102, 0.5, "50%", ha="left", va="center", fontsize=7.5, color="#999999")
ax.grid(True, color="#EEEEEE", linewidth=0.7, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend outside right
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
          fontsize=8.5, framealpha=0.93, edgecolor="#CCCCCC",
          borderpad=0.8, handlelength=2.2)

plt.subplots_adjust(right=0.74)
plt.tight_layout(pad=1.5)
plt.subplots_adjust(right=0.74)
plt.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUT}")
