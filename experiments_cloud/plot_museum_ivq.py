#!/usr/bin/env python3
"""U-curve chart for the museum M0 dense-IVQ experiment (Haiku 4.5).

Small multiples: 2 rows (K=5,10) x 3 cols (N=10,20,50). Each panel shows
ordinal-query accuracy vs query position (the 'kth artwork' line) plus the
semantic 'first'/'last' anchors as separate markers — making the ordinal decay
and the semantic recency-rescue visible at a glance.

Palette: validated blue (#2a78d6) / orange (#eb6834).
Output: results/museum_ivq/claude-haiku/museum_ivq_ucurve.png
"""
import glob
import json
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e6e2"

f = sorted(glob.glob("results/museum_ivq/claude-haiku/ivq_full_*_queries.jsonl"))[-1]
rows = [json.loads(l) for l in open(f)]


def wilson(n, k, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    return z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)


# aggregate: cell -> ordinal {pos: [correct]}, semantic {'first'/'last': [correct]}
data = defaultdict(lambda: {"ord": defaultdict(list), "sem": defaultdict(list)})
for r in rows:
    d = data[(r["num_keys"], r["num_updates"])]
    if r["style"] == "ordinal":
        d["ord"][r["position"]].append(r["correct"])
    else:
        d["sem"][r["anchor"]].append(r["correct"])

KS, NS = [5, 10], [10, 20, 50]
fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), sharey=True)
fig.patch.set_facecolor("white")

for i, K in enumerate(KS):
    for j, N in enumerate(NS):
        ax = axes[i][j]
        d = data[(K, N)]
        xs = sorted(d["ord"])
        acc = [sum(d["ord"][p]) / len(d["ord"][p]) for p in xs]
        err = [wilson(len(d["ord"][p]), sum(d["ord"][p])) for p in xs]

        # ordinal U-curve (line + band)
        ax.fill_between(xs, [a - e for a, e in zip(acc, err)],
                        [a + e for a, e in zip(acc, err)], color=BLUE, alpha=0.12, lw=0)
        ax.plot(xs, acc, color=BLUE, lw=2, marker="o", ms=4, zorder=3)

        # semantic anchors at their true x-positions (first->1, last->N)
        for anchor, xpos in [("first", 1), ("last", N)]:
            v = d["sem"].get(anchor)
            if v:
                a = sum(v) / len(v)
                ax.plot([xpos], [a], marker="D", ms=9, color=ORANGE,
                        markeredgecolor="white", markeredgewidth=1.5, zorder=5)
                ax.annotate(f"{anchor}\n{a:.0%}", (xpos, a),
                            textcoords="offset points",
                            xytext=(6 if anchor == "first" else -6, 10),
                            ha="left" if anchor == "first" else "right",
                            fontsize=8, color=ORANGE, fontweight="bold")

        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(0.3, N + 0.7)
        ax.set_title(f"K={K} visitors,  N={N} updates", fontsize=10,
                     color=INK, pad=6)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        if j == 0:
            ax.set_ylabel("recall accuracy", fontsize=9, color=MUTED)
        if i == 1:
            ax.set_xlabel("query position (kth mention)", fontsize=9, color=MUTED)
        ax.axhline(0, color=GRID, lw=0.8)

# legend
handles = [
    Line2D([0], [0], color=BLUE, lw=2, marker="o", ms=4,
           label="ordinal query — “the kth artwork” (counting)"),
    Line2D([0], [0], marker="D", ms=8, color=ORANGE, lw=0,
           markeredgecolor="white", label="semantic query — “first” / “last” (anchor)"),
]
fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
           fontsize=9.5, bbox_to_anchor=(0.5, 0.055))

fig.suptitle("Atemporal museum narrative (M0) — Claude Haiku 4.5: recall collapses by position, "
             "recency survives only via the “last” anchor",
             fontsize=12.5, color=INK, y=0.99)
fig.text(0.5, 0.945,
         "Ordinal counting decays from primacy to floor with no recency bump; "
         "the semantic “last” query recovers it. 200 trials/cell, 95% Wilson CI.",
         ha="center", fontsize=9, color=MUTED)

fig.tight_layout(rect=[0, 0.09, 1, 0.925])
out = "results/museum_ivq/claude-haiku/museum_ivq_ucurve.png"
fig.savefig(out, dpi=200, facecolor="white")
print("wrote", out)
