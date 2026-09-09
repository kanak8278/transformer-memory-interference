#!/usr/bin/env python3
"""Plots for the fixed-budget N-scaling ladder (haiku, ARBITRARY_SINGLE, K=5).

The experiment holds the thinking budget pinned at 3000 while N grows, so it
measures compute-bounded indexing rather than "how much reasoning does the task
need". Three artifacts land in <results>/_figs/:

  ladder_positions.png  per-position accuracy, one line per N
  ladder_summary.png    cell accuracy vs N, plus the token/failure diagnostics
  ladder_summary.csv    one row per (cell, position)

Reporting conventions carried over from the main sweep:
  * Cell accuracy is EQUAL-WEIGHTED across positions. Per-position Wilson
    retirement gives easy positions ~56 trials and hard ones ~200, so the
    pooled mean is weighted toward whichever positions happened to be hard. At
    N=460 pooling even reverses the trend (0.508 -> 0.549 vs 0.577 -> 0.564).
  * Ordinal position N and the recency query 'last' hit the same target item,
    so 'last' is drawn detached rather than as a point on the curve.
"""
import collections, csv, json, math, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Defaults to the haiku ladder; pass a run directory as argv[1] to plot another
# model. Figures land in <run>/_figs so the three models cannot overwrite each
# other, except for the default haiku call, which keeps its original shared
# path for backward compatibility.
_DEFAULT_RUN = ("experiments_cloud/results/cot_ivq_arbitrary_single/"
                "claude-haiku-4-5-20251001__cot_thinking__ARBITRARY_SINGLE")
RUN = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else _DEFAULT_RUN
FIGS = ("experiments_cloud/results/cot_ivq_arbitrary_single/_figs"
        if RUN == _DEFAULT_RUN else os.path.join(RUN, "_figs"))
MODEL = os.path.basename(RUN).split("__")[0].replace("claude-", "")
# Reference point from the completed SEMANTIC_MULTI sweep, same K and N.
SEMANTIC_K5N50 = 0.975
BUDGET = 3000


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def load():
    pos = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    tok = collections.defaultdict(list)
    outc = collections.defaultdict(collections.Counter)
    trunc = collections.Counter()
    with open(os.path.join(RUN, "trials.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("error"):
                continue
            nu = r["num_updates"]
            cell = pos[nu][r["position"]]
            cell[0] += bool(r["correct"]); cell[1] += 1
            tok[nu].append(r.get("output_tokens") or 0)
            outc[nu][r["outcome"]] += 1
            trunc[nu] += bool(r.get("truncated"))
    return pos, tok, outc, trunc


def main():
    os.makedirs(FIGS, exist_ok=True)
    pos, tok, outc, trunc = load()
    Ns = sorted(pos)
    cmap = plt.cm.viridis
    colours = {n: cmap(i / max(len(Ns) - 1, 1)) for i, n in enumerate(Ns)}

    with open(os.path.join(FIGS, "ladder_summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["num_keys", "num_updates", "position", "position_kind",
                    "n_correct", "n_trials", "accuracy", "wilson_lo", "wilson_hi"])
        for n in Ns:
            for p, (k, t) in pos[n].items():
                m, lo, hi = wilson(k, t)
                w.writerow([5, n, p, "recency" if p == "last" else "ordinal",
                            k, t, f"{m:.4f}", f"{lo:.4f}", f"{hi:.4f}"])

    # ---- figure 1: per-position curves, all N overlaid --------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    for ax, logx in zip(axes, (False, True)):
        for n in Ns:
            nums = sorted((p for p in pos[n] if p != "last"), key=int)
            xs = [int(p) for p in nums]
            ys, lo, hi = [], [], []
            for p in nums:
                m, a, b = wilson(*pos[n][p])
                ys.append(m); lo.append(m - a); hi.append(b - m)
            ax.errorbar(xs, ys, yerr=[lo, hi], color=colours[n], marker="o", ms=3.5,
                        lw=1.5, capsize=1.5, elinewidth=0.7, label=f"N={n}")
            m, a, b = wilson(*pos[n]["last"])
            ax.errorbar([n * (1.13 if not logx else 1.16)], [m],
                        yerr=[[m - a], [b - m]], color=colours[n], marker="o",
                        ms=7, mfc="white", mew=1.6, lw=0, capsize=1.5, elinewidth=0.7)
        ax.set_ylim(-0.04, 1.04); ax.grid(alpha=0.25, lw=0.5)
        ax.set_xlabel("queried occurrence (ordinal)")
        if logx:
            ax.set_xscale("log"); ax.set_xlim(0.8, 700)
            ax.set_title("same data, log x — shallow anchors are flat across all N")
        else:
            ax.set_xlim(0, 540); ax.set_ylabel("accuracy")
            ax.set_title("linear x — the interior collapses as N grows")
            ax.legend(fontsize=9, loc="upper right", ncol=2)
    fig.suptitle(f"Fixed thinking budget ({BUDGET}), {MODEL}, ARBITRARY_SINGLE K=5: accuracy by query depth\n"
                 "filled = ordinal \"the k-th value\";  open marker = recency \"the last value\" (same target item)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    p1 = os.path.join(FIGS, "ladder_positions.png")
    fig.savefig(p1, dpi=160); plt.close(fig)
    print("wrote", p1)

    # ---- figure 2: cell-level + diagnostics ------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    ax = axes[0]
    eq = [sum(k / t for k, t in pos[n].values()) / len(pos[n]) for n in Ns]
    pooled = [sum(k for k, _ in pos[n].values()) / sum(t for _, t in pos[n].values())
              for n in Ns]
    ax.plot(Ns, eq, "-o", color="#1f77b4", lw=1.8, ms=7, label="equal-weight (use this)")
    ax.plot(Ns, pooled, "--s", color="#999999", lw=1.4, ms=5,
            label="pooled (retirement-biased)")
    ax.axhline(SEMANTIC_K5N50, color="#d1495b", ls=":", lw=1.2)
    ax.annotate(f"SEMANTIC_MULTI K5/N50 = {SEMANTIC_K5N50}", (Ns[-1], SEMANTIC_K5N50),
                fontsize=8, color="#d1495b", ha="right", va="bottom")
    ax.set_xlabel("N (updates per key)"); ax.set_ylabel("accuracy")
    ax.set_ylim(0.4, 1.03); ax.grid(alpha=0.25, lw=0.5)
    ax.set_title("cell accuracy vs N"); ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    means = [sum(tok[n]) / len(tok[n]) for n in Ns]
    p95 = [sorted(tok[n])[int(0.95 * len(tok[n]))] for n in Ns]
    ax.plot(Ns, means, "-o", color="#1f77b4", lw=1.8, ms=6, label="mean output tokens")
    ax.plot(Ns, p95, "--^", color="#1f77b4", lw=1.3, ms=6, alpha=0.6, label="p95")
    ax.axhline(BUDGET, color="black", ls=":", lw=1.2)
    ax.annotate(f"thinking budget = {BUDGET}", (Ns[0], BUDGET), fontsize=8,
                va="bottom")
    ax.set_xlabel("N"); ax.set_ylabel("output tokens")
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_title("the median response never approaches the budget")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[2]
    tot = {n: sum(outc[n].values()) for n in Ns}
    for key, col in [("correct", "#1f77b4"), ("in_sequence", "#d1495b"),
                     ("out_of_context", "#edae49")]:
        ax.plot(Ns, [outc[n][key] / tot[n] for n in Ns], "-o", color=col, ms=6,
                lw=1.6, label=key)
    ax.plot(Ns, [trunc[n] / tot[n] for n in Ns], "--x", color="grey", ms=6,
            lw=1.2, label="truncated")
    ax.set_xlabel("N"); ax.set_ylabel("share of responses")
    ax.set_ylim(-0.03, 1.0); ax.grid(alpha=0.25, lw=0.5)
    ax.set_title("failure mode: wrong occurrence of the right key")
    ax.legend(fontsize=8, loc="center right")

    fig.suptitle("Fixed-budget N ladder — diagnostics", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    p2 = os.path.join(FIGS, "ladder_summary.png")
    fig.savefig(p2, dpi=160); plt.close(fig)
    print("wrote", p2)


if __name__ == "__main__":
    main()
