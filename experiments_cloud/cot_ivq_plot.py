#!/usr/bin/env python3
"""Per-position accuracy curves for the CoT vs no-CoT IVQ sweep.

Emits three artifacts into <results>/_figs/ plus a tidy summary CSV:

  cot_ivq_grid.png      3 models x 6 cells, both arms, full position curve
  cot_ivq_deepest.png   the K10/N50 cell alone, one panel per model
  cot_ivq_cells.png     cell-level accuracy (equal-weight) vs stream length
  cot_ivq_summary.csv   one row per (model, arm, cell, position)

Two conventions matter for reading these plots:

  * The ordinal query "the Nth value" and the recency query "the last value"
    target the SAME item, so 'last' is drawn as a detached open marker at
    x = N rather than as a point on the curve. The gap between the two is a
    pure query-phrasing effect on identical retrieval.
  * Cell-level numbers are equal-weighted across positions, never pooled.
    Per-position Wilson retirement stops sampling a position once its CI is
    tight, so easy positions carry ~56 trials and hard ones ~200; a pooled
    mean is therefore weighted toward whichever positions happened to be hard
    and is not comparable across runs.
"""
import argparse, collections, csv, json, math, os, glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = [("haiku", "claude-haiku-4-5-20251001"),
          ("sonnet", "claude-sonnet-4-5-20250929"),
          ("opus", "claude-opus-4-5-20251101")]
CELLS = [(5, 10), (10, 10), (5, 20), (10, 20), (5, 50), (10, 50)]
ARMS = [("nocot", "no CoT", "#d1495b", "o"), ("cot_thinking", "CoT", "#1f77b4", "s")]


def wilson(k, n, z=1.96):
    """Wilson score interval. Behaves sanely at k=0 and k=n, unlike normal approx."""
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def load(run_dir):
    """-> {(K, N): {position: (n_correct, n_total)}}"""
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    with open(os.path.join(run_dir, "trials.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("error"):
                continue
            cell = acc[(r["num_keys"], r["num_updates"])][r["position"]]
            cell[0] += bool(r["correct"])
            cell[1] += 1
    return acc


def split(positions):
    """Ordinal positions sorted numerically, with 'last' pulled out separately."""
    nums = sorted((p for p in positions if p != "last"), key=int)
    return nums, ("last" if "last" in positions else None)


def curve(ax, data, nu, arm_key, label, colour, marker, show_last=True):
    nums, has_last = split(data)
    if not nums:
        return
    xs = [int(p) for p in nums]
    ys, lo, hi = [], [], []
    for p in nums:
        k, n = data[p]
        m, a, b = wilson(k, n)
        ys.append(m); lo.append(m - a); hi.append(b - m)
    ax.errorbar(xs, ys, yerr=[lo, hi], color=colour, marker=marker, ms=3.5,
                lw=1.4, capsize=1.5, elinewidth=0.7, label=label, zorder=3)
    if has_last and show_last:
        k, n = data["last"]
        m, a, b = wilson(k, n)
        # Offset right of the last ordinal so the two markers don't overlap.
        ax.errorbar([nu * 1.09], [m], yerr=[[m - a], [b - m]], color=colour,
                    marker=marker, ms=6, mfc="white", mew=1.6, lw=0,
                    capsize=1.5, elinewidth=0.7, zorder=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="experiments_cloud/results/cot_ivq")
    args = ap.parse_args()
    figs = os.path.join(args.results_dir, "_figs")
    os.makedirs(figs, exist_ok=True)

    runs = {}
    for short, mid in MODELS:
        for arm, *_ in ARMS:
            d = os.path.join(args.results_dir, f"{mid}__{arm}")
            if os.path.isdir(d):
                runs[(short, arm)] = load(d)

    # ---- tidy CSV ----------------------------------------------------------
    csv_path = os.path.join(figs, "cot_ivq_summary.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "arm", "num_keys", "num_updates", "position",
                    "position_kind", "n_correct", "n_trials", "accuracy",
                    "wilson_lo", "wilson_hi"])
        for (short, arm), acc in runs.items():
            for (nk, nu) in CELLS:
                for p, (k, n) in acc.get((nk, nu), {}).items():
                    m, lo, hi = wilson(k, n)
                    w.writerow([short, arm, nk, nu, p,
                                "recency" if p == "last" else "ordinal",
                                k, n, f"{m:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
    print(f"wrote {csv_path}")

    # ---- figure 1: full grid ----------------------------------------------
    fig, axes = plt.subplots(3, 6, figsize=(21, 9.5), sharey=True)
    for i, (short, _) in enumerate(MODELS):
        for j, (nk, nu) in enumerate(CELLS):
            ax = axes[i][j]
            for arm, label, colour, marker in ARMS:
                data = runs.get((short, arm), {}).get((nk, nu))
                if data:
                    curve(ax, data, nu, arm, label, colour, marker)
            ax.set_ylim(-0.04, 1.04)
            ax.set_xlim(0, nu * 1.18)
            ax.grid(alpha=0.25, lw=0.5)
            ax.axhline(1 / nu, color="grey", ls=":", lw=0.8, zorder=1)
            if i == 0:
                ax.set_title(f"K={nk}, N={nu}", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{short}\naccuracy", fontsize=11)
            if i == 2:
                ax.set_xlabel("queried occurrence")
    axes[0][0].legend(fontsize=9, loc="lower left")
    fig.suptitle("IVQ accuracy by queried occurrence — CoT vs no CoT (flat_nolabel, SEMANTIC_MULTI)\n"
                 "filled = ordinal query \"the k-th value\";  open marker at right = recency query "
                 "\"the last value\" (same target item);  dotted = chance",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p1 = os.path.join(figs, "cot_ivq_grid.png")
    fig.savefig(p1, dpi=160); plt.close(fig)
    print(f"wrote {p1}")

    # ---- figure 2: deepest cell -------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for i, (short, _) in enumerate(MODELS):
        ax = axes[i]
        for arm, label, colour, marker in ARMS:
            data = runs.get((short, arm), {}).get((10, 50))
            if data:
                curve(ax, data, 50, arm, label, colour, marker)
        ax.set_ylim(-0.04, 1.04); ax.set_xlim(0, 58)
        ax.grid(alpha=0.25, lw=0.5)
        ax.axhline(1 / 50, color="grey", ls=":", lw=0.8)
        ax.set_title(short, fontsize=12)
        ax.set_xlabel("queried occurrence")
        if i == 0:
            ax.set_ylabel("accuracy"); ax.legend(fontsize=9, loc="center right")
    fig.suptitle("Deepest cell (K=10, N=50): the interior collapses without CoT", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    p2 = os.path.join(figs, "cot_ivq_deepest.png")
    fig.savefig(p2, dpi=160); plt.close(fig)
    print(f"wrote {p2}")

    # ---- figure 3: cell-level ---------------------------------------------
    # Model is encoded by marker shape, not opacity: three overlapping alpha
    # levels on the same colour are not separable at print size.
    shapes = {"haiku": "o", "sonnet": "^", "opus": "s"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, nk in zip(axes, (5, 10)):
        for short, _ in MODELS:
            for arm, label, colour, _ in ARMS:
                xs, ys = [], []
                for (kk, nu) in CELLS:
                    if kk != nk:
                        continue
                    data = runs.get((short, arm), {}).get((kk, nu))
                    if not data:
                        continue
                    # Equal-weight across positions; see module docstring.
                    xs.append(nu)
                    ys.append(sum(k / n for k, n in data.values()) / len(data))
                ax.plot(xs, ys, ls="-" if arm == "cot_thinking" else "--",
                        marker=shapes[short], color=colour, lw=1.5, ms=7,
                        mfc=colour if arm == "cot_thinking" else "white",
                        mew=1.5, label=f"{short} {label}")
        ax.set_xticks([10, 20, 50]); ax.set_xlim(5, 55)
        ax.set_ylim(0, 1.05); ax.grid(alpha=0.25, lw=0.5)
        ax.set_title(f"K = {nk} keys"); ax.set_xlabel("stream length N (updates)")
    axes[0].set_ylabel("accuracy (equal-weight over positions)")
    axes[1].legend(fontsize=8, ncol=2, loc="center right", framealpha=0.95)
    fig.suptitle("Cell-level accuracy: CoT stays near ceiling, no-CoT is flat and low\n"
                 "circle = haiku, triangle = sonnet, square = opus;  "
                 "solid/filled = CoT, dashed/hollow = no CoT", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    p3 = os.path.join(figs, "cot_ivq_cells.png")
    fig.savefig(p3, dpi=160); plt.close(fig)
    print(f"wrote {p3}")


if __name__ == "__main__":
    main()
