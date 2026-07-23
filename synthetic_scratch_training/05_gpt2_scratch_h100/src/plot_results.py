"""Figure: model accuracy across the (K,N) grid + counting-depth, train (iid) vs
test (held-out) distribution. Reads a run's final_eval.json.

Layout (chosen with the dataviz form heuristic):
  - top row: two K x N heatmaps (iid | heldout), sequential 'cividis' ramp
    (accuracy = magnitude), grey = never-tested cells. Shared 0-100% colorbar.
  - bottom: accuracy vs counting-depth (intermediate step index k), iid vs heldout
    as an Okabe-Ito colorblind-safe pair (blue/orange), distinct line styles +
    direct end labels + legend.

Usage: python plot_results.py <path/to/final_eval.json> [out.png]
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

K_VALUES = [2, 4, 6, 8, 10, 12]
N_VALUES = [2, 4, 6, 8, 10, 12]
IID_COLOR = "#0072B2"      # Okabe-Ito blue
HELDOUT_COLOR = "#E69F00"  # Okabe-Ito orange


def cell_matrix(by_cell):
    """by_cell: {'K{k}_N{n}': {'acc':..}} -> K x N matrix (NaN where absent)."""
    M = np.full((len(K_VALUES), len(N_VALUES)), np.nan)
    for i, k in enumerate(K_VALUES):
        for j, n in enumerate(N_VALUES):
            v = by_cell.get(f"K{k}_N{n}")
            if v and v.get("acc") is not None:
                M[i, j] = v["acc"]
    return M


def draw_heatmap(ax, M, title):
    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad("#dcdcdc")  # never-tested cells
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="equal", origin="upper")
    ax.set_xticks(range(len(N_VALUES)), N_VALUES)
    ax.set_yticks(range(len(K_VALUES)), K_VALUES)
    ax.set_xlabel("N  (updates per key)")
    ax.set_ylabel("K  (active keys)")
    ax.set_title(title, fontsize=11, pad=6)
    for i in range(len(K_VALUES)):
        for j in range(len(N_VALUES)):
            if np.isnan(M[i, j]):
                ax.text(j, i, "-", ha="center", va="center", color="#888", fontsize=8)
            else:
                # label ink stays neutral; contrast against the cividis value
                txt = "#000" if M[i, j] > 0.55 else "#fff"
                ax.text(j, i, f"{M[i, j]*100:.0f}", ha="center", va="center",
                        color=txt, fontsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return im


def draw_depth(ax, iid_step, heldout_step):
    def series(d):
        xs = sorted(int(s) for s in d)
        return xs, [d[str(x)]["acc"] for x in xs]
    xi, yi = series(iid_step)
    xh, yh = series(heldout_step)
    ax.plot(xi, yi, "-o", color=IID_COLOR, lw=2, ms=6, label="iid (trained steps)")
    ax.plot(xh, yh, "--s", color=HELDOUT_COLOR, lw=2, ms=6, label="held-out (never trained)")
    ax.axhline(0.10, color="#bbb", lw=1, ls=":", zorder=0)
    ax.text(xi[-1], 0.10, " chance", color="#999", va="bottom", ha="right", fontsize=8)
    if xi:
        ax.text(xi[-1], yi[-1], "  iid", color=IID_COLOR, va="center", fontsize=9, fontweight="bold")
    if xh:
        ax.text(xh[-1], yh[-1], "  held-out", color=HELDOUT_COLOR, va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("counting depth: query step k  (k-th occurrence, intermediate only)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_xticks(range(2, 12))
    ax.set_title("Deeper counting is harder — and generalizes worse", fontsize=11, pad=6)
    ax.grid(axis="y", color="#eee", lw=1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=9)


def main():
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else path.replace("final_eval.json", "figure_performance.png")
    d = json.load(open(path))
    iid, heldout = d["iid_test"], d["heldout_test"]

    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.85], hspace=0.32, wspace=0.28,
                          left=0.08, right=0.90, top=0.90, bottom=0.09)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    im = draw_heatmap(ax1, cell_matrix(iid["by_cell"]), f"TRAIN dist (iid)   overall {iid['overall']*100:.1f}%")
    draw_heatmap(ax2, cell_matrix(heldout["by_cell"]), f"TEST dist (held-out)   overall {heldout['overall']*100:.1f}%")
    cax = fig.add_axes([0.915, 0.52, 0.015, 0.34])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("accuracy", fontsize=9)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.ax.set_yticklabels(["0", "25", "50", "75", "100%"])

    draw_depth(ax3, iid["by_step"], heldout["by_step"])

    tag = d.get("tag", "run")
    fig.suptitle(f"GPT-2-small from scratch ({tag}): where it succeeds across the K x N grid",
                 fontsize=13, fontweight="bold", y=0.965)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
