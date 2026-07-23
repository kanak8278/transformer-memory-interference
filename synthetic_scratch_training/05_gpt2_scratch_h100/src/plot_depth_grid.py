"""Small multiples: accuracy vs step position, one panel per N, a line per K.

Shows how accuracy declines across intermediate positions and how that depends on
BOTH the history length N (panels) and the number of competing keys K (lines) --
the interaction the marginal step-curve hides. iid = solid lines; held-out
(never-trained) (n,step) cells = star markers on a shaded column.

Usage: python plot_depth_grid.py <joint_knstep.json> [out.png]
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

K_VALUES = [2, 4, 6, 8, 10, 12]
N_VALUES = [2, 4, 6, 8, 10, 12]


def parse(d):
    out = {}
    for key, v in d.items():
        k, n, s = (int(x) for x in key.split(","))
        out.setdefault((k, n), {})[s] = v["acc"]
    return out


def main():
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else path.replace("joint_knstep.json", "figure_depth_grid.png")
    d = json.load(open(path))
    iid = parse(d["iid"])
    held = parse(d["heldout"])
    held_steps_by_n = {}
    for (k, n), st in held.items():
        held_steps_by_n.setdefault(n, set()).update(st.keys())

    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=min(K_VALUES), vmax=max(K_VALUES))
    kcolor = {k: cmap(norm(k)) for k in K_VALUES}

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharey=True)
    for ax, n in zip(axes.flat, N_VALUES):
        # shade held-out step columns for this N
        for s in sorted(held_steps_by_n.get(n, [])):
            ax.axvspan(s - 0.4, s + 0.4, color="#f0d98c", alpha=0.35, zorder=0)
        for k in K_VALUES:
            st = iid.get((k, n), {})
            if st:
                xs = sorted(st)
                ax.plot(xs, [st[x] for x in xs], "-o", color=kcolor[k], lw=1.8, ms=4, zorder=3)
            hst = held.get((k, n), {})
            if hst:
                hx = sorted(hst)
                ax.plot(hx, [hst[x] for x in hx], "*", color=kcolor[k], ms=12,
                        markeredgecolor="#333", markeredgewidth=0.5, zorder=4)
        ax.axhline(0.10, color="#bbb", lw=1, ls=":", zorder=1)
        ax.set_title(f"N = {n}", fontsize=11)
        ax.set_xticks(range(1, n + 1))
        ax.set_xlim(0.5, n + 0.5)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", color="#eee", lw=1)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy")
    for ax in axes[1, :]:
        ax.set_xlabel("query step k (1=first … N=last)")

    # shared K colorbar + a marker-legend note
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cb = fig.colorbar(sm, cax=cax, ticks=K_VALUES); cb.set_label("K  (active keys)")
    fig.text(0.5, 0.005, "solid ●  = iid (trained steps)      ★ on shaded column = held-out (never-trained (N,step))",
             ha="center", fontsize=9, color="#444")
    fig.suptitle("Accuracy by step position, across every (K, N): deep counting fails first at high K",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.06, right=0.91, top=0.90, bottom=0.10, hspace=0.30, wspace=0.10)
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
