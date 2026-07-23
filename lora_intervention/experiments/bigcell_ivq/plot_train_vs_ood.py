"""Heatmap figure: existing held-out (endpoint-only) eval vs this session's
big-cell OOD scan (endpoints + interior), Qwen and Gemma LoRA adapters.
Static matplotlib PNG, meant for sharing outside the repo."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "lora_intervention" / "results"
OUT_DIR = Path(__file__).resolve().parent / "out"

# sequential blue ramp, light->dark, from the palette (100..700)
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("acc_blue", BLUE_RAMP)
NORM = Normalize(vmin=0, vmax=1)
MUTED = "#898781"
INK = "#0b0b0b"
GRID_LINE = "#e1e0d9"

# ─── Panel A: existing held-out eval (endpoints only, RI=FVQ-analog, PI=CVQ-analog) ──

PANEL_A_FILES = {
    "Qwen": "main_eval_20260523_055309.json",
    "Gemma": "gemma_main_eval_20260525_165155.json",
}


def load_panel_a(model):
    d = json.loads((RESULTS_DIR / PANEL_A_FILES[model]).read_text())
    cells = d["results"]["ARBITRARY_SINGLE"]
    rows = []
    for c in cells.values():
        rows.append({
            "K": c["num_keys"], "N": c["num_updates"],
            "FVQ": c["stats"]["RI"]["accuracy"],
            "CVQ": c["stats"]["PI"]["accuracy"],
        })
    return rows


# ─── Panel B: this session's OOD big-cell scan (LoRA only) ───────────────────

def load_panel_b(model):
    path = OUT_DIR / model.lower() / "results.jsonl"
    rows = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["model"] == "lora":
            rows.append(r)
    return rows


def build_grid(rows, k_vals, n_vals, value_fn):
    grid = np.full((len(k_vals), len(n_vals)), np.nan)
    for r in rows:
        k = r.get("K", r.get("num_keys"))
        n = r.get("N", r.get("num_updates"))
        if k in k_vals and n in n_vals:
            v = value_fn(r)
            if v is not None:
                grid[k_vals.index(k), n_vals.index(n)] = v
    return grid


def draw_heatmap(ax, grid, k_vals, n_vals, title, show_ylabel):
    ax.imshow(grid, cmap=CMAP, norm=NORM, aspect="equal")
    for i in range(len(k_vals)):
        for j in range(len(n_vals)):
            v = grid[i, j]
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            facecolor="#f2f1ee", edgecolor=GRID_LINE, linewidth=0.6))
                ax.text(j, i, "—", ha="center", va="center", fontsize=7, color=MUTED)
    ax.set_xticks(range(len(n_vals)))
    ax.set_xticklabels(n_vals, fontsize=7.5, color=MUTED)
    ax.set_yticks(range(len(k_vals)))
    ax.set_yticklabels(k_vals if show_ylabel else [], fontsize=7.5, color=MUTED)
    ax.set_title(title, fontsize=8.5, color=INK, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel("N (updates)", fontsize=7, color=MUTED, labelpad=2)
    if show_ylabel:
        ax.set_ylabel("K (keys)", fontsize=7, color=MUTED, labelpad=2)


def main():
    K_A = [7, 10, 15, 20, 25, 30]
    N_A = [10, 15, 20, 30, 50, 75]
    K_B = [15, 20, 25, 30]
    N_B = [10, 15, 20, 30, 50]

    fig = plt.figure(figsize=(13.5, 9.9), facecolor="#fcfcfb")
    gs_outer = fig.add_gridspec(2, 1, height_ratios=[2, 2], hspace=1.15, top=0.815, bottom=0.155)
    gs_a = gs_outer[0].subgridspec(2, 2, wspace=0.35, hspace=0.55)
    gs_b = gs_outer[1].subgridspec(2, 5, wspace=0.25, hspace=0.55)

    pos_a = gs_outer[0].get_position(fig)
    pos_b = gs_outer[1].get_position(fig)

    # training-grid callout — neither panel below trains on these cells;
    # every cell shown is generalization, differing only in how far out.
    fig.patches.append(plt.Rectangle((0.045, 0.895), 0.91, 0.052, transform=fig.transFigure,
                                      facecolor="#f2f1ee", edgecolor=GRID_LINE, linewidth=0.8,
                                      zorder=0))
    fig.text(0.06, 0.930, "Trained on:", fontsize=9.5, color=INK, fontweight="bold")
    fig.text(0.155, 0.930,
             "K ∈ {2, 3, 5, 10} × N ∈ {5, 10, 15, 20}  —  every cell in A and B below is beyond this grid; "
             "A is nearer extrapolation, B is far extrapolation.",
             fontsize=9, color="#33322f")

    fig.text(0.06, pos_a.y1 + 0.055, "A — Existing held-out eval (near-training extrapolation, endpoints only)",
              fontsize=10.5, color=INK, fontweight="bold")
    fig.text(0.06, pos_a.y1 + 0.030,
             "28 held-out cells, K=7–30 × N=10–75 (already past the training grid above) "
             "— near-ceiling accuracy, established before this session",
             fontsize=8.5, color=MUTED)

    for row, model in enumerate(["Qwen", "Gemma"]):
        rows_a = load_panel_a(model)
        for col, cond in enumerate(["FVQ", "CVQ"]):
            grid = build_grid(rows_a, K_A, N_A, value_fn=lambda r, c=cond: r[c])
            ax = fig.add_subplot(gs_a[row, col])
            label = "first value" if cond == "FVQ" else "last value"
            draw_heatmap(ax, grid, K_A, N_A, f"{model} · {cond} ({label})", show_ylabel=(col == 0))

    fig.text(0.06, pos_b.y1 + 0.055, "B — This session's OOD big-cell scan (far extrapolation, endpoints + interior, LoRA)",
              fontsize=10.5, color=INK, fontweight="bold")
    fig.text(0.06, pos_b.y1 + 0.030,
             "20 new cells, K=15–30 × N=10–50 (further past the training grid than A), n=200 Wilson CIs "
             "— endpoints still hold, but interior collapses",
             fontsize=8.5, color=MUTED)

    conds_b = ["FVQ", "CVQ", "IVQ@0.10", "IVQ@0.50", "IVQ@0.90"]
    cond_labels = {"FVQ": "first", "CVQ": "last", "IVQ@0.10": "interior 10%",
                   "IVQ@0.50": "interior 50%", "IVQ@0.90": "interior 90%"}
    for row, model in enumerate(["Qwen", "Gemma"]):
        rows_b = load_panel_b(model)
        for col, cond in enumerate(conds_b):
            sub = [r for r in rows_b if r["condition"] == cond]
            grid = build_grid(sub, K_B, N_B, value_fn=lambda r: r["accuracy"])
            ax = fig.add_subplot(gs_b[row, col])
            draw_heatmap(ax, grid, K_B, N_B, f"{model} · {cond_labels[cond]}", show_ylabel=(col == 0))

    # shared colorbar
    cax = fig.add_axes([0.30, 0.035, 0.40, 0.020])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, orientation="horizontal",
                       ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb.ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    cb.set_label("Accuracy — legend", fontsize=9, color=INK, labelpad=6)
    cb.ax.tick_params(labelsize=8, color=MUTED, labelcolor=MUTED, length=0)
    cb.outline.set_visible(False)

    fig.suptitle("LoRA generalization: near-ceiling in-distribution vs. big-cell out-of-distribution",
                 fontsize=13, color=INK, y=0.985, fontweight="bold")

    out_path = OUT_DIR / "train_vs_ood_heatmap.png"
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
