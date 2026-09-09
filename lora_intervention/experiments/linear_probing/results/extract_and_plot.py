"""Extract raw per-layer probe values (Qwen2.5-3B-Instruct, v3 grid) to CSV,
and plot AUC-by-layer small multiples (base vs LoRA) for FVQ, CVQ, and each
IVQ depth, at a fixed set of representative K,N cells.

Two metrics per condition/cell/state:
  - correctness probe: cv_auc per layer (needs both classes; degenerate cells
    are skipped / shown as gaps)
  - retrieval probe, wrong-trials-only: retrieval_auc per layer (defined even
    where correctness probe is degenerate)

Raw values only -- no smoothing, no interpretation baked into the numbers.
"""
import json
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RESULTS_DIR = Path(__file__).parent
OUT_DIR = RESULTS_DIR / "plots_for_mentor"
OUT_DIR.mkdir(exist_ok=True)

CONDITIONS = ["FVQ", "CVQ", "IVQ_d10", "IVQ_d25", "IVQ_d50", "IVQ_d75", "IVQ_d90"]
CELLS = [(15, 50), (15, 75), (20, 50), (25, 50), (25, 75), (30, 30), (30, 75)]
# Chosen so LoRA's CVQ wrong-trial retrieval curve is guaranteed to render at every
# cell (hard requirement): the retrieval probe needs >=20 usable (true, distractor)
# pairs per run_probing.py's min_pairs threshold, or the entire per-layer array is
# null (not just a flat/degenerate curve -- no curve at all). All 7 cells verified
# directly against the source JSON to have n_wrong in [23, 40] on CVQ, safely above
# that threshold. FVQ wrong trials are allowed to be zero/near-zero at some of these
# -- LoRA doesn't fail FVQ until N is large, that's a real property of the task, not
# a cell-selection gap. (2,5) and (10,30) were dropped: LoRA solves CVQ/FVQ near-
# perfectly there, leaving nothing to plot in the wrong-trials row at all.
STATES = ["base", "lora"]

COLOR = {"base": "#2a78d6", "lora": "#eb6834"}  # dataviz reference palette slot 1 / slot 2

def load(state, K, N):
    p = RESULTS_DIR / f"probe_Qwen2.5-3B-Instruct-{state}-v3_{K}k_{N}u.json"
    if not p.exists():
        return None
    return json.load(open(p))


def _retrieval_subset(c, subset):
    """Per-layer retrieval AUC list + n_trials for one subset (all/wrong/correct)."""
    sub = c.get("retrieval", {}).get(subset, {}) or {}
    sub_layers = sub.get("layers", {}) or {}
    n_layers = c.get("n_layers") or len(c["layers"])
    vals = []
    for L in range(n_layers):
        entry = sub_layers.get(str(L))
        vals.append(entry.get("retrieval_auc") if isinstance(entry, dict) else None)
    return vals, sub.get("n_trials")


def get_series(state, K, N, cond):
    """Returns (layers list, correctness_auc list-or-None,
    {subset: (retrieval_auc list-or-None, n_trials)} for subset in all/wrong/correct,
    behavioral_acc, n_trials)."""
    d = load(state, K, N)
    if d is None or cond not in d["conditions"]:
        return None
    c = d["conditions"][cond]
    n_layers = c.get("n_layers") or len(c["layers"])
    layers = list(range(n_layers))

    corr = []
    for L in range(n_layers):
        r = c["layers"].get(str(L), {})
        corr.append(r.get("cv_auc") if not r.get("degenerate") else None)

    retrieval = {subset: _retrieval_subset(c, subset) for subset in ("all", "wrong", "correct")}

    beh = c.get("behavioral_acc") or c.get("behavioral_accuracy")
    n = c.get("n")
    return layers, corr, retrieval, beh, n


# ---------- 1. Raw CSV dump (every layer, every cell, every condition, all metrics) ----------
csv_path = RESULTS_DIR / "probe_layer_values_raw.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["state", "K", "N", "condition", "layer", "correctness_cv_auc",
                "retrieval_all_auc", "retrieval_wrong_auc", "retrieval_correct_auc",
                "behavioral_acc", "n_trials", "n_all", "n_wrong", "n_correct"])
    for state in STATES:
        for f_ in sorted(RESULTS_DIR.glob(f"probe_Qwen2.5-3B-Instruct-{state}-v3_*.json")):
            d = json.load(open(f_))
            K, N = d["cell"]["keys"], d["cell"]["updates"]
            for cond in CONDITIONS:
                if cond not in d["conditions"]:
                    continue
                out = get_series(state, K, N, cond)
                if out is None:
                    continue
                layers, corr, retrieval, beh, n = out
                all_v, n_all = retrieval["all"]
                wrong_v, n_wrong = retrieval["wrong"]
                correct_v, n_correct = retrieval["correct"]
                for L in layers:
                    w.writerow([state, K, N, cond, L,
                                corr[L] if corr[L] is not None else "",
                                all_v[L] if all_v[L] is not None else "",
                                wrong_v[L] if wrong_v[L] is not None else "",
                                correct_v[L] if correct_v[L] is not None else "",
                                round(beh, 4) if beh is not None else "",
                                n, n_all, n_wrong, n_correct])
print(f"Wrote raw CSV: {csv_path}")

# ---------- 2. Plots: one figure per condition, small multiples across cells ----------
def make_figure(metric_key, metric_label, fname_prefix):
    """metric_key: 'corr', or 'retr:<subset>' where subset in {all, wrong, correct}."""
    retr_subset = metric_key.split(":")[1] if metric_key.startswith("retr:") else None
    for cond in CONDITIONS:
        fig, axes = plt.subplots(1, len(CELLS), figsize=(4.2 * len(CELLS), 3.6), sharey=True)
        for ax, (K, N) in zip(axes, CELLS):
            ax.axhline(0.5, color="#9a988e", linestyle="--", linewidth=1, zorder=0)
            titles = []
            missing_notes = []
            for state in STATES:
                out = get_series(state, K, N, cond)
                if out is None:
                    continue
                layers, corr, retrieval, beh, n = out
                if retr_subset is None:
                    series, n_used = corr, n
                else:
                    series, n_used = retrieval[retr_subset]
                xs = [L for L, v in zip(layers, series) if v is not None]
                ys = [v for v in series if v is not None]
                if xs:
                    ax.plot(xs, ys, color=COLOR[state], linewidth=2,
                             marker="o", markersize=2.5)
                else:
                    reason = "degenerate (single class)" if retr_subset is None else f"n_{retr_subset}=0"
                    missing_notes.append(f"{state}: no curve — {reason}")
                beh_str = f"{beh:.2f}" if beh is not None else "NA"
                titles.append(f"{state}: acc={beh_str}, n={n_used}")
            title_txt = f"K={K}, N={N}\n" + "  |  ".join(titles)
            if missing_notes:
                title_txt += "\n" + "; ".join(missing_notes)
            ax.set_title(title_txt, fontsize=8)
            ax.set_xlabel("layer")
            ax.set_ylim(0.35, 1.02)
            ax.grid(True, alpha=0.25)
        axes[0].set_ylabel(metric_label)
        legend_handles = [Line2D([0], [0], color=COLOR[s], lw=2, label=s) for s in STATES]
        fig.legend(handles=legend_handles, loc="upper left", ncol=2, fontsize=9,
                    frameon=False, bbox_to_anchor=(0.01, 1.0))
        fig.suptitle(f"{cond} — {metric_label} by layer, base vs LoRA (Qwen2.5-3B-Instruct)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        out_path = OUT_DIR / f"{fname_prefix}_{cond}.png"
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        print(f"Wrote {out_path}")

make_figure("corr", "correctness probe AUC", "correctness")


# ---------- 3. Retrieval probe: one figure per condition, 3 rows (all/wrong/correct) x 5 cells ----------
RETRIEVAL_SUBSETS = ["all", "wrong"]

def make_retrieval_figure():
    for cond in CONDITIONS:
        fig, axes = plt.subplots(len(RETRIEVAL_SUBSETS), len(CELLS),
                                   figsize=(4.2 * len(CELLS), 3.2 * len(RETRIEVAL_SUBSETS)),
                                   sharey=True, sharex=True)
        for row, subset in enumerate(RETRIEVAL_SUBSETS):
            for col, (K, N) in enumerate(CELLS):
                ax = axes[row, col]
                ax.axhline(0.5, color="#9a988e", linestyle="--", linewidth=1, zorder=0)
                titles = []
                missing_notes = []
                for state in STATES:
                    out = get_series(state, K, N, cond)
                    if out is None:
                        continue
                    layers, corr, retrieval, beh, n = out
                    series, n_used = retrieval[subset]
                    xs = [L for L, v in zip(layers, series) if v is not None]
                    ys = [v for v in series if v is not None]
                    if xs:
                        ax.plot(xs, ys, color=COLOR[state], linewidth=2,
                                 marker="o", markersize=2.2)
                    else:
                        n_disp = n_used if n_used is not None else 0
                        if n_disp == 0:
                            reason = f"n_{subset}=0"
                        else:
                            reason = f"n_{subset}={n_disp} < min_pairs=20"
                        missing_notes.append(f"{state}: no curve — {reason}")
                    beh_str = f"{beh:.2f}" if beh is not None else "NA"
                    titles.append(f"{state}: acc={beh_str}, n={n_used}")
                title_txt = ("K={}, N={}\n".format(K, N) if row == 0 else "") + "  |  ".join(titles)
                if missing_notes:
                    title_txt += "\n" + "; ".join(missing_notes)
                ax.set_title(title_txt, fontsize=7.5)
                ax.set_ylim(0.35, 1.02)
                ax.grid(True, alpha=0.25)
                if row == len(RETRIEVAL_SUBSETS) - 1:
                    ax.set_xlabel("layer")
            axes[row, 0].set_ylabel(f"retrieval AUC\n({subset} trials)")
        legend_handles = [Line2D([0], [0], color=COLOR[s], lw=2, label=s) for s in STATES]
        fig.legend(handles=legend_handles, loc="upper left", ncol=2, fontsize=9,
                    frameon=False, bbox_to_anchor=(0.01, 1.0))
        fig.suptitle(f"{cond} — retrieval probe AUC by layer, base vs LoRA "
                     f"(rows: {' / '.join(RETRIEVAL_SUBSETS)} trials) — Qwen2.5-3B-Instruct", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out_path = OUT_DIR / f"retrieval_{cond}.png"
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        print(f"Wrote {out_path}")

make_retrieval_figure()

# ---------- cleanup: remove the old one-subset-per-file retrieval images ----------
for subset in RETRIEVAL_SUBSETS:
    for cond in CONDITIONS:
        stale = OUT_DIR / f"retrieval_{subset}_{cond}.png"
        if stale.exists():
            stale.unlink()
            print(f"Removed stale {stale}")
