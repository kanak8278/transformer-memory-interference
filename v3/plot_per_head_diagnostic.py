"""
Per-head diagnostic plots — surface what head-averaging hides.

Three questions we try to answer here:

1. Are there specialist heads (some attend to round 0, others to round N-1)?
   → Plot: per-layer histogram of per-head argmax-round, plus a
     primacy-vs-recency scatter at selected layers.

2. Does any head distinguish FVQ from CVQ, even if head-averaged routing doesn't?
   → Plot: FVQ - CVQ attention difference heatmap [layers × heads] for
     round 0 (primacy) and last round (recency).

3. Which heads/layers carry the most query-discriminating signal?
   → Same FVQ-CVQ plot; large |delta| cells are the suspects.

Usage:
    .venv/bin/python v3/plot_per_head_diagnostic.py \
        --results v3/results_vllm/attention_routing/Qwen2.5-3B-Instruct__normal.json
    .venv/bin/python v3/plot_per_head_diagnostic.py --all
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = _SCRIPT_DIR / "results_vllm" / "attention_routing"
PLOTS_ROOT = _SCRIPT_DIR / "plots" / "attention_routing"


# ═════════════════════════════════════════════════════════════════════════════
# DATA AGGREGATION  (per-head, no head averaging)
# ═════════════════════════════════════════════════════════════════════════════

def aggregate_per_head(data: dict, source: str, query: str) -> np.ndarray:
    """Mean per-head attention to each round, averaged across trials.

    Returns shape: [n_layers, n_heads, n_rounds]
    """
    attn_key = f"attn_from_{'gen' if source == 'gen' else 'qcat'}"
    per_trial = []
    for t in data["trials"]:
        arr = np.asarray(t[f"expA_{query}"][attn_key], dtype=np.float32)  # [L, H, R, 2]
        per_trial.append(arr.sum(axis=-1))  # combine key_span + value attn → [L, H, R]
    return np.stack(per_trial).mean(axis=0)  # [L, H, R]


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 1 — FVQ - CVQ DIFFERENCE HEATMAP (layers × heads)
# ═════════════════════════════════════════════════════════════════════════════

def plot_fvq_cvq_diff(data: dict, out_dir: Path):
    """Per-head, per-layer FVQ-CVQ attention difference at:
       - round 0 (primacy) and
       - last round (recency)
    Two source tokens × two round-targets = 4 panels.

    Large positive cells = FVQ attends more than CVQ at that (layer, head, target).
    Large negative cells = CVQ attends more than FVQ.
    Cells near zero = no FVQ/CVQ distinction at that head — head-averaging would also miss it.
    """
    model = data["model"]
    mode = data["mode"]
    K, N = data["K"], data["N"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for i, src in enumerate(("gen", "qcat")):
        attn_fvq = aggregate_per_head(data, src, "FVQ")  # [L, H, R]
        attn_cvq = aggregate_per_head(data, src, "CVQ")
        diff = attn_fvq - attn_cvq  # [L, H, R]

        for j, (round_label, ridx) in enumerate((("round 0 (primacy)", 0), ("last round (recency)", -1))):
            ax = axes[i, j]
            m = diff[:, :, ridx]  # [L, H]
            vmax = np.abs(m).max()
            im = ax.imshow(m, aspect="auto", cmap="RdBu_r", origin="lower",
                           vmin=-vmax, vmax=vmax)
            src_label = "generation token" if src == "gen" else "query-cat token"
            ax.set_title(f"{src_label} → {round_label}\n(FVQ - CVQ attn)", fontsize=10)
            ax.set_xlabel("head")
            ax.set_ylabel("layer")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"FVQ - CVQ attention difference per head\n"
        f"{model} / {mode}  (K={K}, N={N}, trials={data['n_trials_completed']})\n"
        f"Cells near zero ⇒ no head differentiates between queries",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path = out_dir / f"perhead_fvq_minus_cvq.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 2 — PRIMACY-VS-RECENCY SCATTER per head, at selected layers
# ═════════════════════════════════════════════════════════════════════════════

def plot_primacy_recency_scatter(data: dict, out_dir: Path):
    """For each head at selected layers: x = attn to round 0, y = attn to last round.
    Overlay FVQ (circles) and CVQ (squares) per head.

    Heads in the bottom-right corner = pure primacy specialists.
    Heads in the top-left corner = pure recency specialists.
    Heads on the diagonal = uniform / no preference.
    """
    model = data["model"]
    mode = data["mode"]
    K, N = data["K"], data["N"]

    # Aggregate
    fvq_qcat = aggregate_per_head(data, "qcat", "FVQ")  # [L, H, R]
    cvq_qcat = aggregate_per_head(data, "qcat", "CVQ")
    fvq_gen = aggregate_per_head(data, "gen", "FVQ")
    cvq_gen = aggregate_per_head(data, "gen", "CVQ")

    n_layers = fvq_qcat.shape[0]
    # Pick 4 representative layers: early, mid, late, final
    layer_ids = [
        n_layers // 8,        # early
        n_layers // 2,        # mid
        int(n_layers * 0.75), # late
        n_layers - 1,         # final
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    sources = (("query-cat", fvq_qcat, cvq_qcat), ("gen", fvq_gen, cvq_gen))

    for i, (src_label, fvq_arr, cvq_arr) in enumerate(sources):
        for j, L in enumerate(layer_ids):
            ax = axes[i, j]
            # round 0 vs round N-1
            x_fvq = fvq_arr[L, :, 0]
            y_fvq = fvq_arr[L, :, -1]
            x_cvq = cvq_arr[L, :, 0]
            y_cvq = cvq_arr[L, :, -1]
            ax.scatter(x_fvq, y_fvq, label="FVQ", marker="o", alpha=0.7)
            ax.scatter(x_cvq, y_cvq, label="CVQ", marker="s", alpha=0.7)
            # Diagonal reference
            lim = max(x_fvq.max(), y_fvq.max(), x_cvq.max(), y_cvq.max()) * 1.05
            ax.plot([0, lim], [0, lim], "k--", alpha=0.3, lw=0.7)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_xlabel("attn to round 0 (primacy)")
            ax.set_ylabel("attn to last round (recency)")
            ax.set_title(f"{src_label}, layer {L}", fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

    fig.suptitle(
        f"Per-head primacy vs recency scatter at 4 layer depths\n"
        f"{model} / {mode}  (K={K}, N={N}, trials={data['n_trials_completed']})\n"
        f"Bottom-right = primacy specialist, top-left = recency specialist",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path = out_dir / "perhead_primacy_recency_scatter.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 3 — PER-HEAD ARGMAX ROUND DISTRIBUTION ACROSS LAYERS
# ═════════════════════════════════════════════════════════════════════════════

def plot_argmax_distribution(data: dict, out_dir: Path):
    """For each (layer, head), what round does it attend to most?
    Heatmap: x=layer, y=head, color=argmax round.

    Source: query-cat, query: CVQ (we already know CVQ is the failing case).
    """
    model = data["model"]
    mode = data["mode"]
    K, N = data["K"], data["N"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax_idx, src in enumerate(("gen", "qcat")):
        attn_cvq = aggregate_per_head(data, src, "CVQ")  # [L, H, R]
        argmax_round = attn_cvq.argmax(axis=-1)  # [L, H]

        ax = axes[ax_idx]
        # Use a colormap that highlights extremes (round 0, mid, round N-1)
        im = ax.imshow(argmax_round.T, aspect="auto", cmap="twilight_shifted", origin="lower",
                       vmin=0, vmax=attn_cvq.shape[-1] - 1)
        ax.set_xlabel("layer")
        ax.set_ylabel("head")
        src_label = "generation token" if src == "gen" else "query-cat token"
        ax.set_title(f"{src_label}, CVQ  (argmax round per head per layer)", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="argmax round")

    fig.suptitle(
        f"Per-head argmax round, CVQ — which round each head attends to most\n"
        f"{model} / {mode}  (K={K}, N={N}, trials={data['n_trials_completed']})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path = out_dir / "perhead_argmax_round.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# DRIVER
# ═════════════════════════════════════════════════════════════════════════════

def run_one(path: Path):
    data = json.loads(path.read_text())
    label = path.stem
    print(f"\n══ Per-head diagnostics  {label}  ({data['n_trials_completed']} trials) ══")
    out_dir = PLOTS_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_fvq_cvq_diff(data, out_dir)
    plot_primacy_recency_scatter(data, out_dir)
    plot_argmax_distribution(data, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="*", help="One or more result JSON paths")
    ap.add_argument("--all", action="store_true",
                    help="Plot all JSONs under v3/results_vllm/attention_routing/")
    args = ap.parse_args()

    if args.all:
        paths = sorted(RESULTS_ROOT.glob("*.json"))
    else:
        paths = [Path(p) for p in (args.results or [])]
    if not paths:
        raise SystemExit("No result JSONs provided. Use --results PATH or --all")

    for p in paths:
        run_one(p)


if __name__ == "__main__":
    main()
