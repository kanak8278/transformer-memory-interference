"""
Plot attention routing results produced by v3/attention_routing.py.

Three experiments per (model, mode):

  Exp A — layer × round heatmaps, 4 panels (gen source / qcat source) × (FVQ / CVQ).
          Shows at which depth the model routes to which round, contrasting FVQ vs CVQ
          and the two source tokens.

  Exp B — incremental N, line plot. Source: generation token. Query: CVQ only.
          Two lines: fraction of attention going to round 1 vs the most recent round.
          As N grows: does attention shift to recency or stick on primacy?

  Exp C — incremental N, heatmaps. Source: query-cat. One panel per query (FVQ / CVQ).
          y-axis: n_updates checkpoint; x-axis: round index.
          Shows the "composition" of the query-key — which value it binds to as
          updates accumulate.

Usage:
    .venv/bin/python v3/plot_attention_routing.py \
        --results v3/results_vllm/attention_routing/Qwen2.5-3B-Instruct__normal.json
    .venv/bin/python v3/plot_attention_routing.py --all
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
# DATA LOADING + AGGREGATION
# ═════════════════════════════════════════════════════════════════════════════

def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def trial_round_attn(trial_entry, key: str) -> np.ndarray:
    """Return per-trial total attention per (layer, head, round): sum of key_span_attn + value_attn.
    Shape: [layers, heads, rounds].
    """
    arr = np.asarray(trial_entry[key], dtype=np.float32)  # [L, H, R, 2]
    return arr.sum(axis=-1)  # combine key-span attn + value attn


def aggregate_expA(data: dict, source: str, query: str):
    """Return mean per-layer attention to each round (averaged over heads, then trials).

    source: 'gen' or 'qcat'
    query: 'FVQ' or 'CVQ'

    Returns: dict with:
        per_layer_round: [L, R] - mean attention per layer/round (head-averaged, trial-averaged)
        per_layer_total: [L]    - mean total attention to tracked rounds per layer
        n_trials: int
    """
    attn_key = f"attn_from_{'gen' if source == 'gen' else 'qcat'}"
    accum = []
    for t in data["trials"]:
        entry = t[f"expA_{query}"]
        per_round = trial_round_attn(entry, attn_key)  # [L, H, R]
        accum.append(per_round.mean(axis=1))  # head-average → [L, R]
    stacked = np.stack(accum)  # [n_trials, L, R]
    per_layer_round = stacked.mean(axis=0)  # [L, R]
    per_layer_total = per_layer_round.sum(axis=-1)  # [L]
    return {
        "per_layer_round": per_layer_round,
        "per_layer_total": per_layer_total,
        "n_trials": len(accum),
    }


def aggregate_incr(data: dict, source: str, query: str):
    """Per-checkpoint per-round attention from incremental passes (final layer only).

    Returns: dict with:
        checkpoints: list of n_updates values
        per_ckpt_round: dict {n_updates: [R]} attention to each round at that ckpt
        n_trials: int
    """
    attn_key = f"attn_from_{'gen' if source == 'gen' else 'qcat'}"
    ckpts_to_round_attn: dict[int, list[np.ndarray]] = {}
    for t in data["trials"]:
        for entry in t[f"incr_{query}"]:
            n_steps = entry["n_updates"]
            arr = np.asarray(entry[attn_key], dtype=np.float32)  # [1, H, R, 2]
            per_round = arr.sum(axis=-1).mean(axis=(0, 1))  # average over (final layer dim=1, heads); → [R]
            ckpts_to_round_attn.setdefault(n_steps, []).append(per_round)
    out = {n: np.stack(arrs) for n, arrs in ckpts_to_round_attn.items()}
    return {
        "checkpoints": sorted(out.keys()),
        "per_ckpt_round": {n: arrs.mean(axis=0) for n, arrs in out.items()},
        "per_ckpt_round_sem": {n: arrs.std(axis=0) / np.sqrt(arrs.shape[0]) for n, arrs in out.items()},
        "n_trials": min(arrs.shape[0] for arrs in out.values()),
    }


# ═════════════════════════════════════════════════════════════════════════════
# EXP A — HEATMAPS
# ═════════════════════════════════════════════════════════════════════════════

def plot_expA(data: dict, out_dir: Path):
    """4 heatmaps: (gen / qcat) × (FVQ / CVQ).
    Each heatmap: y=layer, x=round, color=mean attention (per-layer normalized).
    """
    model = data["model"]
    mode = data["mode"]
    K, N = data["K"], data["N"]
    n_trials = data["n_trials_completed"]

    aggs = {}
    for src in ("gen", "qcat"):
        for q in ("FVQ", "CVQ"):
            aggs[(src, q)] = aggregate_expA(data, src, q)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sources = ("gen", "qcat")
    queries = ("FVQ", "CVQ")

    for i, src in enumerate(sources):
        for j, q in enumerate(queries):
            ax = axes[i, j]
            mat = aggs[(src, q)]["per_layer_round"]  # [L, R]
            # Per-layer normalize so each row sums to 1 (relative routing across rounds)
            row_sums = mat.sum(axis=-1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            normed = mat / row_sums

            im = ax.imshow(normed, aspect="auto", cmap="viridis", origin="lower",
                           vmin=0, vmax=normed.max())
            ax.set_xlabel("round index (1..N, in stream order)")
            ax.set_ylabel("layer")
            src_label = "generation token" if src == "gen" else "query-cat token"
            ax.set_title(f"{src_label}, {q}  (per-layer normalized)", fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Exp A — Attention routing by depth\n"
        f"{model} / {mode}  (K={K}, N={N}, trials={n_trials})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = out_dir / f"expA_{model}__{mode}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")

    # ─── Companion plot: per-layer total attention to tracked rounds ────────
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for (src, q), agg in aggs.items():
        label = f"{src}, {q}"
        ax2.plot(agg["per_layer_total"], label=label, marker=".", markersize=3)
    ax2.set_xlabel("layer")
    ax2.set_ylabel("total attention to test-category rounds  (head-avg)")
    ax2.set_title(
        f"Exp A — total attention budget per layer\n"
        f"{model} / {mode}  (K={K}, N={N}, trials={n_trials})",
        fontsize=10,
    )
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    out_path2 = out_dir / f"expA_{model}__{mode}__budget.png"
    fig2.savefig(out_path2, dpi=150)
    plt.close(fig2)
    print(f"  saved {out_path2}")


# ═════════════════════════════════════════════════════════════════════════════
# EXP B — INCREMENTAL N FROM GEN TOKEN (CVQ)
# ═════════════════════════════════════════════════════════════════════════════

def plot_expB(data: dict, out_dir: Path):
    """Line plot: x = n_updates, y = attention fraction.
    Two lines: fraction to round 1, fraction to most recent round.
    Source = gen token, query = CVQ (failing case).
    """
    model = data["model"]
    mode = data["mode"]
    K, N = data["K"], data["N"]

    agg = aggregate_incr(data, source="gen", query="CVQ")
    ckpts = agg["checkpoints"]

    # Normalize per checkpoint so attention to tracked rounds sums to 1
    frac_first = []
    frac_last = []
    sem_first = []
    sem_last = []
    for n in ckpts:
        row = agg["per_ckpt_round"][n]  # [R]
        sem_row = agg["per_ckpt_round_sem"][n]
        total = row.sum() if row.sum() > 0 else 1.0
        frac_first.append(row[0] / total)
        frac_last.append(row[-1] / total)
        sem_first.append(sem_row[0] / total)
        sem_last.append(sem_row[-1] / total)
    frac_first = np.array(frac_first)
    frac_last = np.array(frac_last)
    sem_first = np.array(sem_first)
    sem_last = np.array(sem_last)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Normalized: fraction of tracked-round attention
    ax = axes[0]
    ax.errorbar(ckpts, frac_first, yerr=sem_first, marker="o", capsize=3,
                label="round 1 (primacy)")
    ax.errorbar(ckpts, frac_last, yerr=sem_last, marker="s", capsize=3,
                label="most recent round (recency)")
    ax.axhline(1 / np.array(ckpts).max(), ls=":", c="gray", alpha=0.5)
    ax.set_xlabel("n_updates accumulated")
    ax.set_ylabel("attention fraction (of test-cat rounds)")
    ax.set_title(f"normalized — fraction of test-cat attention\n"
                 f"(dashed = uniform 1/N at largest checkpoint)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xscale("symlog")

    # Raw attention
    ax2 = axes[1]
    raw_first = [agg["per_ckpt_round"][n][0] for n in ckpts]
    raw_last = [agg["per_ckpt_round"][n][-1] for n in ckpts]
    raw_first_sem = [agg["per_ckpt_round_sem"][n][0] for n in ckpts]
    raw_last_sem = [agg["per_ckpt_round_sem"][n][-1] for n in ckpts]
    ax2.errorbar(ckpts, raw_first, yerr=raw_first_sem, marker="o", capsize=3,
                 label="round 1 (raw attn)")
    ax2.errorbar(ckpts, raw_last, yerr=raw_last_sem, marker="s", capsize=3,
                 label="most recent round (raw attn)")
    ax2.set_xlabel("n_updates accumulated")
    ax2.set_ylabel("raw attention weight  (head-avg, final layer)")
    ax2.set_title("raw — absolute attention weights", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xscale("symlog")

    fig.suptitle(
        f"Exp B — gen-token attention, CVQ, final layer\n"
        f"{model} / {mode}  (K={K}, N={N}, trials={agg['n_trials']})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path = out_dir / f"expB_{model}__{mode}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# EXP C — INCREMENTAL N FROM QUERY-CAT (FVQ + CVQ)
# ═════════════════════════════════════════════════════════════════════════════

def plot_expC(data: dict, out_dir: Path):
    """Heatmap: y = n_updates checkpoint, x = round index.
    Source = query-cat token, one panel per query.
    """
    model = data["model"]
    mode = data["mode"]
    K, N = data["K"], data["N"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for col, q in enumerate(("FVQ", "CVQ")):
        agg = aggregate_incr(data, source="qcat", query=q)
        ckpts = agg["checkpoints"]
        max_rounds = max(len(agg["per_ckpt_round"][n]) for n in ckpts)

        mat = np.full((len(ckpts), max_rounds), np.nan)
        for i, n in enumerate(ckpts):
            row = agg["per_ckpt_round"][n]
            total = row.sum() if row.sum() > 0 else 1.0
            mat[i, : len(row)] = row / total  # normalize per checkpoint

        ax = axes[col]
        im = ax.imshow(mat, aspect="auto", cmap="viridis", origin="lower",
                       vmin=0, vmax=np.nanmax(mat))
        ax.set_yticks(range(len(ckpts)))
        ax.set_yticklabels(ckpts)
        ax.set_xlabel("round index (1..n_updates)")
        ax.set_ylabel("n_updates accumulated")
        ax.set_title(f"query-cat → values, {q}", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Exp C — query-key 'composition' across accumulating updates  (final layer)\n"
        f"{model} / {mode}  (K={K}, N={N}, trials={agg['n_trials']})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path = out_dir / f"expC_{model}__{mode}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# DRIVER
# ═════════════════════════════════════════════════════════════════════════════

def run_one(path: Path):
    data = load_results(path)
    model = data["model"]
    mode = data["mode"]
    K, N = data["K"], data["N"]
    # Use the JSON filename stem as the output dir — disambiguates K/N overrides
    label = path.stem
    print(f"\n══ Plotting  {label}  ({data['n_trials_completed']} trials, K={K}/N={N}) ══")

    out_dir = PLOTS_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_expA(data, out_dir)
    plot_expB(data, out_dir)
    plot_expC(data, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="*", help="One or more result JSON paths")
    ap.add_argument("--all", action="store_true",
                    help="Plot all JSONs under v3/results_vllm/attention_routing/")
    ap.add_argument("--smoke", action="store_true",
                    help="Plot the smoke run only")
    args = ap.parse_args()

    if args.smoke:
        paths = list((RESULTS_ROOT / "smoke").glob("*.json"))
    elif args.all:
        paths = sorted(p for p in RESULTS_ROOT.glob("*.json"))
    else:
        paths = [Path(p) for p in (args.results or [])]

    if not paths:
        raise SystemExit("No result JSONs provided. Use --results PATH, --all, or --smoke")

    for p in paths:
        run_one(p)


if __name__ == "__main__":
    main()
