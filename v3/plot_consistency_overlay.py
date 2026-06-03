"""
Cross-config consistency overlay.

Loads multiple attention_routing JSONs for the same model and overlays
the per-layer "attention to round 0" (primacy attention) and
"attention to last round" (recency attention) curves.

If the curves collapse across configs that span normal → reversal behaviorally,
the attention routing is invariant to operating zone — i.e., the behavioral
flip is not driven by changes in routing.

Usage:
    .venv/bin/python v3/plot_consistency_overlay.py \\
        --results v3/results_vllm/attention_routing/Qwen2.5-3B-Instruct__*.json \\
        --out v3/plots/attention_routing/consistency_overlay.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_ROOT = _SCRIPT_DIR / "plots" / "attention_routing"


def per_layer_round_attn(data, source: str, query: str, round_idx: int):
    """Return mean per-layer attention to a specific round, averaged over heads + trials."""
    attn_key = f"attn_from_{'gen' if source == 'gen' else 'qcat'}"
    layer_vals = []
    for t in data["trials"]:
        arr = np.asarray(t[f"expA_{query}"][attn_key], dtype=np.float32)  # [L, H, R, 2]
        per_round = arr.sum(axis=-1).mean(axis=1)  # [L, R] head-avg
        # Normalize per layer (fraction of tracked-round attention)
        row_sums = per_round.sum(axis=-1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        normed = per_round / row_sums
        # round_idx: 0 = first, -1 = last
        layer_vals.append(normed[:, round_idx])
    stacked = np.stack(layer_vals)  # [trials, L]
    return stacked.mean(0), stacked.std(0) / np.sqrt(stacked.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", default=str(PLOTS_ROOT / "consistency_overlay.png"))
    args = ap.parse_args()

    configs = []
    for path in args.results:
        p = Path(path)
        d = json.loads(p.read_text())
        # Distinguish models when present
        models_in_set = {json.loads(Path(x).read_text())["model"] for x in args.results}
        if len(models_in_set) > 1:
            label = f"{d['model']}  (K={d['K']}, N={d['N']}, {d['mode']})"
        else:
            label = f"K={d['K']}, N={d['N']}  ({d['mode']})"
        configs.append((label, d))

    # 4 panels: (gen / qcat) × (round 0 / round last) for CVQ
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    sources = ("gen", "qcat")
    round_choice = (("round 0 (primacy)", 0), ("last round (recency)", -1))

    for i, src in enumerate(sources):
        for j, (round_label, ridx) in enumerate(round_choice):
            ax = axes[i, j]
            for label, data in configs:
                mean, sem = per_layer_round_attn(data, src, "CVQ", ridx)
                ax.plot(mean, label=label, marker=".", markersize=3)
                ax.fill_between(np.arange(len(mean)), mean - sem, mean + sem, alpha=0.15)
            src_label = "generation token" if src == "gen" else "query-cat token"
            ax.set_xlabel("layer")
            ax.set_ylabel("normalized attention to chosen round")
            ax.set_title(f"{src_label} → {round_label}  (CVQ)", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

    fig.suptitle(
        "Consistency: attention routing across configs spanning normal → reversal\n"
        "If curves collapse, attention routing is invariant to operating zone",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
