"""
Plot SAE feature analysis from Gemma Scope.

Usage:
    cd /Users/kanak.raj/workspace/hobby/research_work_ri
    .venv/bin/python v3/scripts/plotting/plot_sae_features.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.family": "serif",
})

_V3_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = _V3_DIR / "results" / "sae"
FIGURES_DIR = _V3_DIR / "figures"


def main():
    results_file = RESULTS_DIR / "gemma_scope_sae_analysis.json"
    if not results_file.exists():
        print(f"Results not found: {results_file}")
        return

    d = json.load(open(results_file))
    fa = d["feature_analysis"]
    layers = sorted([int(k) for k in fa.keys()])
    ri_acc = d["behavioral"]["ri_acc"]
    pi_acc = d["behavioral"]["pi_acc"]

    print(f"Behavioral: RI={ri_acc:.0%} PI={pi_acc:.0%} gap={ri_acc-pi_acc:+.0%}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, layer in enumerate(layers):
        ax = axes[idx // 2, idx % 2]
        info = fa[str(layer)]

        # Plot top RI-preferring vs PI-preferring feature strengths
        ri_pref = info["top_ri_preferring"]  # [(idx, diff), ...]
        pi_pref = info["top_pi_preferring"]

        if ri_pref and pi_pref:
            ri_diffs = [x[1] for x in ri_pref[:10]]
            pi_diffs = [x[1] for x in pi_pref[:10]]

            x = np.arange(10)
            width = 0.35
            ax.bar(x - width/2, ri_diffs, width, label="RI-preferring features",
                  color="#2196F3", alpha=0.8)
            ax.bar(x + width/2, pi_diffs, width, label="PI-preferring features",
                  color="#F44336", alpha=0.8)

            ax.set_xlabel("Feature rank", fontsize=10)
            ax.set_ylabel("Mean activation difference", fontsize=10)
            ax.set_title(
                f"Layer {layer} — RI vs PI discriminating features\n"
                f"(RI active: {info['ri_mean_active']:.1f}/token, "
                f"PI active: {info['pi_mean_active']:.1f}/token)",
                fontsize=11, fontweight="bold"
            )
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Gemma Scope 2 SAE Features: RI vs PI Asymmetry\n"
        f"Model: gemma-3-1b-it | RI={ri_acc:.0%} PI={pi_acc:.0%} (gap={ri_acc-pi_acc:+.0%})",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    save_path = FIGURES_DIR / "gemma_scope_sae_features.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

    # Feature summary table
    print("\n=== Feature Summary ===")
    print(f"{'Layer':>7} {'RI active':>10} {'PI active':>10} {'Top RI feat':>12} {'Top PI feat':>12}")
    print("-" * 55)
    for layer in layers:
        info = fa[str(layer)]
        ri_top = info["top_ri_preferring"][0][0] if info["top_ri_preferring"] else "—"
        pi_top = info["top_pi_preferring"][0][0] if info["top_pi_preferring"] else "—"
        print(f"{layer:>7} {info['ri_mean_active']:>10.1f} {info['pi_mean_active']:>10.1f} "
              f"{str(ri_top):>12} {str(pi_top):>12}")


if __name__ == "__main__":
    main()
