"""
Plot training dynamics: PI > RI emergence across training steps.

Usage:
    cd /Users/kanak.raj/workspace/hobby/research_work_ri
    .venv/bin/python v3/scripts/plotting/plot_training_dynamics.py
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
RESULTS_DIR = _V3_DIR / "results" / "training_dynamics"
FIGURES_DIR = _V3_DIR / "figures"


def main():
    results_file = RESULTS_DIR / "smollm2_1.7b_dynamics.json"
    if not results_file.exists():
        print(f"Results not found: {results_file}")
        return

    d = json.load(open(results_file))
    results = d["results"]

    # Sort by step
    data = []
    for ckpt, r in sorted(results.items(),
                           key=lambda x: int(x[0].split("-")[1]) if x[0].startswith("step-") else 2e6):
        step = int(ckpt.split("-")[1]) if ckpt.startswith("step-") else 2000000
        data.append((step, r["ri_acc"], r["pi_acc"], r["gap"], r.get("garbage_rate", 0)))

    steps = [d[0] for d in data]
    ri_vals = [d[1] for d in data]
    pi_vals = [d[2] for d in data]
    gaps = [d[3] for d in data]
    garb = [d[4] for d in data]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: RI and PI accuracy over training
    ax = axes[0]
    ax.plot(steps, ri_vals, "o-", color="#2196F3", linewidth=2.5, markersize=7,
            label="RI (retrieve first)")
    ax.plot(steps, pi_vals, "s-", color="#F44336", linewidth=2.5, markersize=7,
            label="PI (retrieve last)")
    ax.fill_between(steps, pi_vals, ri_vals, alpha=0.1, color="gray", label="PI > RI gap")
    ax.set_xlabel("Training steps", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("(a) RI and PI accuracy over training", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    # Annotate first checkpoint
    if data:
        ax.annotate(f"Step {steps[0]//1000}K\ngap={gaps[0]:+.0%}",
                   xy=(steps[0], pi_vals[0]),
                   xytext=(steps[0] + (steps[-1]-steps[0])*0.05, pi_vals[0] - 0.15),
                   fontsize=8, arrowprops=dict(arrowstyle="->", color="gray"))

    # Panel 2: Gap over training
    ax = axes[1]
    colors = ["#F44336" if g > 0.1 else "#FF9800" if g > 0 else "#4CAF50" for g in gaps]
    ax.bar(range(len(steps)), gaps, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.axhline(y=0.1, color="gray", linestyle="--", alpha=0.5, label="gap=10%")
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([f"{s//1000}K" for s in steps], rotation=45, fontsize=8)
    ax.set_xlabel("Training checkpoint", fontsize=11)
    ax.set_ylabel("PI > RI Gap (RI - PI)", fontsize=11)
    ax.set_title("(b) Gap magnitude across training", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: Garbage rate
    ax = axes[2]
    ax.plot(steps, [g * 100 for g in garb], "D-", color="#9C27B0", linewidth=2, markersize=7)
    ax.set_xlabel("Training steps", fontsize=11)
    ax.set_ylabel("Garbage rate (%)", fontsize=11)
    ax.set_title("(c) Instruction-following quality", fontsize=13, fontweight="bold")
    ax.set_ylim(-2, 100)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    # Determine conclusion
    if data:
        first_gap = gaps[0]
        last_gap = gaps[-1]
        if first_gap > 0.1:
            conclusion = "PI > RI exists from earliest checkpoint → ARCHITECTURAL"
        elif last_gap > 0.1 and first_gap < 0.05:
            conclusion = "PI > RI emerges during training → LEARNED"
        else:
            conclusion = "Pattern unclear (check garbage rates)"

        fig.suptitle(
            f"Training Dynamics: SmolLM2-1.7B\n{conclusion}",
            fontsize=14, fontweight="bold", y=1.02
        )

    plt.tight_layout()
    save_path = FIGURES_DIR / "training_dynamics_smollm2.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

    # Print summary
    print("\n=== Summary ===")
    print(f"{'Step':>12} {'RI':>6} {'PI':>6} {'Gap':>7} {'Garb':>6}")
    print("-" * 45)
    for step, ri, pi, gap, gb in data:
        print(f"{step:>12,} {ri:>5.0%} {pi:>5.0%} {gap:>+6.0%} {gb:>5.0%}")


if __name__ == "__main__":
    main()
