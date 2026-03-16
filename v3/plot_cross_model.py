"""
Cross-model comparison figures for the paper.

Generates multi-panel figures showing PI > RI consistency across models.

Usage:
    cd v3
    python plot_cross_model.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from collections import Counter

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "serif",
})

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"


def load_stage1(model_short):
    """Load Stage 1 results. Prefers trials file (has trial_details), falls back to sweep/checkpoint."""
    model_dir = RESULTS_DIR / model_short
    # Try trials file first (has trial_details for error position analysis)
    trials_files = sorted(model_dir.glob("stage1_trials_*.json"))
    if trials_files:
        return json.load(open(trials_files[-1]))
    # Fall back to checkpoint (also has trial_details)
    ckpt = model_dir / "stage1_checkpoint.json"
    if ckpt.exists():
        return json.load(open(ckpt))
    # Last resort: sweep summary (no trial_details)
    sweep_files = sorted(model_dir.glob("stage1_sweep_*.json"))
    if sweep_files:
        return json.load(open(sweep_files[-1]))
    return None


def load_stage2(model_short):
    model_dir = RESULTS_DIR / model_short
    files = sorted(model_dir.glob("stage2_logit_lens_*.json"))
    if files:
        return json.load(open(files[-1]))
    ckpt = model_dir / "stage2_checkpoint.json"
    if ckpt.exists():
        return json.load(open(ckpt))
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE A: PI Accuracy vs N (updates) — multi-model overlay
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pi_vs_n():
    """Plot PI and RI accuracy vs N for multiple models at fixed key count."""
    models = [
        ("Qwen2.5-0.5B-Instruct", "Qwen 0.5B", "#e41a1c"),
        ("Qwen2.5-1.5B-Instruct", "Qwen 1.5B", "#377eb8"),
        ("Qwen2.5-3B-Instruct", "Qwen 3B", "#4daf4a"),
        ("gemma-3-1b-it", "Gemma 1B", "#984ea3"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for model_short, label, color in models:
        s1 = load_stage1(model_short)
        if s1 is None:
            continue

        cells = s1["cells"]

        # Extract 2-key data
        ri_data, pi_data = {}, {}
        for ck, cell in cells.items():
            nk, nu = int(ck.split("_")[0]), int(ck.split("_")[1])
            if nk != 2:
                continue
            ri_data[nu] = cell["stats"]["RI"]["accuracy"]
            pi_data[nu] = cell["stats"]["PI"]["accuracy"]

        if not ri_data:
            continue

        updates = sorted(ri_data.keys())
        ri_vals = [ri_data[u] for u in updates]
        pi_vals = [pi_data[u] for u in updates]

        axes[0].plot(updates, ri_vals, color=color, linewidth=2, marker="o",
                     markersize=4, label=label, alpha=0.8)
        axes[1].plot(updates, pi_vals, color=color, linewidth=2, marker="s",
                     markersize=4, label=label, alpha=0.8)

    axes[0].set_title("RI Accuracy (ask for first)")
    axes[0].set_xlabel("Updates per key (N)")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)

    axes[1].set_title("PI Accuracy (ask for last)")
    axes[1].set_xlabel("Updates per key (N)")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)

    fig.suptitle("PI > RI Across Model Sizes and Architectures (2 keys)", fontsize=14, y=1.02)
    plt.tight_layout()
    path = FIGURES_DIR / "cross_model_pi_vs_n.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE B: Logit Lens Comparison — suppression across models
# ═══════════════════════════════════════════════════════════════════════════════

def plot_logit_lens_comparison():
    """Side-by-side P(v_last) and P(v_first) trajectories for all models."""
    models = [
        ("Qwen2.5-0.5B-Instruct", "Qwen 0.5B", 24),
        ("Qwen2.5-1.5B-Instruct", "Qwen 1.5B", 28),
        ("Qwen2.5-3B-Instruct", "Qwen 3B", 36),
    ]

    fig, axes = plt.subplots(2, len(models), figsize=(5 * len(models), 8))

    for col, (model_short, label, n_layers) in enumerate(models):
        s2 = load_stage2(model_short)
        if s2 is None:
            continue

        analyses = s2["analyses"]
        actual_n_layers = s2["n_layers"]

        # Find best operating point (most PI failures with reasonable data)
        from collections import defaultdict
        groups = defaultdict(list)
        for a in analyses:
            groups[(a["num_keys"], a["num_updates"], a["condition"])].append(a)

        best_point = None
        best_count = 0
        for (k, u, cond), items in groups.items():
            if cond == "PI":
                failures = [a for a in items if not a["correct"]
                           and a.get("total_value_prob", 0) > 0.05]
                if len(failures) > best_count:
                    best_count = len(failures)
                    best_point = (k, u)

        if best_point is None:
            continue

        k, u = best_point

        pi_failures = [a for a in analyses
                       if a["condition"] == "PI" and a["num_keys"] == k
                       and a["num_updates"] == u
                       and not a["correct"] and a.get("total_value_prob", 0) > 0.05]
        ri_correct = [a for a in analyses
                      if a["condition"] == "RI" and a["num_keys"] == k
                      and a["num_updates"] == u and a["correct"]]

        # Top row: PI failure trajectories
        ax = axes[0, col]
        if pi_failures:
            avg_last = np.mean([np.array(a["value_probs_by_layer"])[-1]
                               for a in pi_failures], axis=0)
            avg_penult = np.mean([np.array(a["value_probs_by_layer"])[-2]
                                 for a in pi_failures], axis=0)

            # Normalize to relative depth
            x = np.linspace(0, 1, actual_n_layers)
            ax.plot(x, avg_last, color="#4CAF50", linewidth=2.5,
                    label=f"P(v_last) correct")
            ax.plot(x, avg_penult, color="#FF5722", linewidth=2.5,
                    label=f"P(v_penult) competitor")

            peak = np.argmax(avg_last)
            ax.annotate(f"peak: {avg_last[peak]:.2f}",
                       xy=(x[peak], avg_last[peak]),
                       xytext=(x[peak] - 0.1, avg_last[peak] + 0.03),
                       fontsize=8, arrowprops=dict(arrowstyle="->", color="gray"))

        ax.set_title(f"{label} — PI failures ({len(pi_failures)})\n{k}k_{u}u",
                     fontsize=11)
        ax.set_xlabel("Relative depth (0=input, 1=output)")
        ax.set_ylabel("P(value)")
        ax.legend(fontsize=8)
        ax.set_xlim(0.6, 1.0)  # Zoom to active region
        ax.grid(True, alpha=0.3)

        # Bottom row: RI correct trajectories
        ax = axes[1, col]
        if ri_correct:
            avg_first = np.mean([np.array(a["value_probs_by_layer"])[0]
                                for a in ri_correct], axis=0)
            x = np.linspace(0, 1, actual_n_layers)
            ax.plot(x, avg_first, color="#4CAF50", linewidth=2.5,
                    label=f"P(v_first) correct")

        ax.set_title(f"{label} — RI correct ({len(ri_correct)})\n{k}k_{u}u",
                     fontsize=11)
        ax.set_xlabel("Relative depth")
        ax.set_ylabel("P(value)")
        ax.legend(fontsize=8)
        ax.set_xlim(0.6, 1.0)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Logit Lens: P(v_last) Suppressed vs P(v_first) Clean",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    path = FIGURES_DIR / "cross_model_logit_lens.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE C: Error Position Comparison — off-by-one across models
# ═══════════════════════════════════════════════════════════════════════════════

def plot_error_positions_comparison():
    """Bar charts of PI error positions for multiple models at matched N."""
    models = [
        ("Qwen2.5-0.5B-Instruct", "Qwen 0.5B"),
        ("Qwen2.5-1.5B-Instruct", "Qwen 1.5B"),
        ("Qwen2.5-3B-Instruct", "Qwen 3B"),
        ("gemma-3-1b-it", "Gemma 1B"),
    ]

    # Show at N=5 (low) and first available higher N
    target_configs = [
        (2, 5, "2k_5u (N=5)"),
        (2, 10, "2k_10u (N=10)"),
    ]

    fig, axes = plt.subplots(len(target_configs), len(models),
                             figsize=(4 * len(models), 4 * len(target_configs)))

    for row, (target_k, target_u, config_label) in enumerate(target_configs):
        for col, (model_short, model_label) in enumerate(models):
            ax = axes[row, col]

            s1 = load_stage1(model_short)
            if s1 is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                       transform=ax.transAxes)
                continue

            details = s1.get("trial_details", {})
            cell_key = f"{target_k}_{target_u}"
            if cell_key not in details:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                       transform=ax.transAxes)
                continue

            pi_trials = details[cell_key]["PI"]
            failures = [t for t in pi_trials if not t["correct"]
                       and t.get("predicted_idx") is not None]

            if not failures:
                ax.text(0.5, 0.5, "No failures", ha="center", va="center",
                       transform=ax.transAxes)
                continue

            n_vals = target_u
            idx_counts = Counter([t["predicted_idx"] for t in failures])
            positions = list(range(n_vals))
            counts = [idx_counts.get(p, 0) for p in positions]

            # Color: green for correct (last), red for first, blue for others
            colors = ["#2196F3"] * n_vals
            colors[-1] = "#4CAF50"
            colors[0] = "#FF5722"
            if n_vals >= 2:
                colors[-2] = "#FF9800"  # penultimate in orange

            ax.bar(positions, counts, color=colors, edgecolor="white", linewidth=0.5)
            ax.set_xlabel("Value position")
            ax.set_ylabel("Count")

            # Annotate penultimate %
            total = len(failures)
            penult_pct = idx_counts.get(n_vals - 2, 0) / total * 100 if total else 0
            ax.set_title(f"{model_label}\npenult: {penult_pct:.0f}% ({total} failures)")

            ax.set_xticks(positions)
            ax.set_xticklabels([f"v{i}" for i in positions], fontsize=7, rotation=45)

    # Row labels
    for row, (_, _, config_label) in enumerate(target_configs):
        axes[row, 0].set_ylabel(f"{config_label}\nCount", fontsize=10)

    fig.suptitle("PI Error Positions: Off-by-One Across Models",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    path = FIGURES_DIR / "cross_model_error_positions.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("Generating cross-model comparison figures...\n")
    plot_pi_vs_n()
    plot_logit_lens_comparison()
    plot_error_positions_comparison()
    print("\nDone!")
