"""
V3 Paper Figures from Stage 1 + Stage 2 results.

Usage:
    cd v3
    python plot_results.py --model Qwen2.5-3B-Instruct

Generates figures in v3/figures/{model}/
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from collections import Counter

matplotlib.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"


def load_stage1(model_short):
    """Load latest Stage 1 results."""
    model_dir = RESULTS_DIR / model_short
    # Try timestamped first, then checkpoint
    trials_files = sorted(model_dir.glob("stage1_trials_*.json"))
    if trials_files:
        return json.load(open(trials_files[-1]))
    ckpt = model_dir / "stage1_checkpoint.json"
    if ckpt.exists():
        return json.load(open(ckpt))
    raise FileNotFoundError(f"No Stage 1 results in {model_dir}")


def load_stage2(model_short):
    """Load latest Stage 2 results."""
    model_dir = RESULTS_DIR / model_short
    files = sorted(model_dir.glob("stage2_logit_lens_*.json"))
    if files:
        return json.load(open(files[-1]))
    ckpt = model_dir / "stage2_checkpoint.json"
    if ckpt.exists():
        return json.load(open(ckpt))
    raise FileNotFoundError(f"No Stage 2 results in {model_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Regime Map Heatmap (Stage 1)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_regime_map(s1, model_short, fig_dir):
    cells = s1["cells"]
    kl = sorted(set(int(ck.split("_")[0]) for ck in cells))
    ul = sorted(set(int(ck.split("_")[1]) for ck in cells))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (label, key) in zip(axes, [("RI Accuracy", "RI"), ("PI Accuracy", "PI"), ("Gap (RI - PI)", None)]):
        grid = np.full((len(kl), len(ul)), np.nan)
        for i, nk in enumerate(kl):
            for j, nu in enumerate(ul):
                ck = f"{nk}_{nu}"
                if ck in cells:
                    ri = cells[ck]["stats"]["RI"]["accuracy"]
                    pi = cells[ck]["stats"]["PI"]["accuracy"]
                    if key == "RI":
                        grid[i, j] = ri
                    elif key == "PI":
                        grid[i, j] = pi
                    else:
                        grid[i, j] = ri - pi

        if key is None:
            im = ax.imshow(grid, cmap="RdYlGn", vmin=-0.3, vmax=0.7, aspect="auto")
        else:
            im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

        ax.set_xticks(range(len(ul)))
        ax.set_xticklabels(ul, rotation=45)
        ax.set_yticks(range(len(kl)))
        ax.set_yticklabels(kl)
        ax.set_xlabel("Updates per key")
        ax.set_ylabel("Number of keys")
        ax.set_title(label)
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Annotate
        for i in range(len(kl)):
            for j in range(len(ul)):
                if not np.isnan(grid[i, j]):
                    val = grid[i, j]
                    color = "white" if val < 0.3 or val > 0.7 else "black"
                    ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                            fontsize=8, color=color)

    fig.suptitle(f"Stage 1: Behavioral Sweep — {model_short}", fontsize=16, y=1.02)
    plt.tight_layout()
    path = fig_dir / "fig1_regime_map.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: PI Failure Position Distribution (Stage 1)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_failure_positions(s1, model_short, fig_dir):
    details = s1.get("trial_details", {})
    cells = s1["cells"]

    # Pick regime B/C cells with enough failures
    interesting = {ck: c for ck, c in cells.items()
                   if c["stats"]["PI"]["n_failures"] > 20
                   and c["stats"]["RI"]["accuracy"] > 0.5}

    if not interesting:
        print("  No interesting cells for failure position plot")
        return

    n_plots = min(6, len(interesting))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, ck in enumerate(sorted(interesting.keys(),
                                     key=lambda x: (int(x.split("_")[0]), int(x.split("_")[1])))[:n_plots]):
        ax = axes[idx]
        if ck not in details:
            continue

        cell = cells[ck]
        pi_trials = details[ck]["PI"]
        n_updates = cell["num_updates"]

        pos_counts = Counter()
        n_garbage = 0
        for t in pi_trials:
            # Skip OOM/empty — infrastructure failures, not model behavior
            if t.get("error_type") in ("oom", "empty"):
                continue
            if t["predicted"].strip() == "":
                continue  # old data without oom/empty classification
            if t["predicted_idx"] is not None:
                pos_counts[t["predicted_idx"]] += 1
            elif not t["correct"]:
                n_garbage += 1

        positions = list(range(n_updates))
        counts = [pos_counts.get(p, 0) for p in positions]
        n_total = len(pi_trials)

        colors = ["#2196F3"] * n_updates
        colors[-1] = "#4CAF50"  # correct answer (last) in green
        colors[0] = "#FF5722"   # first value in red

        ax.bar(positions, counts, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(y=0, color="black", linewidth=0.5)

        # Add garbage bar
        ax.bar(n_updates, n_garbage, color="#9E9E9E", edgecolor="white", linewidth=0.5)

        labels = [f"v{i}" for i in positions] + ["garb"]
        ax.set_xticks(list(range(n_updates + 1)))
        ax.set_xticklabels(labels, fontsize=8, rotation=45)
        ax.set_ylabel("Count")
        ax.set_title(f"{ck} (RI={cell['stats']['RI']['accuracy']:.0%}, "
                     f"PI={cell['stats']['PI']['accuracy']:.0%})")

    # Hide unused axes
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"PI Failure Position Distribution — {model_short}", fontsize=16, y=1.02)
    plt.tight_layout()
    path = fig_dir / "fig2_failure_positions.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: RI vs PI Layer Trajectories (Stage 2) — the core figure
# ═══════════════════════════════════════════════════════════════════════════════

def plot_layer_trajectories(s2, model_short, fig_dir):
    points = s2["points"]
    analyses = s2["analyses"]
    n_layers = s2["n_layers"]

    n_points = len(points)
    # Two rows: full view + zoomed to active region
    fig, axes = plt.subplots(3, n_points, figsize=(6 * n_points, 15))
    if n_points == 1:
        axes = axes.reshape(3, 1)

    for col, pt in enumerate(points):
        k, u = pt["keys"], pt["updates"]

        # Gather data for both conditions
        pi_trials = [a for a in analyses
                     if a["condition"] == "PI" and a["num_keys"] == k
                     and a["num_updates"] == u
                     and not a["correct"] and a["total_value_prob"] > 0.05]
        ri_trials = [a for a in analyses
                     if a["condition"] == "RI" and a["num_keys"] == k
                     and a["num_updates"] == u and a["correct"]]

        for row, (trials, cond, title_prefix) in enumerate([
            (ri_trials, "RI", "RI (ask first)"),
            (pi_trials, "PI", "PI (ask last) — full"),
            (pi_trials, "PI_zoom", "PI (ask last) — zoomed to active layers"),
        ]):
            ax = axes[row, col]
            label_suffix = f" ({len(trials)} {'correct' if cond == 'RI' else 'failures'})"

            if not trials:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue

            n_values = len(trials[0]["all_values"])
            avg = np.mean([a["value_probs_by_layer"] for a in trials], axis=0)

            # Find active region: first layer where any value > 0.01
            max_per_layer = avg.max(axis=0)
            active_start = 0
            for L in range(n_layers):
                if max_per_layer[L] > 0.01:
                    active_start = max(0, L - 2)
                    break

            if cond == "PI_zoom":
                layer_range = range(active_start, n_layers)
                cond_for_logic = "PI"
            else:
                layer_range = range(n_layers)
                cond_for_logic = cond

            layers = list(layer_range)
            correct_idx = n_values - 1 if cond_for_logic == "PI" else 0

            # Find which wrong value is dominant at the final layer
            last_probs = avg[:, -1].copy()
            last_probs_wrong = last_probs.copy()
            last_probs_wrong[correct_idx] = -1
            dominant_wrong = np.argmax(last_probs_wrong)

            # Plot each value
            cmap = plt.cm.coolwarm
            for vi in range(n_values):
                color = cmap(vi / max(n_values - 1, 1))

                is_correct = vi == correct_idx
                is_dominant_wrong = vi == dominant_wrong

                if is_correct:
                    lw, ls, alpha = 3.0, "-", 1.0
                    color = "#4CAF50"  # green for correct
                elif is_dominant_wrong:
                    lw, ls, alpha = 2.5, "-", 1.0
                    color = "#FF5722"  # red for dominant wrong
                else:
                    lw, ls, alpha = 1.0, "--", 0.4

                label = f"v{vi}"
                if is_correct:
                    label += " ✓ (correct)"
                elif is_dominant_wrong:
                    label += " ✗ (dominant wrong)"

                values = avg[vi, layers]
                ax.plot(layers, values, color=color, linewidth=lw,
                        linestyle=ls, alpha=alpha, label=label)

            # Mark crossover point for PI: where correct was leading then lost
            if cond_for_logic == "PI":
                p_correct = avg[correct_idx, :]
                p_dominant = avg[dominant_wrong, :]
                # Find crossover
                for L in range(n_layers - 1):
                    if p_correct[L] > p_dominant[L] and p_correct[L + 1] <= p_dominant[L + 1]:
                        if L in layers:
                            ax.axvline(x=L, color="black", linewidth=1.5, linestyle=":",
                                      alpha=0.7, label=f"crossover @ L{L}")
                        break

            # Shade the suppression region for PI
            if cond_for_logic == "PI" and cond == "PI_zoom":
                # Find where P(correct) peaks then drops
                p_correct = avg[correct_idx, :]
                peak_layer = active_start + np.argmax(p_correct[active_start:])
                if peak_layer < n_layers - 1 and p_correct[peak_layer] > p_correct[-1] + 0.02:
                    ax.axvspan(peak_layer, n_layers - 1, alpha=0.1, color="red",
                              label=f"suppression zone (L{peak_layer}→L{n_layers-1})")

            ax.set_xlabel("Layer")
            ax.set_ylabel("P(value)")
            ax.set_title(f"{title_prefix} — {k}k_{u}u{label_suffix}")
            ax.legend(fontsize=8, loc="upper left")
            ax.set_xlim(layers[0], layers[-1])
            ax.set_ylim(-0.02, min(1.0, avg[:, layers].max() * 1.3 + 0.05))
            ax.grid(True, alpha=0.3)

    fig.suptitle(f"Layer-by-Layer P(v_i) — {model_short}", fontsize=16, y=1.01)
    plt.tight_layout()
    path = fig_dir / "fig3_layer_trajectories.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Last-Layer Value Probability Landscape (Stage 2)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_last_layer_landscape(s2, model_short, fig_dir):
    points = s2["points"]
    analyses = s2["analyses"]

    n_points = len(points)
    fig, axes = plt.subplots(1, n_points, figsize=(5 * n_points, 5))
    if n_points == 1:
        axes = [axes]

    for col, pt in enumerate(points):
        ax = axes[col]
        k, u = pt["keys"], pt["updates"]

        for cond, color, label in [("RI", "#2196F3", "RI correct"),
                                    ("PI", "#FF5722", "PI failures")]:
            if cond == "PI":
                trials = [a for a in analyses
                          if a["condition"] == cond and a["num_keys"] == k
                          and a["num_updates"] == u
                          and not a["correct"] and a["total_value_prob"] > 0.05]
            else:
                trials = [a for a in analyses
                          if a["condition"] == cond and a["num_keys"] == k
                          and a["num_updates"] == u and a["correct"]]

            if not trials:
                continue

            n_values = len(trials[0]["all_values"])
            all_last = [[t["value_probs_by_layer"][vi][-1] for vi in range(n_values)]
                        for t in trials]
            avg = np.mean(all_last, axis=0)
            sem = np.std(all_last, axis=0) / np.sqrt(len(all_last))

            positions = np.arange(n_values)
            offset = -0.15 if cond == "RI" else 0.15
            ax.bar(positions + offset, avg, width=0.3, color=color,
                   alpha=0.8, label=f"{label} (n={len(trials)})")
            ax.errorbar(positions + offset, avg, yerr=sem, fmt="none",
                       ecolor="black", capsize=2, linewidth=0.8)

        ax.set_xticks(range(u))
        ax.set_xticklabels([f"v{i}" for i in range(u)], fontsize=9)
        ax.set_xlabel("Value position (v0=first, v_last=correct for PI)")
        ax.set_ylabel("P(value) at final layer")
        ax.set_title(f"{k}k_{u}u")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Last-Layer P(v_i): RI vs PI — {model_short}", fontsize=16, y=1.02)
    plt.tight_layout()
    path = fig_dir / "fig4_last_layer_landscape.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: P(v_last) Suppression — zoomed comparison across operating points
# ═══════════════════════════════════════════════════════════════════════════════

def plot_suppression(s2, model_short, fig_dir):
    points = s2["points"]
    analyses = s2["analyses"]
    n_layers = s2["n_layers"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: P(v_last) across layers for each operating point (PI failures)
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(points)))

    for i, pt in enumerate(points):
        k, u = pt["keys"], pt["updates"]
        trials = [a for a in analyses
                  if a["condition"] == "PI" and a["num_keys"] == k
                  and a["num_updates"] == u
                  and not a["correct"] and a["total_value_prob"] > 0.05]
        if not trials:
            continue

        n_values = len(trials[0]["all_values"])
        avg = np.mean([a["value_probs_by_layer"] for a in trials], axis=0)
        p_last = avg[-1, :]  # P(v_last) across layers

        ax.plot(range(n_layers), p_last, color=colors[i], linewidth=2,
                label=f"{k}k_{u}u (n={len(trials)})")

    ax.set_xlabel("Layer")
    ax.set_ylabel("P(v_last)")
    ax.set_title("P(v_last) across layers — PI failures")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right panel: P(v_0) across layers for RI correct (comparison)
    ax = axes[1]
    for i, pt in enumerate(points):
        k, u = pt["keys"], pt["updates"]
        trials = [a for a in analyses
                  if a["condition"] == "RI" and a["num_keys"] == k
                  and a["num_updates"] == u and a["correct"]]
        if not trials:
            continue

        avg = np.mean([a["value_probs_by_layer"] for a in trials], axis=0)
        p_first = avg[0, :]  # P(v_0) across layers

        ax.plot(range(n_layers), p_first, color=colors[i], linewidth=2,
                label=f"{k}k_{u}u (n={len(trials)})")

    ax.set_xlabel("Layer")
    ax.set_ylabel("P(v_first)")
    ax.set_title("P(v_first) across layers — RI correct")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Retrieval Asymmetry: P(v_last) suppressed vs P(v_first) clean — {model_short}",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    path = fig_dir / "fig5_suppression.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate paper figures from V3 results")
    parser.add_argument("--model", required=True, help="Model short name (e.g. Qwen2.5-3B-Instruct)")
    args = parser.parse_args()

    model_short = args.model
    fig_dir = FIGURES_DIR / model_short
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating figures for {model_short}")
    print(f"Output: {fig_dir}/\n")

    # Stage 1
    try:
        s1 = load_stage1(model_short)
        print("Stage 1 figures:")
        plot_regime_map(s1, model_short, fig_dir)
        plot_failure_positions(s1, model_short, fig_dir)
    except FileNotFoundError as e:
        print(f"  Stage 1: {e}")

    # Stage 2
    try:
        s2 = load_stage2(model_short)
        print("\nStage 2 figures:")
        plot_layer_trajectories(s2, model_short, fig_dir)
        plot_last_layer_landscape(s2, model_short, fig_dir)
        plot_suppression(s2, model_short, fig_dir)
    except FileNotFoundError as e:
        print(f"  Stage 2: {e}")

    print(f"\nDone. Figures in {fig_dir}/")


if __name__ == "__main__":
    main()
