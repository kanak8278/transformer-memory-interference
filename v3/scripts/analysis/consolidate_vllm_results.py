"""
Consolidate all vLLM sweep results into unified summary files.

Creates:
  - v3/results_vllm/summary/all_models_arbitrary_single.json
  - v3/results_vllm/summary/all_models_semantic_multi.json
  - v3/results_vllm/summary/cross_model_comparison.json
  - v3/results_vllm/summary/results_table.txt (printable table)

Usage:
    python v3/scripts/analysis/consolidate_vllm_results.py
"""

import json
import numpy as np
from pathlib import Path

RESULTS_BASE = Path(__file__).resolve().parent.parent.parent / "results_vllm"
SUMMARY_DIR = RESULTS_BASE / "summary"


def load_all_sweeps(dataset_type):
    """Load all sweep results for a dataset type."""
    ds_dir = RESULTS_BASE / dataset_type.lower()
    if not ds_dir.exists():
        return {}

    results = {}
    for model_dir in sorted(ds_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        sweeps = sorted(model_dir.glob("stage1_sweep_*.json"))
        if not sweeps:
            continue
        # Use the latest sweep file
        with open(sweeps[-1]) as f:
            data = json.load(f)
        results[model_dir.name] = data
    return results


def compute_summary_stats(data):
    """Extract key summary statistics from a model's sweep data."""
    cells = data.get("cells", {})
    if not cells:
        return None

    ri_accs = [c["stats"]["RI"]["accuracy"] for c in cells.values()]
    pi_accs = [c["stats"]["PI"]["accuracy"] for c in cells.values()]

    # Per-operating-point stats
    operating_points = {}
    for cell_key, cell in cells.items():
        nk, nu = map(int, cell_key.split("_"))
        ri = cell["stats"]["RI"]
        pi = cell["stats"]["PI"]
        operating_points[cell_key] = {
            "num_keys": nk, "num_updates": nu,
            "ri_accuracy": ri["accuracy"], "pi_accuracy": pi["accuracy"],
            "ri_ci": [ri["ci_lower"], ri["ci_upper"]],
            "pi_ci": [pi["ci_lower"], pi["ci_upper"]],
            "gap": round(ri["accuracy"] - pi["accuracy"], 4),
            "regime": cell["regime"],
            "ri_garbage_rate": round(ri["n_garbage"] / max(ri["n"], 1), 4),
            "pi_garbage_rate": round(pi["n_garbage"] / max(pi["n"], 1), 4),
            "n_trials": ri["n"],
        }

    return {
        "model": data["model"],
        "dataset_type": data["dataset_type"],
        "mean_ri": round(float(np.mean(ri_accs)), 4),
        "mean_pi": round(float(np.mean(pi_accs)), 4),
        "mean_gap": round(float(np.mean(ri_accs)) - float(np.mean(pi_accs)), 4),
        "n_cells": len(cells),
        "n_regime_A": sum(1 for c in cells.values() if c["regime"] == "A"),
        "n_regime_B": sum(1 for c in cells.values() if c["regime"] in ("B", "AB")),
        "n_regime_C": sum(1 for c in cells.values() if c["regime"] == "C"),
        "n_regime_D": sum(1 for c in cells.values() if c["regime"] == "D"),
        "operating_points": operating_points,
    }


def generate_cross_model_comparison(arbi_results, sem_results):
    """Generate cross-model comparison at standard operating points."""
    standard_ops = ["2_5", "2_10", "2_20", "5_10", "5_20", "10_10"]

    comparison = {}
    all_models = set(list(arbi_results.keys()) + list(sem_results.keys()))

    for model_name in sorted(all_models):
        model_cmp = {"model": model_name}

        for ds_name, ds_results in [("arbitrary_single", arbi_results),
                                     ("semantic_multi", sem_results)]:
            if model_name not in ds_results:
                continue
            data = ds_results[model_name]
            cells = data.get("cells", {})

            ds_stats = {}
            for op in standard_ops:
                if op in cells:
                    c = cells[op]
                    ri = c["stats"]["RI"]["accuracy"]
                    pi = c["stats"]["PI"]["accuracy"]
                    ds_stats[op] = {
                        "ri": ri, "pi": pi, "gap": round(ri - pi, 4),
                        "regime": c["regime"]
                    }
            model_cmp[ds_name] = ds_stats

        comparison[model_name] = model_cmp

    return comparison


def print_results_table(arbi_summaries, sem_summaries):
    """Print a formatted results table."""
    lines = []
    lines.append("=" * 120)
    lines.append("COMPLETE RESULTS TABLE — vLLM Stage 1 Behavioral Sweep")
    lines.append("=" * 120)
    lines.append("")

    header = f"{'Model':<30} {'ARBI RI':>8} {'ARBI PI':>8} {'Gap':>8} {'SEM RI':>8} {'SEM PI':>8} {'Gap':>8} {'Cells':>6}"
    lines.append(header)
    lines.append("-" * len(header))

    all_models = sorted(set(list(arbi_summaries.keys()) + list(sem_summaries.keys())))

    for model in all_models:
        arbi = arbi_summaries.get(model)
        sem = sem_summaries.get(model)

        arbi_ri = f"{arbi['mean_ri']:.1%}" if arbi else "—"
        arbi_pi = f"{arbi['mean_pi']:.1%}" if arbi else "—"
        arbi_gap = f"{arbi['mean_gap']:+.1%}" if arbi else "—"
        sem_ri = f"{sem['mean_ri']:.1%}" if sem else "—"
        sem_pi = f"{sem['mean_pi']:.1%}" if sem else "—"
        sem_gap = f"{sem['mean_gap']:+.1%}" if sem else "—"
        cells = str(arbi['n_cells']) if arbi else "—"

        lines.append(f"{model:<30} {arbi_ri:>8} {arbi_pi:>8} {arbi_gap:>8} "
                     f"{sem_ri:>8} {sem_pi:>8} {sem_gap:>8} {cells:>6}")

    lines.append("")
    lines.append("PI > RI asymmetry: positive gap = RI better (expected)")
    lines.append("Regime D (negative gap) = PI better than RI (reversed)")

    return "\n".join(lines)


def main():
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    # Load all results
    arbi_results = load_all_sweeps("arbitrary_single")
    sem_results = load_all_sweeps("semantic_multi")

    print(f"Loaded: {len(arbi_results)} ARBITRARY_SINGLE, {len(sem_results)} SEMANTIC_MULTI")

    # Compute summaries
    arbi_summaries = {}
    for name, data in arbi_results.items():
        s = compute_summary_stats(data)
        if s:
            arbi_summaries[name] = s

    sem_summaries = {}
    for name, data in sem_results.items():
        s = compute_summary_stats(data)
        if s:
            sem_summaries[name] = s

    # Save per-dataset summaries
    with open(SUMMARY_DIR / "all_models_arbitrary_single.json", "w") as f:
        json.dump(arbi_summaries, f, indent=2)
    with open(SUMMARY_DIR / "all_models_semantic_multi.json", "w") as f:
        json.dump(sem_summaries, f, indent=2)

    # Cross-model comparison
    comparison = generate_cross_model_comparison(arbi_results, sem_results)
    with open(SUMMARY_DIR / "cross_model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # Printable table
    table = print_results_table(arbi_summaries, sem_summaries)
    print(table)
    with open(SUMMARY_DIR / "results_table.txt", "w") as f:
        f.write(table)

    print(f"\nSaved to {SUMMARY_DIR}/")


if __name__ == "__main__":
    main()
