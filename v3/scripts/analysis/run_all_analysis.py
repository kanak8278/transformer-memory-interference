"""
Run all analysis on vLLM Stage 1 results: scaling laws, error positions, cross-model comparison.

Reads from: v3/results_vllm/arbitrary_single/ and v3/results_vllm/semantic_multi/
Saves to: v3/results_vllm/analysis/

Usage:
    python v3/scripts/analysis/run_all_analysis.py
"""

import json
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

RESULTS_BASE = Path(__file__).resolve().parent.parent.parent / "results_vllm"
ANALYSIS_DIR = RESULTS_BASE / "analysis"


def load_all_sweeps(dataset="arbitrary_single"):
    """Load all Stage 1 sweep results for a dataset."""
    ds_dir = RESULTS_BASE / dataset
    results = {}
    for model_dir in sorted(ds_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        sweeps = sorted(model_dir.glob("stage1_sweep_*.json"))
        if not sweeps:
            continue
        with open(sweeps[-1]) as f:
            results[model_dir.name] = json.load(f)
    return results


def load_all_trials(dataset="arbitrary_single"):
    """Load trial details for error position analysis."""
    ds_dir = RESULTS_BASE / dataset
    results = {}
    for model_dir in sorted(ds_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        trials = sorted(model_dir.glob("stage1_trials_*.json"))
        if not trials:
            continue
        with open(trials[-1]) as f:
            results[model_dir.name] = json.load(f)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: SCALING LAW FITS
# ═══════════════════════════════════════════════════════════════════════════════

def exp_decay(N, a, b, c):
    return a * np.exp(-b * N) + c

def log_decay(N, d, e):
    return d - e * np.log(N)

def fit_scaling_laws(all_sweeps):
    """Fit PI(N) = a*exp(-b*N)+c and RI(N) = d - e*log(N) per model."""
    fits = {}

    for model_name, data in all_sweeps.items():
        cells = data.get("cells", {})
        if not cells:
            continue

        # Group by key count, fit per key count
        by_keys = {}
        for ck, cv in cells.items():
            nk = cv["num_keys"]
            nu = cv["num_updates"]
            by_keys.setdefault(nk, []).append((nu, cv))

        model_fits = {}
        for nk, cell_list in sorted(by_keys.items()):
            cell_list.sort(key=lambda x: x[0])
            if len(cell_list) < 4:
                continue

            N_vals = np.array([c[0] for c in cell_list], dtype=float)
            ri_vals = np.array([c[1]["stats"]["RI"]["accuracy"] for c in cell_list])
            pi_vals = np.array([c[1]["stats"]["PI"]["accuracy"] for c in cell_list])

            # Fit PI: exponential decay
            pi_fit = None
            try:
                popt, pcov = curve_fit(exp_decay, N_vals, pi_vals,
                                       p0=[0.5, 0.05, 0.05], maxfev=5000,
                                       bounds=([0, 0, 0], [1, 1, 1]))
                pi_pred = exp_decay(N_vals, *popt)
                ss_res = np.sum((pi_vals - pi_pred)**2)
                ss_tot = np.sum((pi_vals - np.mean(pi_vals))**2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                pi_fit = {"a": round(popt[0], 4), "b": round(popt[1], 4),
                          "c": round(popt[2], 4), "r2": round(r2, 4)}
            except Exception:
                pass

            # Fit RI: log decay
            ri_fit = None
            try:
                popt, pcov = curve_fit(log_decay, N_vals, ri_vals,
                                       p0=[0.8, 0.1], maxfev=5000)
                ri_pred = log_decay(N_vals, *popt)
                ss_res = np.sum((ri_vals - ri_pred)**2)
                ss_tot = np.sum((ri_vals - np.mean(ri_vals))**2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                ri_fit = {"d": round(popt[0], 4), "e": round(popt[1], 4),
                          "r2": round(r2, 4)}
            except Exception:
                pass

            model_fits[f"keys_{nk}"] = {
                "n_points": len(cell_list),
                "N_range": [int(N_vals[0]), int(N_vals[-1])],
                "ri_mean": round(float(np.mean(ri_vals)), 4),
                "pi_mean": round(float(np.mean(pi_vals)), 4),
                "gap_mean": round(float(np.mean(ri_vals - pi_vals)), 4),
                "pi_fit_exp": pi_fit,
                "ri_fit_log": ri_fit,
                "data": [{"N": int(n), "ri": round(float(r), 4), "pi": round(float(p), 4)}
                         for n, r, p in zip(N_vals, ri_vals, pi_vals)],
            }

        fits[model_name] = model_fits

    return fits


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: ERROR POSITION HISTOGRAMS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_error_positions(all_trials):
    """Analyze WHERE PI failures land in the value sequence."""
    results = {}

    for model_name, data in all_trials.items():
        trial_details = data.get("trial_details", {})
        if not trial_details:
            continue

        model_errors = {}
        for cell_key, cell_data in trial_details.items():
            pi_trials = cell_data.get("PI", [])
            if not pi_trials:
                continue

            positions = []
            error_types = {"correct": 0, "primacy_intrusion": 0, "recency_intrusion": 0,
                          "intermediate_intrusion": 0, "garbage": 0}

            for t in pi_trials:
                et = t.get("error_type", "garbage")
                error_types[et] = error_types.get(et, 0) + 1

                if not t.get("correct", False) and t.get("predicted_relative_pos") is not None:
                    positions.append(t["predicted_relative_pos"])

            n_total = len(pi_trials)
            n_failures = n_total - error_types.get("correct", 0)

            model_errors[cell_key] = {
                "n_total": n_total,
                "n_failures": n_failures,
                "n_garbage": error_types.get("garbage", 0),
                "error_types": error_types,
                "failure_positions": positions,
                "mean_failure_pos": round(float(np.mean(positions)), 4) if positions else None,
                "median_failure_pos": round(float(np.median(positions)), 4) if positions else None,
                "near_last_fraction": round(
                    sum(1 for p in positions if p >= 0.7) / len(positions), 4
                ) if positions else None,
                "primacy_fraction": round(
                    sum(1 for p in positions if p <= 0.2) / len(positions), 4
                ) if positions else None,
            }

        # Aggregate across all cells
        all_positions = []
        total_failures = 0
        total_garbage = 0
        for v in model_errors.values():
            all_positions.extend(v["failure_positions"])
            total_failures += v["n_failures"]
            total_garbage += v["n_garbage"]

        results[model_name] = {
            "per_cell": model_errors,
            "aggregate": {
                "total_failures": total_failures,
                "total_garbage": total_garbage,
                "n_with_position": len(all_positions),
                "mean_failure_pos": round(float(np.mean(all_positions)), 4) if all_positions else None,
                "near_last_fraction": round(
                    sum(1 for p in all_positions if p >= 0.7) / len(all_positions), 4
                ) if all_positions else None,
                "position_histogram": np.histogram(all_positions, bins=10, range=(0, 1))[0].tolist()
                if all_positions else None,
            }
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: CROSS-MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_SIZES = {
    "Qwen2.5-0.5B-Instruct": 494e6,
    "Qwen2.5-1.5B-Instruct": 1.5e9,
    "Qwen2.5-3B-Instruct": 3e9,
    "Qwen2.5-3B": 3e9,
    "gemma-3-270m-it": 270e6,
    "gemma-3-1b-it": 1e9,
    "gemma-3-4b-it": 4e9,
    "TinyLlama-1.1B-Chat-v1.0": 1.1e9,
    "stablelm-2-1_6b-chat": 1.6e9,
    "pythia-410m": 410e6,
    "mamba-1.4b-hf": 1.4e9,
}

def cross_model_scaling(all_sweeps):
    """Test RI/PI scaling with model size at fixed operating points."""
    standard_ops = ["2_5", "2_10", "2_20", "5_10", "5_20"]
    results = {}

    for op in standard_ops:
        models_with_data = []
        for model_name, data in all_sweeps.items():
            cells = data.get("cells", {})
            if op not in cells:
                continue
            cell = cells[op]
            size = MODEL_SIZES.get(model_name)
            if size is None:
                continue

            models_with_data.append({
                "model": model_name,
                "size": size,
                "ri": cell["stats"]["RI"]["accuracy"],
                "pi": cell["stats"]["PI"]["accuracy"],
                "gap": cell["stats"]["RI"]["accuracy"] - cell["stats"]["PI"]["accuracy"],
            })

        if len(models_with_data) < 4:
            continue

        sizes = np.array([m["size"] for m in models_with_data])
        ri_vals = np.array([m["ri"] for m in models_with_data])
        pi_vals = np.array([m["pi"] for m in models_with_data])
        log_sizes = np.log10(sizes)

        ri_corr, ri_p = pearsonr(log_sizes, ri_vals) if len(sizes) > 2 else (0, 1)
        pi_corr, pi_p = pearsonr(log_sizes, pi_vals) if len(sizes) > 2 else (0, 1)

        results[op] = {
            "n_models": len(models_with_data),
            "ri_vs_size_r": round(float(ri_corr), 4),
            "ri_vs_size_p": round(float(ri_p), 4),
            "pi_vs_size_r": round(float(pi_corr), 4),
            "pi_vs_size_p": round(float(pi_p), 4),
            "models": models_with_data,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in ["arbitrary_single", "semantic_multi"]:
        print(f"\n{'='*60}")
        print(f"ANALYSIS: {dataset}")
        print(f"{'='*60}")

        sweeps = load_all_sweeps(dataset)
        print(f"  Loaded {len(sweeps)} models")

        # 1. Scaling law fits
        print("\n  Fitting scaling laws...")
        fits = fit_scaling_laws(sweeps)
        with open(ANALYSIS_DIR / f"scaling_law_fits_{dataset}.json", "w") as f:
            json.dump(fits, f, indent=2)
        for model, mf in fits.items():
            for kname, kdata in mf.items():
                pi_r2 = kdata["pi_fit_exp"]["r2"] if kdata["pi_fit_exp"] else "N/A"
                print(f"    {model} {kname}: gap={kdata['gap_mean']:+.1%}, PI R²={pi_r2}")

        # 2. Error positions (needs trial data)
        print("\n  Analyzing error positions...")
        try:
            trials = load_all_trials(dataset)
            errors = analyze_error_positions(trials)
            with open(ANALYSIS_DIR / f"error_positions_{dataset}.json", "w") as f:
                json.dump(errors, f, indent=2, default=str)
            for model, ed in errors.items():
                agg = ed["aggregate"]
                print(f"    {model}: {agg['total_failures']} failures, "
                      f"mean_pos={agg['mean_failure_pos']}, "
                      f"near_last={agg['near_last_fraction']}")
        except Exception as e:
            print(f"    Error: {e}")

        # 3. Cross-model scaling
        print("\n  Cross-model scaling analysis...")
        scaling = cross_model_scaling(sweeps)
        with open(ANALYSIS_DIR / f"cross_model_scaling_{dataset}.json", "w") as f:
            json.dump(scaling, f, indent=2)
        for op, sd in scaling.items():
            print(f"    {op}: RI~size r={sd['ri_vs_size_r']:.3f} (p={sd['ri_vs_size_p']:.3f}), "
                  f"PI~size r={sd['pi_vs_size_r']:.3f} (p={sd['pi_vs_size_p']:.3f})")

    print(f"\n  All analysis saved to {ANALYSIS_DIR}/")


if __name__ == "__main__":
    main()
