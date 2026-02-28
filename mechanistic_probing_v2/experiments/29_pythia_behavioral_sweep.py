"""
Experiment 29: Pythia-160M Behavioral Sweep (Completion Format)

Tests whether PI > RI exists in a base completion model (no instruction tuning).
This rules out RLHF/SFT as the cause of the interference asymmetry.

Key differences from experiment 11:
  - Uses few-shot completion format (no chat template, no system prompt)
  - Uses short category names ("shape", "stone", not "visual art", "gemstone")
  - Uses single-token English words as values (verified against Pythia tokenizer)
  - Smaller grid: Pythia-160M has 2048 context, 12 layers, 12 heads

Usage:
    cd mechanistic_probing_v2
    python experiments/29_pythia_behavioral_sweep.py
    python experiments/29_pythia_behavioral_sweep.py --model EleutherAI/pythia-410m
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_loader import load_model_hf, get_context_limit
from core.dataset_configs import (
    generate_completion_trial, COMPLETION_CATEGORIES,
)
from core.model_loader import verify_single_token


# Smaller grid for 160M model with 2048 context
KEY_LEVELS_PYTHIA = [2, 3, 5, 7, 10]
UPDATE_LEVELS_PYTHIA = [1, 2, 3, 5, 7, 10, 15, 20, 30]


def classify_error(predicted, expected, initial_value, final_value, all_values, condition):
    """Classify error type (same logic as experiment 11)."""
    pred_lower = predicted.lower().strip()
    exp_lower = expected.lower().strip()

    if exp_lower in pred_lower or pred_lower.startswith(exp_lower):
        return "correct"

    init_lower = initial_value.lower()
    final_lower = final_value.lower()

    if condition == "PI" and (init_lower in pred_lower or pred_lower.startswith(init_lower)):
        return "primacy_intrusion"
    if condition == "RI" and (final_lower in pred_lower or pred_lower.startswith(final_lower)):
        return "recency_intrusion"

    for val in all_values:
        if val.lower() != exp_lower and (val.lower() in pred_lower or pred_lower.startswith(val.lower())):
            return "intermediate_intrusion"

    return "garbage"


def bootstrap_ci(data, n_bootstrap=2000, ci=0.95):
    if not data:
        return 0.0, 0.0, 0.0
    arr = np.array(data, dtype=float)
    mean = arr.mean()
    if len(arr) < 3:
        return mean, 0.0, 1.0
    rng = np.random.RandomState(42)
    boot_means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_bootstrap)]
    alpha = (1 - ci) / 2
    return mean, np.percentile(boot_means, alpha * 100), np.percentile(boot_means, (1 - alpha) * 100)


def run_single_trial(model, tokenizer, trial, max_new_tokens=10):
    """Run a single trial with a completion model (no chat template)."""
    import torch

    # Completion model: use the prompt directly, no chat wrapping
    inputs = tokenizer(trial.prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        gen_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None,
        )

    new_ids = gen_ids[0, inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip().split("\n")[0].strip()

    error_type = classify_error(
        answer, trial.expected_answer,
        trial.initial_value, trial.final_value,
        trial.all_values, trial.condition,
    )

    return {
        "seed": trial.seed,
        "num_keys": trial.num_keys,
        "num_updates": trial.num_updates,
        "condition": trial.condition,
        "test_category": trial.test_category,
        "expected": trial.expected_answer,
        "predicted": answer,
        "correct": error_type == "correct",
        "error_type": error_type,
        "initial_value": trial.initial_value,
        "final_value": trial.final_value,
    }


def preflight_check(num_keys, num_updates, value_pool, tokenizer, context_limit, safety=0.85):
    """Check if a (keys, updates) cell fits in Pythia's 2048 context."""
    trial = generate_completion_trial(
        num_keys, num_updates, "RI", seed=0, value_pool=value_pool,
    )
    n_tokens = len(tokenizer.encode(trial.prompt))
    limit = int(context_limit * safety)
    return n_tokens <= limit, n_tokens


def run_sweep(model, tokenizer, model_name, value_pool, feasible_grid, trials_per_cell=30):
    results = {
        "model": model_name,
        "prompt_format": "completion_few_shot",
        "config": {
            "key_levels": KEY_LEVELS_PYTHIA,
            "update_levels": UPDATE_LEVELS_PYTHIA,
            "trials_per_cell": trials_per_cell,
            "categories": COMPLETION_CATEGORIES[5:],  # test categories
            "demo_categories": COMPLETION_CATEGORIES[:5],
            "value_pool_size": len(value_pool),
        },
        "feasible_cells": {f"{k}_{u}": t for (k, u), t in feasible_grid.items()},
        "cells": {},
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    total_cells = len(feasible_grid)
    cell_idx = 0
    saturation_tracker = {}

    for nk in KEY_LEVELS_PYTHIA:
        saturation_tracker.setdefault(nk, {"RI": 0, "PI": 0})

        for nu in UPDATE_LEVELS_PYTHIA:
            if (nk, nu) not in feasible_grid:
                continue

            cell_key = f"{nk}_{nu}"

            if saturation_tracker[nk]["RI"] >= 5 and saturation_tracker[nk]["PI"] >= 5:
                print(f"  [{cell_idx+1}/{total_cells}] keys={nk}, updates={nu}: SKIPPED (saturated)")
                cell_idx += 1
                continue

            cell_start = time.time()
            cell_results = {"RI": [], "PI": []}

            for condition in ["RI", "PI"]:
                for trial_idx in range(trials_per_cell):
                    seed = hash((nk, nu, condition, trial_idx)) % (2**31)
                    test_cat_idx = trial_idx % min(nk, len(COMPLETION_CATEGORIES[5:]))

                    trial = generate_completion_trial(
                        num_keys=nk,
                        num_updates=nu,
                        condition=condition,
                        seed=seed,
                        value_pool=value_pool,
                        test_category_idx=test_cat_idx,
                    )

                    result = run_single_trial(model, tokenizer, trial)
                    cell_results[condition].append(result)

            cell_stats = {}
            for cond in ["RI", "PI"]:
                corrects = [r["correct"] for r in cell_results[cond]]
                mean, ci_lo, ci_hi = bootstrap_ci(corrects)

                error_counts = {}
                for r in cell_results[cond]:
                    et = r["error_type"]
                    error_counts[et] = error_counts.get(et, 0) + 1

                cell_stats[cond] = {
                    "accuracy": mean,
                    "ci_lower": ci_lo,
                    "ci_upper": ci_hi,
                    "n": len(corrects),
                    "error_types": error_counts,
                }

                if mean == 0.0:
                    saturation_tracker[nk][cond] += 1
                else:
                    saturation_tracker[nk][cond] = 0

            ri_acc = cell_stats["RI"]["accuracy"]
            pi_acc = cell_stats["PI"]["accuracy"]
            if ri_acc >= 0.5 and pi_acc >= 0.5:
                regime = "A"
            elif ri_acc >= 0.5 and pi_acc < 0.5:
                regime = "B"
            elif ri_acc < 0.5 and pi_acc < 0.5:
                regime = "C"
            else:
                regime = "D"

            cell_data = {
                "num_keys": nk,
                "num_updates": nu,
                "stats": cell_stats,
                "regime": regime,
                "trials": cell_results,
                "elapsed_sec": round(time.time() - cell_start, 1),
            }
            results["cells"][cell_key] = cell_data

            cell_idx += 1
            elapsed = cell_data["elapsed_sec"]
            print(f"  [{cell_idx}/{total_cells}] keys={nk:>2}, updates={nu:>3}: "
                  f"RI={ri_acc:.0%} [{cell_stats['RI']['ci_lower']:.0%}-{cell_stats['RI']['ci_upper']:.0%}] "
                  f"PI={pi_acc:.0%} [{cell_stats['PI']['ci_lower']:.0%}-{cell_stats['PI']['ci_upper']:.0%}] "
                  f"regime={regime} ({elapsed:.1f}s)")

            if cell_idx % 5 == 0:
                _save_results(results, model_name, partial=True)

    results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return results


def _save_results(results, model_name, partial=False):
    from datetime import datetime, timezone
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    model_short = model_name.split("/")[-1]
    suffix = "_partial" if partial else ""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Summary (no trials)
    summary = {k: v for k, v in results.items() if k != "cells"}
    summary["cells"] = {}
    for cell_key, cell_data in results["cells"].items():
        summary["cells"][cell_key] = {k: v for k, v in cell_data.items() if k != "trials"}

    # Timestamped (archival)
    ts_path = results_dir / f"behavioral_sweep_{model_short}{suffix}_{ts}.json"
    with open(ts_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Latest (stable name)
    latest_path = results_dir / f"behavioral_sweep_{model_short}{suffix}.json"
    with open(latest_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Full (with trials) — timestamped + latest
    full_ts = results_dir / f"behavioral_sweep_{model_short}_full{suffix}_{ts}.json"
    with open(full_ts, "w") as f:
        json.dump(results, f, indent=2)

    full_latest = results_dir / f"behavioral_sweep_{model_short}_full{suffix}.json"
    with open(full_latest, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  -> Saved to {ts_path}")


def print_summary(results):
    print("\n" + "=" * 70)
    print("PYTHIA BEHAVIORAL SWEEP SUMMARY (Completion Format)")
    print("=" * 70)

    for label, cond in [("RI", "RI"), ("PI", "PI")]:
        print(f"\n--- {label} Accuracy ---")
        print(f"{'Keys':>6}", end="")
        for nu in UPDATE_LEVELS_PYTHIA:
            print(f" {nu:>5}", end="")
        print()
        for nk in KEY_LEVELS_PYTHIA:
            print(f"{nk:>6}", end="")
            for nu in UPDATE_LEVELS_PYTHIA:
                cell_key = f"{nk}_{nu}"
                if cell_key in results["cells"]:
                    acc = results["cells"][cell_key]["stats"][cond]["accuracy"]
                    print(f" {acc:>5.0%}" if acc > 0 else f" {'0%':>5}", end="")
                else:
                    print(f" {'---':>5}", end="")
            print()

    # Regime map
    print(f"\n--- Regime Map (A=both, B=PI fails, C=both fail, D=PI>RI) ---")
    print(f"{'Keys':>6}", end="")
    for nu in UPDATE_LEVELS_PYTHIA:
        print(f" {nu:>5}", end="")
    print()
    for nk in KEY_LEVELS_PYTHIA:
        print(f"{nk:>6}", end="")
        for nu in UPDATE_LEVELS_PYTHIA:
            cell_key = f"{nk}_{nu}"
            if cell_key in results["cells"]:
                print(f" {results['cells'][cell_key]['regime']:>5}", end="")
            else:
                print(f" {'---':>5}", end="")
        print()

    # Error types in regime B
    regime_b = [(k, v) for k, v in results["cells"].items() if v["regime"] == "B"]
    if regime_b:
        print(f"\n--- PI Error Types in Regime B ({len(regime_b)} cells) ---")
        all_errors = {}
        for _, cell in regime_b:
            for et, count in cell["stats"]["PI"]["error_types"].items():
                all_errors[et] = all_errors.get(et, 0) + count
        total = sum(all_errors.values())
        for et, count in sorted(all_errors.items(), key=lambda x: -x[1]):
            print(f"  {et}: {count}/{total} = {count/total:.1%}")

    # Cracking points
    print(f"\n--- Cracking Points (< 50%) ---")
    for nk in KEY_LEVELS_PYTHIA:
        for cond in ["RI", "PI"]:
            for nu in UPDATE_LEVELS_PYTHIA:
                cell_key = f"{nk}_{nu}"
                if cell_key in results["cells"]:
                    if results["cells"][cell_key]["stats"][cond]["accuracy"] < 0.5:
                        print(f"  keys={nk:>2}, {cond}: cracks at updates={nu}")
                        break


def main():
    parser = argparse.ArgumentParser(description="Pythia Behavioral Sweep (Completion Format)")
    parser.add_argument("--model", default="EleutherAI/pythia-410m")
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()

    model, tokenizer, info = load_model_hf(args.model)

    # Verify single-token values against Pythia's tokenizer
    print("\nVerifying single-token values against Pythia tokenizer...")
    value_to_tid = verify_single_token(tokenizer)
    value_pool = list(value_to_tid.keys())
    print(f"  {len(value_pool)} single-token values verified (of 2300 candidates)")

    if len(value_pool) < 100:
        print(f"  WARNING: Only {len(value_pool)} values are single-token in this tokenizer.")
        print(f"  This may limit max operating points.")

    # Pre-flight context check
    ctx_limit = get_context_limit(args.model, info)
    print(f"\nContext limit: {ctx_limit}")

    feasible = {}
    for nk in KEY_LEVELS_PYTHIA:
        for nu in UPDATE_LEVELS_PYTHIA:
            ok, n_tokens = preflight_check(nk, nu, value_pool, tokenizer, ctx_limit)
            if ok:
                feasible[(nk, nu)] = n_tokens
            else:
                break  # higher updates will also fail

    print(f"Feasible cells: {len(feasible)} / {len(KEY_LEVELS_PYTHIA) * len(UPDATE_LEVELS_PYTHIA)}")

    print(f"\n{'Keys':>6} | Max updates | Est. tokens")
    print(f"{'-'*6}-+-{'-'*11}-+-{'-'*11}")
    for nk in KEY_LEVELS_PYTHIA:
        feasible_updates = sorted([nu for (k, nu) in feasible if k == nk])
        if feasible_updates:
            max_nu = feasible_updates[-1]
            print(f"{nk:>6} | {max_nu:>11} | {feasible[(nk, max_nu)]:>11,}")
        else:
            print(f"{nk:>6} | {'NONE':>11} | {'N/A':>11}")

    # Show example prompt
    sample_trial = generate_completion_trial(2, 3, "PI", seed=42, value_pool=value_pool)
    print(f"\n--- Example completion prompt (2 keys, 3 updates, PI) ---")
    print(sample_trial.prompt)
    print(f"--- Expected: {sample_trial.expected_answer} ---\n")

    # Run sweep
    results = run_sweep(model, tokenizer, args.model, value_pool, feasible, args.trials)

    _save_results(results, args.model, partial=False)
    print_summary(results)


if __name__ == "__main__":
    main()
