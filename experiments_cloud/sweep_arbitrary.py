"""
Cloud Behavioral Sweep — ARBITRARY_MULTI dataset.

Tests PI/RI interference on cloud models (Claude, GPT, Gemini, Bedrock).
Dataset: real category members (alexandrite, baroque, ...) — 46 categories, ~52 values/cat.
Max updates: 300 (synthetic Prefix+Number, pool of 500).

Each trial is fully independent:
  - Fresh random categories, values, sequence, and test category per trial
  - 200 trials per cell (configurable)
  - 2 API calls per trial (RI + PI on same sequence)

Saturation thresholds derived from Clopper-Pearson exact binomial CI, n=200:
  near-zero  : acc <= 1.5%  (CI_upper <= 4.3%, i.e. 95% confident true p <= 5%)
  neutral    : 1.5% < acc < 6%  (ambiguous, do nothing)
  recovery   : acc >= 6%   (CI_lower >= 3.1%, i.e. 95% confident true p > 3%)
  stop when zero_count >= 3 for either RI or PI.

Usage:
  # Full run
  python sweep_arbitrary.py --model claude-haiku

  # Quick smoke test
  python sweep_arbitrary.py --model claude-haiku --trials 3 \\
      --key-levels 2 3 --update-levels 1 3

  # Resume interrupted run
  python sweep_arbitrary.py --model claude-haiku \\
      --resume results/semantic/claude-haiku/sweep_partial.json
"""

import os
import sys
import json
import time
import random
import threading
import argparse
import warnings
import numpy as np

# Suppress google-cloud-aiplatform FutureWarnings about Python 3.10 EOL
warnings.filterwarnings("ignore", category=FutureWarning, module="google")
warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.model_factory import create_model
from mechanistic_probing_v2.core.dataset_configs import (
    get_eligible_categories,
    generate_values_for_trial,
    build_interleaved_sequence,
)
from mechanistic_probing_v2.core.evaluation import classify_error, bootstrap_ci


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_TYPE  = "ARBITRARY_MULTI"
DEFAULT_SAVE  = "experiments_cloud/results/arbitrary"

# Tiered grids — choose with --tier small|medium|large
# Update levels capped at ~50 by SEMANTIC_MULTI pool size
TIER_GRIDS = {
    "small": {
        "key_levels":    [2, 3, 5, 7, 10, 12, 15, 20, 25, 30, 45],
        "update_levels": [1, 3, 5, 10, 15, 20, 30, 40, 50, 100, 200, 300],
    },
    "medium": {
        "key_levels":    [2, 5, 10, 15, 20, 25, 30, 40, 45],
        "update_levels": [1, 5, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300],
    },
    "large": {
        "key_levels":    [2, 10, 15, 20, 25, 30, 35, 37, 40, 42, 45],
        "update_levels": [1, 20, 50, 75, 100, 125, 150, 175, 200, 225, 275, 300],
    },
}
DEFAULT_TIER       = "small"
DEFAULT_TRIALS     = 200
DEFAULT_WORKERS    = 40    # concurrent trials per cell = batch size per convergence check
CI_THRESHOLD       = 0.07  # Wilson CI half-width threshold for early stopping (±7%)
MIN_TRIALS_CHECK   = DEFAULT_WORKERS  # first check after first full batch — naturally aligned

# Saturation thresholds (Clopper-Pearson, n=200, 95% CI — see PLAN.md for derivation)
NEAR_ZERO = 0.015   # acc <= 1.5% → near-zero, increment counter
RECOVERY  = 0.06    # acc >= 6%   → recovered, reset counter
SAT_COUNT = 3       # stop after this many near-zero hits without recovery


# ═══════════════════════════════════════════════════════════════════════════════
# THREAD-LOCAL MODEL INSTANCES
# Each worker thread gets its own model client — no shared state, no race conditions.
# ═══════════════════════════════════════════════════════════════════════════════

_thread_local = threading.local()

def get_thread_model(model_name):
    """Return a model instance owned by the current thread.

    Created on first call per thread, reused for all subsequent calls
    in that thread. Completely isolated from other threads.
    Silent init — startup verification already confirmed credentials.
    """
    if not hasattr(_thread_local, 'model'):
        _thread_local.model = create_model(model_name, {'verbose': False})
    return _thread_local.model


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Cloud behavioral sweep — ARBITRARY_MULTI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model",         default="claude-haiku",
                   help="Model name routed via create_model() (default: claude-haiku)")
    p.add_argument("--trials",   type=int, default=DEFAULT_TRIALS,
                   help=f"Trials per cell (default: {DEFAULT_TRIALS})")
    p.add_argument("--workers",  type=int, default=DEFAULT_WORKERS,
                   help=f"Concurrent trials per cell (default: {DEFAULT_WORKERS})")
    p.add_argument("--save-dir",      default=DEFAULT_SAVE,
                   help=f"Output directory (default: {DEFAULT_SAVE})")
    p.add_argument("--resume",        default=None,
                   help="Path to sweep_partial.json to resume from")
    p.add_argument("--tier",          default=DEFAULT_TIER, choices=list(TIER_GRIDS.keys()),
                   help=f"Model tier — sets default grid (default: {DEFAULT_TIER})")
    p.add_argument("--key-levels",    type=int, nargs="+", default=None,
                   help="Override key levels (default: tier grid)")
    p.add_argument("--update-levels", type=int, nargs="+", default=None,
                   help="Override update levels (default: tier grid)")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_prompt(sequence, test_category, condition):
    """Build a single-key prompt for one API call.

    Args:
        sequence:      list of {"category": ..., "value": ..., "update_idx": ...}
        test_category: the category to ask about
        condition:     "RI" (first) or "PI" (last)

    Returns:
        prompt string
    """
    stream = "\n".join(f"{item['category']}: {item['value']}" for item in sequence)
    query_word = "first" if condition == "RI" else "last"
    return (
        f"Read the following key-value stream. "
        f"Each key gets updated multiple times.\n\n"
        f"{stream}\n\n"
        f"What was the {query_word} value of {test_category}?\n"
        f"Answer with ONLY the exact value. No explanation."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR DETAIL
# ═══════════════════════════════════════════════════════════════════════════════

def get_error_detail(predicted, all_values):
    """Return the index and relative position of the predicted value in all_values.

    all_values: values for test_category in sequence order (first to last).
    relative_pos: 0.0 = first occurrence, 1.0 = last occurrence.
    Returns None values if predicted doesn't match any value in sequence.
    """
    pred_lower = predicted.lower().strip()
    for idx, val in enumerate(all_values):
        v = val.lower()
        if v in pred_lower or pred_lower.startswith(v):
            return {
                "predicted_idx": idx,
                "predicted_relative_pos": round(idx / max(len(all_values) - 1, 1), 4),
            }
    return {"predicted_idx": None, "predicted_relative_pos": None}


# ═══════════════════════════════════════════════════════════════════════════════
# SATURATION
# ═══════════════════════════════════════════════════════════════════════════════

def init_saturation(key_levels):
    return {nk: {"RI": {"zero_count": 0}, "PI": {"zero_count": 0}}
            for nk in key_levels}


def update_saturation(sat, nk, cond, acc):
    """Three-zone saturation update. Returns True if cell should stop."""
    if acc <= NEAR_ZERO:
        sat[nk][cond]["zero_count"] += 1
    elif acc >= RECOVERY:
        sat[nk][cond]["zero_count"] = 0
    # neutral zone: do nothing


def is_saturated(sat, nk):
    """True if either RI or PI has hit SAT_COUNT near-zero cells."""
    for cond in ["RI", "PI"]:
        if sat[nk][cond]["zero_count"] >= SAT_COUNT:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# SEQUENTIAL CONVERGENCE CHECK
# Stop early when CI is tight — saves trials on ceiling/floor cells,
# spends full budget on uncertain transition cells near the inflection point.
#
# Uses Wilson score interval (handles p near 0/1 better than normal approx).
# Check after every batch of `workers` trials (natural batch boundary).
# Stop when BOTH RI and PI have CI half-width <= CI_THRESHOLD.
#
# Paper justification:
#   "Trials continued until the 95% Wilson CI half-width fell below ±7%
#    for both RI and PI, or until 200 trials were completed, whichever
#    came first. This concentrates the trial budget on uncertain cells
#    (near the inflection point) while stopping early on ceiling/floor cells."
# ═══════════════════════════════════════════════════════════════════════════════

def wilson_half_width(n, k, z=1.96):
    """95% Wilson score interval half-width for k successes in n trials."""
    if n == 0:
        return 1.0
    p = k / n
    return z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)


def has_converged(all_results, threshold=CI_THRESHOLD, min_trials=MIN_TRIALS_CHECK):
    """Return True if BOTH RI and PI have CI half-width <= threshold."""
    for cond in ["RI", "PI"]:
        corrects = [r["correct"] for r in all_results[cond]]
        n = len(corrects)
        if n < min_trials:
            return False
        k  = sum(corrects)
        hw = wilson_half_width(n, k)
        if hw > threshold:
            return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# TRIAL RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_trial(model_name, num_keys, num_updates, trial_idx, eligible_cats):
    """Run one fully-independent trial (2 API calls: RI + PI).

    Gets a thread-local model instance — safe to call from multiple threads.
    Returns dict with RI and PI results.
    """
    # Each thread has its own model client — no shared state
    model = get_thread_model(model_name)

    seed = abs(hash((num_keys, num_updates, trial_idx))) % (2 ** 31)
    rng = random.Random(seed)

    # Fresh random categories + test category
    categories    = rng.sample(eligible_cats, num_keys)
    test_category = rng.choice(categories)

    # Fresh values and sequence
    values_per_cat = generate_values_for_trial(DATASET_TYPE, categories, num_updates, rng)
    sequence       = build_interleaved_sequence(categories, values_per_cat, rng)

    # Values for test category in sequence order (for error classification)
    all_values    = [item["value"] for item in sequence if item["category"] == test_category]
    initial_value = all_values[0]
    final_value   = all_values[-1]

    results = {}
    for condition in ["RI", "PI"]:
        expected = initial_value if condition == "RI" else final_value
        prompt   = build_prompt(sequence, test_category, condition)

        response = model.generate(prompt)

        # Capture token usage immediately — this thread owns this model instance
        # so model.last_* is safe to read here with no race risk
        token_usage = {
            "input_tokens":  model.last_input_tokens,
            "output_tokens": model.last_output_tokens,
            "was_truncated": model.last_was_truncated,
            "error":         model.last_error,
        }

        predicted    = response.strip()
        error_type   = classify_error(predicted, expected, initial_value, final_value,
                                      all_values, condition)
        error_detail = get_error_detail(predicted, all_values)

        results[condition] = {
            "trial_idx":              trial_idx,
            "seed":                   seed,
            "condition":              condition,
            "test_category":          test_category,
            "categories_in_sequence": categories,
            "num_keys":               num_keys,
            "num_updates":            num_updates,
            "expected":               expected,
            "predicted":              predicted,
            "correct":                error_type == "correct",
            "error_type":             error_type,
            "error_detail":           error_detail,
            "token_usage":            token_usage,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CELL RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_cell(model_name, num_keys, num_updates, trials_per_cell, workers):
    """Run trials for one cell in batches of `workers`, stopping early if CI converges.

    Batch structure:
      - Each batch = workers trials submitted concurrently
      - After each batch: check Wilson CI half-width for both RI and PI
      - Stop if both <= CI_THRESHOLD, else submit next batch
      - Maximum batches = ceil(trials_per_cell / workers)

    This concentrates the trial budget on uncertain cells (p≈0.5) while
    stopping early on ceiling/floor cells (p≈0.0 or p≈1.0).
    """
    eligible = get_eligible_categories(DATASET_TYPE, min_values=num_updates)
    if len(eligible) < num_keys:
        return None, None

    all_trial_results = {"RI": [], "PI": []}
    cell_start  = time.time()
    trial_idx   = 0
    stopped_early = False

    with ThreadPoolExecutor(max_workers=workers) as executor:
        while trial_idx < trials_per_cell:
            batch_size = min(workers, trials_per_cell - trial_idx)

            # Submit one batch of concurrent trials
            futures = [
                executor.submit(run_trial, model_name, num_keys, num_updates,
                                trial_idx + i, eligible)
                for i in range(batch_size)
            ]

            # Collect this batch
            for future in as_completed(futures):
                trial = future.result()
                for cond in ["RI", "PI"]:
                    all_trial_results[cond].append(trial[cond])

            trial_idx += batch_size
            n_done = trial_idx

            ri_acc = np.mean([r["correct"] for r in all_trial_results["RI"]])
            pi_acc = np.mean([r["correct"] for r in all_trial_results["PI"]])

            ri_hw = wilson_half_width(n_done, sum(r["correct"] for r in all_trial_results["RI"]))
            pi_hw = wilson_half_width(n_done, sum(r["correct"] for r in all_trial_results["PI"]))

            print(f"    batch {n_done//workers}/{trials_per_cell//workers} "
                  f"({n_done} trials): "
                  f"RI={ri_acc:.0%}±{ri_hw:.0%}  PI={pi_acc:.0%}±{pi_hw:.0%}", end="")

            # Check convergence after each batch
            if has_converged(all_trial_results):
                print(f"  → converged, stopping early")
                stopped_early = True
                break
            else:
                print()

    # Compute stats
    cell_stats = {}
    for cond in ["RI", "PI"]:
        corrects = [r["correct"] for r in all_trial_results[cond]]
        mean, ci_lo, ci_hi = bootstrap_ci(corrects)
        error_counts = {}
        for r in all_trial_results[cond]:
            et = r["error_type"]
            error_counts[et] = error_counts.get(et, 0) + 1
        cell_stats[cond] = {
            "accuracy":  round(mean, 4),
            "ci_lower":  round(ci_lo, 4),
            "ci_upper":  round(ci_hi, 4),
            "n":         len(corrects),
            "error_types": error_counts,
        }

    ri_acc = cell_stats["RI"]["accuracy"]
    pi_acc = cell_stats["PI"]["accuracy"]
    regime = ("A" if ri_acc >= 0.5 and pi_acc >= 0.5 else
              "B" if ri_acc >= 0.5 and pi_acc <  0.5 else
              "D" if ri_acc <  0.5 and pi_acc >= 0.5 else "C")

    n_actual = len(all_trial_results["RI"])
    summary = {
        "num_keys":       num_keys,
        "num_updates":    num_updates,
        "regime":         regime,
        "elapsed_sec":    round(time.time() - cell_start, 1),
        "n_trials":       n_actual,
        "n_observations": n_actual,
        "stopped_early":  stopped_early,
        "max_trials":     trials_per_cell,
        "stats":          cell_stats,
    }
    return summary, all_trial_results


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE / LOAD
# ═══════════════════════════════════════════════════════════════════════════════

def model_short_name(model_name):
    return model_name.replace("/", "_").replace(":", "_")


def save_results(results, full_trials, model_name, save_dir, partial=False):
    model_dir = Path(save_dir) / model_short_name(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    if partial:
        summary_path = model_dir / "sweep_partial.json"
        full_path    = model_dir / "sweep_full_partial.json"
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        summary_path = model_dir / f"sweep_{ts}.json"
        full_path    = model_dir / f"sweep_full_{ts}.json"

    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    # Full file: summary + all trial details
    full_data = dict(results)
    full_data["trial_details"] = full_trials
    with open(full_path, "w") as f:
        json.dump(full_data, f, indent=2)

    if partial:
        print(f"  -> Checkpoint: {summary_path}")
    else:
        print(f"  -> Saved: {summary_path}")
        print(f"  -> Full:  {full_path}")


def load_resume(path):
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY PRINTER
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(results, key_levels, update_levels):
    print("\n" + "=" * 70)
    print(f"SWEEP SUMMARY — {results['model']}  [{results['dataset_type']}]")
    print("=" * 70)

    for cond in ["RI", "PI"]:
        print(f"\n--- {cond} Accuracy ---")
        print(f"{'keys':>6}", end="")
        for nu in update_levels:
            print(f" {nu:>5}", end="")
        print()
        for nk in key_levels:
            print(f"{nk:>6}", end="")
            for nu in update_levels:
                ck = f"{nk}_{nu}"
                if ck in results["cells"]:
                    acc = results["cells"][ck]["stats"][cond]["accuracy"]
                    print(f" {acc:>5.0%}", end="")
                else:
                    print(f" {'---':>5}", end="")
            print()

    print(f"\n--- Regime Map (A=both good, B=RI good PI bad, C=both bad, D=PI good RI bad) ---")
    print(f"{'keys':>6}", end="")
    for nu in update_levels:
        print(f" {nu:>5}", end="")
    print()
    for nk in key_levels:
        print(f"{nk:>6}", end="")
        for nu in update_levels:
            ck = f"{nk}_{nu}"
            if ck in results["cells"]:
                print(f" {results['cells'][ck]['regime']:>5}", end="")
            else:
                print(f" {'---':>5}", end="")
        print()

    cells = list(results["cells"].values())
    if cells:
        ri_accs = [c["stats"]["RI"]["accuracy"] for c in cells]
        pi_accs = [c["stats"]["PI"]["accuracy"] for c in cells]
        print(f"\nOverall: mean RI={np.mean(ri_accs):.1%}, "
              f"mean PI={np.mean(pi_accs):.1%}, "
              f"gap={np.mean(ri_accs) - np.mean(pi_accs):.1%}")
        print(f"Cells completed: {len(cells)}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    tier          = args.tier
    grid          = TIER_GRIDS[tier]
    key_levels    = args.key_levels    or grid["key_levels"]
    update_levels = args.update_levels or grid["update_levels"]
    model_name    = args.model

    print("=" * 70)
    print("CLOUD BEHAVIORAL SWEEP — ARBITRARY_MULTI")
    print(f"  Model:    {model_name}")
    print(f"  Tier:     {tier}")
    print(f"  Dataset:  {DATASET_TYPE}")
    print(f"  Trials:   {args.trials} per cell")
    print(f"  Workers:  {args.workers} concurrent trials")
    print(f"  Grid:     {len(key_levels)} key levels × {len(update_levels)} update levels")
    print(f"  Save:     {args.save_dir}/{model_short_name(model_name)}/")
    print(f"  Saturation thresholds: near-zero={NEAR_ZERO:.1%}, recovery={RECOVERY:.0%}")
    print("=" * 70)

    # Verify credentials by creating one model instance up front
    print("\nVerifying model connection...")
    create_model(model_name)   # will raise clearly if key is missing

    # Resume or fresh start
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        results     = load_resume(args.resume)
        full_trials = {}
    else:
        results = {
            "model":        model_name,
            "dataset_type": DATASET_TYPE,
            "config": {
                "trials_per_cell":  args.trials,
                "workers":          args.workers,
                "key_levels":       key_levels,
                "update_levels":    update_levels,
                "near_zero_thresh": NEAR_ZERO,
                "recovery_thresh":  RECOVERY,
                "sat_count":        SAT_COUNT,
            },
            "cells":      {},
            "start_time": datetime.now(timezone.utc).isoformat(),
        }
        full_trials = {}

    sat  = init_saturation(key_levels)
    done = 0
    total_cells = len(key_levels) * len(update_levels)

    for nk in key_levels:
        for nu in update_levels:
            cell_key = f"{nk}_{nu}"

            # Skip if already done (resume)
            if cell_key in results["cells"]:
                # Restore saturation state from completed cell
                cell = results["cells"][cell_key]
                for cond in ["RI", "PI"]:
                    update_saturation(sat, nk, cond, cell["stats"][cond]["accuracy"])
                done += 1
                continue

            # Skip if saturated
            if is_saturated(sat, nk):
                print(f"  [{done+1}/{total_cells}] keys={nk:>2}, updates={nu:>3}: "
                      f"SKIPPED (saturated)")
                done += 1
                continue

            # Skip if not enough categories in pool
            eligible = get_eligible_categories(DATASET_TYPE, min_values=nu)
            if len(eligible) < nk:
                print(f"  [{done+1}/{total_cells}] keys={nk:>2}, updates={nu:>3}: "
                      f"SKIPPED (only {len(eligible)} eligible categories, need {nk})")
                done += 1
                continue

            print(f"\n  [{done+1}/{total_cells}] keys={nk:>2}, updates={nu:>3} "
                  f"({args.trials} trials × 2 conditions = {args.trials*2} API calls)")

            cell_summary, cell_trials = run_cell(
                model_name, nk, nu, args.trials, args.workers
            )

            if cell_summary is None:
                print(f"    SKIPPED (insufficient data)")
                done += 1
                continue

            results["cells"][cell_key]  = cell_summary
            full_trials[cell_key]       = cell_trials

            ri_acc = cell_summary["stats"]["RI"]["accuracy"]
            pi_acc = cell_summary["stats"]["PI"]["accuracy"]
            ri_ci  = cell_summary["stats"]["RI"]
            pi_ci  = cell_summary["stats"]["PI"]

            print(f"  -> RI={ri_acc:.0%} [{ri_ci['ci_lower']:.0%}-{ri_ci['ci_upper']:.0%}]  "
                  f"PI={pi_acc:.0%} [{pi_ci['ci_lower']:.0%}-{pi_ci['ci_upper']:.0%}]  "
                  f"regime={cell_summary['regime']}  ({cell_summary['elapsed_sec']:.1f}s)")

            # Update saturation counters
            for cond in ["RI", "PI"]:
                update_saturation(sat, nk, cond, cell_summary["stats"][cond]["accuracy"])
            if is_saturated(sat, nk):
                print(f"  -> Saturated at keys={nk}, stopping further updates for this key count")

            done += 1

            # Checkpoint every 3 cells
            if done % 3 == 0:
                save_results(results, full_trials, model_name, args.save_dir, partial=True)

    results["end_time"] = datetime.now(timezone.utc).isoformat()

    # Final save
    save_results(results, full_trials, model_name, args.save_dir, partial=False)

    print_summary(results, key_levels, update_levels)


if __name__ == "__main__":
    main()
