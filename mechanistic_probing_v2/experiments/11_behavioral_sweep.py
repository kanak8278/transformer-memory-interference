"""
Step 1.2: Full Behavioral Sweep.

Maps the complete interference landscape for a model:
- Grid: 12 key levels x 22 update levels x 2 conditions
- 30 trials per cell with different random seeds
- Pre-flight context length check
- Bootstrap 95% CIs
- Error type classification (primacy intrusion, recency intrusion, other)
- Regime mapping (A: both work, B: PI fails/RI works, C: both fail)
- Auto-truncation: stop when 5+ consecutive update levels hit 0%

Prompt format: NO few-shot, matching original ACL paper.

Usage:
    python 11_behavioral_sweep.py [--model MODEL] [--trials N] [--resume PATH]
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_loader import load_model_hf, get_context_limit
from core.dataset import (
    generate_trial, compute_feasible_grid, format_for_chat,
    KEY_LEVELS, UPDATE_LEVELS, ORIGINAL_CATEGORIES,
)


def classify_error(
    predicted: str,
    expected: str,
    initial_value: str,
    final_value: str,
    all_values: list[str],
    condition: str,
) -> str:
    """Classify error type.

    Returns one of:
        "correct" — predicted matches expected
        "primacy_intrusion" — predicted the initial value when asking for last (PI error)
        "recency_intrusion" — predicted the final value when asking for first (RI error)
        "intermediate_intrusion" — predicted a middle value
        "other_category" — predicted something that looks like a value from another category
        "garbage" — predicted nonsense or refusal
    """
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

    # Check if it's an intermediate value
    for val in all_values:
        if val.lower() != exp_lower and (val.lower() in pred_lower or pred_lower.startswith(val.lower())):
            return "intermediate_intrusion"

    return "garbage"


def bootstrap_ci(data: list[bool], n_bootstrap: int = 2000, ci: float = 0.95) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for accuracy.

    Returns (mean, ci_lower, ci_upper).
    """
    if not data:
        return 0.0, 0.0, 0.0

    arr = np.array(data, dtype=float)
    mean = arr.mean()

    if len(arr) < 3:
        return mean, 0.0, 1.0

    rng = np.random.RandomState(42)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(sample.mean())

    alpha = (1 - ci) / 2
    ci_lower = np.percentile(boot_means, alpha * 100)
    ci_upper = np.percentile(boot_means, (1 - alpha) * 100)
    return mean, ci_lower, ci_upper


def run_single_trial(model, tokenizer, trial, max_new_tokens=20, backend="huggingface"):
    """Run a single trial and return result dict."""
    import torch

    if backend == "huggingface":
        formatted = format_for_chat(trial.prompt, tokenizer)
        inputs = tokenizer(formatted, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=None, top_p=None,
            )
        # Decode only the generated tokens (strip prompt)
        new_ids = gen_ids[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip().split("\n")[0].strip()
    else:
        # TransformerLens backend
        gen_tokens = model.generate(
            trial.prompt, max_new_tokens=max_new_tokens, temperature=0, verbose=False
        )
        gen_text = model.to_string(gen_tokens[0]) if hasattr(gen_tokens, 'shape') else gen_tokens
        answer = gen_text[len(trial.prompt):].strip().split("\n")[0].strip()

    error_type = classify_error(
        answer, trial.expected_answer,
        trial.initial_value, trial.final_value,
        trial.all_values, trial.condition,
    )
    correct = error_type == "correct"

    return {
        "seed": trial.seed,
        "num_keys": trial.num_keys,
        "num_updates": trial.num_updates,
        "condition": trial.condition,
        "test_category": trial.test_category,
        "expected": trial.expected_answer,
        "predicted": answer,
        "correct": correct,
        "error_type": error_type,
        "initial_value": trial.initial_value,
        "final_value": trial.final_value,
    }


def run_sweep(
    model,
    tokenizer,
    model_name: str,
    feasible_grid: dict,
    trials_per_cell: int = 30,
    resume_data: dict = None,
    backend: str = "huggingface",
):
    """Run the full behavioral sweep.

    Args:
        model: HF model or HookedTransformer
        tokenizer: tokenizer
        model_name: model identifier
        feasible_grid: dict mapping (num_keys, num_updates) -> est_tokens
        trials_per_cell: number of trials per (keys, updates, condition) cell
        resume_data: optional previous results to resume from
        backend: "huggingface" or "transformer_lens"
    """
    results = resume_data or {
        "model": model_name,
        "config": {
            "key_levels": KEY_LEVELS,
            "update_levels": UPDATE_LEVELS,
            "trials_per_cell": trials_per_cell,
            "num_categories": len(ORIGINAL_CATEGORIES),
        },
        "feasible_cells": {f"{k}_{u}": t for (k, u), t in feasible_grid.items()},
        "cells": {},  # key: "nk_nu" -> cell results
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    total_cells = len(feasible_grid)
    completed_cells = len(results.get("cells", {}))
    print(f"\nSweep: {total_cells} feasible cells, {trials_per_cell} trials/cell, "
          f"2 conditions = {total_cells * trials_per_cell * 2} total trials")
    if completed_cells > 0:
        print(f"  Resuming from {completed_cells} completed cells")

    cell_idx = 0
    saturation_tracker = {}  # nk -> consecutive_zero_count for each condition

    for nk in KEY_LEVELS:
        saturation_tracker.setdefault(nk, {"RI": 0, "PI": 0})

        for nu in UPDATE_LEVELS:
            if (nk, nu) not in feasible_grid:
                continue

            cell_key = f"{nk}_{nu}"

            # Skip already completed cells (resume support)
            if cell_key in results.get("cells", {}):
                cell_idx += 1
                continue

            # Check saturation: skip if 5+ consecutive zeros for BOTH conditions
            if (saturation_tracker[nk]["RI"] >= 5 and
                saturation_tracker[nk]["PI"] >= 5):
                print(f"  [{cell_idx+1}/{total_cells}] keys={nk}, updates={nu}: "
                      f"SKIPPED (saturated — 5+ consecutive zeros for both)")
                cell_idx += 1
                continue

            cell_start = time.time()
            cell_results = {"RI": [], "PI": []}

            for condition in ["RI", "PI"]:
                for trial_idx in range(trials_per_cell):
                    # Use deterministic seed: hash of (nk, nu, condition, trial_idx)
                    seed = hash((nk, nu, condition, trial_idx)) % (2**31)
                    # Vary which category is tested
                    test_cat_idx = trial_idx % min(nk, len(ORIGINAL_CATEGORIES))

                    trial = generate_trial(
                        num_keys=nk,
                        num_updates=nu,
                        condition=condition,
                        seed=seed,
                        test_category_idx=test_cat_idx,
                    )

                    result = run_single_trial(model, tokenizer, trial, backend=backend)
                    cell_results[condition].append(result)

            # Compute cell statistics
            cell_stats = {}
            for cond in ["RI", "PI"]:
                corrects = [r["correct"] for r in cell_results[cond]]
                mean, ci_lo, ci_hi = bootstrap_ci(corrects)

                error_types = [r["error_type"] for r in cell_results[cond]]
                error_counts = {}
                for et in error_types:
                    error_counts[et] = error_counts.get(et, 0) + 1

                cell_stats[cond] = {
                    "accuracy": mean,
                    "ci_lower": ci_lo,
                    "ci_upper": ci_hi,
                    "n": len(corrects),
                    "error_types": error_counts,
                }

                # Update saturation tracker
                if mean == 0.0:
                    saturation_tracker[nk][cond] += 1
                else:
                    saturation_tracker[nk][cond] = 0

            # Classify regime
            ri_acc = cell_stats["RI"]["accuracy"]
            pi_acc = cell_stats["PI"]["accuracy"]
            if ri_acc >= 0.5 and pi_acc >= 0.5:
                regime = "A"  # both work
            elif ri_acc >= 0.5 and pi_acc < 0.5:
                regime = "B"  # PI fails, RI works
            elif ri_acc < 0.5 and pi_acc < 0.5:
                regime = "C"  # both fail
            else:
                regime = "D"  # PI works, RI fails (unexpected)

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

            # Periodic save (every 5 cells)
            if cell_idx % 5 == 0:
                _save_results(results, model_name, partial=True)

    results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return results


def _save_results(results, model_name, partial=False):
    """Save results to JSON with timestamp to avoid overwriting."""
    from datetime import datetime, timezone
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    model_short = model_name.split("/")[-1]
    suffix = "_partial" if partial else ""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Strip individual trials for the summary file to keep it manageable
    summary = {k: v for k, v in results.items() if k != "cells"}
    summary["cells"] = {}
    for cell_key, cell_data in results["cells"].items():
        summary["cells"][cell_key] = {
            k: v for k, v in cell_data.items() if k != "trials"
        }

    # Timestamped (archival)
    ts_path = results_dir / f"behavioral_sweep_{model_short}{suffix}_{ts}.json"
    with open(ts_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Latest (stable name for downstream code)
    latest_path = results_dir / f"behavioral_sweep_{model_short}{suffix}.json"
    with open(latest_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Full data with trials
    full_ts_path = results_dir / f"behavioral_sweep_{model_short}_full{suffix}_{ts}.json"
    with open(full_ts_path, "w") as f:
        json.dump(results, f, indent=2)

    full_latest = results_dir / f"behavioral_sweep_{model_short}_full{suffix}.json"
    with open(full_latest, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  -> Saved to {ts_path}")


def print_summary(results):
    """Print summary tables."""
    print("\n" + "=" * 80)
    print("BEHAVIORAL SWEEP SUMMARY")
    print("=" * 80)

    # Build grid display
    print(f"\n{'':>6}", end="")
    for nu in UPDATE_LEVELS:
        print(f" {nu:>5}", end="")
    print()

    for label, cond in [("RI", "RI"), ("PI", "PI")]:
        print(f"\n--- {label} Accuracy ---")
        print(f"{'Keys':>6}", end="")
        for nu in UPDATE_LEVELS:
            print(f" {nu:>5}", end="")
        print()

        for nk in KEY_LEVELS:
            print(f"{nk:>6}", end="")
            for nu in UPDATE_LEVELS:
                cell_key = f"{nk}_{nu}"
                if cell_key in results["cells"]:
                    acc = results["cells"][cell_key]["stats"][cond]["accuracy"]
                    print(f" {acc:>5.0%}" if acc > 0 else f" {'0%':>5}", end="")
                else:
                    print(f" {'---':>5}", end="")
            print()

    # Regime map
    print(f"\n--- Regime Map (A=both work, B=PI fails, C=both fail, D=PI>RI) ---")
    print(f"{'Keys':>6}", end="")
    for nu in UPDATE_LEVELS:
        print(f" {nu:>5}", end="")
    print()

    for nk in KEY_LEVELS:
        print(f"{nk:>6}", end="")
        for nu in UPDATE_LEVELS:
            cell_key = f"{nk}_{nu}"
            if cell_key in results["cells"]:
                regime = results["cells"][cell_key]["regime"]
                print(f" {regime:>5}", end="")
            else:
                print(f" {'---':>5}", end="")
        print()

    # Error type summary across regime B cells
    regime_b_cells = [
        (k, v) for k, v in results["cells"].items()
        if v["regime"] == "B"
    ]
    if regime_b_cells:
        print(f"\n--- Error Types in Regime B (PI fails, RI works): {len(regime_b_cells)} cells ---")
        all_pi_errors = {}
        for _, cell in regime_b_cells:
            for et, count in cell["stats"]["PI"]["error_types"].items():
                all_pi_errors[et] = all_pi_errors.get(et, 0) + count
        total = sum(all_pi_errors.values())
        for et, count in sorted(all_pi_errors.items(), key=lambda x: -x[1]):
            print(f"  {et}: {count}/{total} = {count/total:.1%}")

    # Cracking points
    print(f"\n--- Cracking Points (accuracy < 50%) ---")
    for nk in KEY_LEVELS:
        for cond in ["RI", "PI"]:
            crack_nu = None
            for nu in UPDATE_LEVELS:
                cell_key = f"{nk}_{nu}"
                if cell_key in results["cells"]:
                    acc = results["cells"][cell_key]["stats"][cond]["accuracy"]
                    if acc < 0.5:
                        crack_nu = nu
                        break
            if crack_nu is not None:
                print(f"  keys={nk:>2}, {cond}: cracks at updates={crack_nu}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Behavioral Sweep")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to partial results JSON to resume from")
    args = parser.parse_args()

    # Load model (HF backend for full context window)
    model, tokenizer, info = load_model_hf(args.model)

    # Pre-flight context check
    ctx_limit = get_context_limit(args.model, info)
    print(f"\nContext limit: {ctx_limit}")
    feasible = compute_feasible_grid(KEY_LEVELS, UPDATE_LEVELS, tokenizer, ctx_limit)
    print(f"Feasible cells: {len(feasible)} / {len(KEY_LEVELS) * len(UPDATE_LEVELS)}")

    # Print feasibility summary
    print(f"\n{'Keys':>6} | Max updates | Est. tokens")
    print(f"{'-'*6}-+-{'-'*11}-+-{'-'*11}")
    for nk in KEY_LEVELS:
        feasible_updates = sorted([nu for (k, nu) in feasible if k == nk])
        if feasible_updates:
            max_nu = feasible_updates[-1]
            est = feasible[(nk, max_nu)]
            print(f"{nk:>6} | {max_nu:>11} | {est:>11,}")
        else:
            print(f"{nk:>6} | {'NONE':>11} | {'N/A':>11}")

    # Resume if requested
    resume_data = None
    if args.resume:
        print(f"\nResuming from {args.resume}")
        with open(args.resume) as f:
            resume_data = json.load(f)

    # Run sweep
    results = run_sweep(
        model, tokenizer, args.model,
        feasible, trials_per_cell=args.trials,
        resume_data=resume_data,
        backend=info.backend,
    )

    # Save final results
    _save_results(results, args.model, partial=False)

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
