"""
Behavioral Sweep with SEMANTIC dataset (KV format).

Same pattern as 11_behavioral_sweep.py but uses generate_semantic_trial()
with real English words verified single-token in Qwen+Gemma tokenizers.

Provides direct KV-format comparison for the narrative dataset results.
Output files include timestamps to avoid overwriting.

Usage:
    cd mechanistic_probing_v2
    ../.venv/bin/python experiments/51_semantic_behavioral_sweep.py \
        --model Qwen/Qwen2.5-1.5B-Instruct --trials 10

    # Match only the narrative dataset grid points:
    ../.venv/bin/python experiments/51_semantic_behavioral_sweep.py \
        --model Qwen/Qwen2.5-1.5B-Instruct --trials 10 --narrative-grid
"""

import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_loader import load_model_hf, get_context_limit
from core.dataset_configs import (
    generate_semantic_trial, compute_feasible_grid,
    KEY_LEVELS, UPDATE_LEVELS, SEMANTIC_CATEGORIES,
)
from core.dataset_configs import format_for_chat
from core.evaluation import classify_error, bootstrap_ci

# Grid that matches narrative dataset configs
NARRATIVE_KEY_LEVELS = [2, 3, 5, 7, 10]
NARRATIVE_UPDATE_LEVELS = [3, 5, 7, 10, 15]


def timestamp_str():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def run_single_trial(model, tokenizer, trial, max_new_tokens=20):
    import torch
    formatted = format_for_chat(trial.prompt, tokenizer)
    inputs = tokenizer(formatted, return_tensors="pt")
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


def main():
    parser = argparse.ArgumentParser(description="Semantic KV Behavioral Sweep")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--trials", type=int, default=10,
                        help="Trials per (keys, updates, condition) cell")
    parser.add_argument("--narrative-grid", action="store_true",
                        help="Only run grid points matching the narrative dataset")
    args = parser.parse_args()

    ts = timestamp_str()
    model_short = args.model.split("/")[-1]

    # Load model
    model, tokenizer, info = load_model_hf(args.model)
    ctx_limit = get_context_limit(args.model, info)
    print(f"Model: {args.model}")
    print(f"Context limit: {ctx_limit}")

    # Determine grid
    if args.narrative_grid:
        key_levels = NARRATIVE_KEY_LEVELS
        update_levels = NARRATIVE_UPDATE_LEVELS
        print("Grid: narrative-matched (keys=[2,3,5,7,10] x updates=[3,5,7,10,15])")
    else:
        key_levels = KEY_LEVELS
        update_levels = UPDATE_LEVELS
        print(f"Grid: full semantic ({len(KEY_LEVELS)} keys x {len(UPDATE_LEVELS)} updates)")

    feasible = compute_feasible_grid(key_levels, update_levels, tokenizer, ctx_limit,
                                     dataset_type="SEMANTIC_SINGLE")
    print(f"Feasible cells: {len(feasible)}")

    # Run sweep
    results = {
        "model": args.model,
        "model_short": model_short,
        "dataset": "semantic_kv",
        "timestamp": ts,
        "config": {
            "key_levels": key_levels,
            "update_levels": update_levels,
            "trials_per_cell": args.trials,
            "num_categories": len(SEMANTIC_CATEGORIES),
        },
        "feasible_cells": {f"{k}_{u}": t for (k, u), t in feasible.items()},
        "cells": {},
    }

    total_cells = len(feasible)
    cell_idx = 0
    saturation = {}

    for nk in key_levels:
        saturation.setdefault(nk, {"RI": 0, "PI": 0})

        for nu in update_levels:
            if (nk, nu) not in feasible:
                continue

            # Check saturation
            if saturation[nk]["RI"] >= 5 and saturation[nk]["PI"] >= 5:
                print(f"  [{cell_idx+1}/{total_cells}] keys={nk}, updates={nu}: SKIPPED (saturated)")
                cell_idx += 1
                continue

            cell_key = f"{nk}_{nu}"
            cell_start = time.time()
            cell_results = {"RI": [], "PI": []}

            for condition in ["RI", "PI"]:
                for trial_idx in range(args.trials):
                    seed = hash((nk, nu, condition, trial_idx)) % (2**31)
                    test_cat_idx = trial_idx % min(nk, len(SEMANTIC_CATEGORIES))

                    trial = generate_semantic_trial(
                        num_keys=nk,
                        num_updates=nu,
                        condition=condition,
                        seed=seed,
                        test_category_idx=test_cat_idx,
                    )

                    result = run_single_trial(model, tokenizer, trial)
                    cell_results[condition].append(result)

            # Compute stats
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
                    saturation[nk][cond] += 1
                else:
                    saturation[nk][cond] = 0

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

            results["cells"][cell_key] = {
                "num_keys": nk,
                "num_updates": nu,
                "stats": cell_stats,
                "regime": regime,
                "trials": cell_results,
                "elapsed_sec": round(time.time() - cell_start, 1),
            }

            cell_idx += 1
            elapsed = results["cells"][cell_key]["elapsed_sec"]
            print(f"  [{cell_idx}/{total_cells}] keys={nk:>2}, updates={nu:>3}: "
                  f"RI={ri_acc:.0%} PI={pi_acc:.0%} regime={regime} ({elapsed:.1f}s)")

    # Summary
    all_ri = []
    all_pi = []
    for cell in results["cells"].values():
        if "trials" in cell:
            all_ri.extend([r["correct"] for r in cell["trials"]["RI"]])
            all_pi.extend([r["correct"] for r in cell["trials"]["PI"]])

    ri_total = sum(all_ri) / max(len(all_ri), 1)
    pi_total = sum(all_pi) / max(len(all_pi), 1)

    results["summary"] = {
        "ri_accuracy": ri_total,
        "pi_accuracy": pi_total,
        "ri_gt_pi": ri_total > pi_total,
        "diff": ri_total - pi_total,
        "ri_n": len(all_ri),
        "pi_n": len(all_pi),
    }

    print("\n" + "=" * 60)
    print(f"SEMANTIC KV SWEEP — {model_short}")
    print("=" * 60)
    print(f"  RI accuracy: {ri_total:.1%} ({sum(all_ri)}/{len(all_ri)})")
    print(f"  PI accuracy: {pi_total:.1%} ({sum(all_pi)}/{len(all_pi)})")
    print(f"  RI > PI: {ri_total > pi_total} (diff = {ri_total - pi_total:+.1%})")

    # Per-cell summary
    print("\nGrid:")
    print(f"{'':>6}", end="")
    for nu in update_levels:
        print(f" {nu:>5}", end="")
    print()
    for cond in ["RI", "PI"]:
        print(f"\n  {cond}:")
        for nk in key_levels:
            print(f"  {nk:>4}", end="")
            for nu in update_levels:
                ck = f"{nk}_{nu}"
                if ck in results["cells"]:
                    acc = results["cells"][ck]["stats"][cond]["accuracy"]
                    print(f" {acc:>5.0%}", end="")
                else:
                    print(f" {'---':>5}", end="")
            print()

    # Error types in PI
    pi_errors = {}
    for cell in results["cells"].values():
        for et, count in cell["stats"]["PI"]["error_types"].items():
            pi_errors[et] = pi_errors.get(et, 0) + count
    pi_total_errs = sum(pi_errors.values())
    print(f"\nPI error distribution:")
    for et, count in sorted(pi_errors.items(), key=lambda x: -x[1]):
        print(f"  {et}: {count}/{pi_total_errs} = {count/pi_total_errs:.1%}")

    # Save — with timestamp
    results_dir = Path(__file__).parent.parent / "results" / model_short
    results_dir.mkdir(parents=True, exist_ok=True)

    # Strip trials for summary file
    summary = {k: v for k, v in results.items() if k != "cells"}
    summary["cells"] = {}
    for ck, cd in results["cells"].items():
        summary["cells"][ck] = {k: v for k, v in cd.items() if k != "trials"}

    summary_path = results_dir / f"semantic_sweep_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    full_path = results_dir / f"semantic_sweep_{ts}_full.json"
    with open(full_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to: {full_path}")


if __name__ == "__main__":
    main()
