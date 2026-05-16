"""
Remedy Sweep — Format Intervention Experiment.

Tests whether explicit stream structure (labeled, block, landmark) restores
CVQ accuracy that fails on the plain interleaved stream.

Three remedy formats, all using the same FVQ/CVQ queries as the plain sweep:
  labeled  — inline update counter: "key (update j): value"
  block    — round header: [Round j] groups all keys per round
  landmark — round boundary: --- separator between rounds, no numbers

Dataset is configurable (default: SEMANTIC_MULTI). The same trial seed and
category/value draws are used across formats so results are comparable.

Usage:
  # All 3 formats, default grid, SEMANTIC_MULTI
  python remedy_sweep.py --model claude-haiku

  # Single format
  python remedy_sweep.py --model gpt-4.1 --formats labeled

  # Arbitrary-Single dataset
  python remedy_sweep.py --model claude-sonnet --dataset ARBITRARY_SINGLE

  # Resume
  python remedy_sweep.py --model claude-haiku --formats block \\
      --resume experiments_cloud/results/remedy/claude-haiku/block/sweep_partial.json
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

warnings.filterwarnings("ignore", category=FutureWarning, module="google")
warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")

from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

_SCRIPT_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.model_factory import create_model
from mechanistic_probing_v2.core.dataset_configs import (
    get_eligible_categories,
    generate_values_for_trial,
)
from mechanistic_probing_v2.core.evaluation import classify_error, bootstrap_ci


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

SUPPORTED_DATASETS = ["SEMANTIC_MULTI", "ARBITRARY_SINGLE"]
DEFAULT_DATASET    = "SEMANTIC_MULTI"
DEFAULT_SAVE       = "experiments_cloud/results/remedy"
ALL_FORMATS        = ["labeled", "block", "landmark"]

TIER_GRIDS = {
    "small": {
        "key_levels":    [2, 3, 5, 7, 10, 12, 15, 20, 25, 30, 45],
        "update_levels": [1, 3, 5, 10, 15, 20, 30, 40, 50],
    },
    "medium": {
        "key_levels":    [2, 5, 10, 15, 20, 25, 30, 40, 45],
        "update_levels": [1, 5, 10, 15, 20, 30, 50],
    },
}
DEFAULT_TIER     = "medium"
DEFAULT_TRIALS   = 200
DEFAULT_WORKERS  = 40
CI_THRESHOLD     = 0.07
MIN_TRIALS_CHECK = DEFAULT_WORKERS

NEAR_ZERO = 0.015
RECOVERY  = 0.06
SAT_COUNT = 3


# ═══════════════════════════════════════════════════════════════════════════════
# THREAD-LOCAL MODEL
# ═══════════════════════════════════════════════════════════════════════════════

_thread_local = threading.local()

def get_thread_model(model_name):
    if not hasattr(_thread_local, "model"):
        _thread_local.model = create_model(model_name, {"verbose": False})
    return _thread_local.model


# ═══════════════════════════════════════════════════════════════════════════════
# STREAM GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def shuffle_no_consecutive(items, rng):
    """Shuffle so no two adjacent entries share the same category."""
    items = items[:]
    for _ in range(200):
        rng.shuffle(items)
        if all(items[i]["category"] != items[i+1]["category"]
               for i in range(len(items)-1)):
            return items
    return items


def build_trial_streams(categories, values_per_cat, rng):
    """
    Build flat (interleaved) and block (round-grouped) streams for one trial.

    Returns:
        flat_items  — shuffled list of {category, value, update_idx}
                      update_idx is chronological (1 = first update to that key)
        block_items — list of rounds; each round is a shuffled list of
                      {category, value, update_idx}
    """
    num_updates = len(values_per_cat[categories[0]])

    # Flat interleaved — shuffle, no consecutive same category
    flat_items = []
    for cat in categories:
        for k, val in enumerate(values_per_cat[cat], start=1):
            flat_items.append({"category": cat, "value": val, "update_idx": k})
    flat_items = shuffle_no_consecutive(flat_items, rng)

    # Block — one list per round, keys shuffled within round
    block_items = []
    for k in range(1, num_updates + 1):
        block = [{"category": cat, "value": values_per_cat[cat][k-1], "update_idx": k}
                 for cat in categories]
        rng.shuffle(block)
        block_items.append(block)

    return flat_items, block_items


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDERS — same FVQ/CVQ query, different stream structure
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "Output ONLY a single word — the exact value requested. "
    "No explanation, no punctuation, no extra words."
)


def _query(test_category, condition):
    word = "first" if condition == "RI" else "last"
    return (f"What was the {word} value of {test_category}?\n"
            f"Answer with ONLY the exact value. No explanation.")


def build_prompt_labeled(flat_items, test_category, condition):
    """
    Labeled stream: 'key (update j): value' per line, randomly interleaved.
    Preamble explains the update counter notation.
    Query: first / last value of target key.
    """
    stream = "\n".join(
        f"{item['category']} (update {item['update_idx']}): {item['value']}"
        for item in flat_items
    )
    return (
        f"Read the following key-value stream. Each key is updated multiple times. "
        f"The number in parentheses after each key name indicates which update of that "
        f"key it is — for example, 'color (update 3): green' means the 3rd time "
        f"color was updated, its value was 'green'.\n\n"
        f"{stream}\n\n"
        f"{_query(test_category, condition)}"
    )


def build_prompt_block(block_items, test_category, condition):
    """
    Block stream: [Round j] header groups all keys per round.
    Keys are shuffled within each round.
    Query: first / last value of target key.
    """
    lines = []
    for block in block_items:
        lines.append(f"[Round {block[0]['update_idx']}]")
        for item in block:
            lines.append(f"  {item['category']}: {item['value']}")
    stream = "\n".join(lines)
    return (
        f"Read the following key-value stream. Each key is updated multiple times, "
        f"grouped by update round.\n\n"
        f"{stream}\n\n"
        f"{_query(test_category, condition)}"
    )


def build_prompt_landmark(block_items, test_category, condition):
    """
    Landmark stream: keys shuffled within each round, rounds separated by '---'.
    No explicit round numbers — only the boundary marker is structural.
    Query: first / last value of target key.
    """
    rounds = []
    for block in block_items:
        rounds.append("\n".join(
            f"{item['category']}: {item['value']}" for item in block
        ))
    stream = "\n---\n".join(rounds)
    return (
        f"Read the following key-value stream. Each key is updated multiple times. "
        f"The '---' marker separates consecutive update rounds.\n\n"
        f"{stream}\n\n"
        f"{_query(test_category, condition)}"
    )


PROMPT_BUILDERS = {
    "labeled":  build_prompt_labeled,
    "block":    build_prompt_block,
    "landmark": build_prompt_landmark,
}

# labeled uses flat_items; block and landmark use block_items
USES_BLOCK = {"block", "landmark"}


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR DETAIL
# ═══════════════════════════════════════════════════════════════════════════════

def get_error_detail(predicted, all_values):
    pred_lower = predicted.lower().strip()
    for idx, val in enumerate(all_values):
        v = val.lower()
        if v in pred_lower or pred_lower.startswith(v):
            return {
                "predicted_idx": idx,
                "predicted_relative_pos": round(idx / max(len(all_values)-1, 1), 4),
            }
    return {"predicted_idx": None, "predicted_relative_pos": None}


# ═══════════════════════════════════════════════════════════════════════════════
# SATURATION + WILSON CI
# ═══════════════════════════════════════════════════════════════════════════════

def init_saturation(key_levels):
    return {nk: {"RI": {"zero_count": 0}, "PI": {"zero_count": 0}}
            for nk in key_levels}


def update_saturation(sat, nk, cond, acc):
    if acc <= NEAR_ZERO:
        sat[nk][cond]["zero_count"] += 1
    elif acc >= RECOVERY:
        sat[nk][cond]["zero_count"] = 0


def is_saturated(sat, nk):
    return any(sat[nk][c]["zero_count"] >= SAT_COUNT for c in ["RI", "PI"])


def wilson_half_width(n, k, z=1.96):
    if n == 0: return 1.0
    p = k / n
    return z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / (1 + z**2/n)


def has_converged(all_results, threshold=CI_THRESHOLD, min_trials=MIN_TRIALS_CHECK):
    for cond in ["RI", "PI"]:
        corrects = [r["correct"] for r in all_results[cond]]
        n = len(corrects)
        if n < min_trials: return False
        if wilson_half_width(n, sum(corrects)) > threshold: return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# TRIAL RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_trial(model_name, fmt, dataset_type, num_keys, num_updates, trial_idx, eligible_cats):
    model  = get_thread_model(model_name)
    seed   = abs(hash((num_keys, num_updates, trial_idx, fmt))) % (2**31)
    rng    = random.Random(seed)

    categories    = rng.sample(eligible_cats, num_keys)
    test_category = rng.choice(categories)

    values_per_cat = generate_values_for_trial(dataset_type, categories, num_updates, rng)
    flat_items, block_items = build_trial_streams(categories, values_per_cat, rng)

    all_values    = values_per_cat[test_category]       # chronological order
    initial_value = all_values[0]
    final_value   = all_values[-1]

    builder = PROMPT_BUILDERS[fmt]
    items   = block_items if fmt in USES_BLOCK else flat_items

    results = {}
    for condition in ["RI", "PI"]:
        expected = initial_value if condition == "RI" else final_value
        prompt   = builder(items, test_category, condition)
        response = model.generate(prompt)

        token_usage = {
            "input_tokens":  model.last_input_tokens,
            "output_tokens": model.last_output_tokens,
            "was_truncated": model.last_was_truncated,
            "error":         model.last_error,
        }

        predicted  = response.strip()
        error_type = classify_error(predicted, expected, initial_value, final_value,
                                    all_values, condition)
        results[condition] = {
            "trial_idx":   trial_idx,
            "seed":        seed,
            "condition":   condition,
            "test_category": test_category,
            "num_keys":    num_keys,
            "num_updates": num_updates,
            "expected":    expected,
            "predicted":   predicted,
            "correct":     error_type == "correct",
            "error_type":  error_type,
            "error_detail": get_error_detail(predicted, all_values),
            "token_usage": token_usage,
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CELL RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_cell(model_name, fmt, dataset_type, num_keys, num_updates,
             trials_per_cell, workers, eligible_cats):
    all_trial_results = {"RI": [], "PI": []}
    cell_start = time.time()
    max_batches = (trials_per_cell + workers - 1) // workers
    trial_idx = 0

    for batch_num in range(1, max_batches + 1):
        batch_size = min(workers, trials_per_cell - (batch_num-1)*workers)
        futures = {}
        with ThreadPoolExecutor(max_workers=batch_size) as ex:
            for _ in range(batch_size):
                f = ex.submit(run_trial, model_name, fmt, dataset_type,
                              num_keys, num_updates, trial_idx, eligible_cats)
                futures[f] = trial_idx
                trial_idx += 1

            for future in as_completed(futures):
                try:
                    res = future.result()
                    for cond in ["RI", "PI"]:
                        all_trial_results[cond].append(res[cond])
                except Exception as e:
                    for cond in ["RI", "PI"]:
                        all_trial_results[cond].append({
                            "correct": False, "error_type": f"exception:{e}",
                            "trial_idx": futures[future],
                        })

        n_done = len(all_trial_results["RI"])
        ri_acc = sum(r["correct"] for r in all_trial_results["RI"]) / n_done
        pi_acc = sum(r["correct"] for r in all_trial_results["PI"]) / n_done
        ri_hw  = wilson_half_width(n_done, sum(r["correct"] for r in all_trial_results["RI"]))
        pi_hw  = wilson_half_width(n_done, sum(r["correct"] for r in all_trial_results["PI"]))

        converged = has_converged(all_trial_results)
        print(f"    batch {batch_num}/{max_batches} ({n_done} trials): "
              f"RI={ri_acc:.0%}±{ri_hw:.0%}  PI={pi_acc:.0%}±{pi_hw:.0%}"
              + ("  → converged, stopping early" if converged else ""))
        if converged:
            break

    # Build cell summary
    n = len(all_trial_results["RI"])
    stats = {}
    for cond, label in [("RI", "RI"), ("PI", "PI")]:
        corrects = [r["correct"] for r in all_trial_results[cond]]
        k = sum(corrects)
        acc = k / n if n else 0.0
        _, ci_lo, ci_hi = bootstrap_ci(corrects)
        regime = "A"
        if acc < 0.50 and (sum(r["correct"] for r in all_trial_results["RI"])/n) >= 0.80:
            regime = "B"
        stats[label] = {
            "accuracy": round(acc, 4),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
            "n": n,
            "error_types": {
                et: sum(1 for r in all_trial_results[cond] if r.get("error_type") == et)
                for et in set(r.get("error_type","") for r in all_trial_results[cond])
            }
        }

    ri_acc_final = stats["RI"]["accuracy"]
    pi_acc_final = stats["PI"]["accuracy"]
    if ri_acc_final >= 0.80 and pi_acc_final < 0.50:
        regime = "B"
    elif ri_acc_final < 0.50 and pi_acc_final < 0.50:
        regime = "C"
    elif ri_acc_final < 0.50 and pi_acc_final >= 0.80:
        regime = "D"
    else:
        regime = "A"

    cell_summary = {
        "num_keys":    num_keys,
        "num_updates": num_updates,
        "format":      fmt,
        "dataset":     dataset_type,
        "regime":      regime,
        "elapsed_sec": round(time.time() - cell_start, 1),
        "n_trials":    n,
        "stats":       stats,
    }
    return cell_summary, all_trial_results


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT
# ═══════════════════════════════════════════════════════════════════════════════

def load_checkpoint(path):
    with open(path) as f:
        return json.load(f)


def save_checkpoint(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Remedy format sweep")
    p.add_argument("--model",    default="claude-haiku")
    p.add_argument("--dataset",  default=DEFAULT_DATASET,
                   choices=SUPPORTED_DATASETS,
                   help=f"Dataset to use (default: {DEFAULT_DATASET})")
    p.add_argument("--formats",  nargs="+", default=ALL_FORMATS,
                   choices=ALL_FORMATS, metavar="FORMAT",
                   help=f"Formats to run (default: all). Choices: {ALL_FORMATS}")
    p.add_argument("--tier",     default=DEFAULT_TIER,
                   choices=list(TIER_GRIDS.keys()),
                   help=f"Grid tier (default: {DEFAULT_TIER})")
    p.add_argument("--key-levels",    type=int, nargs="+", default=None)
    p.add_argument("--update-levels", type=int, nargs="+", default=None)
    p.add_argument("--trials",   type=int, default=DEFAULT_TRIALS)
    p.add_argument("--workers",  type=int, default=DEFAULT_WORKERS)
    p.add_argument("--save-dir", default=DEFAULT_SAVE)
    p.add_argument("--resume",   default=None,
                   help="Path to sweep_partial.json to resume from")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    grid = TIER_GRIDS[args.tier]
    key_levels    = args.key_levels    or grid["key_levels"]
    update_levels = args.update_levels or grid["update_levels"]
    dataset_type  = args.dataset

    print("=" * 70)
    print("REMEDY FORMAT SWEEP")
    print(f"  Model:    {args.model}")
    print(f"  Dataset:  {dataset_type}")
    print(f"  Formats:  {args.formats}")
    print(f"  Tier:     {args.tier}")
    print(f"  Trials:   {args.trials} per cell")
    print(f"  Workers:  {args.workers} concurrent trials")
    print(f"  Grid:     {len(key_levels)} key levels × {len(update_levels)} update levels")
    print("=" * 70)

    # Verify model
    print("\nVerifying model connection...", end=" ", flush=True)
    test_model = create_model(args.model, {"verbose": False})
    _ = test_model.generate("Reply with: ok")
    print(f"✓  ({test_model.model_id})\n")

    # Run each format independently with its own checkpoint
    for fmt in args.formats:
        save_dir  = Path(args.save_dir) / args.model / fmt
        ckpt_path = save_dir / "sweep_partial.json"

        # Load or init checkpoint
        if args.resume and Path(args.resume).exists():
            results = load_checkpoint(args.resume)
            print(f"[{fmt}] Resuming from {args.resume}")
        elif ckpt_path.exists():
            results = load_checkpoint(ckpt_path)
            print(f"[{fmt}] Resuming from {ckpt_path}")
        else:
            results = {
                "model":      args.model,
                "dataset":    dataset_type,
                "format":     fmt,
                "config": {
                    "key_levels":    key_levels,
                    "update_levels": update_levels,
                    "max_trials":    args.trials,
                    "ci_threshold":  CI_THRESHOLD,
                    "tier":          args.tier,
                },
                "start_time": datetime.now(timezone.utc).isoformat(),
                "cells":      {},
            }

        sat = init_saturation(key_levels)
        # Restore saturation from completed cells
        for cell in results["cells"].values():
            nk = cell["num_keys"]
            for cond in ["RI", "PI"]:
                update_saturation(sat, nk, cond, cell["stats"][cond]["accuracy"])

        total_cells = len(key_levels) * len(update_levels)
        done_cells  = len(results["cells"])
        cell_num    = 0

        print(f"\n{'='*70}")
        print(f"FORMAT: {fmt.upper()}")
        print(f"{'='*70}")

        for nk in key_levels:
            if is_saturated(sat, nk):
                print(f"  [SKIPPED K={nk} — saturated]")
                cell_num += len(update_levels)
                continue

            eligible = get_eligible_categories(dataset_type, min_values=max(update_levels))

            for nu in update_levels:
                cell_num += 1
                cell_key = f"{nk}_{nu}"

                if cell_key in results["cells"]:
                    c = results["cells"][cell_key]
                    update_saturation(sat, nk, "RI", c["stats"]["RI"]["accuracy"])
                    update_saturation(sat, nk, "PI", c["stats"]["PI"]["accuracy"])
                    continue

                eligible_nu = get_eligible_categories(dataset_type, min_values=nu)
                if len(eligible_nu) < nk:
                    print(f"  [{cell_num}/{total_cells}] K={nk:>3} N={nu:>3}: "
                          f"SKIPPED (only {len(eligible_nu)} eligible categories, need {nk})")
                    continue

                print(f"  [{cell_num}/{total_cells}] K={nk:>3} N={nu:>3} "
                      f"({args.trials} trials × 2 conditions = {args.trials*2} API calls)",
                      flush=True)

                cell_sum, _ = run_cell(
                    args.model, fmt, dataset_type,
                    nk, nu, args.trials, args.workers, eligible_nu,
                )

                ri_acc = cell_sum["stats"]["RI"]["accuracy"]
                pi_acc = cell_sum["stats"]["PI"]["accuracy"]
                ri_ci  = cell_sum["stats"]["RI"]
                pi_ci  = cell_sum["stats"]["PI"]
                print(f"  -> RI={ri_acc:.0%} [{ri_ci['ci_lower']:.0%}-{ri_ci['ci_upper']:.0%}]  "
                      f"PI={pi_acc:.0%} [{pi_ci['ci_lower']:.0%}-{pi_ci['ci_upper']:.0%}]  "
                      f"regime={cell_sum['regime']}  ({cell_sum['elapsed_sec']:.1f}s)")

                results["cells"][cell_key] = cell_sum
                update_saturation(sat, nk, "RI", ri_acc)
                update_saturation(sat, nk, "PI", pi_acc)

                # Checkpoint every 3 cells
                if len(results["cells"]) % 3 == 0:
                    save_checkpoint(results, ckpt_path)
                    print(f"  -> Checkpoint: {ckpt_path}")

        results["end_time"] = datetime.now(timezone.utc).isoformat()
        save_checkpoint(results, ckpt_path)

        # Save timestamped final
        ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        final = save_dir / f"sweep_{ts}.json"
        save_checkpoint(results, final)
        print(f"\n  -> Saved: {final}")

        # Print summary table
        print(f"\n{'─'*70}")
        print(f"SUMMARY — {args.model} / {fmt} / {dataset_type}")
        print(f"{'─'*70}")
        print(f"{'keys':>6}", end="")
        for nu in update_levels:
            print(f"  N={nu:<3}", end="")
        print()
        for label, cond in [("RI", "RI"), ("PI", "PI")]:
            print(f"\n--- {label} Accuracy ---")
            for nk in key_levels:
                print(f"  {nk:>4}", end="")
                for nu in update_levels:
                    cell = results["cells"].get(f"{nk}_{nu}")
                    if cell:
                        acc = cell["stats"][cond]["accuracy"]
                        print(f"  {acc:>4.0%}", end="")
                    else:
                        print(f"  {'---':>4}", end="")
                print()

        ri_vals = [c["stats"]["RI"]["accuracy"] for c in results["cells"].values()]
        pi_vals = [c["stats"]["PI"]["accuracy"] for c in results["cells"].values()]
        if ri_vals:
            mean_gap = sum(r-p for r,p in zip(ri_vals,pi_vals)) / len(ri_vals)
            print(f"\nOverall: mean RI={sum(ri_vals)/len(ri_vals):.1%}, "
                  f"mean PI={sum(pi_vals)/len(pi_vals):.1%}, gap={mean_gap:+.1%}")
        print(f"Cells completed: {len(results['cells'])}")


if __name__ == "__main__":
    main()
