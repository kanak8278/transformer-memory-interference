"""
V3 Stage 1: Behavioral Sweep + Error Characterization

Combined behavioral sweep and failure analysis in a single pass.
Every trial records WHERE the model's prediction lands in the value sequence,
raw output text, and output length — enabling post-hoc analysis of failure modes
without re-running experiments.

Framing: PI failure = model fails to retrieve the final value.
What it outputs instead is tracked empirically per trial.

Constraints:
  - Minimum 5 updates per key (avoids floor artifacts)
  - ARBITRARY_SINGLE dataset for local models (single-token values)
  - Results saved to v3/results/{model}/

Usage:
    cd v3
    python stage1_sweep.py --model Qwen/Qwen2.5-1.5B-Instruct --gpu 0
    python stage1_sweep.py --model Qwen/Qwen2.5-3B-Instruct --gpu 1
    python stage1_sweep.py --model google/gemma-3-1b-it --gpu 2
    python stage1_sweep.py --model EleutherAI/pythia-410m --gpu 3

    # Quick test
    python stage1_sweep.py --model Qwen/Qwen2.5-1.5B-Instruct --trials 10 --gpu 0
"""

import sys
import json
import time
import random
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mechanistic_probing_v2.core.dataset_configs import (
    load_dataset_config,
    get_eligible_categories,
    get_max_updates,
    generate_values_for_trial,
    build_interleaved_sequence,
    FIXED_COMPLETION_DEMOS,
)
from mechanistic_probing_v2.core.model_loader import (
    load_model_hf,
    clear_accelerator_cache,
    is_instruct_model,
    model_short_name,
)
from mechanistic_probing_v2.core.inference import run_batch_with_oom_fallback
from mechanistic_probing_v2.core.evaluation import classify_error, bootstrap_ci


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_TYPE = "ARBITRARY_SINGLE"
SYSTEM_PROMPT = (
    "You are a precise data extraction tool. "
    "Output ONLY a single word - the exact value requested. "
    "No other text, no explanation, no punctuation."
)
MIN_UPDATES = 5  # Minimum updates per key — avoids floor artifacts

# ── Grid presets ──────────────────────────────────────────────────────────────
# Edit these directly for your use case, or override via CLI --key-levels / --update-levels.
# All update levels below MIN_UPDATES are auto-filtered at runtime.
#
# "small" is the default for base/completion models (short context, weaker).
# "instruct" is the default for instruction-tuned models (32K context, stronger).
# Switch presets via the ACTIVE_GRID variable below, or just use CLI overrides.

GRID_PRESETS = {
    "small": {
        "key_levels":    [5, 7, 10, 12, 15, 20, 25, 30],
        "update_levels": [2, 5, 7, 10, 12, 15, 20, 30, 40, 50],
    },
    "instruct": {
        "key_levels":    [2, 3, 5, 7, 10, 15, 20, 25, 30],
        "update_levels": [5, 10, 15, 20, 30, 50, 100],
    },
    "quick": {
        # Smoke test: 3 key levels × 4 update levels = 12 cells
        "key_levels":    [3, 5, 10],
        "update_levels": [5, 10, 20, 50],
    },
    "focused": {
        # Regime B hunting: few keys, many update levels to find the transition
        "key_levels":    [2, 3, 5],
        "update_levels": [5, 7, 10, 12, 15, 20, 25, 30, 40, 50],
    },
}

DEFAULT_TRIALS = 200
DEFAULT_BATCH_SIZE = 8

# ── Early stopping: Wilson CI convergence ─────────────────────────────────────
# Stop a cell early when CI is tight enough — saves trials on ceiling/floor cells,
# spends full budget on uncertain transition cells near the inflection point.
CI_THRESHOLD = 0.05      # Wilson CI half-width threshold (±7%)
MIN_TRIALS_CHECK = 25    # First convergence check after this many trials

# ── Three-zone saturation ─────────────────────────────────────────────────────
# Skip remaining update levels for a key count once accuracy has been near-zero
# for SAT_COUNT consecutive update levels.
NEAR_ZERO = 0.015   # acc <= 1.5% → near-zero
RECOVERY  = 0.06    # acc >= 6%   → reset counter
SAT_COUNT = 3       # stop after this many near-zero without recovery


# ═══════════════════════════════════════════════════════════════════════════════
# WILSON CI
# ═══════════════════════════════════════════════════════════════════════════════

def wilson_half_width(n, k, z=1.96):
    """95% Wilson score interval half-width for k successes in n trials."""
    if n == 0:
        return 1.0
    p = k / n
    return z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)


def check_convergence(trial_results, threshold=CI_THRESHOLD, min_trials=MIN_TRIALS_CHECK):
    """Return True if BOTH RI and PI have CI half-width <= threshold."""
    for cond in ["RI", "PI"]:
        if cond not in trial_results:
            return False
        corrects = [r["correct"] for r in trial_results[cond]]
        n = len(corrects)
        if n < min_trials:
            return False
        k = sum(corrects)
        hw = wilson_half_width(n, k)
        if hw > threshold:
            return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# SATURATION
# ═══════════════════════════════════════════════════════════════════════════════

def init_saturation(key_levels):
    return {nk: {"RI": {"zero_count": 0}, "PI": {"zero_count": 0}}
            for nk in key_levels}


def update_saturation(sat, nk, cond, acc):
    """Three-zone saturation update."""
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
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="V3 Stage 1: Behavioral sweep + error characterization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--gpu", type=int, default=None,
                   help="Physical GPU index (e.g. 0 for first GPU). Default: auto-detect.")
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                   help=f"Trials per cell per condition (default: {DEFAULT_TRIALS})")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--max-new-tokens", type=int, default=30,
                   help="Max tokens to generate (default: 30, enough to detect multi-token outputs)")
    p.add_argument("--grid", default=None, choices=list(GRID_PRESETS.keys()),
                   help="Grid preset (default: auto based on model type). See GRID_PRESETS in source.")
    p.add_argument("--key-levels", type=int, nargs="+", default=None,
                   help="Override key levels (highest priority)")
    p.add_argument("--update-levels", type=int, nargs="+", default=None,
                   help="Override update levels (highest priority)")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to partial results JSON to resume from")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# INTERLEAVING
# ═══════════════════════════════════════════════════════════════════════════════

def shuffle_no_consecutive(items, rng, max_attempts=100):
    """Shuffle items so no two consecutive items share the same category."""
    for _ in range(max_attempts):
        candidate = items.copy()
        rng.shuffle(candidate)
        ok = all(candidate[i]["category"] != candidate[i - 1]["category"]
                 for i in range(1, len(candidate)))
        if ok:
            return candidate
    # Fallback: greedy
    remaining = items.copy()
    rng.shuffle(remaining)
    result = []
    last_cat = None
    while remaining:
        valid = [i for i, item in enumerate(remaining) if item["category"] != last_cat]
        if not valid:
            result.extend(remaining)
            break
        idx = rng.choice(valid)
        item = remaining.pop(idx)
        result.append(item)
        last_cat = item["category"]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# TRIAL GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_trial(num_keys, num_updates, condition, seed,
                   use_chat_format=False, tokenizer=None):
    """Generate a single trial with full metadata for error analysis."""
    rng = random.Random(seed)

    eligible = get_eligible_categories(DATASET_TYPE, min_values=num_updates)
    categories = rng.sample(eligible, min(num_keys, len(eligible)))
    test_category = categories[seed % num_keys]

    values_per_cat = generate_values_for_trial(
        DATASET_TYPE, categories, num_updates, rng
    )

    items = []
    for cat in categories:
        for val in values_per_cat[cat]:
            items.append({"category": cat, "value": val})
    items = shuffle_no_consecutive(items, rng)

    stream = "\n".join(f"{it['category']}: {it['value']}" for it in items)
    query_word = "first" if condition == "RI" else "last"

    # All values for test category in sequence order
    cat_values = [it["value"] for it in items if it["category"] == test_category]
    expected = cat_values[0] if condition == "RI" else cat_values[-1]

    raw_prompt = (
        f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{stream}\n\n"
        f"What was the {query_word} value of {test_category}?"
    )

    if use_chat_format:
        if tokenizer and hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": raw_prompt},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = raw_prompt
    else:
        prompt = f"{FIXED_COMPLETION_DEMOS}{stream}\nThe {query_word} value of {test_category} was:"

    return {
        "prompt": prompt,
        "condition": condition,
        "expected": expected,
        "initial_value": cat_values[0],
        "final_value": cat_values[-1],
        "all_values": cat_values,
        "test_category": test_category,
        "num_keys": num_keys,
        "num_updates": num_updates,
        "seed": seed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR DETAIL — where does the prediction land?
# ═══════════════════════════════════════════════════════════════════════════════

def get_error_detail(predicted, all_values):
    """Return index and relative position of predicted value in the sequence.

    all_values: values for test_category in sequence order (v_0 to v_{N-1}).
    Returns:
        predicted_idx: 0 to N-1 if matched, None if garbage
        predicted_relative_pos: 0.0 = first, 1.0 = last, None if garbage
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
# PREFLIGHT
# ═══════════════════════════════════════════════════════════════════════════════

def preflight_check(num_keys, num_updates, tokenizer, context_limit,
                    use_chat_format=False):
    """Check if a cell fits in context and has enough values."""
    eligible = get_eligible_categories(DATASET_TYPE, min_values=num_updates)
    if len(eligible) < num_keys:
        return False, 0, f"need {num_keys} categories with {num_updates}+ values, only {len(eligible)}"

    trial = generate_trial(num_keys, num_updates, "RI", seed=0,
                           use_chat_format=use_chat_format, tokenizer=tokenizer)
    n_tokens = len(tokenizer.encode(trial["prompt"]))
    limit = int(context_limit * 0.85)
    if n_tokens > limit:
        return False, n_tokens, f"~{n_tokens} tokens > {limit} limit"

    return True, n_tokens, "ok"


# ═══════════════════════════════════════════════════════════════════════════════
# CELL RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_cell(model, tokenizer, model_name, num_keys, num_updates,
             trials_per_cell, batch_size, max_new_tokens, device, use_chat):
    """Run trials for one (keys, updates) cell with early stopping on CI convergence.

    Runs RI and PI interleaved in batches. After each batch, checks Wilson CI
    convergence. Stops early when both RI and PI have tight CIs (±7%), concentrating
    the trial budget on uncertain transition cells.

    Returns (cell_summary, trial_details).
    """
    cell_start = time.time()
    cell_trials = {"RI": [], "PI": []}
    stopped_early = False

    # Pre-generate all trials for both conditions
    all_trials = {"RI": [], "PI": []}
    for condition in ["RI", "PI"]:
        for t_idx in range(trials_per_cell):
            seed = hash((num_keys, num_updates, condition, t_idx, "v3")) % (2**31)
            trial = generate_trial(num_keys, num_updates, condition, seed,
                                   use_chat_format=use_chat, tokenizer=tokenizer)
            all_trials[condition].append(trial)

    # Run in interleaved batches, checking convergence after each
    trial_idx = 0
    while trial_idx < trials_per_cell:
        for condition in ["RI", "PI"]:
            batch_end = min(trial_idx + batch_size, trials_per_cell)
            batch_trials = all_trials[condition][trial_idx:batch_end]
            if not batch_trials:
                continue

            prompts = [t["prompt"] for t in batch_trials]
            answers, effective_bs = run_batch_with_oom_fallback(
                model, tokenizer, prompts,
                max_new_tokens=max_new_tokens, device=device
            )
            if effective_bs < batch_size:
                batch_size = effective_bs  # Sticky reduction

            for trial, answer in zip(batch_trials, answers):
                error_type = classify_error(
                    answer, trial["expected"],
                    trial["initial_value"], trial["final_value"],
                    trial["all_values"], trial["condition"],
                )
                error_detail = get_error_detail(answer, trial["all_values"])

                # Expected position: 0 for RI (first), N-1 for PI (last)
                n_vals = len(trial["all_values"])
                expected_idx = 0 if condition == "RI" else n_vals - 1
                expected_rel = 0.0 if condition == "RI" else 1.0

                cell_trials[condition].append({
                    "seed": trial["seed"],
                    "condition": condition,
                    "query_word": "first" if condition == "RI" else "last",
                    "expected": trial["expected"],
                    "expected_idx": expected_idx,
                    "expected_relative_pos": expected_rel,
                    "predicted": answer,
                    "correct": error_type == "correct",
                    "error_type": error_type,
                    "predicted_idx": error_detail["predicted_idx"],
                    "predicted_relative_pos": error_detail["predicted_relative_pos"],
                    "all_values": trial["all_values"],
                    "output_length": len(answer.split()),
                    "output_raw": answer,
                })

        trial_idx += batch_size

        # Check convergence after each batch
        n_done = len(cell_trials["RI"])
        if n_done >= MIN_TRIALS_CHECK and check_convergence(cell_trials):
            ri_n = len(cell_trials["RI"])
            ri_k = sum(r["correct"] for r in cell_trials["RI"])
            pi_k = sum(r["correct"] for r in cell_trials["PI"])
            print(f"    converged at {n_done} trials: "
                  f"RI={ri_k/ri_n:.0%}±{wilson_half_width(ri_n, ri_k):.0%}  "
                  f"PI={pi_k/ri_n:.0%}±{wilson_half_width(ri_n, pi_k):.0%}")
            stopped_early = True
            break

    # Aggregate stats
    cell_stats = {}
    for cond in ["RI", "PI"]:
        results = cell_trials[cond]
        corrects = [r["correct"] for r in results]
        mean, ci_lo, ci_hi = bootstrap_ci(corrects)

        # Error type counts
        error_counts = {}
        for r in results:
            et = r["error_type"]
            error_counts[et] = error_counts.get(et, 0) + 1

        # Failure position analysis (PI only, non-garbage failures)
        failure_positions = [r["predicted_relative_pos"]
                            for r in results
                            if not r["correct"] and r["predicted_relative_pos"] is not None]
        garbage_count = sum(1 for r in results
                          if not r["correct"] and r["predicted_idx"] is None)

        cell_stats[cond] = {
            "accuracy": round(mean, 4),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
            "n": len(corrects),
            "error_types": error_counts,
            "n_failures": sum(1 for r in results if not r["correct"]),
            "n_garbage": garbage_count,
            "failure_avg_relative_pos": (
                round(float(np.mean(failure_positions)), 4)
                if failure_positions else None
            ),
        }

    ri_acc = cell_stats["RI"]["accuracy"]
    pi_acc = cell_stats["PI"]["accuracy"]
    gap = ri_acc - pi_acc  # positive = RI better than PI (expected)

    # Regime classification based on RI-PI gap
    if pi_acc > ri_acc:
        regime = "D"       # reversed: PI > RI
    elif gap >= 0.25:
        regime = "C"       # large gap: PI failing hard (≥25%)
    elif gap >= 0.15:
        regime = "B"       # moderate gap: PI starting to fail (≥15%)
    elif gap < 0.05:
        regime = "A"       # no gap: both similar (<5%)
    else:
        regime = "AB"      # transition zone: 5-15% gap

    summary = {
        "num_keys": num_keys,
        "num_updates": num_updates,
        "stats": cell_stats,
        "regime": regime,
        "n_trials": len(cell_trials["RI"]),
        "max_trials": trials_per_cell,
        "stopped_early": stopped_early,
        "elapsed_sec": round(time.time() - cell_start, 1),
    }

    return summary, cell_trials


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════════

def get_save_dir(model_name):
    m_short = model_short_name(model_name)
    return Path(__file__).resolve().parent / "results" / m_short


def save_results(results, all_trials, model_name, partial=False):
    save_dir = get_save_dir(model_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Summary (no trial details — small file for quick loading)
    summary = {k: v for k, v in results.items() if k != "cells"}
    summary["cells"] = {}
    for ck, cd in results["cells"].items():
        summary["cells"][ck] = cd  # cell summary, no trials

    # Full data (with all trial details including raw outputs)
    full_data = dict(results)
    full_data["trial_details"] = all_trials

    if partial:
        # Single checkpoint file — overwritten every 3 cells, use with --resume
        checkpoint_path = save_dir / "stage1_checkpoint.json"
        with open(checkpoint_path, "w") as f:
            json.dump(full_data, f, indent=2)
        print(f"  -> Checkpoint: {checkpoint_path}")
    else:
        # Final save — timestamped, never overwritten
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        summary_path = save_dir / f"stage1_sweep_{ts}.json"
        full_path = save_dir / f"stage1_trials_{ts}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        with open(full_path, "w") as f:
            json.dump(full_data, f, indent=2)
        print(f"  -> Summary: {summary_path}")
        print(f"  -> Full:    {full_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(results):
    kl = results["config"]["key_levels"]
    ul = results["config"]["update_levels"]

    print("\n" + "=" * 70)
    print(f"V3 STAGE 1 SUMMARY — {results['model']}")
    print(f"Dataset: {DATASET_TYPE} | Min updates: {MIN_UPDATES}")
    print("=" * 70)

    for cond in ["RI", "PI"]:
        print(f"\n--- {cond} Accuracy ---")
        print(f"{'Keys':>6}", end="")
        for nu in ul:
            print(f" {nu:>5}", end="")
        print()
        for nk in kl:
            print(f"{nk:>6}", end="")
            for nu in ul:
                ck = f"{nk}_{nu}"
                if ck in results["cells"]:
                    acc = results["cells"][ck]["stats"][cond]["accuracy"]
                    print(f" {acc:>5.0%}", end="")
                else:
                    print(f" {'---':>5}", end="")
            print()

    print(f"\n--- Regime Map ---")
    print(f"{'Keys':>6}", end="")
    for nu in ul:
        print(f" {nu:>5}", end="")
    print()
    for nk in kl:
        print(f"{nk:>6}", end="")
        for nu in ul:
            ck = f"{nk}_{nu}"
            if ck in results["cells"]:
                print(f" {results['cells'][ck]['regime']:>5}", end="")
            else:
                print(f" {'---':>5}", end="")
        print()

    # PI failure analysis — the key new output
    print(f"\n--- PI Failure Position Analysis ---")
    print(f"{'Cell':>8} {'PI Acc':>7} {'Fail AvgPos':>11} {'Garbage':>8} {'Near-last':>10}")
    print("-" * 50)
    for nk in kl:
        for nu in ul:
            ck = f"{nk}_{nu}"
            if ck not in results["cells"]:
                continue
            cell = results["cells"][ck]
            pi = cell["stats"]["PI"]
            if pi["n_failures"] == 0:
                continue
            avg_pos = pi.get("failure_avg_relative_pos")
            avg_str = f"{avg_pos:.2f}" if avg_pos is not None else "n/a"
            garbage_pct = pi["n_garbage"] / pi["n_failures"] if pi["n_failures"] > 0 else 0
            print(f" {nk}k_{nu}u {pi['accuracy']:>6.0%} {avg_str:>11} "
                  f"{garbage_pct:>7.0%} ", end="")
            # near-last: failures with relative_pos >= 0.7 (excluding garbage)
            # can't compute from summary, just show what we have
            print()

    # Overall
    cells = list(results["cells"].values())
    if cells:
        ri_accs = [c["stats"]["RI"]["accuracy"] for c in cells]
        pi_accs = [c["stats"]["PI"]["accuracy"] for c in cells]
        print(f"\nOverall: mean RI={np.mean(ri_accs):.1%}, "
              f"mean PI={np.mean(pi_accs):.1%}, "
              f"gap={np.mean(ri_accs) - np.mean(pi_accs):.1%}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    use_chat = is_instruct_model(args.model)
    m_short = model_short_name(args.model)
    fmt = "chat_template" if use_chat else "completion_few_shot"

    # Grid: CLI --key/update-levels > --grid preset > model-type default
    if args.grid:
        grid = GRID_PRESETS[args.grid]
    elif use_chat:
        grid = GRID_PRESETS["instruct"]
    else:
        grid = GRID_PRESETS["small"]

    kl = args.key_levels or grid["key_levels"]
    ul = args.update_levels or grid["update_levels"]

    # Enforce minimum updates
    ul = [u for u in ul if u >= MIN_UPDATES]

    # Load model
    model, tokenizer, info = load_model_hf(args.model, gpu_idx=args.gpu)
    device = info.device
    ctx_limit = info.n_ctx

    print("=" * 70)
    print("V3 STAGE 1: BEHAVIORAL SWEEP + ERROR CHARACTERIZATION")
    print(f"  Model:       {args.model} ({m_short})")
    print(f"  Type:        {'Instruct' if use_chat else 'Base'} ({fmt})")
    print(f"  Dataset:     {DATASET_TYPE}")
    print(f"  Device:      {device}")
    print(f"  Trials:      {args.trials} per cell per condition")
    print(f"  Batch:       {args.batch_size}")
    print(f"  Max tokens:  {args.max_new_tokens}")
    print(f"  Grid:        {len(kl)} keys × {len(ul)} updates")
    print(f"  Min updates: {MIN_UPDATES}")
    print(f"  Save:        v3/results/{m_short}/")
    print("=" * 70)

    # Preflight
    print(f"\nPreflight check...")
    feasible = {}
    for nk in kl:
        for nu in ul:
            ok, n_tokens, reason = preflight_check(
                nk, nu, tokenizer, ctx_limit, use_chat_format=use_chat
            )
            if ok:
                feasible[(nk, nu)] = n_tokens
            else:
                print(f"  keys={nk:>2}, updates={nu:>3}: SKIPPED ({reason})")
                break  # Higher updates will also fail

    print(f"\nFeasible cells: {len(feasible)} / {len(kl) * len(ul)}")

    if not feasible:
        print("\nNo feasible cells!")
        return

    # Resume
    resume_data = None
    all_trials = {}
    if args.resume:
        print(f"\nResuming from {args.resume}")
        with open(args.resume) as f:
            resume_data = json.load(f)
        all_trials = resume_data.get("trial_details", {})

    # Build results structure
    results = resume_data or {
        "model": args.model,
        "dataset_type": DATASET_TYPE,
        "prompt_format": fmt,
        "min_updates": MIN_UPDATES,
        "config": {
            "key_levels": kl,
            "update_levels": ul,
            "trials_per_cell": args.trials,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "gpu": args.gpu,
            "device": str(device),
        },
        "cells": {},
        "start_time": datetime.now(timezone.utc).isoformat(),
    }

    # Run sweep
    total_cells = len(feasible)
    cell_idx = 0
    sat = init_saturation(kl)
    initial_batch_size = args.batch_size

    for nk in kl:
        for nu in ul:
            if (nk, nu) not in feasible:
                continue

            # Reset batch size for each cell — OOM reduction from a large cell
            # shouldn't penalize smaller cells that follow
            batch_size = initial_batch_size

            cell_key = f"{nk}_{nu}"

            # Skip if already done (resume)
            if cell_key in results.get("cells", {}):
                cell = results["cells"][cell_key]
                for cond in ["RI", "PI"]:
                    update_saturation(sat, nk, cond, cell["stats"][cond]["accuracy"])
                cell_idx += 1
                continue

            # Skip if saturated (three-zone)
            if is_saturated(sat, nk):
                print(f"  [{cell_idx + 1}/{total_cells}] keys={nk}, updates={nu}: "
                      f"SKIPPED (saturated)")
                cell_idx += 1
                continue

            clear_accelerator_cache(device)

            print(f"\n  [{cell_idx + 1}/{total_cells}] keys={nk:>2}, updates={nu:>3}")

            cell_summary, cell_trial_data = run_cell(
                model, tokenizer, args.model,
                nk, nu, args.trials, batch_size,
                args.max_new_tokens, device, use_chat,
            )

            results["cells"][cell_key] = cell_summary
            all_trials[cell_key] = cell_trial_data

            ri = cell_summary["stats"]["RI"]
            pi = cell_summary["stats"]["PI"]

            # Print cell result with failure position info
            early = " (early)" if cell_summary["stopped_early"] else ""
            pi_avg = pi.get("failure_avg_relative_pos")
            pi_pos_str = f"fail_avg_pos={pi_avg:.2f}" if pi_avg is not None else ""
            print(f"    RI={ri['accuracy']:.0%} [{ri['ci_lower']:.0%}-{ri['ci_upper']:.0%}]  "
                  f"PI={pi['accuracy']:.0%} [{pi['ci_lower']:.0%}-{pi['ci_upper']:.0%}]  "
                  f"regime={cell_summary['regime']}  "
                  f"garbage={pi['n_garbage']}/{pi['n_failures']}  "
                  f"{pi_pos_str}  "
                  f"n={cell_summary['n_trials']}{early}  "
                  f"({cell_summary['elapsed_sec']:.1f}s)")

            # Three-zone saturation
            for cond in ["RI", "PI"]:
                update_saturation(sat, nk, cond, cell_summary["stats"][cond]["accuracy"])
            if is_saturated(sat, nk):
                print(f"    -> Saturated at keys={nk}, skipping higher updates")

            cell_idx += 1

            # Checkpoint every 3 cells
            if cell_idx % 3 == 0:
                save_results(results, all_trials, args.model, partial=True)

    results["end_time"] = datetime.now(timezone.utc).isoformat()

    # Final save
    save_results(results, all_trials, args.model, partial=False)

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
