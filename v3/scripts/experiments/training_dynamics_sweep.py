"""
Training Dynamics: Stage 1 sweep across SmolLM2-1.7B intermediate checkpoints.

Runs the behavioral sweep (RI/PI) on all 41 training checkpoints + final model
to see WHEN the PI > RI asymmetry emerges during training.

Grid: keys=[2,3,5,7] x updates=[2,3,5,7,10,15,20,30] = 32 cells x 100 trials
41 checkpoints + 1 final = 42 models
Total: 32 x 42 x 2 x 100 = 268,800 prompts

Usage:
    export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
    export HF_TOKEN=${HF_TOKEN}
    python training_dynamics_sweep.py --gpu 0
    python training_dynamics_sweep.py --gpu 0 --trials 50   # faster
"""

import os
import sys
import gc
import json
import time
import random
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timezone

# Fix GLIBCXX + HF token
os.environ.setdefault("LD_LIBRARY_PATH", "")
if "/opt/conda/lib" not in os.environ["LD_LIBRARY_PATH"]:
    os.environ["LD_LIBRARY_PATH"] = f"/opt/conda/lib:{os.environ['LD_LIBRARY_PATH']}"
if "HF_TOKEN" not in os.environ:
    pass  # Set HF_TOKEN env var before running

_SCRIPT_DIR = Path(__file__).resolve().parent
_V3_SCRIPTS = _SCRIPT_DIR.parent
_REPO_ROOT = _V3_SCRIPTS.parent.parent
for p in [str(_V3_SCRIPTS), str(_REPO_ROOT), str(_SCRIPT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from mechanistic_probing_v2.core.dataset_configs import (
    get_eligible_categories, generate_values_for_trial, FIXED_COMPLETION_DEMOS,
)
from mechanistic_probing_v2.core.model_loader import model_short_name, CONTEXT_LIMITS
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

# Grid — dense at low end to catch transition
KEY_LEVELS = [2, 3, 5, 7]
UPDATE_LEVELS = [2, 3, 5, 7, 10, 15, 20, 30]
MIN_UPDATES = 2  # Lower than usual — we want to see baseline before interference

# SmolLM2-1.7B checkpoints
BASE_REPO = "HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints"
FINAL_MODEL = "HuggingFaceTB/SmolLM2-1.7B"

CHECKPOINTS = sorted([
    "step-125000", "step-250000", "step-375000", "step-500000",
    "step-625000", "step-750000", "step-875000", "step-1000000",
    "step-1125000", "step-1250000", "step-1375000", "step-1500000",
    "step-1625000", "step-1750000", "step-1875000", "step-2000000",
    "step-2125000", "step-2250000", "step-2375000", "step-2500000",
    "step-2625000", "step-2750000", "step-2875000", "step-3000000",
    "step-3125000", "step-3250000", "step-3375000", "step-3500000",
    "step-3625000", "step-3750000", "step-3875000", "step-4000000",
    "step-4125000", "step-4250000", "step-4375000", "step-4500000",
    "step-4625000", "step-4750000", "step-4875000", "step-5000000",
    "step-5125000",
], key=lambda x: int(x.split("-")[1]))

RESULTS_DIR = _V3_SCRIPTS.parent / "results_vllm" / "training_dynamics"


# ═══════════════════════════════════════════════════════════════════════════════
# TRIAL GENERATION (base model — completion format)
# ═══════════════════════════════════════════════════════════════════════════════

def shuffle_no_consecutive(items, rng, max_attempts=100):
    for _ in range(max_attempts):
        candidate = items.copy()
        rng.shuffle(candidate)
        ok = all(candidate[i]["category"] != candidate[i - 1]["category"]
                 for i in range(1, len(candidate)))
        if ok:
            return candidate
    remaining = items.copy()
    rng.shuffle(remaining)
    result, last_cat = [], None
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


def generate_trial(num_keys, num_updates, condition, seed, tokenizer=None):
    """Generate trial. SmolLM2 is a base model — use completion format."""
    rng = random.Random(seed)
    eligible = get_eligible_categories(DATASET_TYPE, min_values=num_updates)
    categories = rng.sample(eligible, min(num_keys, len(eligible)))
    test_category = categories[seed % num_keys]

    values_per_cat = generate_values_for_trial(DATASET_TYPE, categories, num_updates, rng)

    items = []
    for cat in categories:
        for val in values_per_cat[cat]:
            items.append({"category": cat, "value": val})
    items = shuffle_no_consecutive(items, rng)

    stream = "\n".join(f"{it['category']}: {it['value']}" for it in items)
    query_word = "first" if condition == "RI" else "last"
    cat_values = [it["value"] for it in items if it["category"] == test_category]
    expected = cat_values[0] if condition == "RI" else cat_values[-1]

    # Base model: few-shot completion format
    prompt = f"{FIXED_COMPLETION_DEMOS}{stream}\nThe {query_word} value of {test_category} was:"

    return {
        "prompt": prompt, "condition": condition, "expected": expected,
        "initial_value": cat_values[0], "final_value": cat_values[-1],
        "all_values": cat_values, "test_category": test_category,
        "num_keys": num_keys, "num_updates": num_updates, "seed": seed,
    }


def get_error_detail(predicted, all_values):
    pred_lower = predicted.lower().strip()
    for idx, val in enumerate(all_values):
        v = val.lower()
        if v in pred_lower or pred_lower.startswith(v):
            return {"predicted_idx": idx,
                    "predicted_relative_pos": round(idx / max(len(all_values) - 1, 1), 4)}
    return {"predicted_idx": None, "predicted_relative_pos": None}


def compute_cell_stats(cell_trials):
    cell_stats = {}
    for cond in ["RI", "PI"]:
        results = cell_trials[cond]
        corrects = [r["correct"] for r in results]
        mean, ci_lo, ci_hi = bootstrap_ci(corrects)
        error_counts = {}
        for r in results:
            error_counts[r["error_type"]] = error_counts.get(r["error_type"], 0) + 1
        garbage_count = sum(1 for r in results if not r["correct"] and r["predicted_idx"] is None)

        cell_stats[cond] = {
            "accuracy": round(mean, 4), "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4), "n": len(corrects),
            "error_types": error_counts, "n_garbage": garbage_count,
        }

    ri_acc = cell_stats["RI"]["accuracy"]
    pi_acc = cell_stats["PI"]["accuracy"]
    gap = ri_acc - pi_acc
    regime = "D" if pi_acc > ri_acc else "C" if gap >= 0.25 else "B" if gap >= 0.15 else "A" if gap < 0.05 else "AB"

    return cell_stats, regime


# ═══════════════════════════════════════════════════════════════════════════════
# vLLM INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def load_checkpoint_vllm(model_id, revision=None, gpu_idx=0):
    """Load a checkpoint via vLLM."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    from vllm import LLM
    from transformers import AutoTokenizer

    print(f"\nLoading {model_id}" + (f" @ {revision}" if revision else ""))

    llm = LLM(
        model=model_id,
        revision=revision,
        gpu_memory_utilization=0.95,
        max_model_len=2048,
        max_num_batched_tokens=32768,
        trust_remote_code=True,
        dtype="half",
        enable_prefix_caching=True,
        seed=42,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Engine ready.")
    return llm, tokenizer


def run_vllm_generate(llm, prompts, max_new_tokens=30):
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens, temperature=0.0, top_p=1.0, top_k=-1,
        repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
    )
    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text.strip().split("\n")[0].strip() for o in outputs]


def destroy_vllm_model(llm):
    """Properly shutdown vLLM engine (subprocess) and free GPU."""
    try:
        # vLLM 0.19+ has a shutdown method on the engine
        if hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'shutdown'):
            llm.llm_engine.shutdown()
        elif hasattr(llm, 'shutdown'):
            llm.shutdown()
    except Exception:
        pass
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Give OS time to reclaim GPU memory from subprocess
    time.sleep(5)


# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP ONE CHECKPOINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_sweep_for_checkpoint(llm, tokenizer, checkpoint_name, trials, kl, ul):
    """Run the full grid for one checkpoint. Cell-by-cell with vLLM."""
    cells = {}
    all_cell_trials = {}
    total_prompts = 0
    sweep_start = time.time()

    for nk in kl:
        for nu in ul:
            if nu < MIN_UPDATES:
                continue

            # Check dataset feasibility
            eligible = get_eligible_categories(DATASET_TYPE, min_values=nu)
            if len(eligible) < nk:
                continue

            cell_key = f"{nk}_{nu}"

            # Generate and run all prompts for this cell
            cell_prompts = []
            for condition in ["RI", "PI"]:
                for t_idx in range(trials):
                    seed = hash((nk, nu, condition, t_idx, "v3_td")) % (2**31)
                    trial = generate_trial(nk, nu, condition, seed, tokenizer=tokenizer)
                    cell_prompts.append((condition, t_idx, trial))

            prompts_list = [t[2]["prompt"] for t in cell_prompts]
            try:
                answers = run_vllm_generate(llm, prompts_list)
            except Exception as e:
                print(f"  {cell_key}: ERROR {e}")
                continue

            total_prompts += len(prompts_list)

            # Evaluate
            cell_trials = {"RI": [], "PI": []}
            for (condition, t_idx, trial), answer in zip(cell_prompts, answers):
                error_type = classify_error(
                    answer, trial["expected"], trial["initial_value"],
                    trial["final_value"], trial["all_values"], trial["condition"])
                error_detail = get_error_detail(answer, trial["all_values"])
                n_vals = len(trial["all_values"])

                cell_trials[condition].append({
                    "seed": trial["seed"], "condition": condition,
                    "expected": trial["expected"], "predicted": answer,
                    "correct": error_type == "correct", "error_type": error_type,
                    "predicted_idx": error_detail["predicted_idx"],
                    "all_values": trial["all_values"],
                })

            cell_stats, regime = compute_cell_stats(cell_trials)
            cells[cell_key] = {
                "num_keys": nk, "num_updates": nu,
                "stats": cell_stats, "regime": regime,
                "n_trials": len(cell_trials["RI"]),
            }
            all_cell_trials[cell_key] = cell_trials

            ri = cell_stats["RI"]
            pi = cell_stats["PI"]
            print(f"  {cell_key:>6}  RI={ri['accuracy']:.0%} PI={pi['accuracy']:.0%} "
                  f"gap={ri['accuracy']-pi['accuracy']:+.0%} regime={regime} "
                  f"garbage={pi['n_garbage']}/{pi['n']}")

    elapsed = time.time() - sweep_start
    return cells, all_cell_trials, total_prompts, elapsed


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Training dynamics sweep for SmolLM2-1.7B")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--single-checkpoint", type=str, default=None,
                        help="Run only one checkpoint (e.g. 'step-125000' or 'final')")
    args = parser.parse_args()

    kl = KEY_LEVELS
    ul = UPDATE_LEVELS
    n_cells = len(kl) * len([u for u in ul if u >= MIN_UPDATES])

    # Build model list: all checkpoints + final
    if args.single_checkpoint:
        # Single checkpoint mode (called from bash wrapper for clean GPU)
        name = args.single_checkpoint
        if name == "final":
            models = [(FINAL_MODEL, None, "final")]
        else:
            models = [(BASE_REPO, name, name)]
    else:
        models = [(BASE_REPO, ckpt, ckpt) for ckpt in CHECKPOINTS]
        models.append((FINAL_MODEL, None, "final"))

    print("=" * 70)
    print("TRAINING DYNAMICS: SmolLM2-1.7B Intermediate Checkpoints")
    print(f"  Checkpoints: {len(models)} ({len(CHECKPOINTS)} intermediate + final)")
    print(f"  Grid: {len(kl)} keys x {len(ul)} updates = {n_cells} cells")
    print(f"  Trials: {args.trials} per cell per condition")
    print(f"  Total prompts: ~{n_cells * len(models) * 2 * args.trials:,}")
    print(f"  Output: {RESULTS_DIR}")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    total_start = time.time()

    for i, (model_id, revision, name) in enumerate(models, 1):
        print(f"\n{'#' * 70}")
        print(f"# [{i}/{len(models)}] {name}")
        print(f"{'#' * 70}")

        # Check if already done
        result_file = RESULTS_DIR / f"{name}.json"
        if result_file.exists():
            print(f"  SKIP: {result_file} already exists")
            with open(result_file) as f:
                all_results[name] = json.load(f)
            continue

        # Load model
        try:
            llm, tokenizer = load_checkpoint_vllm(model_id, revision=revision, gpu_idx=args.gpu)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            all_results[name] = {"error": str(e)}
            continue

        # Run sweep
        cells, cell_trials, total_prompts, elapsed = run_sweep_for_checkpoint(
            llm, tokenizer, name, args.trials, kl, ul)

        # Compute summary
        if cells:
            ri_accs = [c["stats"]["RI"]["accuracy"] for c in cells.values()]
            pi_accs = [c["stats"]["PI"]["accuracy"] for c in cells.values()]
            mean_ri = float(np.mean(ri_accs))
            mean_pi = float(np.mean(pi_accs))
            mean_gap = mean_ri - mean_pi
        else:
            mean_ri = mean_pi = mean_gap = 0.0

        result = {
            "checkpoint": name,
            "model_id": model_id,
            "revision": revision,
            "step": int(name.split("-")[1]) if name.startswith("step-") else None,
            "dataset_type": DATASET_TYPE,
            "config": {
                "key_levels": kl, "update_levels": ul,
                "trials_per_cell": args.trials, "min_updates": MIN_UPDATES,
            },
            "cells": cells,
            "summary": {
                "mean_ri": round(mean_ri, 4), "mean_pi": round(mean_pi, 4),
                "mean_gap": round(mean_gap, 4), "n_cells": len(cells),
            },
            "total_prompts": total_prompts,
            "elapsed_sec": round(elapsed, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Save per-checkpoint result
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

        # Save trial details separately (larger file)
        trials_file = RESULTS_DIR / f"{name}_trials.json"
        with open(trials_file, "w") as f:
            json.dump({"checkpoint": name, "trial_details": cell_trials}, f, indent=2)

        all_results[name] = result
        print(f"\n  Summary: RI={mean_ri:.1%} PI={mean_pi:.1%} gap={mean_gap:+.1%} "
              f"({total_prompts} prompts, {elapsed:.0f}s)")

        # Free GPU
        destroy_vllm_model(llm)

    # Save combined summary
    total_elapsed = time.time() - total_start
    summary = {
        "experiment": "training_dynamics",
        "model": "SmolLM2-1.7B",
        "n_checkpoints": len(models),
        "grid": {"key_levels": kl, "update_levels": ul},
        "trials_per_cell": args.trials,
        "total_elapsed_sec": round(total_elapsed, 1),
        "checkpoints": {},
    }
    for name, result in all_results.items():
        if "error" in result:
            summary["checkpoints"][name] = {"error": result["error"]}
        else:
            summary["checkpoints"][name] = result.get("summary", {})
            summary["checkpoints"][name]["step"] = result.get("step")

    with open(RESULTS_DIR / "training_dynamics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print trajectory
    print(f"\n{'=' * 70}")
    print(f"TRAINING DYNAMICS COMPLETE — {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")
    print(f"{'=' * 70}")
    print(f"\n{'Step':>12} {'RI':>6} {'PI':>6} {'Gap':>7}")
    print("-" * 35)
    for name in sorted(all_results.keys(), key=lambda x: int(x.split("-")[1]) if x.startswith("step-") else 99999999):
        r = all_results[name]
        if "error" in r:
            print(f"{name:>12} FAILED")
        else:
            s = r["summary"]
            print(f"{name:>12} {s['mean_ri']:>5.1%} {s['mean_pi']:>5.1%} {s['mean_gap']:>+6.1%}")


if __name__ == "__main__":
    main()
