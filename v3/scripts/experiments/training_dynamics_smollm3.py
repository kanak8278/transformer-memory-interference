"""
Training Dynamics: Stage 1 sweep across SmolLM3-3B checkpoints.

Covers full pipeline: pretraining (stage1/2/3) → SFT → alignment → final.
40 checkpoints, same grid as SmolLM2-1.7B experiment.

Grid: keys=[2,3,5,7] x updates=[2,3,5,7,10,15,20,30] = 32 cells x 100 trials

Usage:
    export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
    export HF_TOKEN=${HF_TOKEN}

    # Run single checkpoint (for bash wrapper)
    python training_dynamics_smollm3.py --gpu 0 --single-checkpoint stage1-step-40000

    # Run all (not recommended — use bash wrapper for GPU isolation)
    python training_dynamics_smollm3.py --gpu 0
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
from mechanistic_probing_v2.core.evaluation import classify_error, bootstrap_ci


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_TYPE = "ARBITRARY_SINGLE"
KEY_LEVELS = [2, 3, 5, 7]
UPDATE_LEVELS = [2, 3, 5, 7, 10, 15, 20, 30]
MIN_UPDATES = 2

CHECKPOINT_REPO = "HuggingFaceTB/SmolLM3-3B-checkpoints"
FINAL_MODEL = "HuggingFaceTB/SmolLM3-3B"

# 40 checkpoints: 25 stage1 + 5 stage2 + 5 stage3 + 4 IT + final
CHECKPOINTS = [
    # Stage 1 pretraining (25 evenly sampled from 86)
    "stage1-step-40000", "stage1-step-160000", "stage1-step-320000",
    "stage1-step-440000", "stage1-step-600000", "stage1-step-720000",
    "stage1-step-880000", "stage1-step-1000000", "stage1-step-1160000",
    "stage1-step-1280000", "stage1-step-1440000", "stage1-step-1560000",
    "stage1-step-1720000", "stage1-step-1880000", "stage1-step-2000000",
    "stage1-step-2160000", "stage1-step-2280000", "stage1-step-2440000",
    "stage1-step-2560000", "stage1-step-2720000", "stage1-step-2840000",
    "stage1-step-3000000", "stage1-step-3120000", "stage1-step-3280000",
    "stage1-step-3440000",
    # Stage 2 continued pretraining (5 from 19)
    "stage2-step-3480000", "stage2-step-3640000", "stage2-step-3840000",
    "stage2-step-4000000", "stage2-step-4200000",
    # Stage 3 final pretraining (5 from 13)
    "stage3-step-4240000", "stage3-step-4360000", "stage3-step-4480000",
    "stage3-step-4600000", "stage3-step-4720000",
    # Instruction tuning stages
    "it-SFT", "it-mid-training", "it-soup-APO", "it-LC-expert",
]

RESULTS_DIR = _V3_SCRIPTS.parent / "results_vllm" / "training_dynamics_smollm3"


# ═══════════════════════════════════════════════════════════════════════════════
# TRIAL GENERATION
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


def generate_trial(num_keys, num_updates, condition, seed, tokenizer=None,
                   use_chat=False):
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

    if use_chat and tokenizer and hasattr(tokenizer, "apply_chat_template"):
        raw_prompt = (
            f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
            f"{stream}\n\n"
            f"What was the {query_word} value of {test_category}?"
        )
        messages = [
            {"role": "system", "content": "You are a precise data extraction tool. Output ONLY a single word."},
            {"role": "user", "content": raw_prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
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
        if val.lower() in pred_lower or pred_lower.startswith(val.lower()):
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
# vLLM
# ═══════════════════════════════════════════════════════════════════════════════

def load_checkpoint_vllm(model_id, revision=None, gpu_idx=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    from vllm import LLM
    from transformers import AutoTokenizer

    print(f"\nLoading {model_id}" + (f" @ {revision}" if revision else ""))
    llm = LLM(
        model=model_id, revision=revision,
        gpu_memory_utilization=0.92, max_model_len=4096,
        max_num_batched_tokens=16384, trust_remote_code=True,
        dtype="bfloat16", enable_prefix_caching=True, seed=42,
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


# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP
# ═══════════════════════════════════════════════════════════════════════════════

def detect_use_chat(tokenizer, checkpoint_name):
    """Detect whether to use chat template based on the tokenizer.

    - stage1/2/3 checkpoints: base model, no chat template -> completion format
    - it-* checkpoints: instruction tuned, has chat template -> chat format
    - final: SmolLM3-3B is instruct, has chat template -> chat format
    """
    has_chat = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None
    if has_chat:
        # Verify it actually works
        try:
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "test"}],
                tokenize=False, add_generation_prompt=True)
            return True
        except Exception:
            return False
    return False


def run_sweep_for_checkpoint(llm, tokenizer, checkpoint_name, trials, kl, ul):
    cells = {}
    all_cell_trials = {}
    total_prompts = 0
    sweep_start = time.time()
    use_chat = detect_use_chat(tokenizer, checkpoint_name)
    fmt = "chat_template" if use_chat else "completion_few_shot"
    print(f"  Prompt format: {fmt}")

    for nk in kl:
        for nu in ul:
            if nu < MIN_UPDATES:
                continue
            eligible = get_eligible_categories(DATASET_TYPE, min_values=nu)
            if len(eligible) < nk:
                continue

            cell_key = f"{nk}_{nu}"
            cell_prompts = []
            for condition in ["RI", "PI"]:
                for t_idx in range(trials):
                    seed = hash((nk, nu, condition, t_idx, "v3_td_sm3")) % (2**31)
                    trial = generate_trial(nk, nu, condition, seed,
                                           tokenizer=tokenizer, use_chat=use_chat)
                    cell_prompts.append((condition, t_idx, trial))

            prompts_list = [t[2]["prompt"] for t in cell_prompts]
            try:
                answers = run_vllm_generate(llm, prompts_list)
            except Exception as e:
                print(f"  {cell_key}: ERROR {e}")
                continue

            total_prompts += len(prompts_list)

            cell_trials = {"RI": [], "PI": []}
            for (condition, t_idx, trial), answer in zip(cell_prompts, answers):
                error_type = classify_error(
                    answer, trial["expected"], trial["initial_value"],
                    trial["final_value"], trial["all_values"], trial["condition"])
                error_detail = get_error_detail(answer, trial["all_values"])
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
                  f"gap={ri['accuracy']-pi['accuracy']:+.0%} regime={regime}")

    elapsed = time.time() - sweep_start
    return cells, all_cell_trials, total_prompts, elapsed


def main():
    parser = argparse.ArgumentParser(description="Training dynamics: SmolLM3-3B")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--single-checkpoint", type=str, default=None)
    args = parser.parse_args()

    if args.single_checkpoint:
        name = args.single_checkpoint
        if name == "final":
            models = [(FINAL_MODEL, None, "final")]
        elif name.startswith("it-"):
            models = [(CHECKPOINT_REPO, name, name)]
        else:
            models = [(CHECKPOINT_REPO, name, name)]
    else:
        models = [(CHECKPOINT_REPO, ckpt, ckpt) for ckpt in CHECKPOINTS]
        models.append((FINAL_MODEL, None, "final"))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Training Dynamics: SmolLM3-3B ({len(models)} checkpoints)")
    print(f"  Grid: {len(KEY_LEVELS)} keys x {len(UPDATE_LEVELS)} updates, {args.trials} trials")
    print("=" * 70)

    for i, (model_id, revision, name) in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] {name}")

        result_file = RESULTS_DIR / f"{name}.json"
        if result_file.exists():
            if json.load(open(result_file)).get("cells"):
                print(f"  SKIP: already done")
                continue
            else:
                os.remove(result_file)

        try:
            llm, tokenizer = load_checkpoint_vllm(model_id, revision=revision, gpu_idx=args.gpu)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            with open(result_file, "w") as f:
                json.dump({"checkpoint": name, "error": str(e)}, f)
            continue

        cells, cell_trials, total_prompts, elapsed = run_sweep_for_checkpoint(
            llm, tokenizer, name, args.trials, KEY_LEVELS, UPDATE_LEVELS)

        if cells:
            ri_accs = [c["stats"]["RI"]["accuracy"] for c in cells.values()]
            pi_accs = [c["stats"]["PI"]["accuracy"] for c in cells.values()]
            summary = {
                "mean_ri": round(float(np.mean(ri_accs)), 4),
                "mean_pi": round(float(np.mean(pi_accs)), 4),
                "mean_gap": round(float(np.mean(ri_accs) - np.mean(pi_accs)), 4),
                "n_cells": len(cells),
            }
        else:
            summary = {"mean_ri": 0, "mean_pi": 0, "mean_gap": 0, "n_cells": 0}

        # Detect format used (for logging)
        use_chat_flag = detect_use_chat(tokenizer, name)

        result = {
            "checkpoint": name, "model_id": model_id, "revision": revision,
            "dataset_type": DATASET_TYPE,
            "prompt_format": "chat_template" if use_chat_flag else "completion_few_shot",
            "config": {"key_levels": KEY_LEVELS, "update_levels": UPDATE_LEVELS,
                       "trials_per_cell": args.trials, "min_updates": MIN_UPDATES},
            "cells": cells, "summary": summary,
            "total_prompts": total_prompts, "elapsed_sec": round(elapsed, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        trials_file = RESULTS_DIR / f"{name}_trials.json"
        with open(trials_file, "w") as f:
            json.dump({"checkpoint": name, "trial_details": cell_trials}, f, indent=2)

        print(f"  Summary: RI={summary['mean_ri']:.1%} PI={summary['mean_pi']:.1%} "
              f"gap={summary['mean_gap']:+.1%} ({total_prompts} prompts, {elapsed:.0f}s)")

        del llm, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(3)

        # Clean HF cache for this checkpoint to avoid disk fill
        # (each 3B checkpoint is ~6GB, 40 checkpoints = 240GB without cleanup)
        import shutil
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--HuggingFaceTB--SmolLM3-3B-checkpoints"
        if cache_dir.exists():
            # Remove blob files for old revisions but keep the latest
            blobs_dir = cache_dir / "blobs"
            if blobs_dir.exists():
                total_before = sum(f.stat().st_size for f in blobs_dir.iterdir() if f.is_file()) / 1e9
                if total_before > 15:  # Only clean if > 15GB accumulated
                    print(f"  Cleaning HF cache ({total_before:.1f}GB)...")
                    shutil.rmtree(str(cache_dir), ignore_errors=True)
                    print(f"  Cache cleaned.")


if __name__ == "__main__":
    main()
