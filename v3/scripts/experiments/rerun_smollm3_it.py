"""
Re-run SmolLM3-3B IT checkpoints in BOTH formats:
  1. Completion format (few-shot, max_tokens=30)
  2. Chat template format (with thinking, max_tokens=1024)

Saves FULL raw response for later extraction.

Usage:
    export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
    export HF_TOKEN=${HF_TOKEN}
    python rerun_smollm3_it.py --gpu 0
"""

import os, sys, gc, json, time, random, re
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

DATASET_TYPE = "ARBITRARY_SINGLE"
KEY_LEVELS = [2, 3, 5, 7]
UPDATE_LEVELS = [2, 3, 5, 7, 10, 15, 20, 30]
MIN_UPDATES = 2

CHECKPOINT_REPO = "HuggingFaceTB/SmolLM3-3B-checkpoints"
FINAL_MODEL = "HuggingFaceTB/SmolLM3-3B"

# IT checkpoints + final
IT_CHECKPOINTS = ["it-SFT", "it-mid-training", "it-soup-APO", "it-LC-expert"]

RESULTS_DIR = _V3_SCRIPTS.parent / "results_vllm" / "training_dynamics_smollm3"

SYSTEM_PROMPT = (
    "You are a precise data extraction tool. "
    "Output ONLY a single word - the exact value requested. "
    "No other text, no explanation, no punctuation."
)


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


def generate_trial_both_formats(num_keys, num_updates, condition, seed, tokenizer):
    """Generate trial in BOTH completion and chat format."""
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

    # Completion format
    completion_prompt = f"{FIXED_COMPLETION_DEMOS}{stream}\nThe {query_word} value of {test_category} was:"

    # Chat format
    raw_prompt = (
        f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{stream}\n\n"
        f"What was the {query_word} value of {test_category}?"
    )
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": raw_prompt},
        ]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    else:
        chat_prompt = completion_prompt  # fallback

    return {
        "completion_prompt": completion_prompt,
        "chat_prompt": chat_prompt,
        "condition": condition, "expected": expected,
        "initial_value": cat_values[0], "final_value": cat_values[-1],
        "all_values": cat_values, "test_category": test_category,
        "num_keys": num_keys, "num_updates": num_updates, "seed": seed,
    }


def extract_answer_from_thinking(raw_output):
    """Extract the actual answer from a thinking model's output.

    Handles:
    - <think>...</think>answer
    - Direct answer (no thinking)
    - answer\n (take first line)
    """
    text = raw_output.strip()

    # Try to extract after </think> tag
    think_match = re.search(r'</think>\s*(.*)', text, re.DOTALL)
    if think_match:
        answer = think_match.group(1).strip().split("\n")[0].strip()
        return answer

    # No thinking block — take first non-empty line
    for line in text.split("\n"):
        line = line.strip()
        if line:
            return line

    return text


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


def run_vllm_generate(llm, prompts, max_new_tokens=30):
    """Returns list of FULL raw outputs (no truncation)."""
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens, temperature=0.0, top_p=1.0, top_k=-1,
        repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
    )
    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text for o in outputs]  # NO stripping — save full raw


def run_checkpoint_both_formats(checkpoint_name, gpu_idx=0, trials=100):
    """Run one checkpoint in both completion and chat format."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    from vllm import LLM
    from transformers import AutoTokenizer

    is_final = checkpoint_name == "final"
    model_id = FINAL_MODEL if is_final else CHECKPOINT_REPO
    revision = None if is_final else checkpoint_name

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

    results = {}

    for fmt_name, max_tokens, prompt_key in [
        ("completion", 100, "completion_prompt"),
        ("chat_1024", 1024, "chat_prompt"),
    ]:
        print(f"\n  --- Format: {fmt_name} (max_tokens={max_tokens}) ---")
        cells = {}
        all_cell_trials = {}
        total_prompts = 0
        sweep_start = time.time()

        for nk in KEY_LEVELS:
            for nu in UPDATE_LEVELS:
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
                        trial = generate_trial_both_formats(
                            nk, nu, condition, seed, tokenizer)
                        cell_prompts.append((condition, t_idx, trial))

                prompts_list = [t[2][prompt_key] for t in cell_prompts]
                try:
                    raw_outputs = run_vllm_generate(llm, prompts_list, max_new_tokens=max_tokens)
                except Exception as e:
                    print(f"    {cell_key}: ERROR {e}")
                    continue

                total_prompts += len(prompts_list)

                cell_trials = {"RI": [], "PI": []}
                for (condition, t_idx, trial), raw_output in zip(cell_prompts, raw_outputs):
                    # Extract answer from raw output
                    if fmt_name == "chat_1024":
                        answer = extract_answer_from_thinking(raw_output)
                    else:
                        answer = raw_output.strip().split("\n")[0].strip()

                    error_type = classify_error(
                        answer, trial["expected"], trial["initial_value"],
                        trial["final_value"], trial["all_values"], trial["condition"])
                    error_detail = get_error_detail(answer, trial["all_values"])

                    cell_trials[condition].append({
                        "seed": trial["seed"], "condition": condition,
                        "expected": trial["expected"],
                        "predicted": answer,
                        "output_raw": raw_output,  # FULL raw response
                        "output_length_tokens": len(raw_output.split()),
                        "correct": error_type == "correct",
                        "error_type": error_type,
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
                print(f"    {cell_key:>6}  RI={ri['accuracy']:.0%} PI={pi['accuracy']:.0%} "
                      f"gap={ri['accuracy']-pi['accuracy']:+.0%}")

        elapsed = time.time() - sweep_start

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

        result = {
            "checkpoint": checkpoint_name, "prompt_format": fmt_name,
            "max_new_tokens": max_tokens,
            "model_id": model_id, "revision": revision,
            "dataset_type": DATASET_TYPE,
            "config": {"key_levels": KEY_LEVELS, "update_levels": UPDATE_LEVELS,
                       "trials_per_cell": trials, "min_updates": MIN_UPDATES},
            "cells": cells, "summary": summary,
            "total_prompts": total_prompts, "elapsed_sec": round(elapsed, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Save per-format result
        result_file = RESULTS_DIR / f"{checkpoint_name}_{fmt_name}.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

        trials_file = RESULTS_DIR / f"{checkpoint_name}_{fmt_name}_trials.json"
        with open(trials_file, "w") as f:
            json.dump({"checkpoint": checkpoint_name, "format": fmt_name,
                       "trial_details": all_cell_trials}, f, indent=2)

        print(f"  {fmt_name}: RI={summary['mean_ri']:.1%} PI={summary['mean_pi']:.1%} "
              f"gap={summary['mean_gap']:+.1%} ({elapsed:.0f}s)")
        results[fmt_name] = summary

    # Cleanup
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # All IT checkpoints + final
    checkpoints = IT_CHECKPOINTS + ["final"]

    print("=" * 70)
    print(f"SmolLM3-3B IT Re-run: {len(checkpoints)} checkpoints × 2 formats")
    print(f"  Completion (max_tokens=30) + Chat (max_tokens=1024)")
    print(f"  Full raw response saved for later extraction")
    print("=" * 70)

    for i, name in enumerate(checkpoints, 1):
        print(f"\n{'#' * 60}")
        print(f"# [{i}/{len(checkpoints)}] {name}")
        print(f"{'#' * 60}")

        # Skip if both formats already done
        comp_done = (RESULTS_DIR / f"{name}_completion.json").exists()
        chat_done = (RESULTS_DIR / f"{name}_chat_1024.json").exists()
        if comp_done and chat_done:
            print(f"  SKIP: both formats done")
            continue

        try:
            run_checkpoint_both_formats(name, gpu_idx=args.gpu, trials=args.trials)
        except Exception as e:
            print(f"  FAILED: {e}")

    print(f"\n{'=' * 70}")
    print("ALL DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
