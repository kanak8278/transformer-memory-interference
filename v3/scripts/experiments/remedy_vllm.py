"""
Remedy Experiment — vLLM Backend

Tests 5 prompt interventions to reduce PI > RI asymmetry:
  control:     Standard "color: red" format (baseline)
  numbered:    "Update 1 — color: red" (positional confusability)
  landmark:    "---" separators between rounds (capacity saturation)
  recency_cue: Explicit "MOST RECENT = LAST" instruction (cumulative reinforcement)
  combined:    Landmark + round numbers (all three)

Same unified grid as Stage 1: keys=[2,3,5,7,10,15,20,25,30] x updates=[5,7,10,15,20,30,50,75,100]
Models: Qwen 3B-Instruct, Gemma 4b, SmolLM3-3B

Usage:
    export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
    export HF_TOKEN=<your_token>
    python remedy_vllm.py --all --trials 100
    python remedy_vllm.py --model Qwen/Qwen2.5-3B-Instruct --trials 50
"""

import os, sys, gc, json, time, random, re
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timezone

os.environ.setdefault("LD_LIBRARY_PATH", "")
if "/opt/conda/lib" not in os.environ["LD_LIBRARY_PATH"]:
    os.environ["LD_LIBRARY_PATH"] = f"/opt/conda/lib:{os.environ['LD_LIBRARY_PATH']}"

_SCRIPT_DIR = Path(__file__).resolve().parent
_V3_SCRIPTS = _SCRIPT_DIR.parent
_REPO_ROOT = _V3_SCRIPTS.parent.parent
for p in [str(_V3_SCRIPTS), str(_REPO_ROOT), str(_SCRIPT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from mechanistic_probing_v2.core.dataset_configs import (
    get_eligible_categories, generate_values_for_trial,
)
from mechanistic_probing_v2.core.model_loader import (
    is_instruct_model, model_short_name, CONTEXT_LIMITS,
)
from mechanistic_probing_v2.core.evaluation import classify_error, bootstrap_ci

DATASET_TYPE = "ARBITRARY_SINGLE"
RESULTS_DIR = _V3_SCRIPTS.parent / "results_vllm" / "remedy"

SYSTEM_PROMPT = (
    "You are a precise data extraction tool. "
    "Output ONLY a single word - the exact value requested. "
    "No other text, no explanation, no punctuation."
)

# Same unified grid as Stage 1
KEY_LEVELS = [2, 3, 5, 7, 10, 15, 20, 25, 30]
UPDATE_LEVELS = [5, 7, 10, 15, 20, 30, 50, 75, 100]
MIN_UPDATES = 5
DATASET_POOL_SIZE = 2300
DEFAULT_TRIALS = 100

STYLES = ["control", "numbered", "landmark", "recency_cue", "combined"]

ALL_MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "google/gemma-3-4b-it",
    "HuggingFaceTB/SmolLM3-3B",
]

MODEL_ENGINE_CONFIG = {
    "Qwen/Qwen2.5-3B-Instruct":  (0.92, 16384, 8192, "half"),
    "google/gemma-3-4b-it":       (0.92, 16384, 8192, "bfloat16"),
    "HuggingFaceTB/SmolLM3-3B":  (0.92, 16384, 4096, "bfloat16"),
}
DEFAULT_CONFIG = (0.90, 16384, 8192, "half")


# ═══════════════════════════════════════════════════════════════════════════════
# KV STREAM BUILDERS (5 styles)
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


def build_kv_stream(items, num_keys, style="control"):
    """Build KV stream with different formatting styles."""
    if style == "control":
        return "\n".join(f"{it['category']}: {it['value']}" for it in items)

    elif style == "numbered":
        return "\n".join(
            f"Update {idx + 1} — {it['category']}: {it['value']}"
            for idx, it in enumerate(items)
        )

    elif style == "landmark":
        lines = []
        for idx, it in enumerate(items):
            lines.append(f"{it['category']}: {it['value']}")
            if (idx + 1) % num_keys == 0 and idx + 1 < len(items):
                lines.append("---")
        return "\n".join(lines)

    elif style == "recency_cue":
        stream = "\n".join(f"{it['category']}: {it['value']}" for it in items)
        cue = ("\n\nIMPORTANT: Each category is updated multiple times above. "
               "The MOST RECENT value (listed LAST) is the current value. "
               "The FIRST value listed is the oldest.")
        return stream + cue

    elif style == "combined":
        lines = []
        round_num = 0
        for idx, it in enumerate(items):
            lines.append(f"{it['category']}: {it['value']}")
            if (idx + 1) % num_keys == 0 and idx + 1 < len(items):
                round_num += 1
                lines.append(f"--- round {round_num + 1} ---")
        return "\n".join(lines)

    raise ValueError(f"Unknown style: {style}")


def generate_trial(num_keys, num_updates, condition, seed, style, tokenizer, use_chat):
    """Generate one trial with the specified intervention style."""
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

    stream = build_kv_stream(items, num_keys, style=style)
    query_word = "first" if condition == "RI" else "last"
    cat_values = [it["value"] for it in items if it["category"] == test_category]
    expected = cat_values[0] if condition == "RI" else cat_values[-1]

    raw_prompt = (
        f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{stream}\n\n"
        f"What was the {query_word} value of {test_category}?"
    )

    if use_chat and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": raw_prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        from mechanistic_probing_v2.core.dataset_configs import FIXED_COMPLETION_DEMOS
        prompt = f"{FIXED_COMPLETION_DEMOS}{stream}\nThe {query_word} value of {test_category} was:"

    return {
        "prompt": prompt, "condition": condition, "expected": expected,
        "initial_value": cat_values[0], "final_value": cat_values[-1],
        "all_values": cat_values, "test_category": test_category,
        "num_keys": num_keys, "num_updates": num_updates,
        "seed": seed, "style": style,
    }


def get_error_detail(predicted, all_values):
    pred_lower = predicted.lower().strip()
    for idx, val in enumerate(all_values):
        if val.lower() in pred_lower or pred_lower.startswith(val.lower()):
            return {"predicted_idx": idx,
                    "predicted_relative_pos": round(idx / max(len(all_values) - 1, 1), 4)}
    return {"predicted_idx": None, "predicted_relative_pos": None}


# ═══════════════════════════════════════════════════════════════════════════════
# vLLM
# ═══════════════════════════════════════════════════════════════════════════════

def load_vllm_model(model_name, gpu_idx=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    from vllm import LLM
    from transformers import AutoTokenizer

    gpu_mem, max_batched, max_model_len, dtype = MODEL_ENGINE_CONFIG.get(
        model_name, DEFAULT_CONFIG)
    ctx_limit = CONTEXT_LIMITS.get(model_name, 4096)
    max_model_len = min(max_model_len, ctx_limit)

    print(f"\nLoading {model_name} (dtype={dtype}, max_model_len={max_model_len})")
    llm = LLM(
        model=model_name, gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len, max_num_batched_tokens=max_batched,
        trust_remote_code=True, dtype=dtype,
        enable_prefix_caching=True, seed=42,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return llm, tokenizer, ctx_limit


def run_vllm_generate(llm, prompts, max_new_tokens=30):
    from vllm import SamplingParams
    params = SamplingParams(
        max_tokens=max_new_tokens, temperature=0.0, top_p=1.0, top_k=-1,
        repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
    )
    outputs = llm.generate(prompts, params)
    return [o.outputs[0].text.strip().split("\n")[0].strip() for o in outputs]


# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP
# ═══════════════════════════════════════════════════════════════════════════════

def run_remedy_sweep(model_name, args):
    m_short = model_short_name(model_name)
    use_chat = is_instruct_model(model_name)

    save_dir = RESULTS_DIR / m_short
    if save_dir.exists() and list(save_dir.glob("remedy_*.json")):
        print(f"\n  SKIP {model_name}: already done")
        return True

    try:
        llm, tokenizer, ctx_limit = load_vllm_model(model_name, gpu_idx=args.gpu)
    except Exception as e:
        print(f"\n  FAILED to load {model_name}: {e}")
        return False

    kl = KEY_LEVELS
    ul = [u for u in UPDATE_LEVELS if u >= MIN_UPDATES]

    print(f"\n{'='*70}")
    print(f"  REMEDY: {model_name}")
    print(f"  Grid: {len(kl)} keys x {len(ul)} updates x {len(STYLES)} styles x {args.trials} trials")
    print(f"{'='*70}")

    all_style_results = {}
    sweep_start = time.time()

    for style in STYLES:
        print(f"\n  --- Style: {style} ---")
        cells = {}
        total_prompts = 0

        for nk in kl:
            for nu in ul:
                needed = nk * nu
                if needed > DATASET_POOL_SIZE:
                    continue

                cell_key = f"{nk}_{nu}"
                cell_prompts = []

                for condition in ["RI", "PI"]:
                    for t_idx in range(args.trials):
                        seed = hash((nk, nu, condition, t_idx, style, "remedy")) % (2**31)
                        try:
                            trial = generate_trial(nk, nu, condition, seed, style,
                                                   tokenizer, use_chat)
                        except Exception:
                            continue

                        n_tokens = len(tokenizer.encode(trial["prompt"]))
                        if n_tokens > ctx_limit - 50:
                            continue

                        cell_prompts.append((condition, t_idx, trial))

                if not cell_prompts:
                    continue

                prompts_list = [t[2]["prompt"] for t in cell_prompts]
                try:
                    answers = run_vllm_generate(llm, prompts_list)
                except Exception as e:
                    print(f"    {cell_key}: ERROR {e}")
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
                    })

                cell_stats = {}
                for cond in ["RI", "PI"]:
                    trials = cell_trials[cond]
                    if trials:
                        corrects = [t["correct"] for t in trials]
                        mean, ci_lo, ci_hi = bootstrap_ci(corrects)
                        cell_stats[cond] = {"accuracy": round(mean, 4), "n": len(trials),
                                            "ci_lower": round(ci_lo, 4), "ci_upper": round(ci_hi, 4)}
                    else:
                        cell_stats[cond] = {"accuracy": 0, "n": 0}

                ri = cell_stats["RI"]["accuracy"]
                pi = cell_stats["PI"]["accuracy"]
                cells[cell_key] = {"stats": cell_stats, "gap": round(ri - pi, 4)}

        # Style summary
        if cells:
            ri_accs = [c["stats"]["RI"]["accuracy"] for c in cells.values()]
            pi_accs = [c["stats"]["PI"]["accuracy"] for c in cells.values()]
            mean_ri = float(np.mean(ri_accs))
            mean_pi = float(np.mean(pi_accs))
            gap = mean_ri - mean_pi
        else:
            mean_ri = mean_pi = gap = 0

        all_style_results[style] = {
            "cells": cells,
            "summary": {"mean_ri": round(mean_ri, 4), "mean_pi": round(mean_pi, 4),
                         "mean_gap": round(gap, 4), "n_cells": len(cells)},
            "total_prompts": total_prompts,
        }

        print(f"    {style}: RI={mean_ri:.1%} PI={mean_pi:.1%} gap={gap:+.1%} ({len(cells)} cells)")

    elapsed = time.time() - sweep_start

    # Print comparison
    print(f"\n{'='*70}")
    print(f"  REMEDY COMPARISON — {model_name}")
    print(f"{'='*70}")
    print(f"  {'Style':<15} {'RI':>6} {'PI':>6} {'Gap':>7} {'vs Control':>10}")
    print(f"  {'-'*45}")

    control_gap = all_style_results.get("control", {}).get("summary", {}).get("mean_gap", 0)
    for style in STYLES:
        s = all_style_results.get(style, {}).get("summary", {})
        gap = s.get("mean_gap", 0)
        delta = gap - control_gap if style != "control" else 0
        print(f"  {style:<15} {s.get('mean_ri',0):>5.1%} {s.get('mean_pi',0):>5.1%} "
              f"{gap:>+6.1%} {delta:>+9.1%}")

    # Save
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    result = {
        "model": model_name, "dataset_type": DATASET_TYPE,
        "styles": STYLES,
        "config": {"key_levels": kl, "update_levels": ul,
                   "trials_per_cell": args.trials, "min_updates": MIN_UPDATES},
        "style_results": all_style_results,
        "elapsed_sec": round(elapsed, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(save_dir / f"remedy_{ts}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {save_dir}")

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Remedy experiment (vLLM)")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--model", default=None)
    g.add_argument("--all", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    args = parser.parse_args()

    models = ALL_MODELS if args.all else [args.model or ALL_MODELS[0]]

    print("=" * 70)
    print(f"REMEDY EXPERIMENT (vLLM) — 5 styles x {len(models)} models")
    print("=" * 70)

    status = {}
    for i, model_name in enumerate(models, 1):
        print(f"\n{'#'*60}")
        print(f"# [{i}/{len(models)}] {model_name}")
        print(f"{'#'*60}")
        ok = run_remedy_sweep(model_name, args)
        status[model_name] = "OK" if ok else "FAILED"

    print(f"\n{'='*70}")
    print("ALL DONE")
    for m, s in status.items():
        print(f"  {s:>8}  {m}")


if __name__ == "__main__":
    main()
