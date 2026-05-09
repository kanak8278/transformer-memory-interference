"""
V3 Stage 1: Behavioral Sweep + Error Characterization — vLLM Backend

Drop-in replacement for stage1_sweep.py using vLLM for ~10x faster inference.
All trial generation, evaluation, and result formats are identical to the
HuggingFace version so analysis scripts work on both.

Architecture:
  - Generates ALL prompts for the entire grid upfront
  - Submits them in ONE llm.generate() call (vLLM handles continuous batching)
  - Slices results back into cells for evaluation + saving
  - Model-size-aware engine tuning (gpu_memory_utilization, max_num_batched_tokens)

Inference config (deterministic, explicitly set to override generation_config.json):
  - temperature=0 (greedy), top_p=1.0, top_k=-1
  - repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0
  - enable_prefix_caching=True (shared system prompt across prompts)

Dataset: ARBITRARY_SINGLE — 2,300 single-token values, 46 categories (shared pool)
  - Pool limit: keys * updates <= 2,300 values per trial
  - Infeasible cells auto-skipped: (25k, 100u)=2500, (30k, 100u)=3000

Modes:
  --model: Run a single model (default)
  --all:   Run all 11 models sequentially in one process (load -> sweep -> free -> next)

Usage:
    python stage1_sweep_vllm.py --model Qwen/Qwen2.5-1.5B-Instruct
    python stage1_sweep_vllm.py --all --trials 100
    python stage1_sweep_vllm.py --model Qwen/Qwen2.5-0.5B-Instruct --grid quick --trials 10
"""

import os
import sys
import gc
import json
import time
import random
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# Fix GLIBCXX_3.4.31 not found — conda's GCC 13 has it
os.environ.setdefault("LD_LIBRARY_PATH", "")
if "/opt/conda/lib" not in os.environ["LD_LIBRARY_PATH"]:
    os.environ["LD_LIBRARY_PATH"] = f"/opt/conda/lib:{os.environ['LD_LIBRARY_PATH']}"

# HuggingFace token for gated models (Gemma etc.)
if "HF_TOKEN" not in os.environ:
    pass  # Set HF_TOKEN env var before running

# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent          # v3/scripts/experiments/
_V3_SCRIPTS = _SCRIPT_DIR.parent                       # v3/scripts/
_REPO_ROOT = _V3_SCRIPTS.parent.parent                 # repo root
for p in [str(_V3_SCRIPTS), str(_REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from mechanistic_probing_v2.core.dataset_configs import (
    load_dataset_config,
    get_eligible_categories,
    get_max_updates,
    generate_values_for_trial,
    build_interleaved_sequence,
    FIXED_COMPLETION_DEMOS,
)
from mechanistic_probing_v2.core.model_loader import (
    is_instruct_model,
    model_short_name,
    CONTEXT_LIMITS,
)
from mechanistic_probing_v2.core.evaluation import classify_error, bootstrap_ci


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

VALID_DATASETS = ["ARBITRARY_SINGLE", "SEMANTIC_MULTI"]

SYSTEM_PROMPT = (
    "You are a precise data extraction tool. "
    "Output ONLY a single word - the exact value requested. "
    "No other text, no explanation, no punctuation."
)
MIN_UPDATES = 5  # Minimum updates per key — avoids floor artifacts

GRID_PRESETS = {
    "unified": {
        "key_levels":    [2, 3, 5, 7, 10, 15, 20, 25, 30],
        "update_levels": [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100],
    },
    "small": {
        "key_levels":    [5, 7, 10, 12, 15, 20, 25, 30],
        "update_levels": [2, 5, 7, 10, 12, 15, 20, 30, 40, 50],
    },
    "instruct": {
        "key_levels":    [2, 3, 5, 7, 10, 15, 20, 25, 30],
        "update_levels": [5, 10, 15, 20, 30, 50, 100],
    },
    "quick": {
        "key_levels":    [3, 5, 10],
        "update_levels": [5, 10, 20, 50],
    },
    "focused": {
        "key_levels":    [2, 3, 5],
        "update_levels": [5, 7, 10, 12, 15, 20, 25, 30, 40, 50],
    },
}

DEFAULT_TRIALS = 100

# ── Three-zone saturation (skip remaining updates when accuracy is at floor) ─
NEAR_ZERO = 0.015
RECOVERY  = 0.06
SAT_COUNT = 3

# ── All 11 models from MODELS_TODO ───────────────────────────────────────────
ALL_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-3B",
    "google/gemma-3-270m-it",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "stabilityai/stablelm-2-1_6b-chat",
    "EleutherAI/pythia-410m",
    "state-spaces/mamba-1.4b-hf",
]

# Qwen3.5 models added after environment upgrade (transformers 5.8.0.dev0 + vLLM 0.20.1)
QWEN35_MODELS = [
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
]

# Model-size-aware vLLM engine config.
# (gpu_memory_utilization, max_num_batched_tokens, max_model_len, dtype)
# Gemma 3 requires bfloat16 (float16 causes numerical instability).
MODEL_ENGINE_CONFIG = {
    # <=500M
    "Qwen/Qwen2.5-0.5B-Instruct":          (0.95, 65536, 8192, "half"),
    # Qwen3.5 series (qwen3_5 arch, bfloat16 required)
    "Qwen/Qwen3.5-0.8B":               (0.95, 65536, 8192, "bfloat16"),
    "Qwen/Qwen3.5-4B":                 (0.92, 16384, 8192, "bfloat16"),
    "Qwen/Qwen3.5-9B":                 (0.90, 16384, 8192, "bfloat16"),
    "Qwen/Qwen3.5-0.8B-Base":          (0.95, 65536, 8192, "bfloat16"),
    "Qwen/Qwen3.5-4B-Base":            (0.92, 16384, 8192, "bfloat16"),
    "Qwen/Qwen3.5-9B-Base":            (0.90, 16384, 8192, "bfloat16"),
    # Gemma 3
    "google/gemma-3-270m-it":               (0.95, 65536, 8192, "bfloat16"),
    "EleutherAI/pythia-410m":               (0.95, 65536, 2048, "half"),
    # 1-2B
    "Qwen/Qwen2.5-1.5B-Instruct":          (0.95, 32768, 8192, "half"),
    "google/gemma-3-1b-it":                 (0.95, 32768, 8192, "bfloat16"),
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0":  (0.95, 32768, 2048, "half"),
    "stabilityai/stablelm-2-1_6b-chat":     (0.95, 32768, 4096, "half"),
    "state-spaces/mamba-1.4b-hf":           (0.95, 32768, 2048, "half"),
    # 3-4B
    "Qwen/Qwen2.5-3B-Instruct":            (0.92, 16384, 8192, "half"),
    "Qwen/Qwen2.5-3B":                     (0.92, 16384, 8192, "half"),
    "google/gemma-3-4b-it":                 (0.92, 16384, 8192, "bfloat16"),
}
DEFAULT_ENGINE_CONFIG = (0.90, 16384, 8192, "half")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="V3 Stage 1 (vLLM): Behavioral sweep + error characterization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--model", default=None, help="Run a single model")
    g.add_argument("--all", action="store_true", help="Run all 11 models sequentially")
    p.add_argument("--dataset", default="ARBITRARY_SINGLE,SEMANTIC_MULTI",
                   help="Dataset type(s), comma-separated (default: both)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    p.add_argument("--max-new-tokens", type=int, default=30)
    p.add_argument("--grid", default=None, choices=list(GRID_PRESETS.keys()))
    p.add_argument("--key-levels", type=int, nargs="+", default=None)
    p.add_argument("--update-levels", type=int, nargs="+", default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--output-dir", type=str, default=None)
    args = p.parse_args()
    if not args.all and args.model is None:
        args.model = "Qwen/Qwen2.5-1.5B-Instruct"
    return args


# ═══════════════════════════════════════════════════════════════════════════════
# INTERLEAVING
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


# ═══════════════════════════════════════════════════════════════════════════════
# TRIAL GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_trial(num_keys, num_updates, condition, seed,
                   use_chat_format=False, tokenizer=None, dataset_type="ARBITRARY_SINGLE"):
    rng = random.Random(seed)
    eligible = get_eligible_categories(dataset_type, min_values=num_updates)
    categories = rng.sample(eligible, min(num_keys, len(eligible)))
    test_category = categories[seed % num_keys]

    values_per_cat = generate_values_for_trial(dataset_type, categories, num_updates, rng)

    items = []
    for cat in categories:
        for val in values_per_cat[cat]:
            items.append({"category": cat, "value": val})
    items = shuffle_no_consecutive(items, rng)

    stream = "\n".join(f"{it['category']}: {it['value']}" for it in items)
    query_word = "first" if condition == "RI" else "last"
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
        "prompt": prompt, "condition": condition, "expected": expected,
        "initial_value": cat_values[0], "final_value": cat_values[-1],
        "all_values": cat_values, "test_category": test_category,
        "num_keys": num_keys, "num_updates": num_updates, "seed": seed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR DETAIL
# ═══════════════════════════════════════════════════════════════════════════════

def get_error_detail(predicted, all_values):
    pred_lower = predicted.lower().strip()
    for idx, val in enumerate(all_values):
        v = val.lower()
        if v in pred_lower or pred_lower.startswith(v):
            return {"predicted_idx": idx,
                    "predicted_relative_pos": round(idx / max(len(all_values) - 1, 1), 4)}
    return {"predicted_idx": None, "predicted_relative_pos": None}


# ═══════════════════════════════════════════════════════════════════════════════
# PREFLIGHT — dataset + context feasibility
# ═══════════════════════════════════════════════════════════════════════════════

def check_dataset_feasibility(num_keys, num_updates, dataset_type):
    """Check if dataset pool has enough values for this cell."""
    max_upd = get_max_updates(dataset_type, num_keys)
    if num_updates > max_upd:
        return False, f"max {max_upd} updates for {num_keys} keys in {dataset_type}"
    eligible = get_eligible_categories(dataset_type, min_values=num_updates)
    if len(eligible) < num_keys:
        return False, f"need {num_keys} categories with {num_updates}+ values, only {len(eligible)}"
    return True, "ok"


def preflight_check(num_keys, num_updates, tokenizer, context_limit,
                    use_chat_format=False, dataset_type="ARBITRARY_SINGLE"):
    """Check dataset feasibility AND context length (conservative).

    Tests 3 seeds to catch variance in prompt length. Uses 0.75 margin
    to leave room for max_new_tokens and tokenizer edge cases.
    """
    ok, reason = check_dataset_feasibility(num_keys, num_updates, dataset_type)
    if not ok:
        return False, 0, reason

    # Test multiple seeds — prompt length varies by random interleaving
    max_tokens = 0
    for seed in [0, 42, 99]:
        trial = generate_trial(num_keys, num_updates, "RI", seed=seed,
                               use_chat_format=use_chat_format, tokenizer=tokenizer,
                               dataset_type=dataset_type)
        n_tokens = len(tokenizer.encode(trial["prompt"]))
        max_tokens = max(max_tokens, n_tokens)

    # Conservative: leave 25% headroom (prompt + 30 generated tokens must fit)
    limit = context_limit - 50  # subtract max_new_tokens + safety
    if max_tokens > limit:
        return False, max_tokens, f"~{max_tokens} tokens > {limit} limit"
    return True, max_tokens, "ok"


# ═══════════════════════════════════════════════════════════════════════════════
# vLLM ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def load_vllm_model(model_name, gpu_idx=0, tp_size=1):
    """Load model with model-size-aware engine tuning."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

    from vllm import LLM
    from transformers import AutoTokenizer

    gpu_mem, max_batched_tokens, max_model_len, dtype = MODEL_ENGINE_CONFIG.get(
        model_name, DEFAULT_ENGINE_CONFIG
    )

    # Also enforce context limit from model registry
    ctx_limit = CONTEXT_LIMITS.get(model_name, 4096)
    max_model_len = min(max_model_len, ctx_limit)

    print(f"\nLoading {model_name} via vLLM")
    print(f"  gpu={gpu_idx}, tp={tp_size}, dtype={dtype}")
    print(f"  gpu_memory_utilization={gpu_mem}")
    print(f"  max_model_len={max_model_len}, max_num_batched_tokens={max_batched_tokens}")
    print(f"  enable_prefix_caching=True")

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_batched_tokens,
        trust_remote_code=True,
        dtype=dtype,
        enable_prefix_caching=True,
        seed=42,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Engine ready.")
    return llm, tokenizer, ctx_limit


def destroy_vllm_model(llm):
    """Free GPU memory so the next model can be loaded."""
    import torch
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_vllm_generate(llm, prompts, max_new_tokens=30):
    """Single vLLM generate call with deterministic greedy decoding.

    All params set explicitly to override any generation_config.json defaults.
    vLLM handles continuous batching internally — just pass all prompts.
    """
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    )

    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text.strip().split("\n")[0].strip() for o in outputs]


# ═══════════════════════════════════════════════════════════════════════════════
# CELL STATS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_cell_stats(cell_trials):
    """Compute accuracy, CI, error types, regime for a cell's trial results."""
    cell_stats = {}
    for cond in ["RI", "PI"]:
        results = cell_trials[cond]
        corrects = [r["correct"] for r in results]
        mean, ci_lo, ci_hi = bootstrap_ci(corrects)

        error_counts = {}
        for r in results:
            et = r["error_type"]
            error_counts[et] = error_counts.get(et, 0) + 1

        failure_positions = [r["predicted_relative_pos"] for r in results
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
    gap = ri_acc - pi_acc
    if pi_acc > ri_acc:
        regime = "D"
    elif gap >= 0.25:
        regime = "C"
    elif gap >= 0.15:
        regime = "B"
    elif gap < 0.05:
        regime = "A"
    else:
        regime = "AB"

    return cell_stats, regime


# ═══════════════════════════════════════════════════════════════════════════════
# SATURATION
# ═══════════════════════════════════════════════════════════════════════════════

def init_saturation(key_levels):
    return {nk: {"RI": {"zero_count": 0}, "PI": {"zero_count": 0}} for nk in key_levels}

def update_saturation(sat, nk, cond, acc):
    if acc <= NEAR_ZERO:
        sat[nk][cond]["zero_count"] += 1
    elif acc >= RECOVERY:
        sat[nk][cond]["zero_count"] = 0

def is_saturated(sat, nk):
    return any(sat[nk][cond]["zero_count"] >= SAT_COUNT for cond in ["RI", "PI"])


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════════

def get_save_dir(model_name, dataset_type="ARBITRARY_SINGLE", output_dir=None):
    if output_dir:
        return Path(output_dir)
    m_short = model_short_name(model_name)
    ds_short = dataset_type.lower()  # e.g. "arbitrary_single", "semantic_multi"
    return Path(__file__).resolve().parent.parent.parent / "results_vllm" / ds_short / m_short


def save_results(results, all_trials, model_name, dataset_type="ARBITRARY_SINGLE",
                 partial=False, output_dir=None):
    save_dir = get_save_dir(model_name, dataset_type, output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    full_data = dict(results)
    full_data["trial_details"] = all_trials

    if partial:
        with open(save_dir / "stage1_checkpoint.json", "w") as f:
            json.dump(full_data, f, indent=2)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        with open(save_dir / f"stage1_sweep_{ts}.json", "w") as f:
            json.dump(results, f, indent=2)
        with open(save_dir / f"stage1_trials_{ts}.json", "w") as f:
            json.dump(full_data, f, indent=2)
        print(f"  -> Saved to {save_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(results):
    kl = results["config"]["key_levels"]
    ul = results["config"]["update_levels"]

    print("\n" + "=" * 70)
    print(f"SUMMARY — {results['model']}")
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
                    print(f" {results['cells'][ck]['stats'][cond]['accuracy']:>5.0%}", end="")
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


# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP ONE MODEL — single llm.generate() for the ENTIRE grid
# ═══════════════════════════════════════════════════════════════════════════════

def run_dataset_sweep(llm, tokenizer, ctx_limit, model_name, dataset_type, args):
    """Run one dataset sweep for an already-loaded model. Cell-by-cell with checkpoints.

    Returns True if completed, False if failed/empty.
    """
    use_chat = is_instruct_model(model_name)
    m_short = model_short_name(model_name)
    fmt = "chat_template" if use_chat else "completion_few_shot"

    save_dir = get_save_dir(model_name, dataset_type, args.output_dir)
    if save_dir.exists() and list(save_dir.glob("stage1_sweep_*.json")):
        print(f"\n  SKIP {model_name} [{dataset_type}]: already done")
        return True

    grid = GRID_PRESETS[args.grid] if args.grid else GRID_PRESETS["unified"]
    kl = args.key_levels or grid["key_levels"]
    ul = args.update_levels or grid["update_levels"]
    ul = [u for u in ul if u >= MIN_UPDATES]

    print(f"\n{'─' * 60}")
    print(f"  Dataset: {dataset_type} | {len(kl)} keys x {len(ul)} updates")
    print(f"  Save:    {save_dir}")

    # Preflight
    feasible = {}
    for nk in kl:
        for nu in ul:
            ok, n_tokens, reason = preflight_check(
                nk, nu, tokenizer, ctx_limit,
                use_chat_format=use_chat, dataset_type=dataset_type)
            if ok:
                feasible[(nk, nu)] = n_tokens
            else:
                print(f"  SKIP keys={nk:>2}, updates={nu:>3}: {reason}")
                if "tokens >" in reason:
                    break

    print(f"  Feasible: {len(feasible)} cells")
    if not feasible:
        return False

    # Resume from checkpoint
    checkpoint_path = save_dir / "stage1_checkpoint.json"
    cells = {}
    all_cell_trials = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        cells = ckpt.get("cells", {})
        all_cell_trials = ckpt.get("trial_details", {})
        print(f"  Resumed {len(cells)} cells from checkpoint")

    # Saturation tracking
    sat = init_saturation(kl)
    for ck, c in cells.items():
        nk_s = int(ck.split("_")[0])
        for cond in ["RI", "PI"]:
            update_saturation(sat, nk_s, cond, c["stats"][cond]["accuracy"])

    sweep_start = time.time()
    total_cells = len(feasible)
    cell_idx = 0
    total_infer_time = 0.0
    total_prompts = 0

    for nk in kl:
        for nu in ul:
            if (nk, nu) not in feasible:
                continue
            cell_key = f"{nk}_{nu}"
            cell_idx += 1

            if cell_key in cells:
                continue

            if is_saturated(sat, nk):
                print(f"  [{cell_idx}/{total_cells}] {cell_key}: SATURATED")
                continue

            # Generate prompts for this cell
            cell_prompts = []
            cell_trial_meta = {"RI": [], "PI": []}
            for condition in ["RI", "PI"]:
                for t_idx in range(args.trials):
                    seed = hash((nk, nu, condition, t_idx, "v3")) % (2**31)
                    trial = generate_trial(nk, nu, condition, seed,
                                           use_chat_format=use_chat, tokenizer=tokenizer,
                                           dataset_type=dataset_type)
                    cell_trial_meta[condition].append(trial)
                    cell_prompts.append((condition, t_idx, trial))

            # Run vLLM for this cell
            prompts_list = [t[2]["prompt"] for t in cell_prompts]
            infer_start = time.time()
            try:
                answers = run_vllm_generate(llm, prompts_list, max_new_tokens=args.max_new_tokens)
            except Exception as e:
                print(f"  [{cell_idx}/{total_cells}] {cell_key:>8}  ERROR: {e}")
                continue
            cell_infer = time.time() - infer_start
            total_infer_time += cell_infer
            total_prompts += len(prompts_list)

            # Evaluate
            cell_trials = {"RI": [], "PI": []}
            for (condition, t_idx, trial), answer in zip(cell_prompts, answers):
                error_type = classify_error(
                    answer, trial["expected"], trial["initial_value"], trial["final_value"],
                    trial["all_values"], trial["condition"])
                error_detail = get_error_detail(answer, trial["all_values"])
                n_vals = len(trial["all_values"])

                cell_trials[condition].append({
                    "seed": trial["seed"], "condition": condition,
                    "query_word": "first" if condition == "RI" else "last",
                    "expected": trial["expected"],
                    "expected_idx": 0 if condition == "RI" else n_vals - 1,
                    "expected_relative_pos": 0.0 if condition == "RI" else 1.0,
                    "predicted": answer, "correct": error_type == "correct",
                    "error_type": error_type,
                    "predicted_idx": error_detail["predicted_idx"],
                    "predicted_relative_pos": error_detail["predicted_relative_pos"],
                    "all_values": trial["all_values"],
                    "output_length": len(answer.split()), "output_raw": answer,
                })

            cell_stats, regime = compute_cell_stats(cell_trials)
            cells[cell_key] = {
                "num_keys": nk, "num_updates": nu,
                "stats": cell_stats, "regime": regime,
                "n_trials": len(cell_trials["RI"]),
                "max_trials": args.trials, "stopped_early": False,
            }
            all_cell_trials[cell_key] = cell_trials

            ri = cell_stats["RI"]
            pi = cell_stats["PI"]
            for cond in ["RI", "PI"]:
                update_saturation(sat, nk, cond, cell_stats[cond]["accuracy"])

            print(f"  [{cell_idx}/{total_cells}] {cell_key:>8}  "
                  f"RI={ri['accuracy']:.0%} PI={pi['accuracy']:.0%} "
                  f"regime={regime} garbage={pi['n_garbage']}/{pi['n_failures']} "
                  f"({cell_infer:.1f}s)")

            # Checkpoint every 5 cells
            if cell_idx % 5 == 0:
                _save_checkpoint(cells, all_cell_trials, model_name, dataset_type,
                                 fmt, kl, ul, args, save_dir)

    # Final save
    elapsed = time.time() - sweep_start
    throughput = total_prompts / total_infer_time if total_infer_time > 0 else 0

    gpu_mem, max_batched_tokens, max_model_len_cfg, dtype_cfg = MODEL_ENGINE_CONFIG.get(
        model_name, DEFAULT_ENGINE_CONFIG)

    results = {
        "model": model_name, "dataset_type": dataset_type,
        "prompt_format": fmt, "min_updates": MIN_UPDATES, "backend": "vllm",
        "config": {
            "key_levels": kl, "update_levels": ul,
            "trials_per_cell": args.trials, "max_new_tokens": args.max_new_tokens,
            "gpu": args.gpu,
            "engine": {"gpu_memory_utilization": gpu_mem,
                       "max_num_batched_tokens": max_batched_tokens,
                       "max_model_len": max_model_len_cfg,
                       "enable_prefix_caching": True, "dtype": dtype_cfg},
            "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": -1,
                         "repetition_penalty": 1.0, "presence_penalty": 0.0,
                         "frequency_penalty": 0.0},
        },
        "cells": cells,
        "total_prompts": total_prompts,
        "inference_time_sec": round(total_infer_time, 1),
        "throughput_prompts_per_sec": round(throughput, 1),
        "start_time": datetime.now(timezone.utc).isoformat(),
        "end_time": datetime.now(timezone.utc).isoformat(),
    }

    save_results(results, all_cell_trials, model_name, dataset_type=dataset_type,
                 output_dir=args.output_dir)
    print_summary(results)
    print(f"  {dataset_type}: {elapsed:.0f}s ({elapsed/60:.1f}m), {throughput:.0f} prompts/sec")
    return True


def _save_checkpoint(cells, all_cell_trials, model_name, dataset_type,
                     fmt, kl, ul, args, save_dir):
    """Save checkpoint after every N cells."""
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model_name, "dataset_type": dataset_type,
        "prompt_format": fmt, "min_updates": MIN_UPDATES, "backend": "vllm",
        "config": {"key_levels": kl, "update_levels": ul,
                   "trials_per_cell": args.trials},
        "cells": cells, "trial_details": all_cell_trials,
    }
    with open(save_dir / "stage1_checkpoint.json", "w") as f:
        json.dump(ckpt, f)


def run_model_both_datasets(model_name, args):
    """Load model ONCE, run BOTH datasets, then free GPU.

    This saves ~30-60s of model loading time per model.
    """
    m_short = model_short_name(model_name)
    datasets = [d.strip() for d in args.dataset.split(",")]

    # Check if all datasets already done
    all_done = True
    for ds in datasets:
        sd = get_save_dir(model_name, ds, args.output_dir)
        if not (sd.exists() and list(sd.glob("stage1_sweep_*.json"))):
            all_done = False
            break
    if all_done:
        print(f"\n  SKIP {model_name}: all datasets already done")
        return True

    # Load model once
    print(f"\n{'=' * 70}")
    print(f"  MODEL: {model_name} ({m_short})")
    try:
        llm, tokenizer, ctx_limit = load_vllm_model(
            model_name, gpu_idx=args.gpu, tp_size=args.tensor_parallel_size)
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return False

    # Run each dataset
    results = {}
    for ds in datasets:
        ok = run_dataset_sweep(llm, tokenizer, ctx_limit, model_name, ds, args)
        results[ds] = "OK" if ok else "FAILED"

    destroy_vllm_model(llm)

    for ds, status in results.items():
        print(f"  {status:>8}  {ds}")
    return all(v == "OK" for v in results.values())


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    datasets = [d.strip() for d in args.dataset.split(",")]
    print(f"Datasets: {datasets}")

    if args.all:
        print("=" * 70)
        print(f"V3 UNIFIED GRID SWEEP — ALL MODELS x ALL DATASETS (vLLM)")
        print(f"  Models: {len(ALL_MODELS)} | GPU: {args.gpu} | Trials: {args.trials}")
        print(f"  Datasets: {datasets}")
        print("=" * 70)

        total_start = time.time()
        status = {}

        for i, model_name in enumerate(ALL_MODELS, 1):
            print(f"\n{'#' * 70}")
            print(f"# [{i}/{len(ALL_MODELS)}] {model_name}")
            print(f"{'#' * 70}")
            ok = run_model_both_datasets(model_name, args)
            status[model_name] = "OK" if ok else "FAILED/SKIPPED"

        elapsed = time.time() - total_start
        print(f"\n{'=' * 70}")
        print(f"ALL DONE — {elapsed:.0f}s ({elapsed/60:.1f}m)")
        for m, s in status.items():
            print(f"  {s:>12}  {m}")
    else:
        run_model_both_datasets(args.model, args)


if __name__ == "__main__":
    main()
