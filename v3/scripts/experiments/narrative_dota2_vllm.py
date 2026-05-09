"""
Dota 2 Narrative Interference Experiment — vLLM Backend

Tests PI > RI on naturalistic Dota 2 match narratives using vLLM.
Generates trials on-the-fly using the narrative_generator.

Grid: keys=[2,3,5,7,10,15,20,25,30] x updates=[5,7,10,15,20,30,50,75,100] (same as Stage 1)
Models: All Qwen + Gemma models

Usage:
    export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
    export HF_TOKEN=<your_token>
    python narrative_dota2_vllm.py --all --trials 100
    python narrative_dota2_vllm.py --model Qwen/Qwen2.5-3B-Instruct --trials 50
"""

import os
import sys
import gc
import re
import json
import time
import random
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

os.environ.setdefault("LD_LIBRARY_PATH", "")
if "/opt/conda/lib" not in os.environ["LD_LIBRARY_PATH"]:
    os.environ["LD_LIBRARY_PATH"] = f"/opt/conda/lib:{os.environ['LD_LIBRARY_PATH']}"

_SCRIPT_DIR = Path(__file__).resolve().parent
_V3_SCRIPTS = _SCRIPT_DIR.parent
_REPO_ROOT = _V3_SCRIPTS.parent.parent
for p in [str(_V3_SCRIPTS), str(_REPO_ROOT), str(_SCRIPT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from mechanistic_probing_v2.core.model_loader import (
    is_instruct_model, model_short_name, CONTEXT_LIMITS,
)
from mechanistic_probing_v2.core.evaluation import bootstrap_ci
from narrative_generator.dota2.generator import DotaTrialGenerator

RESULTS_DIR = _V3_SCRIPTS.parent / "results_vllm" / "narrative_dota2"

SYSTEM_PROMPT = (
    "You are a precise data extraction tool. "
    "Read the match narrative carefully and answer the question. "
    "Output ONLY the exact number. No other text, no units, no explanation."
)

# Same unified grid as Stage 1 sweep (after MIN_UPDATES=5 filter)
KEY_LEVELS = [2, 3, 5, 7, 10, 15, 20, 25, 30]
UPDATE_LEVELS = [5, 7, 10, 15, 20, 30, 50, 75, 100]
DEFAULT_TRIALS = 100

ALL_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-3B",
    "google/gemma-3-270m-it",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
]

# Qwen3.5 series — GDN hybrid architecture (separate list for targeted runs)
QWEN35_MODELS = [
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
]

# vLLM engine config per model
MODEL_ENGINE_CONFIG = {
    "Qwen/Qwen2.5-0.5B-Instruct":   (0.95, 65536, 8192, "half"),
    "Qwen/Qwen2.5-1.5B-Instruct":   (0.95, 32768, 8192, "half"),
    "Qwen/Qwen2.5-3B-Instruct":     (0.92, 16384, 8192, "half"),
    "Qwen/Qwen2.5-3B":              (0.92, 16384, 8192, "half"),
    "google/gemma-3-270m-it":        (0.95, 65536, 8192, "bfloat16"),
    "google/gemma-3-1b-it":          (0.95, 32768, 8192, "bfloat16"),
    "google/gemma-3-4b-it":          (0.92, 16384, 8192, "bfloat16"),
    # Qwen3.5 — GDN hybrid, bfloat16, language_model_only + enforce_eager required
    # Qwen3.5 GDN: max_num_batched_tokens must >= max_model_len (chunked_prefill=False)
    # 4B/9B use 16384 (= max_model_len) to process one long narrative prompt at a time
    # This prevents GDN linear_attn crash on variable-length long sequences
    "Qwen/Qwen3.5-0.8B":             (0.92, 16384, 16384, "bfloat16"),
    "Qwen/Qwen3.5-2B":               (0.92, 65536, 16384, "bfloat16"),
    "Qwen/Qwen3.5-4B":               (0.92, 16384, 16384, "bfloat16"),
    "Qwen/Qwen3.5-9B":               (0.92, 16384, 16384, "bfloat16"),
}
DEFAULT_CONFIG = (0.90, 16384, 8192, "half")


def extract_answer(raw_output, expected):
    """Extract numeric answer from model output."""
    raw = raw_output.strip()
    expected_str = str(expected)
    if raw == expected_str:
        return True, raw
    numbers = re.findall(r'\d+', raw)
    if numbers:
        for n in numbers:
            if n == expected_str:
                return True, n
        return False, numbers[0]
    return False, raw


def load_vllm_model(model_name, gpu_idx=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    from vllm import LLM
    from transformers import AutoTokenizer

    gpu_mem, max_batched, max_model_len, dtype = MODEL_ENGINE_CONFIG.get(
        model_name, DEFAULT_CONFIG)
    ctx_limit = CONTEXT_LIMITS.get(model_name, 4096)
    max_model_len = min(max_model_len, ctx_limit)

    # Qwen3.5 uses GDN — requires special flags to avoid engine crashes
    is_gdn = "Qwen3.5" in model_name
    extra = {}
    if is_gdn:
        extra["language_model_only"] = True      # skip vision encoder
        extra["enforce_eager"] = True            # skip CUDA graph (GDN cache issue)
        extra["enable_prefix_caching"] = False   # experimental for GDN
        extra["enable_chunked_prefill"] = False  # triggers GDN linear_attn bug
    else:
        extra["enable_prefix_caching"] = True

    print(f"\nLoading {model_name} via vLLM (dtype={dtype}, max_model_len={max_model_len}, gdn={is_gdn})")
    llm = LLM(
        model=model_name, gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len, max_num_batched_tokens=max_batched,
        trust_remote_code=True, dtype=dtype, seed=42,
        **extra,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Engine ready.")
    return llm, tokenizer, ctx_limit


def run_vllm_generate(llm, prompts, max_new_tokens=30):
    from vllm import SamplingParams
    params = SamplingParams(
        max_tokens=max_new_tokens, temperature=0.0, top_p=1.0, top_k=-1,
        repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
    )
    outputs = llm.generate(prompts, params)
    return [o.outputs[0].text.strip().split("\n")[0].strip() for o in outputs]


def run_narrative_sweep(model_name, args):
    """Run Dota2 narrative experiment for one model."""
    m_short = model_short_name(model_name)
    use_chat = is_instruct_model(model_name)

    save_dir = RESULTS_DIR / m_short
    if save_dir.exists() and list(save_dir.glob("narrative_dota2_*.json")):
        print(f"\n  SKIP {model_name}: already done")
        return True

    # Load model
    try:
        llm, tokenizer, ctx_limit = load_vllm_model(model_name, gpu_idx=args.gpu)
    except Exception as e:
        print(f"\n  FAILED to load {model_name}: {e}")
        return False

    generator = DotaTrialGenerator()
    kl = args.key_levels or KEY_LEVELS
    ul = args.update_levels or UPDATE_LEVELS

    print(f"\n{'='*60}")
    print(f"  Dota 2 Narrative: {model_name}")
    print(f"  Grid: {len(kl)} keys x {len(ul)} updates, {args.trials} trials")
    print(f"{'='*60}")

    all_results = {}
    all_trials_data = {}
    total_prompts = 0
    sweep_start = time.time()

    for nk in kl:
        for nu in ul:
            cell_key = f"{nk}k_{nu}u"
            cell_prompts = []
            cell_meta = []

            # Generate trials and prompts
            for t_idx in range(args.trials):
                seed = hash((nk, nu, t_idx, "dota2_v3")) % (2**31)

                for condition in ["RI", "PI"]:
                    try:
                        trial = generator.generate_trial(nk, nu, condition, seed)
                    except Exception as e:
                        continue

                    narrative = trial["narrative"]
                    q_data = trial["questions"][condition]
                    question = q_data["question"]
                    expected = str(q_data["expected_answer"])

                    user_msg = f"{narrative}\n\n{question}"

                    if use_chat and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                        messages = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                        ]
                        prompt = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True)
                    else:
                        prompt = f"{SYSTEM_PROMPT}\n\n{user_msg}\n\nAnswer:"

                    # Check context length
                    n_tokens = len(tokenizer.encode(prompt))
                    if n_tokens > ctx_limit - 50:
                        continue

                    cell_prompts.append(prompt)
                    cell_meta.append({
                        "condition": condition, "trial_idx": t_idx,
                        "expected": expected, "n_tokens": n_tokens,
                        "seed": seed,
                    })

            if not cell_prompts:
                print(f"  {cell_key}: no feasible prompts")
                continue

            # Run vLLM
            try:
                answers = run_vllm_generate(llm, cell_prompts, max_new_tokens=30)
            except Exception as e:
                print(f"  {cell_key}: ERROR {e}")
                continue

            total_prompts += len(cell_prompts)

            # Evaluate
            cell_trials = {"RI": [], "PI": []}
            for meta, answer in zip(cell_meta, answers):
                correct, extracted = extract_answer(answer, meta["expected"])
                cell_trials[meta["condition"]].append({
                    "correct": correct, "expected": meta["expected"],
                    "predicted": answer, "extracted": extracted,
                    "n_tokens": meta["n_tokens"], "seed": meta["seed"],
                })

            # Cell stats
            cell_stats = {}
            for cond in ["RI", "PI"]:
                trials = cell_trials[cond]
                if trials:
                    corrects = [t["correct"] for t in trials]
                    mean, ci_lo, ci_hi = bootstrap_ci(corrects)
                    cell_stats[cond] = {
                        "accuracy": round(mean, 4),
                        "ci_lower": round(ci_lo, 4),
                        "ci_upper": round(ci_hi, 4),
                        "n": len(trials),
                    }
                else:
                    cell_stats[cond] = {"accuracy": 0, "n": 0}

            ri_acc = cell_stats["RI"]["accuracy"]
            pi_acc = cell_stats["PI"]["accuracy"]
            gap = ri_acc - pi_acc

            all_results[cell_key] = {"stats": cell_stats, "gap": round(gap, 4)}
            all_trials_data[cell_key] = cell_trials

            print(f"  {cell_key:>8}  RI={ri_acc:.0%} ({cell_stats['RI']['n']}t)  "
                  f"PI={pi_acc:.0%} ({cell_stats['PI']['n']}t)  gap={gap:+.0%}")

    elapsed = time.time() - sweep_start

    # Summary
    if all_results:
        ri_accs = [v["stats"]["RI"]["accuracy"] for v in all_results.values()]
        pi_accs = [v["stats"]["PI"]["accuracy"] for v in all_results.values()]
        mean_ri = float(np.mean(ri_accs))
        mean_pi = float(np.mean(pi_accs))
        mean_gap = mean_ri - mean_pi
    else:
        mean_ri = mean_pi = mean_gap = 0

    print(f"\n  Overall: RI={mean_ri:.1%} PI={mean_pi:.1%} gap={mean_gap:+.1%}")
    print(f"  {total_prompts} prompts, {elapsed:.0f}s")

    # Save
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    result = {
        "model": model_name, "dataset": "dota2_narrative",
        "config": {"key_levels": kl, "update_levels": ul,
                   "trials_per_cell": args.trials},
        "results": all_results,
        "summary": {"mean_ri": round(mean_ri, 4), "mean_pi": round(mean_pi, 4),
                     "mean_gap": round(mean_gap, 4)},
        "total_prompts": total_prompts,
        "elapsed_sec": round(elapsed, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(save_dir / f"narrative_dota2_{ts}.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(save_dir / f"narrative_dota2_{ts}_trials.json", "w") as f:
        json.dump({"model": model_name, "trial_details": all_trials_data}, f, indent=2)

    print(f"  Saved to {save_dir}")

    # Cleanup
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return True


def main():
    parser = argparse.ArgumentParser(description="Dota 2 narrative PI/RI (vLLM)")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--model", default=None)
    g.add_argument("--all", action="store_true", help="Run all Qwen2.5+Gemma models")
    g.add_argument("--qwen35", action="store_true", help="Run all Qwen3.5 models")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--key-levels", type=int, nargs="+", default=None)
    parser.add_argument("--update-levels", type=int, nargs="+", default=None)
    args = parser.parse_args()

    if args.qwen35:
        models = QWEN35_MODELS
    elif args.all:
        models = ALL_MODELS
    else:
        models = [args.model or "Qwen/Qwen2.5-1.5B-Instruct"]

    print("=" * 60)
    print(f"DOTA 2 NARRATIVE EXPERIMENT (vLLM)")
    print(f"  Models: {len(models)} | Trials: {args.trials}")
    print("=" * 60)

    status = {}
    for i, model_name in enumerate(models, 1):
        print(f"\n{'#'*60}")
        print(f"# [{i}/{len(models)}] {model_name}")
        print(f"{'#'*60}")
        ok = run_narrative_sweep(model_name, args)
        status[model_name] = "OK" if ok else "FAILED"

    print(f"\n{'='*60}")
    print("ALL DONE")
    for m, s in status.items():
        print(f"  {s:>8}  {m}")


if __name__ == "__main__":
    main()
