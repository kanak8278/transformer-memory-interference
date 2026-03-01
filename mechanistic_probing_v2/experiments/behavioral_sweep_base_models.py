"""
Behavioral Sweep — Unified script for base and instruct models.

Supports all 4 dataset types from dataset_configs.py:
  --dataset ARBITRARY_MULTI    (default) Synthetic prefix+number (Gem42, Art375)
  --dataset ARBITRARY_SINGLE   Single-token random English words
  --dataset SEMANTIC_SINGLE    Single-token real category members (ruby, jade)
  --dataset SEMANTIC_MULTI     Multi-token real category members (alexandrite)

═══════════════════════════════════════════════════════════════════════════
MODELS SUPPORTED
═══════════════════════════════════════════════════════════════════════════

  Base completion models (few-shot format):
    state-spaces/mamba-370m-hf      370M    Pure Mamba (S6)
    state-spaces/mamba2-370m-hf     370M    Pure Mamba2 (SSD)
    EleutherAI/pythia-410m          405M    Transformer (MHA)
    EleutherAI/pythia-160m          160M    Transformer (MHA)

  Instruction-tuned models (chat template format):
    Qwen/Qwen2.5-{0.5B,1.5B,3B}-Instruct
    google/gemma-{3-1b-it,2-2b-it}
    HuggingFaceTB/SmolLM2-{135M,360M,1.7B}-Instruct

═══════════════════════════════════════════════════════════════════════════
EXAMPLES
═══════════════════════════════════════════════════════════════════════════

  # Default (ARBITRARY_MULTI, same as before)
  python behavioral_sweep_base_models.py --model EleutherAI/pythia-410m --gpu 5

  # Semantic single-token values (for mechanistic probing compatibility)
  python behavioral_sweep_base_models.py --model Qwen/Qwen2.5-3B-Instruct \
      --dataset SEMANTIC_SINGLE --gpu 5

  # All 4 datasets on same model (parallel on 4 GPUs)
  for ds in ARBITRARY_MULTI ARBITRARY_SINGLE SEMANTIC_SINGLE SEMANTIC_MULTI; do
    python behavioral_sweep_base_models.py \
        --model Qwen/Qwen2.5-3B-Instruct --dataset $ds --gpu $((5+i)) &
    i=$((i+1))
  done

═══════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path so we can import dataset_configs
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mechanistic_probing_v2.core.dataset_configs import (
    load_dataset_config,
    get_eligible_categories,
    get_max_updates,
    generate_values_for_trial,
    VALID_DATASET_TYPES,
)
from mechanistic_probing_v2.core.model_loader import (
    clear_accelerator_cache,
    is_instruct_model,
    model_short_name,
    load_model_hf,
)
from mechanistic_probing_v2.core.dataset_configs import FIXED_COMPLETION_DEMOS, format_for_chat
from mechanistic_probing_v2.core.inference import run_batch
from mechanistic_probing_v2.core.evaluation import classify_error, bootstrap_ci


# ═══════════════════════════════════════════════════════════════════════════
# DEFAULT CONFIG — Edit this dict to change defaults. CLI args override.
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    # ── Model ──────────────────────────────────────────────────────────────
    "model": "state-spaces/mamba-370m-hf",

    # ── Dataset ───────────────────────────────────────────────────────────
    "dataset": "ARBITRARY_MULTI",
    # ARBITRARY_MULTI    → Gem42, Art375 (unlimited, default)
    # ARBITRARY_SINGLE   → random English words (2300 pool)
    # SEMANTIC_SINGLE    → ruby, jade, pearl (36 cats, 791 values)
    # SEMANTIC_MULTI     → alexandrite, amethyst (46 cats, 2403 values)

    # ── Hardware ───────────────────────────────────────────────────────────
    "gpu": 5,
    "batch_size": 8,
    "dtype": "float32",

    # ── Experiment grid ────────────────────────────────────────────────────
    "key_levels": [2, 3, 5, 7, 10],
    "update_levels": [1, 3, 5, 10, 20, 30, 50],
    "trials_per_cell": 200,

    # ── Early stopping ─────────────────────────────────────────────────────
    "saturation_threshold": 3,

    # ── Paths ──────────────────────────────────────────────────────────────
    "save_dir": "results",
    "resume": None,
}

# ── Instruct models use a bigger grid (larger context windows)
INSTRUCT_KEY_LEVELS = [2, 3, 5, 7, 10, 15, 20, 25, 30, 46]
INSTRUCT_UPDATE_LEVELS = [1, 3, 5, 10, 20, 50, 100, 200]

SYSTEM_PROMPT = "Answer with ONLY the exact value. No explanation."


# ═══════════════════════════════════════════════════════════════════════════
# CLI ARGS
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Behavioral sweep with unified dataset support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test
  python behavioral_sweep_base_models.py --model EleutherAI/pythia-410m --gpu 5 --trials 20

  # Semantic single-token values
  python behavioral_sweep_base_models.py --model Qwen/Qwen2.5-3B-Instruct \\
      --dataset SEMANTIC_SINGLE --gpu 5

  # Resume interrupted run
  python behavioral_sweep_base_models.py --model EleutherAI/pythia-410m --gpu 5 \\
      --resume results/pythia-410m/behavioral_sweep_partial.json
        """,
    )
    parser.add_argument("--model", default=DEFAULT_CONFIG["model"],
                        help=f"HuggingFace model name (default: {DEFAULT_CONFIG['model']})")
    parser.add_argument("--dataset", default=DEFAULT_CONFIG["dataset"],
                        choices=VALID_DATASET_TYPES,
                        help=f"Dataset type (default: {DEFAULT_CONFIG['dataset']})")
    parser.add_argument("--gpu", type=int, default=DEFAULT_CONFIG["gpu"],
                        help=f"GPU index 0-7 (default: {DEFAULT_CONFIG['gpu']})")
    parser.add_argument("--trials", type=int, default=DEFAULT_CONFIG["trials_per_cell"],
                        help=f"Trials per cell per condition (default: {DEFAULT_CONFIG['trials_per_cell']})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"],
                        help=f"Inference batch size (default: {DEFAULT_CONFIG['batch_size']})")
    parser.add_argument("--save-dir", type=str, default=DEFAULT_CONFIG["save_dir"],
                        help=f"Output directory (default: {DEFAULT_CONFIG['save_dir']})")
    parser.add_argument("--resume", type=str, default=DEFAULT_CONFIG["resume"],
                        help="Path to partial results JSON to resume from")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# INTERLEAVING (shared by trials and demos)
# ═══════════════════════════════════════════════════════════════════════════

def shuffle_no_consecutive(items, rng, max_attempts=100):
    """Shuffle items so no two consecutive items share the same category."""
    for _ in range(max_attempts):
        candidate = items.copy()
        rng.shuffle(candidate)
        ok = all(candidate[i]["category"] != candidate[i-1]["category"]
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


# ═══════════════════════════════════════════════════════════════════════════
# TRIAL GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_trial(dataset_type, num_keys, num_updates, condition, seed,
                   use_chat_format=False, tokenizer=None):
    """Generate a single trial.

    Prompt format auto-detected:
      use_chat_format=True  → chat template (instruct models)
      use_chat_format=False → FIXED_COMPLETION_DEMOS + raw prompt (base models)
    """
    rng = random.Random(seed)

    eligible = get_eligible_categories(dataset_type, min_values=num_updates)
    categories = rng.sample(eligible, min(num_keys, len(eligible)))
    test_category = categories[seed % num_keys]

    values_per_cat = generate_values_for_trial(
        dataset_type, categories, num_updates, rng
    )

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
        "dataset_type": dataset_type,
    }


def preflight_context_check(dataset_type, num_keys, num_updates, tokenizer,
                            context_limit=2048, safety=0.85, use_chat_format=False):
    """Check if a cell fits in context."""
    trial = generate_trial(dataset_type, num_keys, num_updates, "RI", seed=0,
                           use_chat_format=use_chat_format, tokenizer=tokenizer)
    n_tokens = len(tokenizer.encode(trial["prompt"]))
    limit = int(context_limit * safety)
    return n_tokens <= limit, n_tokens


def preflight_pool_check(dataset_type, num_keys, num_updates, use_chat_format=False):
    """Check if a cell has enough values in the pool.

    For ARBITRARY datasets, always True. For SEMANTIC datasets, checks
    that enough categories have sufficient pool size.
    """
    eligible = get_eligible_categories(dataset_type, min_values=num_updates)
    return len(eligible) >= num_keys, len(eligible)


# ═══════════════════════════════════════════════════════════════════════════
# SWEEP ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def run_sweep(model, tokenizer, model_name, args, feasible_grid,
              key_levels, update_levels, dataset_type, device, use_chat,
              resume_data=None):
    import torch

    m_short = model_short_name(model_name)
    trials_per_cell = args.trials
    batch_size = args.batch_size  # Will be reduced on OOM and stay reduced
    prompt_fmt = "chat_template" if use_chat else "completion_few_shot"
    ds_config = load_dataset_config(dataset_type)

    results = resume_data or {
        "model": model_name,
        "dataset_type": dataset_type,
        "dataset_name": ds_config["name"],
        "prompt_format": prompt_fmt,
        "config": {
            "key_levels": key_levels,
            "update_levels": update_levels,
            "trials_per_cell": trials_per_cell,
            "dataset_type": dataset_type,
            "num_categories": len(ds_config["categories"]),
            "demo_format": "none" if use_chat else "fixed_completion_demos",
            "gpu": args.gpu,
            "device": str(device),
            "initial_batch_size": args.batch_size,
        },
        "feasible_cells": {f"{k}_{u}": t for (k, u), t in feasible_grid.items()},
        "cells": {},
        "start_time": datetime.now(timezone.utc).isoformat(),
    }

    total_cells = len(feasible_grid)
    completed = len(results.get("cells", {}))
    print(f"\nSweep: {total_cells} feasible cells, {trials_per_cell} trials/cell, "
          f"2 conditions = {total_cells * trials_per_cell * 2:,} total trials")
    print(f"  Device: {device}, batch_size: {batch_size}")
    if completed > 0:
        print(f"  Resuming from {completed} completed cells")

    cell_idx = 0
    saturation = {}

    for nk in key_levels:
        saturation.setdefault(nk, {"RI": 0, "PI": 0})

        for nu in update_levels:
            if (nk, nu) not in feasible_grid:
                continue

            cell_key = f"{nk}_{nu}"

            if cell_key in results.get("cells", {}):
                cell_idx += 1
                continue

            if saturation[nk]["RI"] >= 3 and saturation[nk]["PI"] >= 3:
                print(f"  [{cell_idx+1}/{total_cells}] keys={nk}, updates={nu}: SKIPPED (saturated)")
                cell_idx += 1
                continue

            # Proactive cache clearing before each cell
            clear_accelerator_cache(device)

            cell_start = time.time()
            cell_results = {"RI": [], "PI": []}

            for condition in ["RI", "PI"]:
                trials = []
                for t_idx in range(trials_per_cell):
                    seed = hash((nk, nu, condition, t_idx)) % (2**31)
                    trial = generate_trial(dataset_type, nk, nu, condition, seed,
                                           use_chat_format=use_chat, tokenizer=tokenizer)
                    trials.append(trial)

                for batch_start in range(0, len(trials), batch_size):
                    batch_trials = trials[batch_start:batch_start + batch_size]
                    prompts = [t["prompt"] for t in batch_trials]

                    # Try current batch size, reduce on OOM
                    answers = None
                    attempt_size = len(prompts)
                    while answers is None:
                        try:
                            answers = run_batch(model, tokenizer,
                                                prompts[:attempt_size],
                                                max_new_tokens=20, device=device)
                            # Process remaining prompts if we had to split
                            remaining = prompts[attempt_size:]
                            while remaining:
                                chunk = remaining[:attempt_size]
                                remaining = remaining[attempt_size:]
                                answers.extend(run_batch(model, tokenizer, chunk,
                                                         max_new_tokens=20, device=device))
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                clear_accelerator_cache(device)
                                new_size = max(1, attempt_size // 2)
                                if new_size < attempt_size:
                                    print(f"    OOM at batch_size={attempt_size}, "
                                          f"reducing to {new_size}")
                                    attempt_size = new_size
                                    batch_size = new_size  # Sticky for all future batches
                                    answers = None  # Retry
                                else:
                                    # Already at 1, skip these prompts
                                    print(f"    OOM even at batch_size=1, skipping batch")
                                    answers = [""] * len(prompts)
                            else:
                                raise

                    for trial, answer in zip(batch_trials, answers):
                        error_type = classify_error(
                            answer, trial["expected"],
                            trial["initial_value"], trial["final_value"],
                            trial["all_values"], trial["condition"],
                        )
                        cell_results[condition].append({
                            "seed": trial["seed"],
                            "expected": trial["expected"],
                            "predicted": answer,
                            "correct": error_type == "correct",
                            "error_type": error_type,
                        })

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
                    "accuracy": mean, "ci_lower": ci_lo, "ci_upper": ci_hi,
                    "n": len(corrects), "error_types": error_counts,
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

            cell_data = {
                "num_keys": nk, "num_updates": nu,
                "stats": cell_stats, "regime": regime,
                "trials": cell_results,
                "elapsed_sec": round(time.time() - cell_start, 1),
            }
            results["cells"][cell_key] = cell_data

            cell_idx += 1
            print(f"  [{cell_idx}/{total_cells}] keys={nk:>2}, updates={nu:>3}: "
                  f"RI={ri_acc:.0%} [{cell_stats['RI']['ci_lower']:.0%}-{cell_stats['RI']['ci_upper']:.0%}] "
                  f"PI={pi_acc:.0%} [{cell_stats['PI']['ci_lower']:.0%}-{cell_stats['PI']['ci_upper']:.0%}] "
                  f"regime={regime} ({cell_data['elapsed_sec']:.1f}s)")

            if cell_idx % 3 == 0:
                save_results(results, m_short, args.save_dir, partial=True)

    results["end_time"] = datetime.now(timezone.utc).isoformat()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# SAVE AND SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

def save_results(results, model_short, save_dir, partial=False):
    model_dir = Path(save_dir) / model_short
    model_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_partial" if partial else ""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    summary = {k: v for k, v in results.items() if k != "cells"}
    summary["cells"] = {}
    for ck, cd in results["cells"].items():
        summary["cells"][ck] = {k: v for k, v in cd.items() if k != "trials"}

    # Timestamped only — never overwritten
    ts_path = model_dir / f"behavioral_sweep{suffix}_{ts}.json"
    with open(ts_path, "w") as f:
        json.dump(summary, f, indent=2)

    full_path = model_dir / f"behavioral_sweep_full{suffix}_{ts}.json"
    with open(full_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  -> Saved to {ts_path}")


def print_summary(results):
    kl = results["config"]["key_levels"]
    ul = results["config"]["update_levels"]

    print("\n" + "=" * 70)
    print(f"BEHAVIORAL SWEEP SUMMARY — {results['model']}")
    print(f"Dataset: {results.get('dataset_type', '?')} | "
          f"Format: {results.get('prompt_format', '?')}")
    print("=" * 70)

    for label, cond in [("RI", "RI"), ("PI", "PI")]:
        print(f"\n--- {label} Accuracy ---")
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
                    print(f" {acc:>5.0%}" if acc > 0 else f" {'0%':>5}", end="")
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

    regime_b = [(k, v) for k, v in results["cells"].items() if v["regime"] == "B"]
    if regime_b:
        print(f"\n--- PI Error Types in Regime B ({len(regime_b)} cells) ---")
        all_errors = {}
        for _, cell in regime_b:
            for et, count in cell["stats"]["PI"]["error_types"].items():
                all_errors[et] = all_errors.get(et, 0) + count
        total = sum(all_errors.values())
        for et, count in sorted(all_errors.items(), key=lambda x: -x[1]):
            print(f"  {et}: {count}/{total} = {count/total:.1%}")

    ri_accs = [c["stats"]["RI"]["accuracy"] for c in results["cells"].values()]
    pi_accs = [c["stats"]["PI"]["accuracy"] for c in results["cells"].values()]
    print(f"\nOverall: mean RI={np.mean(ri_accs):.1%}, mean PI={np.mean(pi_accs):.1%}, "
          f"gap={np.mean(ri_accs)-np.mean(pi_accs):.1%}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Detect model type and dataset
    use_chat = is_instruct_model(args.model)
    m_short = model_short_name(args.model)
    fmt = "chat_template" if use_chat else "completion_few_shot"
    dataset_type = args.dataset
    ds_config = load_dataset_config(dataset_type)

    # Select grid based on model type
    if use_chat:
        kl = INSTRUCT_KEY_LEVELS
        ul = INSTRUCT_UPDATE_LEVELS
    else:
        kl = DEFAULT_CONFIG["key_levels"]
        ul = DEFAULT_CONFIG["update_levels"]

    # Load model via core/model_loader (handles device detection, dtype, etc.)
    model, tokenizer, info = load_model_hf(args.model, gpu_idx=args.gpu)
    device = info.device
    ctx_limit = info.n_ctx

    print("=" * 70)
    print(f"BEHAVIORAL SWEEP")
    print(f"  Model:    {args.model} ({m_short})")
    print(f"  Type:     {'Instruct (chat template)' if use_chat else 'Base (completion few-shot)'}")
    print(f"  Dataset:  {dataset_type} — {ds_config['name']}")
    print(f"  Device:   {device} ({info.device_name})")
    print(f"  Trials:   {args.trials} per cell per condition")
    print(f"  Batch:    {args.batch_size}")
    print(f"  Grid:     {len(kl)} keys × {len(ul)} updates = {len(kl)*len(ul)} cells")
    print(f"  Format:   {fmt}")
    print(f"  Context:  {ctx_limit}")
    print(f"  Save:     {args.save_dir}/{m_short}/")
    print("=" * 70)

    # Show dataset capacity
    print(f"\nDataset capacity ({dataset_type}):")
    print(f"  Categories: {len(ds_config['categories'])}")
    print(f"  Total values: {ds_config['total_values']}")
    for nk in kl:
        mx = get_max_updates(dataset_type, nk)
        if mx > 0:
            print(f"  {nk:>3} keys → max {mx} updates")

    # Preflight: check both pool size AND context fit
    print(f"\nPreflight check (pool size + context)...")
    feasible = {}
    for nk in kl:
        for nu in ul:
            # Check 1: pool size
            pool_ok, n_eligible = preflight_pool_check(
                dataset_type, nk, nu, use_chat_format=use_chat
            )
            if not pool_ok:
                print(f"  keys={nk:>2}, updates={nu:>3}: SKIPPED "
                      f"(need {nk} categories with {nu}+ values, only {n_eligible} eligible)")
                break

            # Check 2: context length
            ctx_ok, n_tokens = preflight_context_check(
                dataset_type, nk, nu, tokenizer, ctx_limit,
                use_chat_format=use_chat
            )
            if ctx_ok:
                feasible[(nk, nu)] = n_tokens
            else:
                print(f"  keys={nk:>2}, updates={nu:>3}: SKIPPED "
                      f"(~{n_tokens} tokens > {int(ctx_limit * 0.85)} limit)")
                break

    print(f"\nFeasible cells: {len(feasible)} / {len(kl) * len(ul)}")
    print(f"\n{'Keys':>6} | Max updates | Est. tokens")
    print(f"{'-'*6}-+-{'-'*11}-+-{'-'*11}")
    for nk in kl:
        feasible_updates = sorted([nu for (k, nu) in feasible if k == nk])
        if feasible_updates:
            max_nu = feasible_updates[-1]
            print(f"{nk:>6} | {max_nu:>11} | {feasible[(nk, max_nu)]:>11,}")
        else:
            print(f"{nk:>6} | {'NONE':>11} | {'N/A':>11}")

    if not feasible:
        print("\nNo feasible cells! Check dataset pool sizes and context limit.")
        return

    # Show example prompt
    sample_nk, sample_nu = min(feasible.keys())
    sample = generate_trial(dataset_type, sample_nk, sample_nu, "PI", seed=42,
                            use_chat_format=use_chat, tokenizer=tokenizer)
    print(f"\n--- Example prompt ({sample_nk}k, {sample_nu}u, PI, {dataset_type}, {fmt}) ---")
    print(sample["prompt"][:600])
    print(f"...\nExpected: {sample['expected']}")

    # Resume
    resume_data = None
    if args.resume:
        print(f"\nResuming from {args.resume}")
        with open(args.resume) as f:
            resume_data = json.load(f)

    # Run sweep
    results = run_sweep(model, tokenizer, args.model, args, feasible,
                        key_levels=kl, update_levels=ul,
                        dataset_type=dataset_type, device=device,
                        use_chat=use_chat, resume_data=resume_data)

    # Save final
    save_results(results, m_short, args.save_dir, partial=False)

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
