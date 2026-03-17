"""
V3 Stage 2: Full Value Logit Lens

Tracks P(v_i) for ALL values in the sequence across ALL layers using TransformerLens.
This is the core mechanistic characterization — shows where in the network the model
commits to the wrong value and what the competition landscape looks like.

Key difference from V2 exp 12: V2 tracked only P(v_0) and P(v_{N-1}).
Stage 2 tracks ALL N values, revealing the near-last error pattern.

Requires: TransformerLens, ARBITRARY_SINGLE dataset (single-token values).

Usage:
    cd v3
    python stage2_logit_lens.py --model Qwen/Qwen2.5-1.5B-Instruct --points "5,5 5,15 5,30" --gpu 0
    python stage2_logit_lens.py --model Qwen/Qwen2.5-3B-Instruct --points "5,10 5,20" --gpu 1

    # Quick test
    python stage2_logit_lens.py --model Qwen/Qwen2.5-1.5B-Instruct --points "5,5" --trials 10 --gpu 0
"""

import sys
import json
import time
import random
import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mechanistic_probing_v2.core.model_loader import (
    load_model, verify_single_token, model_short_name, clear_accelerator_cache,
    is_instruct_model,
)
from mechanistic_probing_v2.core.dataset_configs import (
    format_for_chat, get_value_pool, get_eligible_categories,
    generate_values_for_trial, FIXED_COMPLETION_DEMOS, ORIGINAL_CATEGORIES,
)
from mechanistic_probing_v2.core.evaluation import classify_error

SYSTEM_PROMPT = (
    "You are a precise data extraction tool. "
    "Output ONLY a single word - the exact value requested. "
    "No other text, no explanation, no punctuation."
)
DATASET_TYPE = "ARBITRARY_SINGLE"


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="V3 Stage 2: Full value logit lens")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--points", required=True,
                   help='Operating points as "keys,updates keys,updates ..." e.g. "5,5 5,15 5,30"')
    p.add_argument("--trials", type=int, default=100,
                   help="Trials per condition per operating point (default: 100)")
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--n-ctx", type=int, default=8192)
    return p.parse_args()


def parse_points(points_str):
    """Parse "5,5 5,15 5,30" into [(5,5), (5,15), (5,30)]."""
    points = []
    for p in points_str.strip().split():
        k, u = p.split(",")
        points.append((int(k), int(u)))
    return points


# ═══════════════════════════════════════════════════════════════════════════════
# TRIAL GENERATION (same as stage1 but for TransformerLens)
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


def generate_trial(num_keys, num_updates, condition, seed, value_pool, tokenizer,
                   model_name):
    """Generate a trial with single-token values and format for the model."""
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

    use_chat = is_instruct_model(model_name) if model_name else True

    if use_chat:
        # Instruct model: instruction-style prompt with chat template
        raw_prompt = (
            f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
            f"{stream}\n\n"
            f"What was the {query_word} value of {test_category}?"
        )
        formatted = format_for_chat(raw_prompt, tokenizer, model_name=model_name)
    else:
        # Base model: completion-style prompt with few-shot demos
        # Must match the notebook's format exactly — base models can't follow
        # "What was the last value?" but can complete "The last value of X was:"
        formatted = f"{FIXED_COMPLETION_DEMOS}{stream}\nThe {query_word} value of {test_category} was:"

    return {
        "prompt": formatted,
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
# ERROR DETAIL
# ═══════════════════════════════════════════════════════════════════════════════

def get_error_detail(predicted, all_values):
    pred_lower = predicted.lower().strip()
    for idx, val in enumerate(all_values):
        v = val.lower()
        if v in pred_lower or pred_lower.startswith(v):
            return idx, round(idx / max(len(all_values) - 1, 1), 4)
    return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# LOGIT LENS — CORE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_logit_lens_trial(model, tokenizer, trial, value_to_tid):
    """Run one trial: forward pass with cache, compute P(v_i) at every layer.

    Returns dict with all per-trial data including layer trajectories.
    """
    all_values = trial["all_values"]
    n_values = len(all_values)

    # Get token IDs for each value (space-prefixed + bare, take max prob)
    value_tids = []
    for v in all_values:
        sp = tokenizer.encode(f" {v}", add_special_tokens=False)
        bare = tokenizer.encode(v, add_special_tokens=False)
        tids = set()
        if len(sp) == 1:
            tids.add(sp[0])
        if len(bare) == 1:
            tids.add(bare[0])
        # Fallback: use value_to_tid if available
        if v in value_to_tid:
            tids.add(value_to_tid[v])
        value_tids.append(list(tids))

    # Forward pass with cache
    tokens = model.to_tokens(trial["prompt"])
    seq_len = tokens.shape[1]

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)

    n_layers = model.cfg.n_layers

    # ── Logit lens at every layer ──
    # value_probs_by_layer: shape [n_values, n_layers]
    value_probs_by_layer = np.zeros((n_values, n_layers))

    for layer in range(n_layers):
        resid = cache["resid_post", layer][0, -1, :]  # [d_model]
        layer_logits = resid @ model.W_U + model.b_U   # [vocab_size]
        probs = torch.softmax(layer_logits, dim=-1)

        for vi in range(n_values):
            if value_tids[vi]:
                value_probs_by_layer[vi, layer] = max(
                    probs[tid].item() for tid in value_tids[vi]
                )

    # ── Last layer: global argmax + total value prob ──
    last_probs = torch.softmax(logits[0, -1, :], dim=-1)
    global_argmax_tid = last_probs.argmax().item()
    global_argmax_token = tokenizer.decode([global_argmax_tid]).strip()
    global_argmax_prob = last_probs[global_argmax_tid].item()

    # Check if global argmax is one of our tracked values
    all_value_tids = set()
    for tids in value_tids:
        all_value_tids.update(tids)
    global_argmax_is_value = global_argmax_tid in all_value_tids

    # Total value prob at last layer
    total_value_prob = float(value_probs_by_layer[:, -1].sum())

    # Prediction from logits
    pred_text = global_argmax_token
    correct = pred_text.lower().strip() == trial["expected"].lower().strip()

    error_type = classify_error(
        pred_text, trial["expected"],
        trial["initial_value"], trial["final_value"],
        trial["all_values"], trial["condition"],
    )

    pred_idx, pred_rel_pos = get_error_detail(pred_text, trial["all_values"])

    n_vals = len(trial["all_values"])
    expected_idx = 0 if trial["condition"] == "RI" else n_vals - 1
    expected_rel = 0.0 if trial["condition"] == "RI" else 1.0

    # Clean up cache
    del cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "seed": trial["seed"],
        "condition": trial["condition"],
        "query_word": "first" if trial["condition"] == "RI" else "last",
        "num_keys": trial["num_keys"],
        "num_updates": trial["num_updates"],

        "expected": trial["expected"],
        "expected_idx": expected_idx,
        "expected_relative_pos": expected_rel,

        "predicted": pred_text,
        "correct": correct,
        "error_type": error_type,
        "predicted_idx": pred_idx,
        "predicted_relative_pos": pred_rel_pos,

        "all_values": trial["all_values"],
        "value_probs_by_layer": value_probs_by_layer.tolist(),  # [n_values, n_layers]
        "n_layers": n_layers,

        "total_value_prob": round(total_value_prob, 6),
        "global_argmax_token": global_argmax_token,
        "global_argmax_prob": round(global_argmax_prob, 6),
        "global_argmax_is_value": global_argmax_is_value,

        "seq_len": seq_len,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════════

def get_save_dir(model_name):
    m_short = model_short_name(model_name)
    return Path(__file__).resolve().parent / "results" / m_short


def save_results(results, model_name, partial=False):
    save_dir = get_save_dir(model_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    if partial:
        path = save_dir / "stage2_checkpoint.json"
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = save_dir / f"stage2_logit_lens_{ts}.json"

    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    label = "Checkpoint" if partial else "Saved"
    print(f"  -> {label}: {path}")
    return str(path)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run_stage2(config, model=None, tokenizer=None, info=None):
    """Run Stage 2 logit lens analysis from a config dict.

    Can be called from notebook or CLI. If model/tokenizer/info are provided,
    skips loading (useful in notebook where model is already loaded).

    Args:
        config: dict with keys: model, points (list of (k,u) tuples),
                trials, gpu (optional), n_ctx (optional)
        model, tokenizer, info: pre-loaded TransformerLens model (optional)

    Returns:
        (results, save_path) — results dict and path to saved file
    """
    model_name = config["model"]
    points = config["points"]
    trials = config.get("trials", 100)
    gpu_idx = config.get("gpu", None)
    n_ctx = config.get("n_ctx", 8192)

    print("=" * 70)
    print("V3 STAGE 2: FULL VALUE LOGIT LENS")
    print(f"  Model:    {model_name}")
    print(f"  Points:   {points}")
    print(f"  Trials:   {trials} per condition per point")
    print(f"  Total:    {len(points) * 2 * trials} forward passes with cache")
    print("=" * 70)

    # Load model if not provided
    if model is None:
        model, tokenizer, info = load_model(model_name, n_ctx=n_ctx, gpu_idx=gpu_idx)

    # Build single-token value pool
    candidate_pool = get_value_pool(DATASET_TYPE)
    value_to_tid = verify_single_token(tokenizer, values=candidate_pool)
    value_pool = list(value_to_tid.keys())
    print(f"\nValue pool: {len(candidate_pool)} candidates → {len(value_pool)} single-token verified")

    results = {
        "model": model_name,
        "n_layers": info.n_layers,
        "n_heads": info.n_heads,
        "d_model": info.d_model,
        "dataset_type": DATASET_TYPE,
        "config": config,
        "points": [{"keys": k, "updates": u} for k, u in points],
        "trials_per_condition": trials,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "analyses": [],
    }

    total_trials = len(points) * 2 * trials
    trial_count = 0
    t_start = time.time()

    for num_keys, num_updates in points:
        print(f"\n{'='*60}")
        print(f"Operating point: {num_keys} keys × {num_updates} updates")
        print(f"{'='*60}")

        for condition in ["RI", "PI"]:
            print(f"\n  --- {condition} ({trials} trials) ---")

            for t_idx in range(trials):
                seed = hash((num_keys, num_updates, condition, t_idx, "v3_stage2")) % (2**31)

                trial = generate_trial(
                    num_keys, num_updates, condition, seed,
                    value_pool, tokenizer, model_name,
                )

                t0 = time.time()
                result = run_logit_lens_trial(model, tokenizer, trial, value_to_tid)
                elapsed = time.time() - t0

                results["analyses"].append(result)
                trial_count += 1

                # Print progress
                status = "OK" if result["correct"] else "WRONG"
                vp = result["value_probs_by_layer"]
                last_layer_probs = [vp[vi][-1] for vi in range(len(vp))]
                n_v = len(last_layer_probs)
                if n_v <= 5:
                    prob_str = " ".join(f"{p:.3f}" for p in last_layer_probs)
                else:
                    prob_str = (
                        " ".join(f"{p:.3f}" for p in last_layer_probs[:2])
                        + " ... "
                        + " ".join(f"{p:.3f}" for p in last_layer_probs[-2:])
                    )

                if (t_idx + 1) % 10 == 0 or t_idx == 0:
                    print(f"    [{trial_count}/{total_trials}] t{t_idx}: "
                          f"{status} pred='{result['predicted']}' "
                          f"P(values)=[{prob_str}] "
                          f"total={result['total_value_prob']:.3f} "
                          f"global='{result['global_argmax_token']}' "
                          f"({elapsed:.1f}s)")

            # Per-condition summary
            cond_results = [r for r in results["analyses"]
                           if r["condition"] == condition
                           and r["num_keys"] == num_keys
                           and r["num_updates"] == num_updates]
            n_correct = sum(r["correct"] for r in cond_results)
            n_garbage = sum(1 for r in cond_results if r["predicted_idx"] is None and not r["correct"])

            fail_positions = [r["predicted_relative_pos"] for r in cond_results
                            if not r["correct"] and r["predicted_relative_pos"] is not None]
            avg_fail_pos = np.mean(fail_positions) if fail_positions else None

            print(f"  {condition} summary: {n_correct}/{len(cond_results)} correct, "
                  f"garbage={n_garbage}, "
                  f"fail_avg_pos={avg_fail_pos:.2f}" if avg_fail_pos is not None else
                  f"  {condition} summary: {n_correct}/{len(cond_results)} correct, "
                  f"garbage={n_garbage}, fail_avg_pos=n/a")

        # Checkpoint after each operating point
        save_results(results, model_name, partial=True)

    results["end_time"] = datetime.now(timezone.utc).isoformat()
    results["total_elapsed_sec"] = round(time.time() - t_start, 1)

    # Final save
    save_path = save_results(results, model_name, partial=False)

    # ── Overall summary ──
    print(f"\n{'='*70}")
    print(f"STAGE 2 COMPLETE — {trial_count} trials in {results['total_elapsed_sec']:.0f}s")
    print(f"{'='*70}")

    for num_keys, num_updates in points:
        print(f"\n  {num_keys}k_{num_updates}u:")
        for condition in ["RI", "PI"]:
            cond_results = [r for r in results["analyses"]
                           if r["condition"] == condition
                           and r["num_keys"] == num_keys
                           and r["num_updates"] == num_updates]
            n_correct = sum(r["correct"] for r in cond_results)
            acc = n_correct / len(cond_results) if cond_results else 0

            all_last_probs = []
            for r in cond_results:
                vp = r["value_probs_by_layer"]
                all_last_probs.append([vp[vi][-1] for vi in range(len(vp))])

            if all_last_probs:
                avg_probs = np.mean(all_last_probs, axis=0)
                winner_idx = np.argmax(avg_probs)
                print(f"    {condition}: acc={acc:.0%}, "
                      f"avg P(v_i) winner=v{winner_idx} (P={avg_probs[winner_idx]:.3f}), "
                      f"P(v_last)={avg_probs[-1]:.3f}")

    return results


def main():
    args = parse_args()
    points = parse_points(args.points)
    config = {
        "model": args.model,
        "points": points,
        "trials": args.trials,
        "gpu": args.gpu,
        "n_ctx": args.n_ctx,
    }
    run_stage2(config)


if __name__ == "__main__":
    main()
