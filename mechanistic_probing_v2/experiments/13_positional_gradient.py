"""
Phase 2: Positional gradient analysis.

Question: Is the primacy effect strictly "first value only" or does it decay
gradually across positions?

For a test category with N updates (values v0, v1, ..., v_{N-1}):
  - Track P(v_i) at the answer position across all layers
  - Track attention to each value position from the answer position
  - Compare RI vs PI conditions

This answers whether it's "absolute primacy" (sharp cliff after position 0)
or "positional decay" (gradual decrease with distance from first appearance).

Uses single-token values so each v_i maps to exactly one token ID.

Usage:
    cd mechanistic_probing_v2
    uv run python experiments/13_positional_gradient.py [--trials 5] [--updates 10]
"""

import sys
import json
import time
import random
import argparse
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_loader import load_model
from core.dataset import format_for_chat, ORIGINAL_CATEGORIES
from core.single_token_values import verify_single_token


def build_trial(
    num_keys: int,
    num_updates: int,
    condition: str,
    seed: int,
    value_pool: list[str],
    categories: list[str],
) -> dict:
    """Build a trial. Returns dict with all value metadata."""
    rng = random.Random(seed)

    total_needed = num_keys * num_updates
    if total_needed > len(value_pool):
        raise ValueError(f"Need {total_needed} values, have {len(value_pool)}")

    selected = rng.sample(value_pool, total_needed)
    values_per_cat = {}
    idx = 0
    for cat in categories[:num_keys]:
        values_per_cat[cat] = selected[idx:idx + num_updates]
        idx += num_updates

    test_cat = categories[seed % num_keys]

    # Build interleaved sequence
    items = []
    for cat in categories[:num_keys]:
        for ui, val in enumerate(values_per_cat[cat]):
            items.append({"category": cat, "value": val, "update_idx": ui})

    rng.shuffle(items)
    for _ in range(100):
        ok = all(items[i]["category"] != items[i-1]["category"] for i in range(1, len(items)))
        if ok:
            break
        rng.shuffle(items)

    stream_lines = [f"{it['category']}: {it['value']}" for it in items]
    stream_text = "\n".join(stream_lines)
    query_word = "first" if condition == "RI" else "last"

    cat_values = [it["value"] for it in items if it["category"] == test_cat]

    prompt = (
        f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{stream_text}\n\n"
        f"What was the {query_word} value of {test_cat}?"
    )

    return {
        "prompt": prompt,
        "condition": condition,
        "test_category": test_cat,
        "all_values": cat_values,  # in order of appearance in prompt
        "num_keys": num_keys,
        "num_updates": num_updates,
        "seed": seed,
        "items": items,  # full sequence for position tracking
    }


def find_value_positions(token_ids: list[int], value_tid: int) -> list[int]:
    """Find all positions where a value token appears."""
    return [i for i, t in enumerate(token_ids) if t == value_tid]


def run_positional_analysis(model, tokenizer, trial: dict, value_to_tid: dict) -> dict:
    """For each of the N values of the test category, track P(value) across layers
    and attention to that value's position."""

    all_values = trial["all_values"]
    n_values = len(all_values)

    # Get token IDs (both space-prefixed for in-context, bare for output)
    value_tids_spaced = []  # space-prefixed (in-context form)
    value_tids_bare = []    # bare (output form)
    for v in all_values:
        sp_toks = tokenizer.encode(f" {v}", add_special_tokens=False)
        bare_toks = tokenizer.encode(v, add_special_tokens=False)
        value_tids_spaced.append(sp_toks[0] if len(sp_toks) == 1 else -1)
        value_tids_bare.append(bare_toks[0] if len(bare_toks) == 1 else -1)

    # Tokenize and run
    formatted = format_for_chat(trial["prompt"], tokenizer)
    tokens = model.to_tokens(formatted)
    token_ids = tokens[0].tolist()
    seq_len = tokens.shape[1]

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)

    # What does the model predict?
    pred_tid = logits[0, -1].argmax().item()
    pred_text = tokenizer.decode([pred_tid]).strip()

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # ── Logit lens: P(v_i) at each layer ──
    # Shape: [n_layers, n_values]
    prob_per_value = np.zeros((n_layers, n_values))
    rank_per_value = np.zeros((n_layers, n_values), dtype=int)

    for layer in range(n_layers):
        resid = cache["resid_post", layer]
        h = resid[0, -1, :]  # answer position
        layer_logits = h @ model.W_U + model.b_U
        probs = torch.softmax(layer_logits, dim=-1)
        sorted_indices = torch.argsort(probs, descending=True)

        for vi in range(n_values):
            # Max probability across space-prefixed and bare forms
            tids_to_check = set()
            if value_tids_spaced[vi] >= 0:
                tids_to_check.add(value_tids_spaced[vi])
            if value_tids_bare[vi] >= 0:
                tids_to_check.add(value_tids_bare[vi])

            if tids_to_check:
                best_prob = max(probs[t].item() for t in tids_to_check)
                best_tid = max(tids_to_check, key=lambda t: probs[t].item())
                best_rank = (sorted_indices == best_tid).nonzero(as_tuple=True)[0].item()
                prob_per_value[layer, vi] = best_prob
                rank_per_value[layer, vi] = best_rank

    # ── Attention: per-head attention to each value's position ──
    # Find where each value appears in the token sequence (use space-prefixed TID)
    value_positions = []
    for vi in range(n_values):
        tid = value_tids_spaced[vi]
        if tid >= 0:
            positions = find_value_positions(token_ids, tid)
            # Take first occurrence (each value should appear once in the test category)
            value_positions.append(positions[0] if positions else -1)
        else:
            value_positions.append(-1)

    # Attention from answer position to each value position
    # Shape: [n_layers, n_heads, n_values]
    attn_per_value = np.zeros((n_layers, n_heads, n_values))

    for layer in range(n_layers):
        pattern = cache["pattern", layer]  # [batch, n_heads, seq, seq]
        attn_from_answer = pattern[0, :, -1, :]  # [n_heads, seq]

        for vi in range(n_values):
            pos = value_positions[vi]
            if pos >= 0:
                attn_per_value[layer, :, vi] = attn_from_answer[:, pos].cpu().numpy()

    # ── Summary stats ──
    # Sum attention across all heads for each value position
    # Shape: [n_layers, n_values]
    total_attn_per_value = attn_per_value.sum(axis=1)  # sum over heads

    # For the last layer, which value gets the most attention per head?
    last_layer_attn = attn_per_value[-1]  # [n_heads, n_values]

    # Cleanup
    del cache
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "condition": trial["condition"],
        "seed": trial["seed"],
        "num_updates": trial["num_updates"],
        "all_values": all_values,
        "predicted": pred_text,
        "expected_first": all_values[0],
        "expected_last": all_values[-1],
        "value_positions": value_positions,
        "seq_len": seq_len,
        # Per-value logit lens: P(v_i) at each layer
        "prob_per_value": prob_per_value.tolist(),
        "rank_per_value": rank_per_value.tolist(),
        # Attention to each value position (summed across heads)
        "total_attn_per_value": total_attn_per_value.tolist(),
        # Per-head attention at last layer
        "last_layer_head_attn": last_layer_attn.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Positional Gradient Analysis")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--updates", type=int, default=10,
                        help="Number of updates per category")
    parser.add_argument("--keys", type=int, default=2,
                        help="Number of categories")
    parser.add_argument("--n-ctx", type=int, default=2048)
    args = parser.parse_args()

    print("=" * 70)
    print("POSITIONAL GRADIENT ANALYSIS")
    print(f"  keys={args.keys}, updates={args.updates}, trials={args.trials}")
    print("=" * 70)

    model, tokenizer, info = load_model(args.model, n_ctx=args.n_ctx)
    value_to_tid = verify_single_token(tokenizer)
    value_pool = list(value_to_tid.keys())
    print(f"Value pool: {len(value_pool)} single-token values")

    categories = ["color", "animal", "material", "weather", "weapon"]

    all_results = {
        "model": args.model,
        "n_layers": info.n_layers,
        "n_heads": info.n_heads,
        "num_keys": args.keys,
        "num_updates": args.updates,
        "trials": args.trials,
        "analyses": [],
    }

    t_start = time.time()

    for condition in ["RI", "PI"]:
        print(f"\n{'='*70}")
        print(f"Condition: {condition}")
        print(f"{'='*70}")

        for t_idx in range(args.trials):
            seed = hash((condition, t_idx, args.updates)) % (2**31)
            trial = build_trial(
                num_keys=args.keys,
                num_updates=args.updates,
                condition=condition,
                seed=seed,
                value_pool=value_pool,
                categories=categories,
            )

            print(f"\n  Trial {t_idx+1}/{args.trials} (seed={seed})")
            print(f"    Values: {trial['all_values']}")

            t0 = time.time()
            result = run_positional_analysis(model, tokenizer, trial, value_to_tid)
            elapsed = time.time() - t0
            result["elapsed_sec"] = round(elapsed, 1)
            all_results["analyses"].append(result)

            # Print the positional gradient at the last few layers
            probs = np.array(result["prob_per_value"])
            attns = np.array(result["total_attn_per_value"])
            n_layers = probs.shape[0]

            print(f"    Predicted: '{result['predicted']}' (seq_len={result['seq_len']})")
            print(f"    Positions: {result['value_positions']}")

            # Show P(v_i) at last 3 layers
            for layer_idx in [n_layers - 4, n_layers - 2, n_layers - 1]:
                if layer_idx < 0:
                    continue
                p_row = probs[layer_idx]
                print(f"    Layer {layer_idx} P(v_i): " +
                      " ".join(f"{p:.4f}" for p in p_row))

            # Show attention gradient at last layer
            a_row = attns[-1]
            print(f"    Last layer attn:  " +
                  " ".join(f"{a:.4f}" for a in a_row))
            print(f"    ({elapsed:.1f}s)")

    total = time.time() - t_start
    all_results["total_elapsed_sec"] = round(total, 1)

    # Save
    from core.output import save_results
    out_path = save_results(all_results, args.model, args.keys, args.updates, "positional_gradient")

    # ── Aggregate summary ──
    print(f"\n{'='*70}")
    print("AGGREGATE POSITIONAL GRADIENT")
    print(f"{'='*70}")

    for condition in ["RI", "PI"]:
        cond_results = [r for r in all_results["analyses"] if r["condition"] == condition]
        if not cond_results:
            continue

        n_updates = args.updates
        # Average P(v_i) at last layer across trials
        avg_probs_last = np.zeros(n_updates)
        avg_attn_last = np.zeros(n_updates)

        for r in cond_results:
            probs = np.array(r["prob_per_value"])
            attns = np.array(r["total_attn_per_value"])
            avg_probs_last += probs[-1, :n_updates]
            avg_attn_last += attns[-1, :n_updates]

        avg_probs_last /= len(cond_results)
        avg_attn_last /= len(cond_results)

        print(f"\n{condition} (n={len(cond_results)} trials, last layer avg):")
        print(f"  Position:  " + " ".join(f"  v{i:<3}" for i in range(n_updates)))
        print(f"  P(v_i):   " + " ".join(f"{p:.4f}" for p in avg_probs_last))
        print(f"  Attn:     " + " ".join(f"{a:.4f}" for a in avg_attn_last))

        # Is it first-only or gradual?
        if avg_probs_last[0] > 0.001:
            ratio = avg_probs_last[1] / avg_probs_last[0] if avg_probs_last[0] > 0 else 0
            print(f"  v1/v0 prob ratio: {ratio:.3f} (0=sharp cliff, ~1=gradual)")
        if avg_attn_last[0] > 0.001:
            ratio = avg_attn_last[1] / avg_attn_last[0] if avg_attn_last[0] > 0 else 0
            print(f"  v1/v0 attn ratio: {ratio:.3f}")


if __name__ == "__main__":
    main()
