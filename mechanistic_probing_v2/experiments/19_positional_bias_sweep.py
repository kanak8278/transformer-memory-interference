"""
Step 2.13: Positional recency bias sweep.

Add a linear recency bias to the 8 primacy-biased heads' pre-softmax
attention scores. Sweep λ to find whether there's a sweet spot where
PI improves without destroying RI.

    score[j] = q·k_j / √d  +  λ × (j / seq_len)

Two modes:
  - blind:  bias applied to ALL positions (no oracle knowledge)
  - oracle: bias applied ONLY to value token positions

Usage:
    cd mechanistic_probing_v2
    uv run python experiments/19_positional_bias_sweep.py [--trials 20] [--mode blind]
"""

import sys
import json
import random
import argparse
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_loader import load_model
from core.dataset_configs import format_for_chat, ORIGINAL_CATEGORIES
from core.model_loader import verify_single_token


def build_trial(num_keys, num_updates, condition, seed, value_pool, categories):
    rng = random.Random(seed)
    cats = categories[:num_keys]
    total = num_keys * num_updates
    selected = rng.sample(value_pool, total)
    values_per_cat = {}
    idx = 0
    for cat in cats:
        values_per_cat[cat] = selected[idx:idx + num_updates]
        idx += num_updates

    test_cat = cats[seed % num_keys]
    items = []
    for cat in cats:
        for val in values_per_cat[cat]:
            items.append({"category": cat, "value": val})

    rng.shuffle(items)
    for _ in range(100):
        ok = all(items[i]["category"] != items[i-1]["category"] for i in range(1, len(items)))
        if ok:
            break
        rng.shuffle(items)

    stream = "\n".join(f"{it['category']}: {it['value']}" for it in items)
    query_word = "first" if condition == "RI" else "last"
    cat_values = [it["value"] for it in items if it["category"] == test_cat]
    expected = cat_values[0] if condition == "RI" else cat_values[-1]

    return {
        "prompt": (
            f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
            f"{stream}\n\n"
            f"What was the {query_word} value of {test_cat}?"
        ),
        "condition": condition, "expected": expected,
        "initial_value": cat_values[0], "final_value": cat_values[-1],
        "all_values": cat_values, "seed": seed,
    }


def find_value_positions(token_ids, value_to_tid, values):
    """Find token positions for a list of values."""
    positions = {}
    for val in values:
        tid = value_to_tid.get(val, -1)
        if tid < 0:
            continue
        # Also check space-prefixed variant
        space_tid = value_to_tid.get(val, -1)
        pos_list = [i for i, t in enumerate(token_ids) if t == tid]
        positions[val] = pos_list
    return positions


def run_with_bias(model, tokenizer, trial, value_to_tid,
                  heads_to_bias, lam, mode="blind"):
    """Run trial with positional recency bias on specified heads.

    Args:
        lam: bias strength. score[j] += lam * (j / seq_len)
        mode: "blind" (bias all positions) or "oracle" (bias only value positions)
    """
    formatted = format_for_chat(trial["prompt"], tokenizer)
    tokens = model.to_tokens(formatted)
    seq_len = tokens.shape[1]
    token_ids = tokens[0].tolist()

    # For oracle mode: find which positions are value tokens
    value_positions_set = set()
    if mode == "oracle":
        for val in trial["all_values"]:
            tid = value_to_tid.get(val, -1)
            if tid >= 0:
                for i, t in enumerate(token_ids):
                    if t == tid:
                        value_positions_set.add(i)

    # Build position bias vector
    bias = torch.zeros(seq_len, device=tokens.device)
    if mode == "blind":
        for j in range(seq_len):
            bias[j] = lam * (j / seq_len)
    elif mode == "oracle":
        # Only bias value positions, proportional to their position
        for j in value_positions_set:
            bias[j] = lam * (j / seq_len)

    hooks = []
    for layer, head in heads_to_bias:
        def make_hook(h, bias_vec):
            def hook_fn(scores, hook):
                # scores: [batch, n_heads, seq, seq]
                # Add bias to the last query position attending to all key positions
                scores[0, h, -1, :] = scores[0, h, -1, :] + bias_vec
                return scores
            return hook_fn
        hooks.append((f"blocks.{layer}.attn.hook_attn_scores", make_hook(head, bias)))

    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

    pred_tid = logits[0, -1].argmax().item()
    pred_text = tokenizer.decode([pred_tid]).strip()
    correct = pred_text.lower() == trial["expected"].lower()
    return pred_text, correct


def run_baseline(model, tokenizer, trial):
    formatted = format_for_chat(trial["prompt"], tokenizer)
    tokens = model.to_tokens(formatted)
    with torch.no_grad():
        logits = model(tokens)
    pred_tid = logits[0, -1].argmax().item()
    pred_text = tokenizer.decode([pred_tid]).strip()
    correct = pred_text.lower() == trial["expected"].lower()
    return pred_text, correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--keys", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--mode", default="blind", choices=["blind", "oracle"])
    args = parser.parse_args()

    lambda_values = [0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]

    print("=" * 70)
    print(f"POSITIONAL RECENCY BIAS SWEEP ({args.mode} mode)")
    print(f"  keys={args.keys}, updates={args.updates}, trials={args.trials}")
    print(f"  λ values: {lambda_values}")
    print("=" * 70)

    model, tokenizer, info = load_model(args.model, n_ctx=args.n_ctx)
    value_to_tid = verify_single_token(tokenizer)
    value_pool = list(value_to_tid.keys())
    categories = ORIGINAL_CATEGORIES

    # Load primacy-biased heads
    from core.output import load_head_identification
    try:
        primacy_heads = load_head_identification(args.model, args.keys, args.updates)
        print(f"Loaded {len(primacy_heads)} primacy-biased heads")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No head_identification results for {args.model} at {args.keys}k_{args.updates}u. "
            f"Run exp 16 first."
        )

    for l, h in primacy_heads:
        print(f"  L{l}H{h}")

    # ── Run sweep ──
    results = {}
    for lam in lambda_values:
        results[lam] = {"RI": {"correct": 0, "total": 0},
                        "PI": {"correct": 0, "total": 0}}

    print(f"\nRunning {args.trials} trials × 2 conditions × {len(lambda_values)} λ values...")

    for t_idx in range(args.trials):
        for condition in ["RI", "PI"]:
            seed = hash((condition, t_idx, args.updates, "posbias")) % (2**31)
            trial = build_trial(args.keys, args.updates, condition, seed,
                                value_pool, categories)

            for lam in lambda_values:
                if lam == 0:
                    _, correct = run_baseline(model, tokenizer, trial)
                else:
                    _, correct = run_with_bias(
                        model, tokenizer, trial, value_to_tid,
                        primacy_heads, lam, mode=args.mode)

                results[lam][condition]["correct"] += correct
                results[lam][condition]["total"] += 1

        if (t_idx + 1) % 5 == 0:
            print(f"  {t_idx + 1}/{args.trials} done")

    # ── Print results ──
    print(f"\n{'='*70}")
    print(f"RESULTS ({args.mode} mode)")
    print(f"{'='*70}")
    print(f"{'λ':>8} {'RI':>8} {'PI':>8} {'Gap':>8} {'PI-base':>10}")
    print("-" * 50)

    baseline_ri = None
    baseline_pi = None
    for lam in lambda_values:
        ri_acc = results[lam]["RI"]["correct"] / results[lam]["RI"]["total"]
        pi_acc = results[lam]["PI"]["correct"] / results[lam]["PI"]["total"]
        gap = ri_acc - pi_acc

        if lam == 0:
            baseline_ri = ri_acc
            baseline_pi = pi_acc
            marker = " ← baseline"
        else:
            pi_change = pi_acc - baseline_pi
            marker = f" PI {pi_change:+.0%}"

        print(f"{lam:>8.2f} {ri_acc:>7.0%} {pi_acc:>7.0%} {gap:>+7.0%}  {marker}")

    # ── Find sweet spot ──
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")

    best_lam = 0
    best_score = -999
    for lam in lambda_values:
        ri_acc = results[lam]["RI"]["correct"] / results[lam]["RI"]["total"]
        pi_acc = results[lam]["PI"]["correct"] / results[lam]["PI"]["total"]
        # Score: maximize PI while penalizing RI loss
        # Use: PI_accuracy - max(0, baseline_RI - RI) as the objective
        ri_penalty = max(0, baseline_ri - ri_acc)
        score = pi_acc - ri_penalty
        if score > best_score:
            best_score = score
            best_lam = lam

    best_ri = results[best_lam]["RI"]["correct"] / results[best_lam]["RI"]["total"]
    best_pi = results[best_lam]["PI"]["correct"] / results[best_lam]["PI"]["total"]
    print(f"Best λ = {best_lam:.2f} (RI={best_ri:.0%}, PI={best_pi:.0%}, gap={best_ri-best_pi:+.0%})")
    print(f"Baseline: RI={baseline_ri:.0%}, PI={baseline_pi:.0%}, gap={baseline_ri-baseline_pi:+.0%}")

    if best_pi > baseline_pi + 0.1:
        print(f"\nSweet spot FOUND at λ={best_lam:.2f}")
        print(f"  PI improved from {baseline_pi:.0%} to {best_pi:.0%} ({best_pi-baseline_pi:+.0%})")
        print(f"  RI changed from {baseline_ri:.0%} to {best_ri:.0%} ({best_ri-baseline_ri:+.0%})")
        print(f"  Gap went from {baseline_ri-baseline_pi:+.0%} to {best_ri-best_pi:+.0%}")
        print("  → The PI > RI asymmetry is a tunable positional parameter.")
    else:
        print("\nNo clear sweet spot found.")
        print("  → Primacy bias is too strong for a linear positional correction.")
        print("  → Supports the case for QK fine-tuning (Step 2.11).")

    # ── Save results ──
    save_data = {
        "model": args.model,
        "mode": args.mode,
        "config": {"keys": args.keys, "updates": args.updates, "trials": args.trials},
        "primacy_heads": [{"layer": l, "head": h} for l, h in primacy_heads],
        "lambda_values": lambda_values,
        "results": {
            str(lam): {
                cond: {
                    "accuracy": results[lam][cond]["correct"] / results[lam][cond]["total"],
                    "n": results[lam][cond]["total"]
                }
                for cond in ["RI", "PI"]
            }
            for lam in lambda_values
        },
        "best_lambda": best_lam,
    }
    from core.output import save_results
    out_path = save_results(save_data, args.model, args.keys, args.updates,
                            f"positional_bias_sweep_{args.mode}")


if __name__ == "__main__":
    main()
