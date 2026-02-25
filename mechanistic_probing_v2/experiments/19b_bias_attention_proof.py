"""
Step 2.14b: Prove WHY oracle bias works and blind bias doesn't.

Extract attention patterns of the 8 primacy-biased heads under:
  - No bias (baseline)
  - Blind bias at λ=10
  - Oracle bias at λ=10

Show that blind bias shifts attention to query/instruction tokens,
while oracle bias shifts attention to the final value.

Usage:
    cd mechanistic_probing_v2
    uv run python experiments/19b_bias_attention_proof.py [--trials 20]
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
from core.dataset import format_for_chat, ORIGINAL_CATEGORIES
from core.single_token_values import verify_single_token


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


def classify_positions(token_ids, value_to_tid, trial):
    """Classify each token position as initial_value, final_value, other_value, or other."""
    init_tid = value_to_tid.get(trial["initial_value"], -1)
    final_tid = value_to_tid.get(trial["final_value"], -1)
    all_value_tids = set()
    for v in trial["all_values"]:
        tid = value_to_tid.get(v, -1)
        if tid >= 0:
            all_value_tids.add(tid)

    roles = []
    for i, tid in enumerate(token_ids):
        if tid == init_tid:
            roles.append("initial_value")
        elif tid == final_tid:
            roles.append("final_value")
        elif tid in all_value_tids:
            roles.append("other_value")
        else:
            roles.append("other")
    return roles


def run_with_bias_and_extract(model, tokenizer, trial, value_to_tid,
                               heads_to_bias, lam, mode, primacy_heads):
    """Run trial with bias and extract attention patterns for primacy heads."""
    formatted = format_for_chat(trial["prompt"], tokenizer)
    tokens = model.to_tokens(formatted)
    seq_len = tokens.shape[1]
    token_ids = tokens[0].tolist()

    # Classify positions
    roles = classify_positions(token_ids, value_to_tid, trial)

    # Build bias vector
    value_positions_set = set()
    if mode == "oracle":
        for val in trial["all_values"]:
            tid = value_to_tid.get(val, -1)
            if tid >= 0:
                for i, t in enumerate(token_ids):
                    if t == tid:
                        value_positions_set.add(i)

    bias = torch.zeros(seq_len, device=tokens.device)
    if mode == "blind":
        for j in range(seq_len):
            bias[j] = lam * (j / seq_len)
    elif mode == "oracle":
        for j in value_positions_set:
            bias[j] = lam * (j / seq_len)
    # mode == "none": bias stays zero

    # Hook to modify attention scores
    score_hooks = []
    if lam > 0:
        for layer, head in heads_to_bias:
            def make_score_hook(h, bias_vec):
                def hook_fn(scores, hook):
                    scores[0, h, -1, :] = scores[0, h, -1, :] + bias_vec
                    return scores
                return hook_fn
            score_hooks.append((f"blocks.{layer}.attn.hook_attn_scores",
                                make_score_hook(head, bias)))

    # Hook to capture attention patterns
    captured_patterns = {}
    pattern_hooks = []
    for layer, head in primacy_heads:
        def make_pattern_hook(l, h):
            def hook_fn(pattern, hook):
                # pattern: [batch, n_heads, seq, seq]
                captured_patterns[(l, h)] = pattern[0, h, -1, :].detach().cpu().numpy()
                return pattern
            return hook_fn
        hook_name = f"blocks.{layer}.attn.hook_pattern"
        pattern_hooks.append((hook_name, make_pattern_hook(layer, head)))

    all_hooks = score_hooks + pattern_hooks

    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=all_hooks)

    pred_tid = logits[0, -1].argmax().item()
    pred_text = tokenizer.decode([pred_tid]).strip()
    correct = pred_text.lower() == trial["expected"].lower()

    # Aggregate attention by role for each primacy head
    head_attention_by_role = {}
    for (l, h), attn in captured_patterns.items():
        role_attn = {"initial_value": 0.0, "final_value": 0.0,
                     "other_value": 0.0, "other": 0.0}
        for i, role in enumerate(roles):
            role_attn[role] += attn[i]
        head_attention_by_role[(l, h)] = role_attn

    return correct, head_attention_by_role


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--keys", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--lam", type=float, default=10.0)
    args = parser.parse_args()

    print("=" * 70)
    print(f"BIAS ATTENTION PROOF: Where do heads attend under blind vs oracle?")
    print(f"  keys={args.keys}, updates={args.updates}, trials={args.trials}, λ={args.lam}")
    print("=" * 70)

    model, tokenizer, info = load_model(args.model, n_ctx=args.n_ctx)
    value_to_tid = verify_single_token(tokenizer)
    value_pool = list(value_to_tid.keys())
    categories = ORIGINAL_CATEGORIES

    # Load primacy-biased heads
    from core.output import load_head_identification
    primacy_heads = load_head_identification(args.model, args.keys, args.updates)
    print(f"Tracking {len(primacy_heads)} primacy-biased heads")

    if len(primacy_heads) == 0:
        print("  WARNING: 0 primacy heads found. Skipping bias attention proof.")
        print("  This happens at trivial operating points where no head shows primacy bias.")
        from core.output import save_results
        save_results(
            {"model": args.model, "config": {"keys": args.keys, "updates": args.updates},
             "skipped": True, "reason": "0 primacy heads found"},
            args.model, args.keys, args.updates, "bias_attention_proof")
        return

    modes = ["none", "blind", "oracle"]
    # Accumulate attention by role across trials, per mode
    # {mode: {(layer, head): {role: total_attention}}}
    accum = {m: {(l, h): {"initial_value": 0.0, "final_value": 0.0,
                           "other_value": 0.0, "other": 0.0}
                 for l, h in primacy_heads}
             for m in modes}
    accuracy = {m: {"correct": 0, "total": 0} for m in modes}

    print(f"\nRunning {args.trials} PI trials × 3 modes...")

    for t_idx in range(args.trials):
        seed = hash(("PI", t_idx, args.updates, "proof")) % (2**31)
        trial = build_trial(args.keys, args.updates, "PI", seed,
                            value_pool, categories)

        for mode in modes:
            lam = args.lam if mode != "none" else 0.0
            correct, head_attn = run_with_bias_and_extract(
                model, tokenizer, trial, value_to_tid,
                primacy_heads, lam, mode, primacy_heads)

            accuracy[mode]["correct"] += correct
            accuracy[mode]["total"] += 1

            for (l, h), role_attn in head_attn.items():
                for role, val in role_attn.items():
                    accum[mode][(l, h)][role] += val

        if (t_idx + 1) % 5 == 0:
            print(f"  {t_idx + 1}/{args.trials} done")

    # ── Print results ──
    n = args.trials

    print(f"\n{'='*70}")
    print("PI ACCURACY")
    print(f"{'='*70}")
    for mode in modes:
        acc = accuracy[mode]["correct"] / accuracy[mode]["total"]
        print(f"  {mode:<10} {acc:.0%}")

    print(f"\n{'='*70}")
    print("ATTENTION DISTRIBUTION (averaged across 8 primacy heads)")
    print(f"{'='*70}")
    print(f"  {'Mode':<10} {'Initial':>10} {'Final':>10} {'Other Val':>10} {'Non-value':>10}")
    print(f"  {'-'*55}")

    for mode in modes:
        avg = {"initial_value": 0, "final_value": 0, "other_value": 0, "other": 0}
        for (l, h) in primacy_heads:
            for role in avg:
                avg[role] += accum[mode][(l, h)][role] / n
        # Average across heads
        n_heads = len(primacy_heads)
        for role in avg:
            avg[role] /= n_heads
        print(f"  {mode:<10} {avg['initial_value']:>10.3f} {avg['final_value']:>10.3f} {avg['other_value']:>10.3f} {avg['other']:>10.3f}")

    print(f"\n{'='*70}")
    print("PER-HEAD BREAKDOWN")
    print(f"{'='*70}")

    for l, h in primacy_heads:
        print(f"\n  L{l}H{h}:")
        print(f"    {'Mode':<10} {'Initial':>10} {'Final':>10} {'Other Val':>10} {'Non-value':>10}")
        print(f"    {'-'*55}")
        for mode in modes:
            a = {r: accum[mode][(l, h)][r] / n for r in accum[mode][(l, h)]}
            print(f"    {mode:<10} {a['initial_value']:>10.3f} {a['final_value']:>10.3f} {a['other_value']:>10.3f} {a['other']:>10.3f}")

    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")

    # Compare blind vs oracle on where attention goes
    blind_final = sum(accum["blind"][(l, h)]["final_value"] / n for l, h in primacy_heads) / len(primacy_heads)
    blind_other = sum(accum["blind"][(l, h)]["other"] / n for l, h in primacy_heads) / len(primacy_heads)
    oracle_final = sum(accum["oracle"][(l, h)]["final_value"] / n for l, h in primacy_heads) / len(primacy_heads)
    oracle_other = sum(accum["oracle"][(l, h)]["other"] / n for l, h in primacy_heads) / len(primacy_heads)
    baseline_init = sum(accum["none"][(l, h)]["initial_value"] / n for l, h in primacy_heads) / len(primacy_heads)

    print(f"  Baseline: heads attend {baseline_init:.1%} to initial value")
    print(f"  Blind λ={args.lam}: {blind_final:.1%} to final value, {blind_other:.1%} to non-value tokens")
    print(f"  Oracle λ={args.lam}: {oracle_final:.1%} to final value, {oracle_other:.1%} to non-value tokens")

    if blind_other > oracle_other + 0.05:
        print(f"\n  PROVEN: Blind bias diverts {blind_other - oracle_other:.1%} more attention to non-value tokens.")
        print(f"  Oracle keeps attention on values, directing it to the FINAL value instead of initial.")
    else:
        print(f"\n  Difference is small — mechanism may be more subtle.")

    # ── Save ──
    save_data = {
        "model": args.model,
        "lambda": args.lam,
        "config": {"keys": args.keys, "updates": args.updates, "trials": args.trials},
        "accuracy": {m: accuracy[m]["correct"] / accuracy[m]["total"] for m in modes},
        "avg_attention_by_role": {},
    }
    for mode in modes:
        avg = {"initial_value": 0, "final_value": 0, "other_value": 0, "other": 0}
        for (l, h) in primacy_heads:
            for role in avg:
                avg[role] += accum[mode][(l, h)][role] / n
        n_heads = len(primacy_heads)
        for role in avg:
            avg[role] /= n_heads
        save_data["avg_attention_by_role"][mode] = avg

    from core.output import save_results
    out_path = save_results(save_data, args.model, args.keys, args.updates, "bias_attention_proof")


if __name__ == "__main__":
    main()
