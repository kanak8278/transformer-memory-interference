"""
Step 2.10: Forced attention intervention.

Instead of zeroing out primacy-biased heads, FORCE their attention
to the correct position. This distinguishes:
  - QK circuit problem (where the head attends) vs
  - OV circuit problem (what the head outputs)

If forcing L16H3 to attend to the final value fixes PI → QK problem only.
If it doesn't fix PI → OV circuit also encodes primacy bias.

Design:
  - For PI trials: force primacy-biased heads to attend to final value position
  - For RI trials (control): force them to attend to initial value position
  - Measure accuracy change vs baseline and vs knockout

Usage:
    cd mechanistic_probing_v2
    uv run python experiments/18_forced_attention.py [--trials 20]
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


def run_with_forced_attention(model, tokenizer, trial, value_to_tid,
                              heads_to_force, target_position):
    """Run trial with specific heads forced to attend to target_position.

    Forces attention by replacing the attention pattern post-softmax:
    set attention[answer_pos, :] = 0, attention[answer_pos, target_pos] = 1.
    This makes the head copy ONLY from target_position.
    """
    formatted = format_for_chat(trial["prompt"], tokenizer)
    tokens = model.to_tokens(formatted)
    seq_len = tokens.shape[1]

    hooks = []
    for layer, head in heads_to_force:
        def make_hook(h, tgt_pos):
            def hook_fn(pattern, hook):
                # pattern: [batch, n_heads, seq, seq]
                # Force this head at the answer position to attend only to tgt_pos
                pattern[0, h, -1, :] = 0.0
                if tgt_pos >= 0 and tgt_pos < pattern.shape[-1]:
                    pattern[0, h, -1, tgt_pos] = 1.0
                return pattern
            return hook_fn
        hooks.append((f"blocks.{layer}.attn.hook_pattern", make_hook(head, target_position)))

    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

    pred_tid = logits[0, -1].argmax().item()
    pred_text = tokenizer.decode([pred_tid]).strip()
    correct = pred_text.lower() == trial["expected"].lower()
    return pred_text, correct


def run_with_knockout(model, tokenizer, trial, heads_to_knock):
    """Zero out specific heads."""
    formatted = format_for_chat(trial["prompt"], tokenizer)
    tokens = model.to_tokens(formatted)

    hooks = []
    for layer, head in heads_to_knock:
        def make_hook(h):
            def hook_fn(z, hook):
                z[:, :, h, :] = 0.0
                return z
            return hook_fn
        hooks.append((f"blocks.{layer}.attn.hook_z", make_hook(head)))

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
    args = parser.parse_args()

    print("=" * 70)
    print("FORCED ATTENTION INTERVENTION")
    print(f"  keys={args.keys}, updates={args.updates}, trials={args.trials}")
    print("=" * 70)

    model, tokenizer, info = load_model(args.model, n_ctx=args.n_ctx)
    value_to_tid = verify_single_token(tokenizer)
    value_pool = list(value_to_tid.keys())
    categories = ["color", "animal", "material", "weather", "weapon"]

    # Load primacy-biased heads
    from core.output import load_head_identification
    try:
        primacy_heads = load_head_identification(args.model, args.keys, args.updates)
        print(f"Loaded {len(primacy_heads)} primacy-biased heads")
    except FileNotFoundError:
        primacy_heads = [(16, 3)]
        print("head_identification not found for this operating point, using default: L16H3")

    for l, h in primacy_heads:
        print(f"  L{l}H{h}")

    # ── Run interventions ──
    configs = {
        "baseline": "no_intervention",
        "knockout_primacy": "zero_out",
        "force_correct_value": "force_attention",
        "force_wrong_value": "force_wrong",  # control: force to wrong position
    }

    results = {cfg: {"RI": {"correct": 0, "total": 0}, "PI": {"correct": 0, "total": 0}}
               for cfg in configs}

    print(f"\nRunning {args.trials} trials × 2 conditions × {len(configs)} configs...")

    for t_idx in range(args.trials):
        for condition in ["RI", "PI"]:
            seed = hash((condition, t_idx, args.updates, "forced")) % (2**31)
            trial = build_trial(args.keys, args.updates, condition, seed, value_pool, categories)

            # Find value positions
            init_tid = value_to_tid.get(trial["initial_value"], -1)
            final_tid = value_to_tid.get(trial["final_value"], -1)

            formatted = format_for_chat(trial["prompt"], tokenizer)
            tokens = model.to_tokens(formatted)
            token_ids = tokens[0].tolist()

            init_pos = [i for i, t in enumerate(token_ids) if t == init_tid]
            final_pos = [i for i, t in enumerate(token_ids) if t == final_tid]

            # Correct target: initial for RI, final for PI
            if condition == "RI":
                correct_pos = init_pos[0] if init_pos else -1
                wrong_pos = final_pos[-1] if final_pos else -1
            else:
                correct_pos = final_pos[-1] if final_pos else -1
                wrong_pos = init_pos[0] if init_pos else -1

            # Run each config
            for cfg in configs:
                if cfg == "baseline":
                    _, correct = run_baseline(model, tokenizer, trial)
                elif cfg == "knockout_primacy":
                    _, correct = run_with_knockout(model, tokenizer, trial, primacy_heads)
                elif cfg == "force_correct_value":
                    _, correct = run_with_forced_attention(
                        model, tokenizer, trial, value_to_tid,
                        primacy_heads, correct_pos)
                elif cfg == "force_wrong_value":
                    _, correct = run_with_forced_attention(
                        model, tokenizer, trial, value_to_tid,
                        primacy_heads, wrong_pos)

                results[cfg][condition]["correct"] += correct
                results[cfg][condition]["total"] += 1

        if (t_idx + 1) % 5 == 0:
            print(f"  {t_idx + 1}/{args.trials} done")

    # ── Print results ──
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"{'Config':<30} {'RI':>6} {'PI':>6} {'Gap':>8} {'Note'}")
    print("-" * 70)

    baseline_gap = None
    for cfg in configs:
        ri_acc = results[cfg]["RI"]["correct"] / results[cfg]["RI"]["total"]
        pi_acc = results[cfg]["PI"]["correct"] / results[cfg]["PI"]["total"]
        gap = ri_acc - pi_acc

        if cfg == "baseline":
            baseline_gap = gap
            note = "← baseline"
        elif cfg == "knockout_primacy":
            note = f"gap change: {gap - baseline_gap:+.0%}" if baseline_gap else ""
        elif cfg == "force_correct_value":
            note = "QK fix: force heads to correct position"
        elif cfg == "force_wrong_value":
            note = "control: force heads to wrong position"
        else:
            note = ""

        ri_n = results[cfg]["RI"]["total"]
        pi_n = results[cfg]["PI"]["total"]
        print(f"{cfg:<30} {ri_acc:>5.0%} {pi_acc:>5.0%} {gap:>+7.0%}  {note}")

    # Interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")

    force_pi = results["force_correct_value"]["PI"]["correct"] / results["force_correct_value"]["PI"]["total"]
    baseline_pi = results["baseline"]["PI"]["correct"] / results["baseline"]["PI"]["total"]
    knockout_pi = results["knockout_primacy"]["PI"]["correct"] / results["knockout_primacy"]["PI"]["total"]

    if force_pi > baseline_pi + 0.1:
        print("  Forcing primacy-biased heads to attend to final value IMPROVES PI accuracy.")
        print("  → The problem is in the QK circuit (where the head attends).")
        print("  → The OV circuit correctly outputs whatever the head attends to.")
        print("  → FIX: make the QK circuit instruction-sensitive.")
    elif force_pi <= baseline_pi + 0.05:
        print("  Forcing primacy-biased heads to attend to final value does NOT help PI.")
        print("  → The problem is also in the OV circuit (what the head outputs).")
        print("  → Even when attending to the right position, the head promotes the wrong value.")
    else:
        print("  Marginal improvement — mixed QK/OV problem.")

    if force_pi > knockout_pi:
        print("  Forcing attention works BETTER than knockout → head is useful when correctly aimed.")
    else:
        print("  Forcing attention is no better than knockout → head output is harmful regardless.")

    # Save
    save_data = {
        "model": args.model,
        "config": {"keys": args.keys, "updates": args.updates, "trials": args.trials},
        "primacy_heads": [{"layer": l, "head": h} for l, h in primacy_heads],
        "results": {
            cfg: {
                cond: {"accuracy": results[cfg][cond]["correct"] / results[cfg][cond]["total"],
                       "n": results[cfg][cond]["total"]}
                for cond in ["RI", "PI"]
            }
            for cfg in configs
        },
    }
    from core.output import save_results
    out_path = save_results(save_data, args.model, args.keys, args.updates, "forced_attention")


if __name__ == "__main__":
    main()
