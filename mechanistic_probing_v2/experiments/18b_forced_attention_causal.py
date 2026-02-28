"""
Exp 18b: Forced attention with causally-identified heads.

Same design as exp 18, but takes heads via CLI instead of loading from
exp 16's attention-based classification.

Use heads identified by exp 25a (per-head causal knockout).

Tests whether the primacy mechanism is:
  (A) QK routing problem: head attends to wrong position → force to correct fixes PI
  (B) Indirect corruption: head corrupts downstream representations → forcing doesn't help

Configs per trial:
  1. baseline — no intervention
  2. knockout — zero out specified heads
  3. force_correct — force heads to attend to correct value position
  4. force_wrong — force heads to attend to wrong value position (control)

Usage:
    cd mechanistic_probing_v2

    # Gemma — single dominant head:
    uv run python experiments/18b_forced_attention_causal.py \
        --model google/gemma-3-1b-it --keys 2 --updates 2 --trials 100 \
        --heads "14,2"

    # Qwen 1.5B — dominant + supporting heads:
    uv run python experiments/18b_forced_attention_causal.py \
        --model Qwen/Qwen2.5-1.5B-Instruct --keys 1 --updates 3 --trials 100 \
        --heads "8,3 0,7 0,3"
"""

import sys
import time
import random
import argparse
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_loader import load_model
from core.dataset import format_for_chat, ORIGINAL_CATEGORIES
from core.model_loader import verify_single_token
from core.analysis_utils import compute_logit_diff
from core.output import save_results


def parse_heads(heads_str):
    """Parse '14,2 8,3 0,7' into [(14,2), (8,3), (0,7)]."""
    heads = []
    for h in heads_str.strip().split():
        parts = h.split(",")
        heads.append((int(parts[0]), int(parts[1])))
    return heads


def build_trial(num_keys, num_updates, condition, seed, value_pool):
    rng = random.Random(seed)
    categories = rng.sample(ORIGINAL_CATEGORIES, min(num_keys, len(ORIGINAL_CATEGORIES)))
    total_needed = num_keys * num_updates
    selected = rng.sample(value_pool, total_needed)
    values_per_cat = {}
    idx = 0
    for cat in categories:
        values_per_cat[cat] = selected[idx:idx + num_updates]
        idx += num_updates

    test_cat = categories[seed % num_keys]
    items = []
    for cat in categories:
        for val in values_per_cat[cat]:
            items.append({"category": cat, "value": val})

    rng.shuffle(items)
    for _ in range(100):
        ok = all(items[i]["category"] != items[i - 1]["category"] for i in range(1, len(items)))
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
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--keys", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--heads", required=True,
                        help="Heads to intervene on as 'layer,head' pairs. E.g., '14,2 8,3'")
    args = parser.parse_args()

    target_heads = parse_heads(args.heads)

    print("=" * 70)
    print("FORCED ATTENTION — Causally-identified heads")
    print(f"  keys={args.keys}, updates={args.updates}, trials={args.trials}")
    print(f"  Heads: {target_heads}")
    print(f"  Source: exp 25a per-head causal knockout")
    print("=" * 70)

    model, tokenizer, info = load_model(args.model, n_ctx=args.n_ctx)
    value_to_tid = verify_single_token(tokenizer)
    value_pool = list(value_to_tid.keys())

    configs = ["baseline", "knockout", "force_correct", "force_wrong"]
    results = {cfg: {"RI": {"correct": 0, "total": 0, "logit_diffs": []},
                      "PI": {"correct": 0, "total": 0, "logit_diffs": []}}
               for cfg in configs}

    t_start = time.time()

    for t_idx in range(args.trials):
        for condition in ["RI", "PI"]:
            seed = hash((condition, t_idx, args.updates, args.keys, "18b")) % (2**31)
            trial = build_trial(args.keys, args.updates, condition, seed, value_pool)

            # Find value positions
            init_tid = value_to_tid.get(trial["initial_value"], -1)
            final_tid = value_to_tid.get(trial["final_value"], -1)

            formatted = format_for_chat(trial["prompt"], tokenizer)
            tokens = model.to_tokens(formatted)
            token_ids = tokens[0].tolist()

            init_pos = [i for i, t in enumerate(token_ids) if t == init_tid]
            final_pos = [i for i, t in enumerate(token_ids) if t == final_tid]

            if not init_pos or not final_pos:
                continue

            # Token IDs for logit_diff
            init_tid_bare = tokenizer.encode(trial["initial_value"], add_special_tokens=False)[0]
            final_tid_bare = tokenizer.encode(trial["final_value"], add_special_tokens=False)[0]
            correct_tids = [init_tid_bare] if condition == "RI" else [final_tid_bare]
            wrong_tids = [final_tid_bare] if condition == "RI" else [init_tid_bare]

            # Correct target position
            if condition == "RI":
                correct_pos = init_pos[0]
                wrong_pos = final_pos[-1]
            else:
                correct_pos = final_pos[-1]
                wrong_pos = init_pos[0]

            for cfg in configs:
                hooks = []

                if cfg == "knockout":
                    for layer, head in target_heads:
                        def make_ko(h):
                            def hook_fn(z, hook):
                                z[:, :, h, :] = 0.0
                                return z
                            return hook_fn
                        hooks.append((f"blocks.{layer}.attn.hook_z", make_ko(head)))

                elif cfg == "force_correct" or cfg == "force_wrong":
                    tgt = correct_pos if cfg == "force_correct" else wrong_pos
                    for layer, head in target_heads:
                        def make_force(h, pos):
                            def hook_fn(pattern, hook):
                                pattern[0, h, -1, :] = 0.0
                                if 0 <= pos < pattern.shape[-1]:
                                    pattern[0, h, -1, pos] = 1.0
                                return pattern
                            return hook_fn
                        hooks.append((f"blocks.{layer}.attn.hook_pattern", make_force(head, tgt)))

                with torch.no_grad():
                    if hooks:
                        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
                    else:
                        logits = model(tokens)

                pred_tid = logits[0, -1].argmax().item()
                pred = tokenizer.decode([pred_tid]).strip()
                correct = pred.lower() == trial["expected"].lower()
                ld = compute_logit_diff(logits[0, -1], correct_tids, wrong_tids)

                results[cfg][condition]["correct"] += int(correct)
                results[cfg][condition]["total"] += 1
                results[cfg][condition]["logit_diffs"].append(ld)

        if (t_idx + 1) % 25 == 0:
            print(f"  {t_idx + 1}/{args.trials} done ({time.time() - t_start:.0f}s)")

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    total_time = time.time() - t_start

    # ── Results ──
    print(f"\n{'=' * 70}")
    print(f"RESULTS ({total_time:.0f}s)")
    print(f"{'=' * 70}")
    print(f"  Heads: {target_heads}")
    print(f"\n  {'Config':<25} {'RI':>6} {'PI':>6} {'Gap':>7} {'RI ld':>8} {'PI ld':>8}")
    print(f"  {'─' * 60}")

    baseline_gap = None
    for cfg in configs:
        ri_n = results[cfg]["RI"]["total"]
        pi_n = results[cfg]["PI"]["total"]
        ri_acc = results[cfg]["RI"]["correct"] / max(ri_n, 1)
        pi_acc = results[cfg]["PI"]["correct"] / max(pi_n, 1)
        gap = ri_acc - pi_acc
        ri_ld = np.mean(results[cfg]["RI"]["logit_diffs"]) if results[cfg]["RI"]["logit_diffs"] else 0
        pi_ld = np.mean(results[cfg]["PI"]["logit_diffs"]) if results[cfg]["PI"]["logit_diffs"] else 0

        if cfg == "baseline":
            baseline_gap = gap

        print(f"  {cfg:<25} {ri_acc:>5.0%} {pi_acc:>5.0%} {gap:>+6.0%} {ri_ld:>+7.2f} {pi_ld:>+7.2f}")

    # ── Interpretation ──
    print(f"\n{'=' * 70}")
    print("INTERPRETATION")
    print(f"{'=' * 70}")

    bl_pi = results["baseline"]["PI"]["correct"] / max(results["baseline"]["PI"]["total"], 1)
    ko_pi = results["knockout"]["PI"]["correct"] / max(results["knockout"]["PI"]["total"], 1)
    fc_pi = results["force_correct"]["PI"]["correct"] / max(results["force_correct"]["PI"]["total"], 1)
    fw_pi = results["force_wrong"]["PI"]["correct"] / max(results["force_wrong"]["PI"]["total"], 1)

    bl_pi_ld = np.mean(results["baseline"]["PI"]["logit_diffs"]) if results["baseline"]["PI"]["logit_diffs"] else 0
    ko_pi_ld = np.mean(results["knockout"]["PI"]["logit_diffs"]) if results["knockout"]["PI"]["logit_diffs"] else 0
    fc_pi_ld = np.mean(results["force_correct"]["PI"]["logit_diffs"]) if results["force_correct"]["PI"]["logit_diffs"] else 0

    print(f"\n  PI accuracy: baseline={bl_pi:.0%}, knockout={ko_pi:.0%}, force_correct={fc_pi:.0%}, force_wrong={fw_pi:.0%}")
    print(f"  PI logit_diff: baseline={bl_pi_ld:+.2f}, knockout={ko_pi_ld:+.2f}, force_correct={fc_pi_ld:+.2f}")

    if fc_pi > bl_pi + 0.15 and fc_pi > ko_pi + 0.05:
        print(f"\n  → MECHANISM (A): QK routing problem.")
        print(f"    Forcing to correct position improves PI by +{fc_pi - bl_pi:.0%}.")
        print(f"    The OV circuit works — head copies correctly when aimed right.")
        print(f"    The primacy bias is in WHERE the head attends, not WHAT it outputs.")
    elif ko_pi > bl_pi + 0.15 and fc_pi <= ko_pi + 0.05:
        print(f"\n  → MECHANISM (B): Indirect corruption.")
        print(f"    Knockout helps (+{ko_pi - bl_pi:.0%}) but forcing doesn't improve over knockout.")
        print(f"    The head's damage is NOT through direct value retrieval.")
        print(f"    It corrupts downstream representations regardless of where it attends.")
    elif ko_pi > bl_pi + 0.15 and fc_pi > ko_pi + 0.05:
        print(f"\n  → MECHANISM (A+B): Mixed — both QK routing and indirect corruption.")
        print(f"    Knockout helps (+{ko_pi - bl_pi:.0%}), forcing helps more (+{fc_pi - bl_pi:.0%}).")
        print(f"    Part of the damage is through attention routing, part through other pathways.")
    else:
        print(f"\n  → WEAK EFFECT: Neither knockout nor forcing shows strong PI change.")
        print(f"    These heads may not be the primary primacy mechanism at this operating point.")

    if fw_pi < bl_pi - 0.1:
        print(f"  → CONTROL: Forcing to WRONG position hurts PI ({bl_pi:.0%}→{fw_pi:.0%}). Head has real causal power.")
    else:
        print(f"  → CONTROL: Forcing to wrong position has weak effect ({bl_pi:.0%}→{fw_pi:.0%}).")

    # ── Save ──
    save_data = {
        "model": args.model,
        "config": {"keys": args.keys, "updates": args.updates, "trials": args.trials},
        "heads_tested": [{"layer": l, "head": h} for l, h in target_heads],
        "head_source": "exp_25a_per_head_knockout",
        "results": {
            cfg: {
                cond: {
                    "accuracy": results[cfg][cond]["correct"] / max(results[cfg][cond]["total"], 1),
                    "mean_logit_diff": float(np.mean(results[cfg][cond]["logit_diffs"])) if results[cfg][cond]["logit_diffs"] else 0,
                    "n": results[cfg][cond]["total"],
                }
                for cond in ["RI", "PI"]
            }
            for cfg in configs
        },
        "total_time_sec": round(total_time, 1),
    }
    save_results(save_data, args.model, args.keys, args.updates, "forced_attention_causal")


if __name__ == "__main__":
    main()
