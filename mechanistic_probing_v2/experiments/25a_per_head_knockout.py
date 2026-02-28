"""
Exp 25a: Per-head causal knockout — gold standard head identification.

For EVERY head in the model, zero it out and measure the change in PI
logit_diff. This gives a continuous causal ranking — no threshold needed.

Positive causal_effect = head was hurting PI (promoting init) → primacy head
Negative causal_effect = head was helping PI (promoting final) → retrieval head
Near zero = head doesn't matter for PI

Method: Wang et al. 2022 (IOI Circuit) — per-head ablation with logit_diff metric.
Metric: Heimersheim & Nanda 2024 — logit_diff for stable recovery measurement.

Usage:
    cd mechanistic_probing_v2
    uv run python experiments/25a_per_head_knockout.py \
        --model google/gemma-3-1b-it --keys 2 --updates 2 --trials 100
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
from core.dataset_configs import format_for_chat, ORIGINAL_CATEGORIES
from core.model_loader import verify_single_token
from core.analysis_utils import compute_logit_diff
from core.output import save_results


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

    prompt = (
        f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{stream}\n\n"
        f"What was the {query_word} value of {test_cat}?"
    )
    return {
        "prompt": prompt, "condition": condition, "expected": expected,
        "initial_value": cat_values[0], "final_value": cat_values[-1],
    }


def get_token_ids(value, value_to_tid, tokenizer):
    """Get both space-prefixed and bare token IDs for a value."""
    tid_sp = value_to_tid.get(value, -1)
    tid_bare = tokenizer.encode(value, add_special_tokens=False)[0]
    return list(set([t for t in [tid_sp, tid_bare] if t >= 0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--keys", type=int, default=2)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--n-ctx", type=int, default=2048)
    args = parser.parse_args()

    print("=" * 70)
    print("PER-HEAD CAUSAL KNOCKOUT — Gold standard head identification")
    print(f"  keys={args.keys}, updates={args.updates}, trials={args.trials}")
    print("=" * 70)

    model, tokenizer, info = load_model(args.model, n_ctx=args.n_ctx)
    value_to_tid = verify_single_token(tokenizer)
    value_pool = list(value_to_tid.keys())

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    total_heads = n_layers * n_heads

    print(f"  {n_layers} layers × {n_heads} heads = {total_heads} heads to test")

    # ── Build all PI trials upfront ──
    trials = []
    for t_idx in range(args.trials):
        seed = hash(("PI", t_idx, args.updates, args.keys, "25a")) % (2**31)
        trial = build_trial(args.keys, args.updates, "PI", seed, value_pool)
        trials.append(trial)

    # ── Precompute tokens and token IDs for each trial ──
    trial_data = []
    for trial in trials:
        formatted = format_for_chat(trial["prompt"], tokenizer)
        tokens = model.to_tokens(formatted)
        correct_tids = get_token_ids(trial["expected"], value_to_tid, tokenizer)
        wrong_tids = get_token_ids(trial["initial_value"], value_to_tid, tokenizer)
        if not wrong_tids:
            wrong_tids = correct_tids
        trial_data.append({
            "tokens": tokens,
            "correct_tids": correct_tids,
            "wrong_tids": wrong_tids,
        })

    # ── Run baseline (no ablation) ──
    print(f"\n  Running baseline ({args.trials} PI trials)...")
    t_start = time.time()

    baseline_lds = []
    baseline_correct = 0
    for td in trial_data:
        with torch.no_grad():
            logits = model(td["tokens"])
        ld = compute_logit_diff(logits[0, -1], td["correct_tids"], td["wrong_tids"])
        baseline_lds.append(ld)
        pred = logits[0, -1].argmax().item()
        baseline_correct += int(pred in td["correct_tids"])

    baseline_mean_ld = np.mean(baseline_lds)
    baseline_acc = baseline_correct / len(trial_data)
    print(f"  Baseline: PI={baseline_acc:.0%}, mean_ld={baseline_mean_ld:.2f}")

    # ── Sweep all heads ──
    # causal_effect[l, h] = mean(ld_knockout - ld_baseline) across trials
    # Positive = knockout HELPED PI = head was hurting PI (primacy)
    # Negative = knockout HURT PI = head was helping PI (retrieval)
    causal_effect = np.zeros((n_layers, n_heads))
    knockout_acc = np.zeros((n_layers, n_heads))

    print(f"\n  Sweeping {total_heads} heads × {args.trials} trials...")

    for layer in range(n_layers):
        for head in range(n_heads):
            def make_hook(h):
                def hook_fn(activation, hook):
                    activation[:, :, h, :] = 0.0
                    return activation
                return hook_fn

            hook = (f"blocks.{layer}.attn.hook_z", make_hook(head))

            ko_lds = []
            ko_correct = 0
            for i, td in enumerate(trial_data):
                with torch.no_grad():
                    logits = model.run_with_hooks(td["tokens"], fwd_hooks=[hook])
                ld = compute_logit_diff(logits[0, -1], td["correct_tids"], td["wrong_tids"])
                ko_lds.append(ld - baseline_lds[i])
                pred = logits[0, -1].argmax().item()
                ko_correct += int(pred in td["correct_tids"])

            causal_effect[layer, head] = np.mean(ko_lds)
            knockout_acc[layer, head] = ko_correct / len(trial_data)

        elapsed = time.time() - t_start
        done = (layer + 1) * n_heads
        eta = (elapsed / done) * (total_heads - done) if done > 0 else 0
        print(f"  Layer {layer}/{n_layers-1} done ({done}/{total_heads}, "
              f"{elapsed:.0f}s, ETA {eta/60:.0f}min)")

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # ── Results ──
    total_time = time.time() - t_start

    print(f"\n{'=' * 70}")
    print(f"RESULTS ({total_time/60:.1f} minutes)")
    print(f"{'=' * 70}")
    print(f"  Baseline PI: {baseline_acc:.0%}, mean_ld={baseline_mean_ld:.2f}")

    # Top primacy heads (knockout HELPS PI → positive causal effect)
    flat = [(causal_effect[l, h], l, h) for l in range(n_layers) for h in range(n_heads)]
    flat.sort(reverse=True)

    print(f"\n  Top 15 PRIMACY heads (knockout helps PI → head hurts PI):")
    print(f"  {'Rank':>4} {'Head':>8} {'Δld':>8} {'KO acc':>8} {'Baseline':>8}")
    for rank, (effect, l, h) in enumerate(flat[:15]):
        print(f"  {rank+1:>4} L{l}H{h:>2} {effect:>+8.3f} {knockout_acc[l, h]:>7.0%} {baseline_acc:>7.0%}")

    print(f"\n  Top 15 RETRIEVAL heads (knockout hurts PI → head helps PI):")
    for rank, (effect, l, h) in enumerate(flat[-15:][::-1]):
        print(f"  {rank+1:>4} L{l}H{h:>2} {effect:>+8.3f} {knockout_acc[l, h]:>7.0%} {baseline_acc:>7.0%}")

    # ── Save ──
    save_data = {
        "model": args.model,
        "config": {"keys": args.keys, "updates": args.updates, "trials": args.trials},
        "n_layers": n_layers,
        "n_heads": n_heads,
        "baseline": {"accuracy": baseline_acc, "mean_logit_diff": baseline_mean_ld},
        "causal_effect": causal_effect.tolist(),
        "knockout_accuracy": knockout_acc.tolist(),
        "top_primacy_heads": [
            {"layer": int(l), "head": int(h), "causal_effect": float(e)}
            for e, l, h in flat[:20]
        ],
        "top_retrieval_heads": [
            {"layer": int(l), "head": int(h), "causal_effect": float(e)}
            for e, l, h in flat[-20:][::-1]
        ],
        "total_time_sec": round(total_time, 1),
    }
    save_results(save_data, args.model, args.keys, args.updates, "per_head_knockout")


if __name__ == "__main__":
    main()
