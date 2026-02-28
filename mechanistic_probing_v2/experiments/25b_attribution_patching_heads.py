"""
Exp 25b: Attribution patching per head — fast causal approximation.

First-order Taylor approximation of per-head activation patching.
~100x faster than full knockout sweep (25a). Should be validated
against 25a's results for the top-ranked heads.

Method: Syed et al. 2023 "Attribution Patching Outperforms Automated
Circuit Discovery". Also: Nanda 2024 "Attribution Patching at
Industrial Scale".

Formula per head:
  attribution = (clean_output - corrupted_output) · gradient_wrt_output
  where gradient = d(logit_diff) / d(head_output) at corrupted point

Positive attribution = head pushes toward correct PI answer when patched
Negative attribution = head pushes away from correct PI answer when patched

Usage:
    cd mechanistic_probing_v2
    uv run python experiments/25b_attribution_patching_heads.py \
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
from core.dataset import format_for_chat, ORIGINAL_CATEGORIES
from core.model_loader import verify_single_token
from core.output import save_results


def build_matched_pair(num_keys, num_updates, seed, value_pool):
    """Build matched clean/corrupted pair for PI condition."""
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
    test_values = values_per_cat[test_cat]
    initial_value = test_values[0]
    final_value = test_values[-1]

    # Clean: 1 value per key (the correct answer = final for PI)
    clean_items = []
    for cat in categories:
        if cat == test_cat:
            clean_items.append({"category": cat, "value": final_value})
        else:
            clean_items.append({"category": cat, "value": values_per_cat[cat][0]})

    rng2 = random.Random(seed + 1000)
    rng2.shuffle(clean_items)
    clean_stream = "\n".join(f"{it['category']}: {it['value']}" for it in clean_items)
    clean_prompt = (
        f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{clean_stream}\n\n"
        f"What was the last value of {test_cat}?"
    )

    # Corrupted: all updates, interleaved
    corr_items = []
    for cat in categories:
        for val in values_per_cat[cat]:
            corr_items.append({"category": cat, "value": val})

    rng3 = random.Random(seed + 2000)
    rng3.shuffle(corr_items)
    for _ in range(100):
        ok = all(corr_items[i]["category"] != corr_items[i - 1]["category"]
                 for i in range(1, len(corr_items)))
        if ok:
            break
        rng3.shuffle(corr_items)

    corr_stream = "\n".join(f"{it['category']}: {it['value']}" for it in corr_items)
    corr_prompt = (
        f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{corr_stream}\n\n"
        f"What was the last value of {test_cat}?"
    )

    return {
        "clean_prompt": clean_prompt,
        "corrupted_prompt": corr_prompt,
        "expected": final_value,  # PI: correct = final
        "initial_value": initial_value,
        "final_value": final_value,
    }


def compute_attribution(model, tokenizer, pair, value_to_tid):
    """Compute attribution patching scores for all heads on one trial.

    Returns: attribution array [n_layers, n_heads]
    """
    expected = pair["expected"]
    init_val = pair["initial_value"]

    # Token IDs
    exp_tid = tokenizer.encode(expected, add_special_tokens=False)[0]
    init_tid = tokenizer.encode(init_val, add_special_tokens=False)[0]

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # ── Clean forward pass (get per-head outputs) ──
    clean_fmt = format_for_chat(pair["clean_prompt"], tokenizer)
    clean_tokens = model.to_tokens(clean_fmt)

    with torch.no_grad():
        _, clean_cache = model.run_with_cache(clean_tokens)

    # Extract clean head outputs at answer position
    clean_z = {}  # (layer) -> [n_heads, d_head]
    for layer in range(n_layers):
        clean_z[layer] = clean_cache["z", layer][0, -1, :, :].detach().clone()

    del clean_cache

    # ── Corrupted forward pass (with gradients for head outputs) ──
    corr_fmt = format_for_chat(pair["corrupted_prompt"], tokenizer)
    corr_tokens = model.to_tokens(corr_fmt)

    # We need gradients w.r.t. each head's z output at the answer position.
    # Strategy: hook into each layer's z, store references, then backward.
    z_activations = {}

    def make_capture_hook(layer):
        def hook_fn(activation, hook):
            # activation: [batch, seq, n_heads, d_head]
            # Keep only answer position, enable grad
            z = activation[0, -1, :, :].detach().clone().requires_grad_(True)
            z_activations[layer] = z
            # Replace in-place so gradient flows through
            activation[0, -1, :, :] = z
            return activation
        return hook_fn

    hooks = [(f"blocks.{layer}.attn.hook_z", make_capture_hook(layer))
             for layer in range(n_layers)]

    # Forward with hooks — need gradients
    logits = model.run_with_hooks(corr_tokens, fwd_hooks=hooks)

    # Compute logit_diff = logit(final) - logit(init) for PI
    # We want the gradient of this w.r.t. each head's z
    logit_diff = logits[0, -1, exp_tid] - logits[0, -1, init_tid]
    logit_diff.backward()

    # ── Compute attribution per head ──
    attribution = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        if layer not in z_activations or z_activations[layer].grad is None:
            continue

        grad = z_activations[layer].grad  # [n_heads, d_head]
        clean_act = clean_z[layer]  # [n_heads, d_head]
        corr_act = z_activations[layer].detach()  # [n_heads, d_head]

        # Attribution = (clean - corrupted) · gradient
        diff = clean_act - corr_act  # [n_heads, d_head]
        attr = (diff * grad).sum(dim=-1)  # [n_heads]
        attribution[layer] = attr.cpu().numpy()

    # Cleanup
    model.zero_grad()
    del z_activations
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return attribution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--keys", type=int, default=2)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--n-ctx", type=int, default=2048)
    args = parser.parse_args()

    print("=" * 70)
    print("ATTRIBUTION PATCHING PER HEAD — Fast causal approximation")
    print(f"  keys={args.keys}, updates={args.updates}, trials={args.trials}")
    print(f"  Method: Syed et al. 2023 (first-order Taylor approximation)")
    print("=" * 70)

    model, tokenizer, info = load_model(args.model, n_ctx=args.n_ctx)
    value_to_tid = verify_single_token(tokenizer)
    value_pool = list(value_to_tid.keys())

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Accumulate attribution across trials
    total_attr = np.zeros((n_layers, n_heads))
    n_valid = 0

    t_start = time.time()

    for t_idx in range(args.trials):
        seed = hash(("PI", t_idx, args.updates, args.keys, "25b")) % (2**31)
        pair = build_matched_pair(args.keys, args.updates, seed, value_pool)

        try:
            attr = compute_attribution(model, tokenizer, pair, value_to_tid)
            total_attr += attr
            n_valid += 1
        except Exception as e:
            print(f"  Trial {t_idx + 1}: FAILED ({e})")

        if (t_idx + 1) % 25 == 0:
            elapsed = time.time() - t_start
            print(f"  {t_idx + 1}/{args.trials} done ({elapsed:.0f}s)")

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    avg_attr = total_attr / max(n_valid, 1)
    total_time = time.time() - t_start

    # ── Results ──
    print(f"\n{'=' * 70}")
    print(f"RESULTS ({total_time:.0f}s, {n_valid}/{args.trials} valid)")
    print(f"{'=' * 70}")

    # Rank heads
    # Positive attribution = patching this head from clean HELPS PI logit_diff
    # → head is broken in corrupted run, fixing it helps → retrieval head
    # Negative attribution = patching this head HURTS PI
    # → head was doing something useful in corrupted run

    # For primacy identification: heads with most NEGATIVE attribution in
    # corrupted run are the ones where clean→corrupted change HURTS PI most.
    # But we want to compare with 25a's convention:
    # 25a: positive = knockout helps PI = head hurts PI (primacy)
    # 25b: positive attribution = patching helps = head was broken = retrieval
    # So: NEGATIVE attribution ≈ primacy head (patching makes it worse)
    #     POSITIVE attribution ≈ retrieval head (patching fixes it)

    flat = [(avg_attr[l, h], l, h) for l in range(n_layers) for h in range(n_heads)]
    flat.sort(reverse=True)

    print(f"\n  Top 15 heads where patching HELPS PI (retrieval/broken heads):")
    print(f"  {'Rank':>4} {'Head':>8} {'Attribution':>12}")
    for rank, (attr, l, h) in enumerate(flat[:15]):
        print(f"  {rank + 1:>4} L{l}H{h:>2} {attr:>+12.4f}")

    print(f"\n  Top 15 heads where patching HURTS PI (primacy/already-helpful heads):")
    for rank, (attr, l, h) in enumerate(flat[-15:][::-1]):
        print(f"  {rank + 1:>4} L{l}H{h:>2} {attr:>+12.4f}")

    # ── Save ──
    save_data = {
        "model": args.model,
        "config": {"keys": args.keys, "updates": args.updates, "trials": args.trials},
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_valid_trials": n_valid,
        "attribution": avg_attr.tolist(),
        "top_positive_heads": [
            {"layer": int(l), "head": int(h), "attribution": float(a)}
            for a, l, h in flat[:20]
        ],
        "top_negative_heads": [
            {"layer": int(l), "head": int(h), "attribution": float(a)}
            for a, l, h in flat[-20:][::-1]
        ],
        "total_time_sec": round(total_time, 1),
    }
    save_results(save_data, args.model, args.keys, args.updates, "attribution_patching_heads")


if __name__ == "__main__":
    main()
