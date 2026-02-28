"""
Step 2.15b: Ablation + Patching Interaction.

Closes the causal loop: are the 8 primacy heads responsible for the
query-position corruption seen in L16+?

Runs the same granular query-position patching as experiment 22,
but in two modes:
  1. Normal (no ablation) — reproduces exp 22 baseline
  2. With 8 primacy heads ablated — tests if corruption disappears

If ablating the heads removes the need for query patching:
  → The heads CAUSED the query-position corruption
If query patching still helps even with ablation:
  → Two separate mechanisms, each contributing to PI failure

Usage:
    cd mechanistic_probing_v2
    uv run python experiments/23_ablation_patching_interaction.py [--trials 10]
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

from core.model_loader import load_model, verify_single_token
from core.dataset_configs import (
    format_for_chat, ORIGINAL_CATEGORIES, get_value_pool,
)
from core.output import load_results
from core.analysis_utils import compute_logit_diff, compute_recovery, aggregate_recovery


def get_retrieval_onset_layer(model_name, keys, updates, n_layers, threshold=0.10):
    """Find the layer where retrieval begins, using exp 15 activation patching data."""
    try:
        data = load_results(model_name, keys, updates, "activation_patching")
    except FileNotFoundError:
        fallback = 2 * n_layers // 3
        print(f"  WARNING: No activation_patching results. Using fallback: L{fallback}")
        return fallback

    import numpy as np
    pi_trials = [a for a in data["analyses"] if a["condition"] == "PI"]
    if not pi_trials:
        return 2 * n_layers // 3

    layer_recovery = {}
    for a in pi_trials:
        for item in a["patching"]["resid_at_answer"]:
            l = item["layer"]
            layer_recovery.setdefault(l, []).append(item["recovery"])

    for l in sorted(layer_recovery.keys()):
        if np.mean(layer_recovery[l]) > threshold:
            print(f"  Retrieval onset: L{l} ({l/n_layers*100:.0f}% through network, from exp 15)")
            return l

    return n_layers - 1


def build_matched_pair(num_keys, num_updates, condition, seed, value_pool, categories):
    """Build matched clean/corrupted trial pair."""
    rng = random.Random(seed)
    cats = categories[:num_keys]
    test_cat = cats[seed % num_keys]

    total_needed = num_keys * num_updates
    selected = rng.sample(value_pool, total_needed)
    values_per_cat = {}
    idx = 0
    for cat in cats:
        values_per_cat[cat] = selected[idx:idx + num_updates]
        idx += num_updates

    test_values = values_per_cat[test_cat]
    initial_value = test_values[0]
    final_value = test_values[-1]
    expected = initial_value if condition == "RI" else final_value
    query_word = "first" if condition == "RI" else "last"

    # Clean: 1 value per category
    clean_items = []
    for cat in cats:
        if cat == test_cat:
            clean_items.append({"category": cat, "value": expected})
        else:
            clean_items.append({"category": cat, "value": values_per_cat[cat][0]})

    rng2 = random.Random(seed + 1000)
    rng2.shuffle(clean_items)
    clean_stream = "\n".join(f"{it['category']}: {it['value']}" for it in clean_items)
    clean_prompt = (
        f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{clean_stream}\n\n"
        f"What was the {query_word} value of {test_cat}?"
    )

    # Corrupted: all updates
    corr_items = []
    for cat in cats:
        for val in values_per_cat[cat]:
            corr_items.append({"category": cat, "value": val})

    rng3 = random.Random(seed + 2000)
    rng3.shuffle(corr_items)
    for _ in range(100):
        ok = all(corr_items[i]["category"] != corr_items[i-1]["category"]
                 for i in range(1, len(corr_items)))
        if ok:
            break
        rng3.shuffle(corr_items)

    corr_stream = "\n".join(f"{it['category']}: {it['value']}" for it in corr_items)
    corr_prompt = (
        f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{corr_stream}\n\n"
        f"What was the {query_word} value of {test_cat}?"
    )

    clean_trial = {
        "prompt": clean_prompt, "condition": condition,
        "expected": expected, "initial_value": initial_value,
        "final_value": final_value,
    }
    corrupted_trial = {
        "prompt": corr_prompt, "condition": condition,
        "expected": expected, "initial_value": initial_value,
        "final_value": final_value,
    }
    return clean_trial, corrupted_trial


def find_query_positions(model, formatted_text):
    """Find positions of the query (from 'What' to end)."""
    str_tokens = model.to_str_tokens(formatted_text)
    for i in range(len(str_tokens) - 1, -1, -1):
        if "What" in str_tokens[i]:
            return list(range(i, len(str_tokens)))
    return list(range(max(0, len(str_tokens) - 15), len(str_tokens)))


def make_ablation_hooks(primacy_heads):
    """Create hooks that zero out the 8 primacy-biased heads."""
    hooks = []
    for layer, head in primacy_heads:
        def make_hook(h):
            def hook_fn(z, hook):
                z[:, :, h, :] = 0.0
                return z
            return hook_fn
        hooks.append((f"blocks.{layer}.attn.hook_z", make_hook(head)))
    return hooks


def run_with_hooks_and_cache(model, tokens, fwd_hooks):
    """Run model with hooks AND get cache. TransformerLens doesn't natively
    support both, so we capture cache via additional hooks."""
    # We'll use run_with_cache and add ablation hooks on top
    # Actually, run_with_hooks doesn't return cache. We need a workaround.
    # Strategy: add cache-capture hooks alongside the ablation hooks.
    cache_dict = {}

    def make_cache_hook(key):
        def hook_fn(activation, hook):
            cache_dict[key] = activation.detach()
            return activation
        return hook_fn

    all_hooks = list(fwd_hooks)
    n_layers = model.cfg.n_layers
    for layer in range(n_layers):
        all_hooks.append((f"blocks.{layer}.hook_resid_post",
                          make_cache_hook(("resid_post", layer))))
        all_hooks.append((f"blocks.{layer}.hook_attn_out",
                          make_cache_hook(("attn_out", layer))))
        all_hooks.append((f"blocks.{layer}.hook_mlp_out",
                          make_cache_hook(("mlp_out", layer))))

    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=all_hooks)

    return logits, cache_dict


def patch_from_cache_dict(model, corr_tokens, clean_cache_dict,
                          clean_positions, corr_positions,
                          layer, component, extra_hooks=None):
    """Patch clean activations into corrupted run, with optional extra hooks (ablation)."""
    cache_key = (component, layer)
    n_to_patch = min(len(clean_positions), len(corr_positions))

    clean_acts = []
    for i in range(n_to_patch):
        cp = clean_positions[i]
        clean_acts.append(clean_cache_dict[cache_key][0, cp, :].clone())

    if component == "resid_post":
        hook_name = f"blocks.{layer}.hook_resid_post"
    elif component == "attn_out":
        hook_name = f"blocks.{layer}.hook_attn_out"
    elif component == "mlp_out":
        hook_name = f"blocks.{layer}.hook_mlp_out"

    def patch_hook(activation, hook):
        for i in range(n_to_patch):
            if corr_positions[i] < activation.shape[1]:
                activation[0, corr_positions[i], :] = clean_acts[i]
        return activation

    all_hooks = [(hook_name, patch_hook)]
    if extra_hooks:
        all_hooks.extend(extra_hooks)

    with torch.no_grad():
        logits = model.run_with_hooks(corr_tokens, fwd_hooks=all_hooks)
    return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--keys", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--heads", required=True,
                        help="Primacy heads from exp 25a top_primacy_heads. E.g., '12,0 8,8'")
    args = parser.parse_args()

    print("=" * 70)
    print("ABLATION + PATCHING INTERACTION")
    print(f"  keys={args.keys}, updates={args.updates}, trials={args.trials}")
    print("=" * 70)

    model, tokenizer, info = load_model(args.model, n_ctx=args.n_ctx)
    candidate_pool = get_value_pool("ARBITRARY_SINGLE")
    value_to_tid = verify_single_token(tokenizer, values=candidate_pool)
    value_pool = list(value_to_tid.keys())
    categories = ORIGINAL_CATEGORIES

    primacy_heads = [tuple(map(int, h.split(","))) for h in args.heads.strip().split()]
    print(f"Primacy heads to ablate (from exp 25a): {primacy_heads}")

    ablation_hooks = make_ablation_hooks(primacy_heads)

    # Dynamically spaced based on model depth
    n_test = 9
    step = max(1, model.cfg.n_layers // n_test)
    test_layers = sorted(set(list(range(0, model.cfg.n_layers, step)) + [model.cfg.n_layers - 1]))

    # Early/late boundary from exp 15
    onset_layer = get_retrieval_onset_layer(args.model, args.keys, args.updates, model.cfg.n_layers)

    # Modes: normal (no ablation), ablated (8 heads zeroed)
    modes = ["normal", "ablated"]

    # {mode: {layer: [recovery values]}}
    recovery = {m: {l: [] for l in test_layers} for m in modes}
    accuracy = {m: {"correct": 0, "total": 0} for m in modes}

    print(f"\nRunning {args.trials} PI trials × 2 modes × {len(test_layers)} layers...")
    t_start = time.time()

    for t_idx in range(args.trials):
        seed = hash(("PI", t_idx, args.updates, "abl_patch")) % (2**31)
        clean_trial, corr_trial = build_matched_pair(
            args.keys, args.updates, "PI", seed, value_pool, categories)

        expected = corr_trial["expected"]
        exp_tid_sp = value_to_tid.get(expected, -1)
        exp_tid_bare = tokenizer.encode(expected, add_special_tokens=False)[0]
        expected_tids = list(set([t for t in [exp_tid_sp, exp_tid_bare] if t >= 0]))

        # Incorrect token for logit_diff
        wrong_value = corr_trial["initial_value"]  # PI condition: wrong = initial
        wrong_tid_sp = value_to_tid.get(wrong_value, -1)
        wrong_tid_bare = tokenizer.encode(wrong_value, add_special_tokens=False)[0]
        wrong_tids = list(set([t for t in [wrong_tid_sp, wrong_tid_bare] if t >= 0]))
        if not wrong_tids:
            wrong_tids = expected_tids

        clean_fmt = format_for_chat(clean_trial["prompt"], tokenizer)
        corr_fmt = format_for_chat(corr_trial["prompt"], tokenizer)
        clean_tokens = model.to_tokens(clean_fmt)
        corr_tokens = model.to_tokens(corr_fmt)

        clean_query_pos = find_query_positions(model, clean_fmt)
        corr_query_pos = find_query_positions(model, corr_fmt)

        for mode in modes:
            extra = ablation_hooks if mode == "ablated" else []

            # Run clean with mode-specific hooks + cache
            clean_logits, clean_cache = run_with_hooks_and_cache(
                model, clean_tokens, extra)
            clean_ld = compute_logit_diff(clean_logits[0, -1], expected_tids, wrong_tids)

            # Run corrupted with mode-specific hooks (no patching yet)
            with torch.no_grad():
                if extra:
                    corr_logits = model.run_with_hooks(corr_tokens, fwd_hooks=extra)
                else:
                    corr_logits = model(corr_tokens)

            baseline_ld = compute_logit_diff(corr_logits[0, -1], expected_tids, wrong_tids)
            corr_pred = tokenizer.decode([corr_logits[0, -1].argmax().item()]).strip()
            corr_correct = corr_pred.lower() == expected.lower()

            accuracy[mode]["correct"] += corr_correct
            accuracy[mode]["total"] += 1

            # Sweep: patch resid_post at query position for each layer
            for layer in test_layers:
                patched_logits = patch_from_cache_dict(
                    model, corr_tokens, clean_cache,
                    clean_query_pos, corr_query_pos,
                    layer, "resid_post", extra_hooks=extra)

                patched_ld = compute_logit_diff(patched_logits[0, -1], expected_tids, wrong_tids)
                rec = compute_recovery(patched_ld, baseline_ld, clean_ld)

                recovery[mode][layer].append(rec)

            del clean_cache

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        if (t_idx + 1) % 2 == 0:
            print(f"  {t_idx + 1}/{args.trials} done ({time.time() - t_start:.0f}s)")

    # ── Results ──
    total_time = time.time() - t_start

    print(f"\n{'='*70}")
    print("PI ACCURACY (no patching)")
    print(f"{'='*70}")
    for mode in modes:
        acc = accuracy[mode]["correct"] / accuracy[mode]["total"]
        print(f"  {mode:<10} {acc:.0%}")

    print(f"\n{'='*70}")
    print("QUERY-POSITION PATCHING RECOVERY (resid_post)")
    print(f"{'='*70}")
    print(f"  {'Layer':>6}  {'Normal':>10}  {'Ablated':>10}  {'Difference':>10}")
    print(f"  {'-'*45}")

    for layer in test_layers:
        normal_agg = aggregate_recovery(recovery["normal"][layer])
        ablated_agg = aggregate_recovery(recovery["ablated"][layer])
        diff = ablated_agg["mean"] - normal_agg["mean"]
        filt = normal_agg["n_filtered"] + ablated_agg["n_filtered"]
        filt_msg = f"  ({filt} filtered)" if filt > 0 else ""
        print(f"  L{layer:>4}  {normal_agg['mean']:>+9.0%}  {ablated_agg['mean']:>+9.0%}  {diff:>+9.0%}{filt_msg}")

    # Summary
    # Late = post-onset (from exp 15 activation patching)
    late_test = [l for l in test_layers if l >= onset_layer]
    normal_late_agg = aggregate_recovery([v for l in late_test for v in recovery["normal"][l]])
    ablated_late_agg = aggregate_recovery([v for l in late_test for v in recovery["ablated"][l]])
    normal_late = normal_late_agg["mean"]
    ablated_late = ablated_late_agg["mean"]
    normal_acc = accuracy["normal"]["correct"] / accuracy["normal"]["total"]
    ablated_acc = accuracy["ablated"]["correct"] / accuracy["ablated"]["total"]

    print(f"\n  Late-layer (L16-23) average recovery:")
    print(f"    Normal:  {normal_late:+.0%}")
    print(f"    Ablated: {ablated_late:+.0%}")

    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")

    if ablated_acc > normal_acc + 0.15 and ablated_late < normal_late - 0.2:
        print(f"\n  CAUSAL CHAIN CONFIRMED:")
        print(f"  Ablating 8 primacy heads improves PI from {normal_acc:.0%} to {ablated_acc:.0%}")
        print(f"  AND reduces the need for query patching (late recovery {normal_late:+.0%} → {ablated_late:+.0%}).")
        print(f"  → The 8 heads ARE responsible for the query-position corruption at L16+.")
        print(f"  → Complete chain: primacy heads → corrupt query position → wrong answer.")
    elif ablated_acc > normal_acc + 0.15 and ablated_late >= normal_late - 0.2:
        print(f"\n  TWO SEPARATE MECHANISMS:")
        print(f"  Ablating heads improves PI ({normal_acc:.0%} → {ablated_acc:.0%}),")
        print(f"  BUT query patching is still needed (late recovery {normal_late:+.0%} → {ablated_late:+.0%}).")
        print(f"  → The heads cause PI failure through a DIRECT pathway (answer position),")
        print(f"     not through query-position corruption.")
        print(f"  → Query corruption comes from a different source.")
    elif ablated_acc <= normal_acc + 0.15:
        print(f"\n  ABLATION EFFECT IS WEAK in this sample:")
        print(f"  Normal PI: {normal_acc:.0%}, Ablated PI: {ablated_acc:.0%}")
        print(f"  Need more trials or different operating point for clear signal.")
    else:
        print(f"\n  MIXED RESULT — see numbers above.")

    # ── Save ──
    save_data = {
        "model": args.model,
        "config": {"keys": args.keys, "updates": args.updates, "trials": args.trials},
        "primacy_heads": [{"layer": l, "head": h} for l, h in primacy_heads],
        "accuracy": {m: accuracy[m]["correct"] / accuracy[m]["total"] for m in modes},
        "recovery_by_layer": {
            m: {str(l): aggregate_recovery(recovery[m][l])
                for l in test_layers}
            for m in modes
        },
        "summary": {
            "normal_late_mean": float(normal_late),
            "ablated_late_mean": float(ablated_late),
        },
    }
    from core.output import save_results
    out_path = save_results(save_data, args.model, args.keys, args.updates, "ablation_patching_interaction")
    print(f"({total_time:.0f}s)")


if __name__ == "__main__":
    main()
