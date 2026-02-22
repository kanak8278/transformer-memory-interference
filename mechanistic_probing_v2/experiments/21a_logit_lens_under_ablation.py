"""
Exp 21a: Logit lens under ablation.

Does ablating primacy heads change the P(init)/P(final) trajectory?
Runs logit lens in two modes:
  1. Normal — baseline P(init), P(final) per layer
  2. With primacy heads ablated — does P(initial) drop at L16+?

If ablating shifts the trajectory → causal evidence of information flow corruption.

Usage:
    cd mechanistic_probing_v2
    uv run python experiments/21a_logit_lens_under_ablation.py [--keys 1 --updates 5 --trials 100]
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
from core.output import save_results
def build_trial(num_keys, num_updates, condition, seed, value_pool, categories):
    """Build a single trial."""
    rng = random.Random(seed)
    cats = categories[:num_keys]
    total_needed = num_keys * num_updates
    selected = rng.sample(value_pool, total_needed)
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

    stream_lines = [f"{it['category']}: {it['value']}" for it in items]
    stream_text = "\n".join(stream_lines)
    query_word = "first" if condition == "RI" else "last"
    cat_values = [it["value"] for it in items if it["category"] == test_cat]
    expected = cat_values[0] if condition == "RI" else cat_values[-1]
    initial_value = cat_values[0]
    final_value = cat_values[-1]

    prompt = (
        f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{stream_text}\n\n"
        f"What was the {query_word} value of {test_cat}?"
    )

    return {
        "prompt": prompt, "expected": expected, "condition": condition,
        "initial_value": initial_value, "final_value": final_value,
    }


def get_primacy_heads(model_name, keys, updates):
    """Load primacy heads from exp 16 results."""
    from core.output import load_head_identification
    try:
        return load_head_identification(model_name, keys, updates)
    except FileNotFoundError:
        # Fallback: use the heads we consistently see
        print("  WARNING: No head_identification results found, using default primacy heads")
        return [(16, 3), (16, 9), (14, 13)]


def make_ablation_hooks(primacy_heads):
    hooks = []
    for layer, head in primacy_heads:
        def make_hook(h):
            def hook_fn(z, hook):
                z[:, :, h, :] = 0.0
                return z
            return hook_fn
        hooks.append((f"blocks.{layer}.attn.hook_z", make_hook(head)))
    return hooks


def logit_lens_from_cache(model, cache, init_tids, final_tids):
    """Extract P(init), P(final) per layer from cache."""
    n_layers = model.cfg.n_layers
    results = []
    for layer in range(n_layers):
        resid = cache[("resid_post", layer)][0, -1, :]  # [d_model]
        logits = resid @ model.W_U + model.b_U  # [vocab]
        probs = torch.softmax(logits, dim=-1)
        p_init = max(probs[t].item() for t in init_tids)
        p_final = max(probs[t].item() for t in final_tids)
        results.append({"layer": layer, "prob_initial": p_init, "prob_final": p_final})
    return results


def run_with_hooks_and_cache(model, tokens, fwd_hooks):
    """Run model with hooks AND get cache."""
    cache_dict = {}
    def make_cache_hook(layer):
        def hook_fn(activation, hook):
            cache_dict[("resid_post", layer)] = activation.detach()
            return activation
        return hook_fn

    all_hooks = list(fwd_hooks)
    for layer in range(model.cfg.n_layers):
        all_hooks.append((f"blocks.{layer}.hook_resid_post", make_cache_hook(layer)))

    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=all_hooks)
    return logits, cache_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--keys", type=int, default=1)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--n-ctx", type=int, default=2048)
    args = parser.parse_args()

    print("=" * 70)
    print(f"EXP 21a: LOGIT LENS UNDER ABLATION")
    print(f"  keys={args.keys}, updates={args.updates}, trials={args.trials}")
    print("=" * 70)

    model, tokenizer, info = load_model(args.model, n_ctx=args.n_ctx)
    value_to_tid = verify_single_token(tokenizer)
    value_pool = list(value_to_tid.keys())
    categories = ORIGINAL_CATEGORIES

    primacy_heads = get_primacy_heads(args.model, args.keys, args.updates)
    print(f"Primacy heads to ablate: {primacy_heads}")
    ablation_hooks = make_ablation_hooks(primacy_heads)

    modes = ["normal", "ablated"]
    # {mode: {cond: {layer: [p_init, ...], ...}}}
    layer_probs = {m: {c: {"p_init": [[] for _ in range(info.n_layers)],
                           "p_final": [[] for _ in range(info.n_layers)]}
                       for c in ["RI", "PI"]}
                   for m in modes}
    accuracy = {m: {c: {"correct": 0, "total": 0} for c in ["RI", "PI"]} for m in modes}

    t_start = time.time()

    for t_idx in range(args.trials):
        for condition in ["RI", "PI"]:
            seed = hash((condition, t_idx, args.updates, "21a")) % (2**31)
            trial = build_trial(args.keys, args.updates, condition, seed, value_pool, categories)

            init_tid = value_to_tid[trial["initial_value"]]
            final_tid = value_to_tid[trial["final_value"]]
            init_bare = tokenizer.encode(trial["initial_value"], add_special_tokens=False)[0]
            final_bare = tokenizer.encode(trial["final_value"], add_special_tokens=False)[0]
            init_tids = list(set([init_tid, init_bare]))
            final_tids = list(set([final_tid, final_bare]))

            formatted = format_for_chat(trial["prompt"], tokenizer)
            tokens = model.to_tokens(formatted)

            for mode in modes:
                hooks = ablation_hooks if mode == "ablated" else []

                if hooks:
                    logits, cache = run_with_hooks_and_cache(model, tokens, hooks)
                else:
                    with torch.no_grad():
                        logits, cache = model.run_with_cache(tokens)

                pred_tid = logits[0, -1].argmax().item()
                pred_text = tokenizer.decode([pred_tid]).strip()
                correct = pred_text.lower() == trial["expected"].lower()
                accuracy[mode][condition]["correct"] += correct
                accuracy[mode][condition]["total"] += 1

                ll = logit_lens_from_cache(model, cache, init_tids, final_tids)
                for r in ll:
                    layer_probs[mode][condition]["p_init"][r["layer"]].append(r["prob_initial"])
                    layer_probs[mode][condition]["p_final"][r["layer"]].append(r["prob_final"])

                del cache

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        if (t_idx + 1) % 20 == 0:
            print(f"  {t_idx + 1}/{args.trials} done ({time.time() - t_start:.0f}s)")

    total = time.time() - t_start

    # ── Results ──
    print(f"\n{'='*70}")
    print("ACCURACY")
    print(f"{'='*70}")
    for mode in modes:
        for cond in ["RI", "PI"]:
            a = accuracy[mode][cond]
            print(f"  {mode:<8} {cond}: {a['correct']}/{a['total']} = {a['correct']/a['total']:.0%}")

    print(f"\n{'='*70}")
    print("LOGIT LENS: P(init) and P(final) — last 4 layers")
    print(f"{'='*70}")
    print(f"  {'Layer':>6}  {'Normal RI':>10} {'Ablated RI':>11} | {'Normal PI':>10} {'Ablated PI':>11}")
    print(f"  {'-'*55}")

    for layer in range(info.n_layers - 4, info.n_layers):
        n_ri_pinit = np.mean(layer_probs["normal"]["RI"]["p_init"][layer])
        a_ri_pinit = np.mean(layer_probs["ablated"]["RI"]["p_init"][layer])
        n_pi_pfinal = np.mean(layer_probs["normal"]["PI"]["p_final"][layer])
        a_pi_pfinal = np.mean(layer_probs["ablated"]["PI"]["p_final"][layer])
        print(f"  L{layer:>3}  P(i)={n_ri_pinit:.3f}  P(i)={a_ri_pinit:.3f}   | P(f)={n_pi_pfinal:.3f}  P(f)={a_pi_pfinal:.3f}")

    # Full layer comparison for PI
    print(f"\n{'='*70}")
    print("PI CONDITION: Layer-by-layer P(initial) — normal vs ablated")
    print("(Does ablating primacy heads reduce P(initial) in PI?)")
    print(f"{'='*70}")
    print(f"  {'Layer':>6}  {'Normal P(init)':>14} {'Ablated P(init)':>15} {'Delta':>8}")
    print(f"  {'-'*48}")

    for layer in range(0, info.n_layers, 2):
        n_val = np.mean(layer_probs["normal"]["PI"]["p_init"][layer])
        a_val = np.mean(layer_probs["ablated"]["PI"]["p_init"][layer])
        delta = a_val - n_val
        marker = " ***" if abs(delta) > 0.01 else ""
        print(f"  L{layer:>3}  {n_val:>13.4f} {a_val:>14.4f} {delta:>+7.4f}{marker}")

    # Interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")

    # Check last-layer changes
    last = info.n_layers - 1
    n_ri_acc = accuracy["normal"]["RI"]["correct"] / accuracy["normal"]["RI"]["total"]
    a_ri_acc = accuracy["ablated"]["RI"]["correct"] / accuracy["ablated"]["RI"]["total"]
    n_pi_acc = accuracy["normal"]["PI"]["correct"] / accuracy["normal"]["PI"]["total"]
    a_pi_acc = accuracy["ablated"]["PI"]["correct"] / accuracy["ablated"]["PI"]["total"]

    n_pi_pinit_last = np.mean(layer_probs["normal"]["PI"]["p_init"][last])
    a_pi_pinit_last = np.mean(layer_probs["ablated"]["PI"]["p_init"][last])

    print(f"\n  PI accuracy: normal={n_pi_acc:.0%}, ablated={a_pi_acc:.0%} (delta={a_pi_acc-n_pi_acc:+.0%})")
    print(f"  PI P(initial) at last layer: normal={n_pi_pinit_last:.4f}, ablated={a_pi_pinit_last:.4f}")

    if a_pi_pinit_last < n_pi_pinit_last - 0.005:
        print(f"\n  → Ablating primacy heads REDUCES P(initial) in PI condition")
        print(f"  → These heads causally contribute to primacy signal in the residual stream")
    elif a_pi_acc > n_pi_acc + 0.05:
        print(f"\n  → Ablating improves PI accuracy but doesn't change P(initial) much")
        print(f"  → Heads affect output selection, not representation")
    else:
        print(f"\n  → Ablation has minimal effect on both accuracy and representation")
        print(f"  → Primacy heads are not the main bottleneck")

    # Save
    save_data = {
        "model": args.model,
        "config": {"keys": args.keys, "updates": args.updates, "trials": args.trials},
        "primacy_heads": [{"layer": l, "head": h} for l, h in primacy_heads],
        "accuracy": {m: {c: accuracy[m][c]["correct"] / accuracy[m][c]["total"]
                         for c in ["RI", "PI"]} for m in modes},
        "layer_probs": {m: {c: {"p_init_mean": [float(np.mean(layer_probs[m][c]["p_init"][l]))
                                                 for l in range(info.n_layers)],
                                 "p_final_mean": [float(np.mean(layer_probs[m][c]["p_final"][l]))
                                                  for l in range(info.n_layers)]}
                            for c in ["RI", "PI"]} for m in modes},
    }
    save_results(save_data, args.model, args.keys, args.updates, "logit_lens_under_ablation")
    print(f"\n({total:.0f}s total)")


if __name__ == "__main__":
    main()
