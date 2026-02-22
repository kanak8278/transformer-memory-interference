"""
Exp 24b: Force attention on ONLY the true primacy-biased head.

Tests whether the 0.5B vs 1.5B divergence in exp 18 was due to
misclassified heads being forced alongside the real primacy head.

True primacy head criteria:
  1. Retrieval score > 0.3
  2. PI primacy > 0.7
  3. PI DLA > 0

Results:
  0.5B true primacy head: L16H3
  1.5B true primacy head: L19H1
"""

import sys
import random
import torch
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_loader import load_model
from core.dataset import format_for_chat, ORIGINAL_CATEGORIES
from core.single_token_values import verify_single_token
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
        ok = all(items[i]["category"] != items[i-1]["category"] for i in range(1, len(items)))
        if ok:
            break
        rng.shuffle(items)

    stream_lines = [f"{it['category']}: {it['value']}" for it in items]
    stream_text = "\n".join(stream_lines)
    query_word = "first" if condition == "RI" else "last"
    cat_values = [it["value"] for it in items if it["category"] == test_cat]
    expected = cat_values[0] if condition == "RI" else cat_values[-1]
    prompt = (
        f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{stream_text}\n\n"
        f"What was the {query_word} value of {test_cat}?"
    )
    return {
        "prompt": prompt, "expected": expected, "condition": condition,
        "initial_value": cat_values[0], "final_value": cat_values[-1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--keys", type=int, default=1)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--head-layer", type=int, required=True)
    parser.add_argument("--head-idx", type=int, required=True)
    parser.add_argument("--n-ctx", type=int, default=2048)
    args = parser.parse_args()

    model, tokenizer, info = load_model(args.model, n_ctx=args.n_ctx)
    value_to_tid = verify_single_token(tokenizer)
    value_pool = list(value_to_tid.keys())

    layer = args.head_layer
    head = args.head_idx
    model_short = args.model.split("/")[-1]

    print("=" * 60)
    print(f"{model_short} — Force ONLY L{layer}H{head}")
    print(f"keys={args.keys}, updates={args.updates}, {args.trials} trials")
    print("=" * 60)

    configs = ["baseline", "knockout_single", "force_to_final", "force_to_init"]
    results = {cfg: {"RI": 0, "PI": 0, "RI_n": 0, "PI_n": 0} for cfg in configs}

    for t_idx in range(args.trials):
        for condition in ["RI", "PI"]:
            seed = hash((condition, t_idx, args.updates, args.keys, "single_force")) % (2**31)
            trial = build_trial(args.keys, args.updates, condition, seed, value_pool)

            init_tid = value_to_tid.get(trial["initial_value"], -1)
            final_tid = value_to_tid.get(trial["final_value"], -1)

            formatted = format_for_chat(trial["prompt"], tokenizer)
            tokens = model.to_tokens(formatted)
            token_list = tokens[0].tolist()

            init_positions = [i for i, t in enumerate(token_list) if t == init_tid]
            final_positions = [i for i, t in enumerate(token_list) if t == final_tid]
            if not init_positions or not final_positions:
                continue

            init_pos = init_positions[0]
            final_pos = final_positions[-1]

            for cfg in configs:
                if cfg == "baseline":
                    with torch.no_grad():
                        logits = model(tokens)

                elif cfg == "knockout_single":
                    def make_ko(h):
                        def hook_fn(pattern, hook):
                            pattern[0, h, -1, :] = 0
                            return pattern
                        return hook_fn
                    with torch.no_grad():
                        logits = model.run_with_hooks(tokens, fwd_hooks=[
                            (f"blocks.{layer}.attn.hook_pattern", make_ko(head))
                        ])

                elif cfg == "force_to_final":
                    def make_force_final(h, pos):
                        def hook_fn(pattern, hook):
                            pattern[0, h, -1, :] = 0
                            pattern[0, h, -1, pos] = 1.0
                            return pattern
                        return hook_fn
                    with torch.no_grad():
                        logits = model.run_with_hooks(tokens, fwd_hooks=[
                            (f"blocks.{layer}.attn.hook_pattern", make_force_final(head, final_pos))
                        ])

                elif cfg == "force_to_init":
                    def make_force_init(h, pos):
                        def hook_fn(pattern, hook):
                            pattern[0, h, -1, :] = 0
                            pattern[0, h, -1, pos] = 1.0
                            return pattern
                        return hook_fn
                    with torch.no_grad():
                        logits = model.run_with_hooks(tokens, fwd_hooks=[
                            (f"blocks.{layer}.attn.hook_pattern", make_force_init(head, init_pos))
                        ])

                pred_tid = logits[0, -1].argmax().item()
                pred = tokenizer.decode([pred_tid]).strip()
                correct = pred.lower() == trial["expected"].lower()
                results[cfg][condition] += int(correct)
                results[cfg][f"{condition}_n"] += 1

        if (t_idx + 1) % 25 == 0:
            print(f"  {t_idx + 1}/{args.trials} done")

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Print results
    print(f"\n  Config                    RI      PI      Gap     PI delta")
    print(f"  " + "-" * 60)
    baseline_pi = results["baseline"]["PI"] / max(results["baseline"]["PI_n"], 1)
    for cfg in configs:
        ri = results[cfg]["RI"] / max(results[cfg]["RI_n"], 1)
        pi = results[cfg]["PI"] / max(results[cfg]["PI_n"], 1)
        gap = ri - pi
        delta = f"  {pi - baseline_pi:+.0%}" if cfg != "baseline" else ""
        print(f"  {cfg:<25} {ri:>5.0%}   {pi:>5.0%}   {gap:>+5.0%}{delta}")

    # Save
    save_data = {
        "model": args.model,
        "config": {"keys": args.keys, "updates": args.updates, "trials": args.trials},
        "true_primacy_head": {"layer": layer, "head": head},
        "results": {
            cfg: {
                "RI": results[cfg]["RI"] / max(results[cfg]["RI_n"], 1),
                "PI": results[cfg]["PI"] / max(results[cfg]["PI_n"], 1),
                "RI_n": results[cfg]["RI_n"],
                "PI_n": results[cfg]["PI_n"],
            }
            for cfg in configs
        },
    }
    save_results(save_data, args.model, args.keys, args.updates, "single_head_force")


if __name__ == "__main__":
    main()
