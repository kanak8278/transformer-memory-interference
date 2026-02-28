"""
Phase 1.5d: Sweep with neutral category names + single-token values.

Tests whether replacing semantic categories ("culinary herb") with
neutral keys ("key_A", "key_B") fixes the accuracy collapse.

Usage:
    cd mechanistic_probing_v2
    uv run python experiments/11d_neutral_keys_sweep.py
"""

import sys
import time
import random
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_loader import load_model
from core.dataset_configs import format_for_chat
from core.model_loader import verify_single_token
from core.output import save_results


NEUTRAL_CATEGORIES = [f"key_{chr(65+i)}" for i in range(46)]  # key_A .. key_AT


def build_trial(num_keys, num_updates, condition, seed, value_pool):
    rng = random.Random(seed)

    categories = NEUTRAL_CATEGORIES[:num_keys]

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

    return {"prompt": prompt, "expected": expected, "condition": condition}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--n-ctx", type=int, default=2048)
    args = parser.parse_args()

    keys_grid = [1, 2, 3, 5]
    updates_grid = [1, 2, 3, 5, 10, 15, 20]

    total_cells = len(keys_grid) * len(updates_grid)
    print("=" * 70)
    print("NEUTRAL KEYS SWEEP (key_A/key_B + single-token values)")
    print(f"  {len(keys_grid)} key levels x {len(updates_grid)} update levels = {total_cells} cells")
    print(f"  {args.trials} trials per condition per cell")
    print("=" * 70)

    model, tokenizer, info = load_model(args.model, n_ctx=args.n_ctx)
    value_to_tid = verify_single_token(tokenizer)
    value_pool = list(value_to_tid.keys())
    print(f"Value pool: {len(value_pool)} single-token words")

    # Show example prompt
    trial = build_trial(2, 5, "PI", seed=42, value_pool=value_pool)
    formatted = format_for_chat(trial["prompt"], tokenizer)
    print(f"\n--- Example prompt (2k, 5u, PI) ---")
    print(trial["prompt"])
    print(f"--- Expected: {trial['expected']} ---\n")

    results = {
        "model": args.model,
        "category_style": "neutral",
        "trials_per_condition": args.trials,
        "grid": {},
    }

    t_start = time.time()

    for keys in keys_grid:
        for updates in updates_grid:
            total_needed = keys * updates
            if total_needed > len(value_pool):
                continue

            cell_key = f"{keys}k_{updates}u"
            cell = {"keys": keys, "updates": updates}

            for condition in ["RI", "PI"]:
                correct = 0
                for t_idx in range(args.trials):
                    seed = hash((condition, t_idx, updates, keys, "neutral")) % (2**31)
                    trial = build_trial(keys, updates, condition, seed, value_pool)

                    formatted = format_for_chat(trial["prompt"], tokenizer)
                    tokens = model.to_tokens(formatted)

                    with torch.no_grad():
                        logits = model(tokens)

                    pred_tid = logits[0, -1].argmax().item()
                    pred_text = tokenizer.decode([pred_tid]).strip()

                    if pred_text.lower() == trial["expected"].lower():
                        correct += 1

                acc = correct / args.trials
                cell[condition] = {"accuracy": acc, "correct": correct, "total": args.trials}

            ri_acc = cell["RI"]["accuracy"]
            pi_acc = cell["PI"]["accuracy"]
            gap = ri_acc - pi_acc

            print(f"  keys={keys:>2}, updates={updates:>3}: "
                  f"RI={ri_acc:.0%}  PI={pi_acc:.0%}  gap={gap:+.0%}")

            results["grid"][cell_key] = cell

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    total = time.time() - t_start
    results["total_elapsed_sec"] = round(total, 1)

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"\n{'':>8}", end="")
    for u in updates_grid:
        print(f"  upd={u:>2}", end="")
    print()

    for keys in keys_grid:
        print(f"k={keys:>2} RI", end="")
        for updates in updates_grid:
            ck = f"{keys}k_{updates}u"
            if ck in results["grid"]:
                print(f"  {results['grid'][ck]['RI']['accuracy']:>5.0%}", end="")
            else:
                print(f"  {'---':>5}", end="")
        print()
        print(f"     PI", end="")
        for updates in updates_grid:
            ck = f"{keys}k_{updates}u"
            if ck in results["grid"]:
                print(f"  {results['grid'][ck]['PI']['accuracy']:>5.0%}", end="")
            else:
                print(f"  {'---':>5}", end="")
        print()
        print(f"    gap", end="")
        for updates in updates_grid:
            ck = f"{keys}k_{updates}u"
            if ck in results["grid"]:
                g = results["grid"][ck]["RI"]["accuracy"] - results["grid"][ck]["PI"]["accuracy"]
                print(f"  {g:>+4.0%}", end="")
            else:
                print(f"  {'---':>5}", end="")
        print("\n")

    print(f"  ({total:.0f}s total)")

    save_results(results, args.model, 0, 0, "neutral_keys_sweep")


if __name__ == "__main__":
    main()
