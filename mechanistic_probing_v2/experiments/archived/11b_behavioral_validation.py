"""
Phase 1.5: Behavioral validation with mechanistic data setup.

Confirms PI > RI holds when using single-token values + 46 categories
(the data setup used for all Phase 2 mechanistic experiments).

Runs at the 5 operating points selected from the behavioral sweep.
30 trials per condition per point — just checking regime classification matches.

Usage:
    cd mechanistic_probing_v2
    uv run python experiments/11b_behavioral_validation.py
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
from core.output import save_results


OPERATING_POINTS = [
    {"name": "A", "keys": 3,  "updates": 1,  "sweep_ri": 1.00, "sweep_pi": 0.90},
    {"name": "B", "keys": 2,  "updates": 5,  "sweep_ri": 0.90, "sweep_pi": 0.40},
    {"name": "C", "keys": 2,  "updates": 30, "sweep_ri": 0.83, "sweep_pi": 0.03},
    {"name": "D", "keys": 10, "updates": 80, "sweep_ri": 0.47, "sweep_pi": 0.03},
    {"name": "E", "keys": 25, "updates": 3,  "sweep_ri": 0.80, "sweep_pi": 0.03},
]


def build_trial(num_keys, num_updates, condition, seed, value_pool):
    """Build a trial using single-token values + 46 categories."""
    rng = random.Random(seed)

    # Randomly sample categories (matching behavioral sweep approach)
    categories = rng.sample(ORIGINAL_CATEGORIES, min(num_keys, len(ORIGINAL_CATEGORIES)))

    total_needed = num_keys * num_updates
    if total_needed > len(value_pool):
        raise ValueError(f"Need {total_needed} values, have {len(value_pool)}")

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

    # Shuffle with no consecutive same-category
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

    print("=" * 70)
    print("PHASE 1.5: BEHAVIORAL VALIDATION (single-token values + 46 categories)")
    print(f"  {args.trials} trials per condition per operating point")
    print("=" * 70)

    model, tokenizer, info = load_model(args.model, n_ctx=args.n_ctx)
    value_to_tid = verify_single_token(tokenizer)
    value_pool = list(value_to_tid.keys())
    print(f"Value pool: {len(value_pool)} single-token words")

    all_results = {
        "model": args.model,
        "trials_per_condition": args.trials,
        "points": {},
    }

    t_start = time.time()

    for point in OPERATING_POINTS:
        name = point["name"]
        keys = point["keys"]
        updates = point["updates"]

        print(f"\n{'='*50}")
        print(f"Point {name}: keys={keys}, updates={updates}")
        print(f"  Sweep: RI={point['sweep_ri']:.0%}, PI={point['sweep_pi']:.0%}")
        print(f"{'='*50}")

        point_results = {"keys": keys, "updates": updates}

        for condition in ["RI", "PI"]:
            correct = 0
            for t_idx in range(args.trials):
                seed = hash((condition, t_idx, updates, "validation")) % (2**31)
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
            point_results[condition] = {"accuracy": acc, "correct": correct, "total": args.trials}
            print(f"  {condition}: {correct}/{args.trials} = {acc:.0%}")

        # Classify regime
        ri_acc = point_results["RI"]["accuracy"]
        pi_acc = point_results["PI"]["accuracy"]
        if ri_acc >= 0.5 and pi_acc >= 0.5:
            regime = "A"
        elif ri_acc >= 0.5 and pi_acc < 0.5:
            regime = "B"
        elif ri_acc < 0.5 and pi_acc < 0.5:
            regime = "C"
        else:
            regime = "D"

        sweep_regime = "A" if point["sweep_ri"] >= 0.5 and point["sweep_pi"] >= 0.5 else \
                       "B" if point["sweep_ri"] >= 0.5 and point["sweep_pi"] < 0.5 else \
                       "C" if point["sweep_ri"] < 0.5 and point["sweep_pi"] < 0.5 else "D"

        match = "MATCH" if regime == sweep_regime else "MISMATCH"
        point_results["regime"] = regime
        point_results["sweep_regime"] = sweep_regime
        point_results["regime_match"] = regime == sweep_regime

        print(f"  Regime: {regime} (sweep: {sweep_regime}) → {match}")
        print(f"  Gap: RI-PI = {ri_acc - pi_acc:+.0%} (sweep: {point['sweep_ri'] - point['sweep_pi']:+.0%})")

        all_results["points"][name] = point_results

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    total = time.time() - t_start
    all_results["total_elapsed_sec"] = round(total, 1)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Point':>6} {'Keys':>5} {'Upd':>5} | {'Sweep RI':>9} {'Sweep PI':>9} | {'Valid RI':>9} {'Valid PI':>9} | {'Regime':>7} {'Match':>7}")
    print("-" * 80)

    all_match = True
    for point in OPERATING_POINTS:
        name = point["name"]
        r = all_results["points"][name]
        match_str = "OK" if r["regime_match"] else "FAIL"
        if not r["regime_match"]:
            all_match = False
        print(f"  {name:>4} {point['keys']:>5} {point['updates']:>5} | "
              f"{point['sweep_ri']:>8.0%} {point['sweep_pi']:>8.0%} | "
              f"{r['RI']['accuracy']:>8.0%} {r['PI']['accuracy']:>8.0%} | "
              f"{r['regime']:>3}/{r['sweep_regime']:<3} {match_str:>7}")

    print(f"\n  All regimes match: {'YES' if all_match else 'NO'}")
    if not all_match:
        print("  WARNING: Operating points need adjustment for single-token data setup!")
    print(f"  ({total:.0f}s total)")

    save_results(all_results, args.model, 0, 0, "behavioral_validation")


if __name__ == "__main__":
    main()
