"""
Remedy Experiment: Can prompt-level interventions reduce PI > RI?

Tests three interventions that target the three forces causing PI > RI:
1. NUMBERED UPDATES — explicit "Update 1:", "Update 2:" markers
   (targets positional confusability: makes v_{N-1} distinguishable from v_{N-2})
2. LANDMARK SEPARATORS — "---" between updates
   (targets capacity saturation: prevents representational over-mixing)
3. RECENCY CUE — "The MOST RECENT and FINAL value is the one listed LAST"
   (targets cumulative reinforcement: explicitly tells model to prioritize recency)

Control: standard KV stream (same as Stage 1 experiments)

Runs on Claude Haiku (API) to avoid GPU usage.

Usage:
    cd v3 && python remedy_experiment.py --trials 50
    cd v3 && python remedy_experiment.py --trials 100 --model claude-haiku
"""

import sys
import json
import time
import random
import argparse
import os
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env for API keys
load_dotenv(_PROJECT_ROOT / ".env")

from models.model_factory import create_model
from mechanistic_probing_v2.core.dataset_configs import (
    get_eligible_categories,
    generate_values_for_trial,
)

DATASET_TYPE = "ARBITRARY_MULTI"

SYSTEM_PROMPT = (
    "You are a precise data extraction tool. "
    "Output ONLY a single word or value - the exact answer requested. "
    "No other text, no explanation, no punctuation."
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="claude-haiku")
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--num-keys", type=int, default=5)
    p.add_argument("--update-levels", type=int, nargs="+", default=[20, 50, 100])
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def build_kv_stream(categories, values_per_cat, num_updates, style="control"):
    """Build interleaved KV stream with different formatting styles."""
    items = []
    for round_idx in range(num_updates):
        for cat in categories:
            items.append({"category": cat, "value": values_per_cat[cat][round_idx]})

    if style == "control":
        # Standard: "color: red\ncolor: blue\n..."
        return "\n".join(f"{it['category']}: {it['value']}" for it in items)

    elif style == "numbered":
        # Numbered updates: "Update 1 - color: red\nUpdate 2 - color: blue\n..."
        lines = []
        for idx, it in enumerate(items):
            lines.append(f"Update {idx + 1} — {it['category']}: {it['value']}")
        return "\n".join(lines)

    elif style == "landmark":
        # Landmark separators between rounds
        lines = []
        items_per_round = len(categories)
        for idx, it in enumerate(items):
            lines.append(f"{it['category']}: {it['value']}")
            if (idx + 1) % items_per_round == 0 and idx + 1 < len(items):
                lines.append("---")
        return "\n".join(lines)

    elif style == "recency_cue":
        # Standard stream + explicit recency instruction
        stream = "\n".join(f"{it['category']}: {it['value']}" for it in items)
        cue = ("\n\nIMPORTANT: Each category is updated multiple times above. "
               "The MOST RECENT value (listed LAST) is the current value. "
               "The FIRST value listed is the oldest.")
        return stream + cue

    elif style == "combined":
        # Landmark + short round numbers (minimal overhead)
        lines = []
        items_per_round = len(categories)
        round_num = 0
        for idx, it in enumerate(items):
            lines.append(f"{it['category']}: {it['value']}")
            if (idx + 1) % items_per_round == 0 and idx + 1 < len(items):
                round_num += 1
                lines.append(f"--- round {round_num + 1} ---")
        return "\n".join(lines)

    raise ValueError(f"Unknown style: {style}")


def run_single_trial(model, seed, num_keys, num_updates, style):
    """Run one trial with a given intervention style."""
    rng = random.Random(seed)
    eligible = get_eligible_categories(DATASET_TYPE, min_values=num_updates)
    categories = rng.sample(eligible, min(num_keys, len(eligible)))
    test_category = categories[seed % num_keys]
    values_per_cat = generate_values_for_trial(DATASET_TYPE, categories, num_updates, rng)

    stream = build_kv_stream(categories, values_per_cat, num_updates, style=style)

    cat_values = values_per_cat[test_category]
    first_value = cat_values[0]
    last_value = cat_values[-1]

    results = {}
    for condition, query_word, expected in [
        ("RI", "first", first_value),
        ("PI", "last", last_value),
    ]:
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Read the following data stream:\n\n"
            f"{stream}\n\n"
            f"What was the {query_word} value of {test_category}?"
        )

        try:
            response = model.generate(prompt).strip().lower()
            expected_lower = expected.lower()
            correct = (response == expected_lower or
                      expected_lower in response.split() or
                      response.startswith(expected_lower))
        except Exception as e:
            response = f"ERROR: {e}"
            correct = False

        results[condition] = {
            "correct": correct,
            "expected": expected,
            "predicted": response[:80],
        }

    return seed, results


def main():
    args = parse_args()

    print(f"Creating model: {args.model}")
    model = create_model(args.model, {"max_tokens": 50, "temperature": 0.0})
    if not model.is_available():
        print(f"ERROR: {args.model} not available")
        return

    styles = ["control", "numbered", "landmark", "recency_cue", "combined"]

    all_results = {}

    for num_updates in args.update_levels:
        print(f"\n{'='*60}")
        print(f"N={num_updates} updates, {args.num_keys} keys")
        print(f"{'='*60}")

        for style in styles:
            print(f"\n  Style: {style}")

            trial_results = {"RI": [], "PI": []}

            # Use ThreadPoolExecutor for parallel API calls
            seeds = [hash((args.num_keys, num_updates, i, style)) % (2**31)
                    for i in range(args.trials)]

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(run_single_trial, model, seed,
                                   args.num_keys, num_updates, style): seed
                    for seed in seeds
                }

                done = 0
                for future in as_completed(futures):
                    seed, results = future.result()
                    for cond in ["RI", "PI"]:
                        trial_results[cond].append(results[cond])
                    done += 1

                    if done % 10 == 0:
                        ri_acc = sum(r["correct"] for r in trial_results["RI"]) / len(trial_results["RI"])
                        pi_acc = sum(r["correct"] for r in trial_results["PI"]) / len(trial_results["PI"])
                        print(f"    [{done}/{args.trials}] RI={ri_acc:.0%} PI={pi_acc:.0%} gap={ri_acc-pi_acc:+.0%}")

            ri_acc = sum(r["correct"] for r in trial_results["RI"]) / len(trial_results["RI"])
            pi_acc = sum(r["correct"] for r in trial_results["PI"]) / len(trial_results["PI"])
            gap = ri_acc - pi_acc

            cell_key = f"{args.num_keys}k_{num_updates}u"
            if cell_key not in all_results:
                all_results[cell_key] = {}
            all_results[cell_key][style] = {
                "ri_accuracy": float(ri_acc),
                "pi_accuracy": float(pi_acc),
                "gap": float(gap),
                "n_trials": args.trials,
                "trials": trial_results,
            }

            print(f"    RESULT: RI={ri_acc:.0%} PI={pi_acc:.0%} gap={gap:+.0%}")

    # Summary table
    print(f"\n\n{'='*70}")
    print(f"REMEDY EXPERIMENT SUMMARY — {args.model}")
    print(f"{'='*70}")
    print(f"\n{'Cell':<10} {'Style':<15} {'RI':>5} {'PI':>5} {'Gap':>6} {'PI improvement':>15}")
    print("-" * 60)

    for cell_key in sorted(all_results.keys()):
        control_pi = all_results[cell_key].get("control", {}).get("pi_accuracy", 0)
        for style in styles:
            if style not in all_results[cell_key]:
                continue
            r = all_results[cell_key][style]
            improvement = r["pi_accuracy"] - control_pi if style != "control" else 0
            marker = "" if style == "control" else f"{improvement:+.0%}"
            print(f"{cell_key:<10} {style:<15} {r['ri_accuracy']:>4.0%} {r['pi_accuracy']:>4.0%} {r['gap']:>+5.0%} {marker:>15}")
        print()

    # Save
    save_dir = _SCRIPT_DIR / "results" / "remedy"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"remedy_{args.model}.json"
    output = {
        "model": args.model,
        "config": {
            "num_keys": args.num_keys,
            "update_levels": args.update_levels,
            "trials": args.trials,
            "styles": styles,
        },
        "results": all_results,
    }
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
