"""
Exp 21c: Failure output classification.

On PI failures, what does the model actually output?
- Primacy intrusion (outputs the initial value)
- Intermediate value (outputs some middle value)
- Garbage (outputs something not in the value list)

Pure analysis — reads from existing exp 12 results.

Usage:
    uv run python experiments/21c_failure_output_classification.py
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.model_loader import verify_single_token


def classify_failure(trial, value_pool_set):
    """Classify what the model output on a wrong trial."""
    predicted = trial["predicted"].strip().lower()
    initial = trial["initial_value"].lower()
    final = trial["final_value"].lower()
    intermediates = [v.lower() for v in trial.get("intermediate_values", [])]

    if predicted == initial:
        return "primacy_intrusion"
    elif predicted == final:
        return "correct_but_case"  # shouldn't happen if trial is wrong
    elif predicted in intermediates:
        return "intermediate_intrusion"
    elif predicted in value_pool_set:
        return "wrong_value_from_pool"
    else:
        # Check if it's a prefix/partial token
        if any(initial.startswith(predicted) or predicted.startswith(initial[:2]) for _ in [1]):
            if len(predicted) < len(initial) and initial.startswith(predicted):
                return "initial_prefix"
        if any(final.startswith(predicted) or predicted.startswith(final[:2]) for _ in [1]):
            if len(predicted) < len(final) and final.startswith(predicted):
                return "final_prefix"
        return "garbage"


def analyze_point(results_path, value_pool_set):
    with open(results_path) as f:
        data = json.load(f)

    for cond in ["RI", "PI"]:
        trials = [a for a in data["analyses"] if a["condition"] == cond]
        correct = [a for a in trials if a["correct"]]
        wrong = [a for a in trials if not a["correct"]]

        print(f"\n  {cond}: {len(correct)}/{len(trials)} correct ({len(correct)/len(trials):.0%})")

        if not wrong:
            print(f"    No failures to classify")
            continue

        # Classify each failure
        categories = {}
        examples = {}
        for t in wrong:
            cat = classify_failure(t, value_pool_set)
            categories[cat] = categories.get(cat, 0) + 1
            if cat not in examples:
                examples[cat] = f"pred='{t['predicted']}' init='{t['initial_value']}' final='{t['final_value']}'"

        print(f"    Failure breakdown ({len(wrong)} failures):")
        for cat in sorted(categories, key=lambda c: -categories[c]):
            pct = categories[cat] / len(wrong) * 100
            print(f"      {cat:<25} {categories[cat]:>3} ({pct:>5.1f}%)  e.g. {examples[cat]}")

        # For PI: primacy intrusion rate is the key metric
        if cond == "PI":
            pi_intrusion = categories.get("primacy_intrusion", 0)
            init_prefix = categories.get("initial_prefix", 0)
            total_primacy = pi_intrusion + init_prefix
            print(f"\n    PRIMACY INTRUSION RATE: {total_primacy}/{len(wrong)} = {total_primacy/len(wrong):.0%} of PI failures")
            if total_primacy / len(wrong) > 0.3:
                print(f"    → Model actively retrieves INITIAL value when asked for LAST")
                print(f"    → This is direct evidence of primacy bias in the retrieval circuit")
            else:
                print(f"    → Most failures are NOT primacy intrusions")
                print(f"    → Model fails to retrieve anything useful, not actively biased")


def main():
    base = Path("results/Qwen2.5-0.5B-Instruct")
    points = [
        ("A (1k,3u)", "1k_3u"),
        ("B (1k,5u)", "1k_5u"),
        ("C (1k,15u)", "1k_15u"),
        ("D (2k,5u)", "2k_5u"),
    ]

    # Load value pool for classification
    from core.model_loader import load_model
    from core.model_loader import get_single_token_pool
    pool = get_single_token_pool()
    value_pool_set = set(w.lower() for w in pool)

    print("=" * 70)
    print("EXP 21c: FAILURE OUTPUT CLASSIFICATION")
    print("=" * 70)

    for name, path in points:
        results_path = base / path / "logit_lens.json"
        if not results_path.exists():
            print(f"\n{name}: no data")
            continue
        print(f"\n{'='*50}")
        print(f"{name}")
        print(f"{'='*50}")
        analyze_point(results_path, value_pool_set)


if __name__ == "__main__":
    main()
