"""
Exp 21b: DLA split by outcome.

On PI failures vs successes, is primacy DLA larger?
If so → override is stochastic (depends on specific token positions).

Pure analysis — reads from existing exp 12 results.

Usage:
    uv run python experiments/21b_dla_split_by_outcome.py
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_point(results_path):
    with open(results_path) as f:
        data = json.load(f)

    n_layers = data["n_layers"]
    n_heads = data["n_heads"]

    for cond in ["RI", "PI"]:
        trials = [a for a in data["analyses"] if a["condition"] == cond]
        correct = [a for a in trials if a["correct"]]
        wrong = [a for a in trials if not a["correct"]]

        if not correct or not wrong:
            print(f"  {cond}: {len(correct)} correct, {len(wrong)} wrong — skipping (need both)")
            continue

        # DLA: head_logit_diff is [layers, heads], positive = promotes initial
        correct_dla = np.array([a["dla"]["head_logit_diff"] for a in correct])  # [n_correct, L, H]
        wrong_dla = np.array([a["dla"]["head_logit_diff"] for a in wrong])

        correct_mean = correct_dla.mean(axis=0)  # [L, H]
        wrong_mean = wrong_dla.mean(axis=0)

        # For RI: correct answer IS initial, so positive DLA = helpful
        # For PI: correct answer IS final, so negative DLA = helpful
        # "primacy DLA" = sum of positive contributions (pushing toward initial)
        correct_primacy = np.sum(np.maximum(correct_mean, 0))
        wrong_primacy = np.sum(np.maximum(wrong_mean, 0))
        correct_recency = np.sum(np.minimum(correct_mean, 0))
        wrong_recency = np.sum(np.minimum(wrong_mean, 0))

        print(f"\n  {cond}: {len(correct)} correct, {len(wrong)} wrong")
        print(f"    Primacy DLA (sum of + heads):  correct={correct_primacy:+.2f}  wrong={wrong_primacy:+.2f}  diff={wrong_primacy-correct_primacy:+.2f}")
        print(f"    Recency DLA (sum of - heads):  correct={correct_recency:+.2f}  wrong={wrong_recency:+.2f}  diff={wrong_recency-correct_recency:+.2f}")

        if cond == "PI":
            # For PI, "wrong" means model outputs initial value or garbage
            # Higher primacy DLA on wrong trials = primacy override
            if wrong_primacy > correct_primacy + 0.5:
                print(f"    → PI failures have STRONGER primacy DLA ({wrong_primacy:+.2f} vs {correct_primacy:+.2f})")
                print(f"    → Override is stochastic: depends on how strongly primacy heads fire per trial")
            else:
                print(f"    → Primacy DLA similar on correct/wrong trials")
                print(f"    → Failure isn't from stronger primacy — it's from weaker recency signal")

        # Top 5 heads with biggest difference between correct and wrong
        diff = wrong_mean - correct_mean  # [L, H]
        flat_diff = diff.flatten()
        top5 = np.argsort(np.abs(flat_diff))[::-1][:5]
        print(f"    Top 5 heads differing between correct/wrong:")
        for idx in top5:
            l, h = divmod(idx, n_heads)
            print(f"      L{l:>2}H{h:>2}: correct={correct_mean[l,h]:+.3f} wrong={wrong_mean[l,h]:+.3f} diff={flat_diff[idx]:+.3f}")

        # MLP comparison
        correct_mlp = np.array([a["dla"]["mlp_logit_diff"] for a in correct]).mean(axis=0)
        wrong_mlp = np.array([a["dla"]["mlp_logit_diff"] for a in wrong]).mean(axis=0)
        print(f"    MLP total: correct={correct_mlp.sum():+.2f} wrong={wrong_mlp.sum():+.2f}")

    return data


def main():
    base = Path("results/Qwen2.5-0.5B-Instruct")
    points = [
        ("A (1k,3u)", "1k_3u"),
        ("B (1k,5u)", "1k_5u"),
        ("C (1k,15u)", "1k_15u"),
        ("D (2k,5u)", "2k_5u"),
    ]

    print("=" * 70)
    print("EXP 21b: DLA SPLIT BY OUTCOME")
    print("=" * 70)

    for name, path in points:
        results_path = base / path / "logit_lens.json"
        if not results_path.exists():
            print(f"\n{name}: no data")
            continue
        print(f"\n{'='*50}")
        print(f"{name}")
        print(f"{'='*50}")
        analyze_point(results_path)


if __name__ == "__main__":
    main()
