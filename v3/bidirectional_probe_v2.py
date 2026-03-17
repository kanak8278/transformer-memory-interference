"""
Bidirectional Model Probe V2: Does PI > RI exist in encoder models?

REDESIGN: Uses BERT's masked language model (MLM) to directly test retrieval.
Instead of probing representations, we ask BERT to PREDICT the correct value
via masked token prediction — its native task.

Approach:
  1. Construct KV stream: "color: red color: blue color: green"
  2. Append RI query: "The first color was [MASK]" → should predict "red"
  3. Append PI query: "The last color was [MASK]" → should predict "green"
  4. Compare: does BERT get first right more often than last?

If bidirectional models show PI ≈ RI, autoregressive processing is the key.
If they show PI > RI too, the asymmetry is deeper (training data, etc.).

Also tests a CONTROL condition:
  - Same-query control: "The [MASK] was red/green" → predict "first"/"last"
    This checks if BERT can discriminate positions at all.

Usage:
    cd v3 && python bidirectional_probe_v2.py --trials 200
    cd v3 && python bidirectional_probe_v2.py --model google-bert/bert-large-uncased --trials 200
"""

import sys
import json
import random
import argparse
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mechanistic_probing_v2.core.dataset_configs import (
    get_eligible_categories, generate_values_for_trial,
)

DATASET_TYPE = "ARBITRARY_SINGLE"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google-bert/bert-base-uncased")
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--num-keys", type=int, default=2)
    p.add_argument("--num-updates", type=int, default=5)
    p.add_argument("--device", default=None)
    return p.parse_args()


def generate_kv_stream(num_keys, num_updates, seed):
    """Generate a KV stream with N updates per key."""
    rng = random.Random(seed)
    eligible = get_eligible_categories(DATASET_TYPE, min_values=num_updates)
    categories = rng.sample(eligible, min(num_keys, len(eligible)))
    test_category = categories[seed % num_keys]
    values_per_cat = generate_values_for_trial(DATASET_TYPE, categories, num_updates, rng)

    # Build interleaved stream (round-robin so values for each key are spread out)
    items = []
    for round_idx in range(num_updates):
        for cat in categories:
            items.append({"category": cat, "value": values_per_cat[cat][round_idx]})

    stream = " ".join(f"{it['category']}: {it['value']}" for it in items)
    cat_values = values_per_cat[test_category]

    return {
        "stream": stream,
        "test_category": test_category,
        "first_value": cat_values[0],
        "last_value": cat_values[-1],
        "all_values": cat_values,
    }


def main():
    args = parse_args()

    from transformers import AutoTokenizer, AutoModelForMaskedLM

    print(f"Loading {args.model} (MLM head)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    model.eval()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {n_params:.0f}M params, device={device}")

    mask_token = tokenizer.mask_token
    mask_id = tokenizer.mask_token_id

    # ──────────────────────────────────────────────────────────────
    # Run trials: RI (first) and PI (last) via masked prediction
    # ──────────────────────────────────────────────────────────────
    results = {"RI": [], "PI": []}

    print(f"\nRunning {args.trials} trials (2 conditions each)...")

    for t_idx in range(args.trials):
        seed = hash((args.num_keys, args.num_updates, t_idx, "bidir_v2")) % (2**31)
        trial = generate_kv_stream(args.num_keys, args.num_updates, seed)

        for condition, query_word, expected in [
            ("RI", "first", trial["first_value"]),
            ("PI", "last", trial["last_value"]),
        ]:
            prompt = f"{trial['stream']} The {query_word} {trial['test_category']} was {mask_token}."

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=512).to(device)

            # Find mask position
            input_ids = inputs["input_ids"][0]
            mask_positions = (input_ids == mask_id).nonzero(as_tuple=True)[0]

            if len(mask_positions) == 0:
                results[condition].append({
                    "correct": False, "error": "no_mask", "expected": expected,
                })
                continue

            mask_pos = mask_positions[0].item()

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, mask_pos]  # vocab logits at mask position
                probs = torch.softmax(logits, dim=-1)

            # Get probability of correct token
            expected_ids = tokenizer.encode(expected, add_special_tokens=False)
            if not expected_ids:
                results[condition].append({
                    "correct": False, "error": "no_token", "expected": expected,
                })
                continue

            expected_id = expected_ids[0]
            p_correct = probs[expected_id].item()

            # Get top prediction
            top_id = logits.argmax().item()
            top_token = tokenizer.decode([top_id]).strip()
            p_top = probs[top_id].item()

            # Check if any value in the stream is predicted
            value_probs = {}
            for val in trial["all_values"]:
                val_ids = tokenizer.encode(val, add_special_tokens=False)
                if val_ids:
                    value_probs[val] = probs[val_ids[0]].item()

            # Check all values for other keys too
            predicted_correct = (top_token.lower() == expected.lower() or
                               top_id == expected_id)

            results[condition].append({
                "correct": predicted_correct,
                "expected": expected,
                "expected_prob": float(p_correct),
                "top_prediction": top_token,
                "top_prob": float(p_top),
                "value_probs": {k: float(v) for k, v in value_probs.items()},
            })

        if (t_idx + 1) % 50 == 0:
            ri_acc = np.mean([r["correct"] for r in results["RI"]])
            pi_acc = np.mean([r["correct"] for r in results["PI"]])
            print(f"  [{t_idx + 1}/{args.trials}] RI={ri_acc:.0%} PI={pi_acc:.0%}")

    # ──────────────────────────────────────────────────────────────
    # Analysis
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"RESULTS: Bidirectional MLM Probe — {args.model}")
    print(f"{'='*60}")

    ri_acc = np.mean([r["correct"] for r in results["RI"]])
    pi_acc = np.mean([r["correct"] for r in results["PI"]])
    ri_prob = np.mean([r.get("expected_prob", 0) for r in results["RI"]])
    pi_prob = np.mean([r.get("expected_prob", 0) for r in results["PI"]])

    print(f"\n  RI accuracy: {ri_acc:.1%} (mean P(correct)={ri_prob:.4f})")
    print(f"  PI accuracy: {pi_acc:.1%} (mean P(correct)={pi_prob:.4f})")
    print(f"  Gap: {ri_acc - pi_acc:+.1%}")
    print(f"  Prob ratio: {ri_prob / pi_prob:.2f}x" if pi_prob > 0 else "  PI prob = 0")

    # Compare with autoregressive models
    print(f"\n  Comparison:")
    print(f"    Autoregressive (Qwen 1.5B, 2k_5u): RI=99% PI=60% gap=+39%")
    print(f"    Autoregressive (Qwen 3B,   2k_5u): RI=84% PI=73% gap=+11%")
    print(f"    Bidirectional  ({args.model}):      RI={ri_acc:.0%} PI={pi_acc:.0%} gap={ri_acc-pi_acc:+.0%}")

    if abs(ri_acc - pi_acc) < 0.10:
        print(f"\n  VERDICT: Bidirectional model shows NO significant PI > RI")
        print(f"  → Autoregressive processing IS the key factor")
    elif ri_acc > pi_acc + 0.10:
        print(f"\n  VERDICT: Bidirectional model ALSO shows PI > RI")
        print(f"  → Autoregressive processing is NOT the sole cause")
    elif pi_acc > ri_acc + 0.10:
        print(f"\n  VERDICT: Bidirectional model shows REVERSED pattern (PI > RI)")
        print(f"  → Interesting! Bidirectional encoding may favor recency")
    else:
        print(f"\n  VERDICT: Weak asymmetry ({ri_acc-pi_acc:+.0%}), need more trials")

    # Error analysis
    print(f"\n  RI top predictions (when wrong):")
    ri_wrong = [r for r in results["RI"] if not r["correct"] and "top_prediction" in r]
    if ri_wrong:
        from collections import Counter
        preds = Counter(r["top_prediction"] for r in ri_wrong)
        for pred, count in preds.most_common(5):
            print(f"    '{pred}': {count}/{len(ri_wrong)} ({count/len(ri_wrong):.0%})")

    print(f"\n  PI top predictions (when wrong):")
    pi_wrong = [r for r in results["PI"] if not r["correct"] and "top_prediction" in r]
    if pi_wrong:
        from collections import Counter
        preds = Counter(r["top_prediction"] for r in pi_wrong)
        for pred, count in preds.most_common(5):
            print(f"    '{pred}': {count}/{len(pi_wrong)} ({count/len(pi_wrong):.0%})")

    # Save
    save_dir = _SCRIPT_DIR / "results" / "bidirectional"
    save_dir.mkdir(parents=True, exist_ok=True)
    model_short = args.model.split("/")[-1]
    save_path = save_dir / f"bidir_mlm_{model_short}.json"

    output = {
        "model": args.model,
        "config": {
            "num_keys": args.num_keys,
            "num_updates": args.num_updates,
            "trials": args.trials,
        },
        "ri_accuracy": float(ri_acc),
        "pi_accuracy": float(pi_acc),
        "gap": float(ri_acc - pi_acc),
        "ri_mean_prob": float(ri_prob),
        "pi_mean_prob": float(pi_prob),
        "trials": {"RI": results["RI"], "PI": results["PI"]},
    }
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()
