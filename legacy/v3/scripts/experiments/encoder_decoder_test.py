"""
Encoder-Decoder Control: Does T5 (bidirectional encoder) show PI > RI?

T5 has a BIDIRECTIONAL encoder (like BERT) but an AUTOREGRESSIVE decoder.
If PI > RI appears in T5, the decoder's autoregressive nature is responsible.
If PI ≈ RI in T5, the bidirectional encoder helps equalize retrieval.

This is a cleaner test than BERT-MLM because T5 can actually generate text.

Also tests Flan-T5 (instruction-tuned) for better task following.

Usage:
    cd v3 && python encoder_decoder_test.py --trials 100
"""

import sys
import json
import random
import argparse
import torch
import numpy as np
from pathlib import Path

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
    p.add_argument("--model", default="google/flan-t5-base")
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--num-keys", type=int, default=2)
    p.add_argument("--num-updates", type=int, default=5)
    p.add_argument("--device", default=None)
    return p.parse_args()


def generate_kv_stream(num_keys, num_updates, seed):
    rng = random.Random(seed)
    eligible = get_eligible_categories(DATASET_TYPE, min_values=num_updates)
    categories = rng.sample(eligible, min(num_keys, len(eligible)))
    test_category = categories[seed % num_keys]
    values_per_cat = generate_values_for_trial(DATASET_TYPE, categories, num_updates, rng)

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

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.eval()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {n_params:.0f}M params, device={device}")

    results = {"RI": [], "PI": []}

    print(f"\nRunning {args.trials} trials...")

    for t_idx in range(args.trials):
        seed = hash((args.num_keys, args.num_updates, t_idx, "t5_test")) % (2**31)
        trial = generate_kv_stream(args.num_keys, args.num_updates, seed)

        for condition, query_word, expected in [
            ("RI", "first", trial["first_value"]),
            ("PI", "last", trial["last_value"]),
        ]:
            prompt = (f"Given the following data stream:\n{trial['stream']}\n\n"
                     f"What was the {query_word} value assigned to "
                     f"{trial['test_category']}? Answer with just the word.")

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=512).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    num_beams=1,
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
            expected_lower = expected.lower()
            correct = response == expected_lower or expected_lower in response.split()

            # Classify error type
            error_type = "correct" if correct else "wrong"
            if not correct:
                all_vals_lower = [v.lower() for v in trial["all_values"]]
                if response in all_vals_lower:
                    pred_idx = all_vals_lower.index(response)
                    error_type = "intermediate_intrusion"
                else:
                    error_type = "garbage"

            results[condition].append({
                "correct": correct,
                "expected": expected,
                "predicted": response,
                "error_type": error_type,
            })

        if (t_idx + 1) % 25 == 0:
            ri_acc = np.mean([r["correct"] for r in results["RI"]])
            pi_acc = np.mean([r["correct"] for r in results["PI"]])
            ri_garb = np.mean([r["error_type"] == "garbage" for r in results["RI"]])
            pi_garb = np.mean([r["error_type"] == "garbage" for r in results["PI"]])
            print(f"  [{t_idx + 1}/{args.trials}] RI={ri_acc:.0%}(g={ri_garb:.0%}) "
                  f"PI={pi_acc:.0%}(g={pi_garb:.0%}) gap={ri_acc-pi_acc:+.0%}")

    # Analysis
    print(f"\n{'='*60}")
    print(f"RESULTS: Encoder-Decoder — {args.model}")
    print(f"  (Bidirectional encoder + Autoregressive decoder)")
    print(f"{'='*60}")

    ri_acc = np.mean([r["correct"] for r in results["RI"]])
    pi_acc = np.mean([r["correct"] for r in results["PI"]])
    ri_garb = np.mean([r["error_type"] == "garbage" for r in results["RI"]])
    pi_garb = np.mean([r["error_type"] == "garbage" for r in results["PI"]])

    print(f"\n  RI: {ri_acc:.1%} (garbage={ri_garb:.0%})")
    print(f"  PI: {pi_acc:.1%} (garbage={pi_garb:.0%})")
    print(f"  Gap: {ri_acc - pi_acc:+.1%}")

    if abs(ri_acc - pi_acc) < 0.10:
        print(f"\n  VERDICT: Encoder-decoder shows NO significant PI > RI")
        print(f"  → Bidirectional encoder equalizes retrieval")
        print(f"  → Confirms: autoregressive ENCODING (not just decoding) is key")
    elif ri_acc > pi_acc + 0.10:
        print(f"\n  VERDICT: Encoder-decoder shows PI > RI (gap={ri_acc-pi_acc:+.0%})")
        print(f"  → Even with bidirectional encoder, PI > RI persists")
        print(f"  → Autoregressive decoder may be sufficient to cause asymmetry")
    else:
        print(f"\n  VERDICT: Need more trials to determine")

    # Error analysis
    for cond in ["RI", "PI"]:
        wrong = [r for r in results[cond] if not r["correct"]]
        if wrong:
            from collections import Counter
            preds = Counter(r["predicted"] for r in wrong)
            print(f"\n  {cond} top wrong predictions:")
            for pred, count in preds.most_common(5):
                print(f"    '{pred}': {count}/{len(wrong)}")

    # Save
    save_dir = _SCRIPT_DIR / "results" / "bidirectional"
    save_dir.mkdir(parents=True, exist_ok=True)
    model_short = args.model.split("/")[-1]
    save_path = save_dir / f"enc_dec_{model_short}.json"

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
        "ri_garbage": float(ri_garb),
        "pi_garbage": float(pi_garb),
        "trials": {"RI": results["RI"], "PI": results["PI"]},
    }
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()
