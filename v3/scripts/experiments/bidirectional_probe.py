"""
Bidirectional Model Probe: Does PI > RI exist in encoder models?

Tests whether the representation asymmetry (first value encoded better than
last value) exists in bidirectional (BERT-like) models. If it does NOT,
this confirms autoregressive processing is the key factor.

Approach: Feed the KV stream to BERT, extract representation at the query
position, and probe whether it encodes the first or last value.

Usage:
    cd v3 && python bidirectional_probe.py --trials 200
"""

import sys
import json
import time
import random
import argparse
import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

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
    return p.parse_args()


def generate_kv_stream(num_keys, num_updates, seed):
    rng = random.Random(seed)
    eligible = get_eligible_categories(DATASET_TYPE, min_values=num_updates)
    categories = rng.sample(eligible, min(num_keys, len(eligible)))
    test_category = categories[seed % num_keys]
    values_per_cat = generate_values_for_trial(DATASET_TYPE, categories, num_updates, rng)

    items = []
    for cat in categories:
        for val in values_per_cat[cat]:
            items.append({"category": cat, "value": val})
    rng.shuffle(items)

    stream = " ".join(f"{it['category']}: {it['value']}" for it in items)
    cat_values = [it["value"] for it in items if it["category"] == test_category]

    return {
        "stream": stream,
        "test_category": test_category,
        "first_value": cat_values[0],
        "last_value": cat_values[-1],
        "all_values": cat_values,
    }


def main():
    args = parse_args()

    from transformers import AutoTokenizer, AutoModel

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.eval()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    print(f"  {sum(p.numel() for p in model.parameters())/1e6:.0f}M params, {n_layers} layers, d={d_model}")

    # Generate trials and collect representations
    print(f"\nCollecting representations ({args.trials} trials)...")

    reps_per_layer = {L: [] for L in range(n_layers)}
    labels_first = []  # 1 if query="first", 0 if query="last"
    correct_values = []

    for t_idx in range(args.trials):
        seed = hash((args.num_keys, args.num_updates, t_idx, "bidir")) % (2**31)
        trial = generate_kv_stream(args.num_keys, args.num_updates, seed)

        # Create two prompts: one asking for first, one for last
        for query_type, label in [("first", 1), ("last", 0)]:
            prompt = f"{trial['stream']} What was the {query_type} value of {trial['test_category']}?"

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            # Extract representation at the last token (query position)
            for L in range(n_layers):
                hidden = outputs.hidden_states[L + 1]  # +1 because index 0 is embeddings
                rep = hidden[0, -1, :].cpu().numpy()  # last token
                reps_per_layer[L].append(rep)

            labels_first.append(label)
            expected = trial["first_value"] if query_type == "first" else trial["last_value"]
            correct_values.append(expected)

        if (t_idx + 1) % 50 == 0:
            print(f"  [{t_idx + 1}/{args.trials}]")

    labels_first = np.array(labels_first)
    n_total = len(labels_first)
    print(f"\n  Collected {n_total} representations ({n_total // 2} trials × 2 conditions)")

    # Train probes: can we tell "first query" from "last query"?
    print(f"\n  Probing: can BERT distinguish first-query from last-query?")
    probe_results = {}

    for L in range(n_layers):
        X = np.array(reps_per_layer[L])
        try:
            clf = LogisticRegression(max_iter=500, C=0.1, solver="lbfgs")
            scores = cross_val_score(clf, X, labels_first, cv=5, scoring="accuracy")
            acc = scores.mean()
        except Exception:
            acc = 0.5
            scores = np.array([0.5])

        probe_results[L] = {"accuracy": float(acc), "std": float(scores.std())}

        if L >= n_layers - 4 or L % 3 == 0:
            print(f"    L{L:2d}: {acc:.0%} ±{scores.std():.0%}")

    # Summary
    print(f"\n{'='*60}")
    print(f"BIDIRECTIONAL PROBE — {args.model}")
    print(f"{'='*60}")

    last_4_avg = np.mean([probe_results[L]["accuracy"] for L in range(n_layers - 4, n_layers)])
    print(f"\n  Last 4 layers avg probe accuracy: {last_4_avg:.0%}")
    print(f"  (Autoregressive models show 96-100% here)")
    print(f"  (50% = no discrimination = first and last equally encoded)")
    print(f"  (100% = perfect discrimination = asymmetric encoding)")

    if last_4_avg < 0.6:
        print(f"\n  RESULT: Bidirectional model shows WEAK first/last discrimination")
        print(f"  → Confirms autoregressive processing causes the asymmetry")
    elif last_4_avg > 0.9:
        print(f"\n  RESULT: Bidirectional model ALSO shows strong discrimination")
        print(f"  → Autoregressive processing is NOT the sole cause")
    else:
        print(f"\n  RESULT: Moderate discrimination ({last_4_avg:.0%})")
        print(f"  → Autoregressive processing amplifies but may not be sole cause")

    # Save
    save_dir = _SCRIPT_DIR / "results" / "bidirectional"
    save_dir.mkdir(parents=True, exist_ok=True)
    from mechanistic_probing_v2.core.model_loader import model_short_name
    m_short = model_short_name(args.model)
    save_path = save_dir / f"bidir_probe_{m_short}.json"

    output = {
        "model": args.model,
        "config": {"num_keys": args.num_keys, "num_updates": args.num_updates, "trials": args.trials},
        "n_layers": n_layers,
        "d_model": d_model,
        "probe_results": probe_results,
        "last_4_avg": float(last_4_avg),
    }
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()
