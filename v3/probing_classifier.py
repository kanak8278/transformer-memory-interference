"""
Probing Classifier: What value position is encoded in the residual stream?

Trains linear probes at each layer to predict which value index (0..N-1) the
model is "attending to" at the answer position. Compares probing accuracy
between RI and PI conditions.

If the model encodes position 0 strongly for RI but fails to encode position N-1
for PI, the probes should show:
- RI: high accuracy at late layers (correct position encoded)
- PI: low accuracy at late layers (wrong position encoded)

This provides independent evidence beyond logit lens.

Usage:
    cd v3
    python probing_classifier.py --model Qwen/Qwen2.5-1.5B-Instruct --point "2,5" --trials 200
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

from mechanistic_probing_v2.core.model_loader import (
    load_model, verify_single_token, model_short_name, is_instruct_model,
)
from mechanistic_probing_v2.core.dataset_configs import (
    format_for_chat, get_value_pool, get_eligible_categories,
    generate_values_for_trial, FIXED_COMPLETION_DEMOS,
)

DATASET_TYPE = "ARBITRARY_SINGLE"
SYSTEM_PROMPT = (
    "You are a precise data extraction tool. "
    "Output ONLY a single word - the exact value requested. "
    "No other text, no explanation, no punctuation."
)


def parse_args():
    p = argparse.ArgumentParser(description="Probing classifier for value position encoding")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--point", default="2,5", help="Operating point 'keys,updates'")
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--gpu", type=int, default=None)
    return p.parse_args()


def generate_trial(num_keys, num_updates, condition, seed, value_pool, tokenizer, model_name):
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

    stream = "\n".join(f"{it['category']}: {it['value']}" for it in items)
    query_word = "first" if condition == "RI" else "last"
    cat_values = [it["value"] for it in items if it["category"] == test_category]
    expected = cat_values[0] if condition == "RI" else cat_values[-1]

    use_chat = is_instruct_model(model_name) if model_name else True
    if use_chat:
        raw_prompt = (
            f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
            f"{stream}\n\n"
            f"What was the {query_word} value of {test_category}?"
        )
        formatted = format_for_chat(raw_prompt, tokenizer, model_name=model_name)
    else:
        formatted = f"{FIXED_COMPLETION_DEMOS}{stream}\nThe {query_word} value of {test_category} was:"

    return {
        "prompt": formatted, "condition": condition, "expected": expected,
        "all_values": cat_values, "test_category": test_category,
        "num_keys": num_keys, "num_updates": num_updates, "seed": seed,
    }


def collect_representations(model, tokenizer, value_pool, model_name,
                           num_keys, num_updates, trials, value_to_tid):
    """Collect residual stream representations at answer position for all trials."""
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    data = {"RI": [], "PI": []}

    for condition in ["RI", "PI"]:
        print(f"\n  Collecting {condition} representations ({trials} trials)...")

        for t_idx in range(trials):
            seed = hash((num_keys, num_updates, condition, t_idx, "probing")) % (2**31)
            trial = generate_trial(num_keys, num_updates, condition, seed,
                                   value_pool, tokenizer, model_name)

            all_values = trial["all_values"]
            n_values = len(all_values)
            expected_idx = 0 if condition == "RI" else n_values - 1

            tokens = model.to_tokens(trial["prompt"])

            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)

            # Extract residual stream at answer position (last token) for each layer
            reps = np.zeros((n_layers, d_model))
            for layer in range(n_layers):
                resid = cache["resid_post", layer][0, -1, :].cpu().numpy()
                reps[layer] = resid

            # Get model's prediction
            logits = cache["resid_post", n_layers - 1][0, -1, :] @ model.W_U + model.b_U
            pred_tid = logits.argmax().item()
            pred_text = tokenizer.decode([pred_tid]).strip().lower()
            correct = pred_text == trial["expected"].lower()

            # Find which value the model predicted
            pred_value_idx = None
            for vi, val in enumerate(all_values):
                if val.lower() in pred_text or pred_text.startswith(val.lower()):
                    pred_value_idx = vi
                    break

            data[condition].append({
                "reps": reps,  # [n_layers, d_model]
                "expected_idx": expected_idx,
                "pred_value_idx": pred_value_idx,
                "correct": correct,
                "n_values": n_values,
            })

            del cache
            if (t_idx + 1) % 50 == 0:
                acc = sum(d["correct"] for d in data[condition]) / len(data[condition])
                print(f"    [{t_idx + 1}/{trials}] acc={acc:.0%}")

    return data


def train_probes(data, n_layers):
    """Train probing classifiers at each layer.

    Task: Given residual stream at layer L, predict which value index (0..N-1)
    the model will output.

    Returns accuracy per layer per condition.
    """
    results = {"RI": {}, "PI": {}}

    for condition in ["RI", "PI"]:
        trials = data[condition]
        if len(trials) < 20:
            print(f"  {condition}: too few trials ({len(trials)})")
            continue

        n_values = trials[0]["n_values"]

        # Create labels: expected_idx for correct trials, pred_value_idx for all
        X_all = {L: [] for L in range(n_layers)}
        y_expected = []  # ground truth position
        y_predicted = []  # what model actually outputs

        for trial in trials:
            for L in range(n_layers):
                X_all[L].append(trial["reps"][L])
            y_expected.append(trial["expected_idx"])
            y_predicted.append(trial["pred_value_idx"] if trial["pred_value_idx"] is not None else -1)

        y_expected = np.array(y_expected)

        # Train probe at each layer: predict expected_idx
        print(f"\n  {condition} probing ({len(trials)} trials, {n_values} classes):")
        for L in range(n_layers):
            X = np.array(X_all[L])

            # Only use trials with valid predictions for the "predicted" probe
            # For the "expected" probe, use all trials
            try:
                clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                        multi_class="multinomial")
                scores = cross_val_score(clf, X, y_expected, cv=5, scoring="accuracy")
                acc = scores.mean()
            except Exception:
                acc = 0.0

            results[condition][L] = {
                "accuracy": float(acc),
                "std": float(scores.std()) if acc > 0 else 0.0,
            }

            if L >= n_layers - 6 or L % 5 == 0:
                print(f"    L{L:2d}: {acc:.0%} ±{scores.std():.0%}")

    return results


def main():
    args = parse_args()
    num_keys, num_updates = map(int, args.point.split(","))

    print(f"Loading {args.model}...")
    model, tokenizer, info = load_model(args.model, gpu_idx=args.gpu)

    candidate_pool = get_value_pool(DATASET_TYPE)
    value_to_tid = verify_single_token(tokenizer, values=candidate_pool)
    value_pool = list(value_to_tid.keys())
    print(f"Value pool: {len(value_pool)} verified tokens")

    print(f"\nOperating point: {num_keys}k_{num_updates}u, {args.trials} trials")

    t0 = time.time()
    data = collect_representations(model, tokenizer, value_pool, args.model,
                                    num_keys, num_updates, args.trials, value_to_tid)

    probe_results = train_probes(data, model.cfg.n_layers)
    elapsed = time.time() - t0

    # Summary
    n_layers = model.cfg.n_layers
    print(f"\n{'='*60}")
    print(f"PROBING RESULTS — {args.model}")
    print(f"  Point: {num_keys}k_{num_updates}u, {args.trials} trials")
    print(f"{'='*60}")

    print("\nLast 6 layers comparison:")
    print(f"  {'Layer':>6}  {'RI probe':>10}  {'PI probe':>10}  {'Gap':>8}")
    for L in range(max(0, n_layers - 6), n_layers):
        ri_acc = probe_results["RI"].get(L, {}).get("accuracy", 0)
        pi_acc = probe_results["PI"].get(L, {}).get("accuracy", 0)
        print(f"  L{L:>4}  {ri_acc:>9.0%}  {pi_acc:>9.0%}  {ri_acc-pi_acc:>+7.0%}")

    print(f"\n  Elapsed: {elapsed:.1f}s")

    # Save
    m_short = model_short_name(args.model)
    save_dir = _SCRIPT_DIR / "results" / "probing"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"probing_{m_short}_{num_keys}k_{num_updates}u.json"

    output = {
        "model": args.model,
        "point": {"keys": num_keys, "updates": num_updates},
        "trials": args.trials,
        "n_layers": n_layers,
        "probe_results": probe_results,
        "behavioral": {
            "RI": {
                "accuracy": sum(d["correct"] for d in data["RI"]) / len(data["RI"]),
                "n": len(data["RI"]),
            },
            "PI": {
                "accuracy": sum(d["correct"] for d in data["PI"]) / len(data["PI"]),
                "n": len(data["PI"]),
            },
        },
        "elapsed_sec": elapsed,
    }
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()
