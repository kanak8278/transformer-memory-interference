"""
Remedy + Logit Lens: How does the landmark intervention work mechanistically?

Tests whether adding landmark separators changes the P(v_last) suppression
pattern in the logit lens — connecting the behavioral remedy to the mechanism.

Without landmarks: P(v_last) peaks at ~90% depth then suppressed
With landmarks:    Does P(v_last) stay high through final layers?

This is the critical mechanistic validation of the remedy.

Usage:
    cd /Users/kanak.raj/workspace/hobby/research_work_ri
    .venv/bin/python v3/scripts/experiments/remedy_logit_lens.py
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

_SCRIPT_DIR = Path(__file__).resolve().parent
_V3_DIR = _SCRIPT_DIR.parent.parent
_PROJECT_ROOT = _V3_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from mechanistic_probing_v2.core.dataset_configs import (
    get_eligible_categories, generate_values_for_trial,
    format_for_chat, FIXED_COMPLETION_DEMOS,
)
from mechanistic_probing_v2.core.model_loader import (
    load_model, model_short_name, is_instruct_model,
)
import random

DATASET_TYPE = "ARBITRARY_SINGLE"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
NUM_KEYS = 2
NUM_UPDATES = 5
N_TRIALS = 50

SYSTEM_PROMPT = (
    "You are a precise data extraction tool. "
    "Output ONLY a single word - the exact value requested. "
    "No other text, no explanation, no punctuation."
)


def build_kv_stream(categories, values_per_cat, num_updates, style="control"):
    items = []
    for round_idx in range(num_updates):
        for cat in categories:
            items.append({"category": cat, "value": values_per_cat[cat][round_idx]})

    if style == "control":
        return "\n".join(f"{it['category']}: {it['value']}" for it in items)
    elif style == "landmark":
        lines = []
        items_per_round = len(categories)
        for idx, it in enumerate(items):
            lines.append(f"{it['category']}: {it['value']}")
            if (idx + 1) % items_per_round == 0 and idx + 1 < len(items):
                lines.append(f"--- round {(idx+1)//items_per_round + 1} ---")
        return "\n".join(lines)
    raise ValueError(f"Unknown style: {style}")


def generate_trial(seed, condition, style):
    rng = random.Random(seed)
    eligible = get_eligible_categories(DATASET_TYPE, min_values=NUM_UPDATES)
    categories = rng.sample(eligible, NUM_KEYS)
    test_cat = categories[seed % NUM_KEYS]
    values_per_cat = generate_values_for_trial(DATASET_TYPE, categories, NUM_UPDATES, rng)

    stream = build_kv_stream(categories, values_per_cat, NUM_UPDATES, style=style)
    cat_vals = [values_per_cat[cat][i] for cat in categories
                for i in range(NUM_UPDATES) if cat == test_cat]
    # Rebuild in order
    cat_vals_ordered = values_per_cat[test_cat]
    expected = cat_vals_ordered[0] if condition == "RI" else cat_vals_ordered[-1]
    all_values = cat_vals_ordered
    query_word = "first" if condition == "RI" else "last"

    raw = (
        f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{stream}\n\n"
        f"What was the {query_word} value of {test_cat}?"
    )
    prompt = format_for_chat(raw, None, model_name=MODEL_NAME, system_prompt=SYSTEM_PROMPT)

    return {
        "prompt": prompt,
        "expected": expected,
        "all_values": all_values,
        "condition": condition,
        "style": style,
    }


def run_logit_lens(model, tokenizer, trials, value_to_tid):
    n_layers = model.cfg.n_layers
    results = []

    for t in trials:
        tokens = model.to_tokens(t["prompt"])
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        # Get P(v_i) at each layer for each value
        value_probs_by_layer = []
        for layer in range(n_layers):
            resid = cache["resid_post", layer][0, -1, :]
            logits = resid @ model.W_U + model.b_U
            probs = torch.softmax(logits, dim=-1)

            layer_probs = []
            for val in t["all_values"]:
                tid = value_to_tid.get(val.lower())
                p = probs[tid].item() if tid is not None else 0.0
                layer_probs.append(p)
            value_probs_by_layer.append(layer_probs)

        # Model prediction
        final_logits = cache["resid_post", n_layers - 1][0, -1, :] @ model.W_U + model.b_U
        pred_tid = final_logits.argmax().item()
        pred_text = tokenizer.decode([pred_tid]).strip().lower()
        correct = pred_text == t["expected"].lower()

        del cache
        results.append({
            "correct": correct,
            "expected": t["expected"],
            "predicted": pred_text,
            "condition": t["condition"],
            "style": t["style"],
            "value_probs_by_layer": value_probs_by_layer,
        })

    return results


def main():
    print("=== Remedy + Logit Lens ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Task: {NUM_KEYS}k_{NUM_UPDATES}u, {N_TRIALS} trials")
    print(f"Styles: control vs landmark\n")

    print("Loading model...")
    model, tokenizer, info = load_model(MODEL_NAME)
    n_layers = model.cfg.n_layers

    # Build value-to-token-id mapping
    value_pool = get_eligible_categories(DATASET_TYPE, min_values=NUM_UPDATES)
    from mechanistic_probing_v2.core.dataset_configs import get_value_pool
    try:
        all_values = get_value_pool(DATASET_TYPE)
        value_to_tid = {}
        for val in all_values:
            tids = model.to_tokens(val, prepend_bos=False)[0]
            if len(tids) == 1:
                value_to_tid[val.lower()] = tids[0].item()
    except Exception as e:
        print(f"Warning: value pool error: {e}")
        value_to_tid = {}

    all_results = {"control": {"PI": [], "RI": []}, "landmark": {"PI": [], "RI": []}}

    for style in ["control", "landmark"]:
        for condition in ["RI", "PI"]:
            print(f"\n  {style} {condition}:")
            trials = []
            for t_idx in range(N_TRIALS):
                seed = hash((NUM_KEYS, NUM_UPDATES, condition, t_idx, style, "remedy_ll")) % (2 ** 31)
                trial = generate_trial(seed, condition, style)
                trials.append(trial)

            res = run_logit_lens(model, tokenizer, trials, value_to_tid)
            all_results[style][condition] = res
            acc = sum(r["correct"] for r in res) / len(res)
            print(f"    acc={acc:.0%}")

    # Analysis: compare logit lens trajectories across styles
    print(f"\n{'='*60}")
    print("ANALYSIS: P(v_last) trajectory comparison")
    print(f"{'='*60}")

    for style in ["control", "landmark"]:
        pi_fails = [r for r in all_results[style]["PI"]
                   if not r["correct"] and max(r["value_probs_by_layer"][-1]) > 0.01]
        ri_corr = [r for r in all_results[style]["RI"] if r["correct"]]

        if pi_fails:
            # Average P(v_last) = last value at each layer
            avg_vlast = np.mean(
                [[layer_probs[-1] for layer_probs in r["value_probs_by_layer"]]
                 for r in pi_fails], axis=0
            )
            peak_layer = int(np.argmax(avg_vlast))
            peak_val = float(avg_vlast[peak_layer])
            final_val = float(avg_vlast[-1])

            print(f"\n  {style} — PI failures ({len(pi_fails)}):")
            print(f"    Peak P(v_last): {peak_val:.4f} at L{peak_layer} ({peak_layer/n_layers:.0%} depth)")
            print(f"    Final P(v_last): {final_val:.4f}")
            print(f"    Suppression: {peak_val - final_val:.4f}")

    # Save
    save_dir = _V3_DIR / "results" / "remedy"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"remedy_logit_lens_{model_short_name(MODEL_NAME)}.json"

    output = {
        "model": MODEL_NAME,
        "config": {"num_keys": NUM_KEYS, "num_updates": NUM_UPDATES, "n_trials": N_TRIALS},
        "behavioral": {
            style: {
                cond: sum(r["correct"] for r in all_results[style][cond]) / N_TRIALS
                for cond in ["RI", "PI"]
            }
            for style in ["control", "landmark"]
        },
        "trajectories": {
            style: {
                cond: {
                    "avg_vlast_by_layer": np.mean(
                        [[r["value_probs_by_layer"][l][-1] for l in range(n_layers)]
                         for r in all_results[style][cond]], axis=0
                    ).tolist() if all_results[style][cond] else [],
                    "avg_vfirst_by_layer": np.mean(
                        [[r["value_probs_by_layer"][l][0] for l in range(n_layers)]
                         for r in all_results[style][cond]], axis=0
                    ).tolist() if all_results[style][cond] else [],
                }
                for cond in ["RI", "PI"]
            }
            for style in ["control", "landmark"]
        },
    }
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
