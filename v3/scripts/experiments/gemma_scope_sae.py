"""
Gemma Scope SAE Analysis: Which features activate differently for RI vs PI?

Uses Google's Gemma Scope 2 pre-trained Sparse Autoencoders on gemma-3-1b-it
to find interpretable features that distinguish:
1. RI vs PI conditions (same-query comparison)
2. RI-correct vs RI-incorrect representations
3. PI-correct vs PI-incorrect representations

SAEs available at layers 7, 13, 17, 22 (resid_post) of gemma-3-1b-it.
We use the 65k-width, l0_medium SAE — best tradeoff of interpretability and speed.

Key research question:
- Are there specific "first-value retrieval" or "last-value retrieval" features?
- Does PI failure correlate with specific features being absent?
- Do the features that activate in RI-correct trials activate MORE in PI-correct
  than PI-incorrect?

Usage:
    cd /Users/kanak.raj/workspace/hobby/research_work_ri
    .venv/bin/python v3/scripts/experiments/gemma_scope_sae.py
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
MODEL_NAME = "google/gemma-3-1b-it"
NUM_KEYS = 2
NUM_UPDATES = 5
N_TRIALS = 100  # per condition

SYSTEM_PROMPT = (
    "You are a precise data extraction tool. "
    "Output ONLY a single word - the exact value requested. "
    "No other text, no explanation, no punctuation."
)

# Gemma Scope 2 SAE release for gemma-3-1b-it residual post
SAE_RELEASE = "gemma-scope-2-1b-it-res"
SAE_LAYERS = [7, 13, 17, 22]  # Available in gemma-scope-2-1b-it
SAE_WIDTH = "65k"
SAE_L0 = "medium"


def generate_trial(seed, condition):
    rng = random.Random(seed)
    eligible = get_eligible_categories(DATASET_TYPE, min_values=NUM_UPDATES)
    categories = rng.sample(eligible, NUM_KEYS)
    test_cat = categories[seed % NUM_KEYS]
    values_per_cat = generate_values_for_trial(DATASET_TYPE, categories, NUM_UPDATES, rng)

    items = []
    for cat in categories:
        for val in values_per_cat[cat]:
            items.append({"category": cat, "value": val})
    rng.shuffle(items)

    stream = "\n".join(f"{it['category']}: {it['value']}" for it in items)
    cat_vals = [it["value"] for it in items if it["category"] == test_cat]
    expected = cat_vals[0] if condition == "RI" else cat_vals[-1]
    query_word = "first" if condition == "RI" else "last"

    raw = (
        f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{stream}\n\n"
        f"What was the {query_word} value of {test_cat}?"
    )
    from mechanistic_probing_v2.core.model_loader import is_instruct_model
    use_chat = is_instruct_model(MODEL_NAME)
    if use_chat:
        prompt = format_for_chat(raw, None, model_name=MODEL_NAME, system_prompt=SYSTEM_PROMPT)
    else:
        prompt = f"{FIXED_COMPLETION_DEMOS}{stream}\nThe {query_word} value of {test_cat} was:"

    return {"prompt": prompt, "expected": expected, "condition": condition}


def main():
    print(f"=== Gemma Scope SAE Analysis ===")
    print(f"Model: {MODEL_NAME}")
    print(f"SAE: {SAE_RELEASE} layers {SAE_LAYERS}, width={SAE_WIDTH}, l0={SAE_L0}")
    print(f"Task: {NUM_KEYS}k_{NUM_UPDATES}u, {N_TRIALS} trials per condition\n")

    # Load model with TransformerLens
    print("Loading model...")
    model, tokenizer, info = load_model(MODEL_NAME)
    device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device("mps")
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    print(f"  {n_layers} layers, d_model={d_model}, device={device}")

    # Load SAEs for each layer
    print("\nLoading SAEs...")
    saes = {}
    try:
        from sae_lens import SAE
        for layer in SAE_LAYERS:
            sae_id = f"layer_{layer}_width_{SAE_WIDTH}_l0_{SAE_L0}"
            print(f"  Loading {sae_id}...")
            try:
                sae, cfg_dict, sparsity = SAE.from_pretrained(
                    release=SAE_RELEASE,
                    sae_id=sae_id,
                )
                sae = sae.to(device)
                saes[layer] = sae
                print(f"  Layer {layer}: {sae.cfg.d_sae} features, L0≈{sparsity:.1f}")
            except Exception as e:
                print(f"  Layer {layer}: FAILED — {e}")
    except ImportError:
        print("sae-lens not found, cannot run SAE analysis")
        return

    if not saes:
        print("No SAEs loaded, exiting")
        return

    # Collect activations and SAE features
    print(f"\nCollecting features ({N_TRIALS} trials × 2 conditions)...")

    results = {"RI": [], "PI": []}

    for condition in ["RI", "PI"]:
        print(f"\n  {condition}:")
        for t_idx in range(N_TRIALS):
            seed = hash((NUM_KEYS, NUM_UPDATES, condition, t_idx, "sae")) % (2 ** 31)
            trial = generate_trial(seed, condition)

            tokens = model.to_tokens(trial["prompt"])

            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)

            # Get model prediction
            logits = cache["resid_post", n_layers - 1][0, -1, :] @ model.W_U + model.b_U
            pred_tid = logits.argmax().item()
            pred_text = tokenizer.decode([pred_tid]).strip().lower()
            correct = pred_text == trial["expected"].lower()

            # Extract SAE features at answer position (last token)
            trial_features = {}
            for layer, sae in saes.items():
                resid = cache["resid_post", layer][0, -1, :]  # [d_model]
                with torch.no_grad():
                    features = sae.encode(resid.unsqueeze(0))  # [1, n_features]
                    feature_acts = features[0].cpu().numpy()   # [n_features]

                # Store top-K active features (sparse)
                active_mask = feature_acts > 0
                active_indices = np.where(active_mask)[0].tolist()
                active_values = feature_acts[active_mask].tolist()

                trial_features[layer] = {
                    "active_indices": active_indices,
                    "active_values": active_values,
                    "n_active": int(active_mask.sum()),
                }

            del cache

            results[condition].append({
                "seed": seed,
                "correct": correct,
                "expected": trial["expected"],
                "predicted": pred_text,
                "features": trial_features,
            })

            if (t_idx + 1) % 25 == 0:
                acc = sum(r["correct"] for r in results[condition]) / len(results[condition])
                print(f"    [{t_idx+1}/{N_TRIALS}] acc={acc:.0%}")

    # Analysis
    print(f"\n{'='*60}")
    print("ANALYSIS: Discriminating features across conditions")
    print(f"{'='*60}")

    ri_trials = results["RI"]
    pi_trials = results["PI"]
    ri_acc = sum(r["correct"] for r in ri_trials) / len(ri_trials)
    pi_acc = sum(r["correct"] for r in pi_trials) / len(pi_trials)
    print(f"\n  Behavioral: RI={ri_acc:.0%}, PI={pi_acc:.0%}, gap={ri_acc-pi_acc:+.0%}")

    # For each layer, find features that:
    # (a) Activate more in RI than PI (RI-preferring)
    # (b) Activate more in PI-correct than PI-incorrect (retrieval-enabling)
    feature_analysis = {}

    for layer in SAE_LAYERS:
        n_features = saes[layer].cfg.d_sae

        # Compute mean activation per feature for each group
        def mean_activations(trial_list):
            acts = np.zeros(n_features)
            for t in trial_list:
                feat = t["features"][layer]
                for idx, val in zip(feat["active_indices"], feat["active_values"]):
                    acts[idx] += val
            return acts / max(len(trial_list), 1)

        ri_acts = mean_activations(ri_trials)
        pi_acts = mean_activations(pi_trials)
        pi_correct = [t for t in pi_trials if t["correct"]]
        pi_wrong = [t for t in pi_trials if not t["correct"]]
        ri_correct_t = [t for t in ri_trials if t["correct"]]

        pi_correct_acts = mean_activations(pi_correct) if pi_correct else np.zeros(n_features)
        pi_wrong_acts = mean_activations(pi_wrong) if pi_wrong else np.zeros(n_features)

        # RI vs PI difference
        ri_pi_diff = ri_acts - pi_acts
        # PI correct vs wrong
        correct_diff = pi_correct_acts - pi_wrong_acts

        # Top features
        top_ri_pref = np.argsort(-ri_pi_diff)[:10]
        top_pi_pref = np.argsort(ri_pi_diff)[:10]
        top_correct_pref = np.argsort(-correct_diff)[:10] if pi_correct and pi_wrong else []

        print(f"\n  Layer {layer} (n_features={n_features}):")
        print(f"    Mean active features: RI={np.mean([len(t['features'][layer]['active_indices']) for t in ri_trials]):.1f}, "
              f"PI={np.mean([len(t['features'][layer]['active_indices']) for t in pi_trials]):.1f}")
        print(f"    Top RI-preferring features: {top_ri_pref[:5].tolist()}")
        print(f"      (mean act diff: {ri_pi_diff[top_ri_pref[:5]]:.4f})")
        print(f"    Top PI-preferring features: {top_pi_pref[:5].tolist()}")
        if len(top_correct_pref) > 0:
            print(f"    Top PI-correct enabling features: {top_correct_pref[:5].tolist()}")
            print(f"      (n_correct={len(pi_correct)}, n_wrong={len(pi_wrong)})")

        feature_analysis[layer] = {
            "ri_mean_active": float(np.mean([len(t["features"][layer]["active_indices"]) for t in ri_trials])),
            "pi_mean_active": float(np.mean([len(t["features"][layer]["active_indices"]) for t in pi_trials])),
            "top_ri_preferring": [(int(i), float(ri_pi_diff[i])) for i in top_ri_pref[:10]],
            "top_pi_preferring": [(int(i), float(-ri_pi_diff[i])) for i in top_pi_pref[:10]],
            "top_pi_correct_enabling": [(int(i), float(correct_diff[i])) for i in top_correct_pref[:10]] if len(top_correct_pref) else [],
            "ri_acts_mean": float(ri_acts.mean()),
            "pi_acts_mean": float(pi_acts.mean()),
        }

    # Save
    save_dir = _V3_DIR / "results" / "sae"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "gemma_scope_sae_analysis.json"

    output = {
        "model": MODEL_NAME,
        "sae_release": SAE_RELEASE,
        "sae_layers": SAE_LAYERS,
        "config": {"num_keys": NUM_KEYS, "num_updates": NUM_UPDATES, "n_trials": N_TRIALS},
        "behavioral": {"ri_acc": float(ri_acc), "pi_acc": float(pi_acc), "gap": float(ri_acc - pi_acc)},
        "feature_analysis": feature_analysis,
        "trials": {
            "RI": [{k: v for k, v in t.items() if k != "features"} for t in ri_trials],
            "PI": [{k: v for k, v in t.items() if k != "features"} for t in pi_trials],
        },
    }
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
