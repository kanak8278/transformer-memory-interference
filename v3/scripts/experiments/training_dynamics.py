"""
Training Dynamics: When Does PI > RI Emerge?

Tests PI > RI at multiple training checkpoints of SmolLM2-1.7B to answer:
- Does PI > RI exist from the beginning (step 0) → architectural?
- Does it emerge gradually → learned behavior?
- Does training make it better or worse?

SmolLM2-1.7B checkpoints available at:
steps: 125K, 250K (not available), 500K(not available), ...
Actually: 125000, 250000, 375000... let me check

Usage:
    cd /Users/kanak.raj/workspace/hobby/research_work_ri
    .venv/bin/python v3/scripts/experiments/training_dynamics.py
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_V3_DIR = _SCRIPT_DIR.parent.parent
_PROJECT_ROOT = _V3_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from mechanistic_probing_v2.core.dataset_configs import (
    get_eligible_categories, generate_values_for_trial,
    format_for_chat, FIXED_COMPLETION_DEMOS,
)
import random

DATASET_TYPE = "ARBITRARY_SINGLE"
BASE_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
CHECKPOINT_REPO = "HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints"

# Available checkpoint steps
CHECKPOINTS = [
    "step-125000",
    "step-375000",
    "step-625000",
    "step-875000",
    "step-1125000",
    "step-1375000",
    "step-1625000",
    "step-1875000",
    "step-2000000",  # final = main
]

# Operating point: regime B for a 1.7B model
NUM_KEYS = 2
NUM_UPDATES = 10
N_TRIALS = 50  # smaller for speed across many checkpoints

SYSTEM_PROMPT = (
    "You are a precise data extraction tool. "
    "Output ONLY a single word - the exact value requested. "
    "No other text, no explanation, no punctuation."
)


def build_prompt(categories, values_per_cat, test_cat, condition, tokenizer=None, model_name=None):
    items = []
    for cat in categories:
        for val in values_per_cat[cat]:
            items.append({"category": cat, "value": val})
    random.shuffle(items)  # use same seed externally

    stream = "\n".join(f"{it['category']}: {it['value']}" for it in items)
    cat_vals = values_per_cat[test_cat]
    expected = cat_vals[0] if condition == "RI" else cat_vals[-1]
    query_word = "first" if condition == "RI" else "last"

    raw = (
        f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{stream}\n\n"
        f"What was the {query_word} value of {test_cat}?"
    )

    if tokenizer is not None and model_name and "smol" in model_name.lower():
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": raw},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"{SYSTEM_PROMPT}\n\n{raw}"

    return prompt, expected


def run_checkpoint(checkpoint_name, n_trials, rng_seed=42):
    """Load checkpoint and run PI/RI sweep. Returns (ri_acc, pi_acc, garbage_rate)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n  Loading {checkpoint_name}...")

    if checkpoint_name == "step-2000000":
        # Use final trained model (instruct)
        repo = BASE_MODEL
        revision = "main"
    else:
        repo = CHECKPOINT_REPO
        revision = checkpoint_name

    try:
        tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(
            repo,
            revision=revision,
            torch_dtype=torch.float16,
            device_map="mps",
        )
        model.eval()
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return None

    device = next(model.parameters()).device
    print(f"  Loaded. Device: {device}")

    # Run trials
    ri_correct, pi_correct, n_garbage = 0, 0, 0

    eligible = get_eligible_categories(DATASET_TYPE, min_values=NUM_UPDATES)

    for t_idx in range(n_trials):
        seed = hash((NUM_KEYS, NUM_UPDATES, t_idx, "dynamics")) % (2 ** 31)
        rng = random.Random(seed)
        categories = rng.sample(eligible, NUM_KEYS)
        test_cat = categories[seed % NUM_KEYS]
        values_per_cat = generate_values_for_trial(DATASET_TYPE, categories, NUM_UPDATES, rng)

        for condition in ["RI", "PI"]:
            prompt, expected = build_prompt(
                categories, values_per_cat, test_cat, condition,
                tokenizer=tokenizer, model_name=BASE_MODEL
            )

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                              max_length=2048).to(device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=15,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated = out[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(generated, skip_special_tokens=True).strip().lower()

            correct = (response == expected.lower() or
                      expected.lower() in response.split()[:3])

            # Check garbage
            all_vals = [v.lower() for vals in values_per_cat.values() for v in vals]
            is_garbage = not any(v in response for v in all_vals) and response != expected.lower()

            if is_garbage:
                n_garbage += 1
            elif condition == "RI" and correct:
                ri_correct += 1
            elif condition == "PI" and correct:
                pi_correct += 1

    ri_acc = ri_correct / n_trials
    pi_acc = pi_correct / n_trials
    garb_rate = n_garbage / (n_trials * 2)

    print(f"  RI={ri_acc:.0%} PI={pi_acc:.0%} gap={ri_acc-pi_acc:+.0%} garb={garb_rate:.0%}")

    # Free memory
    del model
    torch.mps.empty_cache() if hasattr(torch.mps, 'empty_cache') else None

    return {"ri_acc": float(ri_acc), "pi_acc": float(pi_acc),
            "gap": float(ri_acc - pi_acc), "garbage_rate": float(garb_rate),
            "n_trials": n_trials}


def main():
    print("=== Training Dynamics: When Does PI > RI Emerge? ===")
    print(f"Model: SmolLM2-1.7B")
    print(f"Checkpoints: {CHECKPOINTS}")
    print(f"Task: {NUM_KEYS}k_{NUM_UPDATES}u, {N_TRIALS} trials\n")

    all_results = {}

    for ckpt in CHECKPOINTS:
        result = run_checkpoint(ckpt, N_TRIALS)
        if result:
            all_results[ckpt] = result

    # Summary
    print(f"\n{'='*60}")
    print("TRAINING DYNAMICS SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Checkpoint':<20} {'Step':>10} {'RI':>6} {'PI':>6} {'Gap':>7}")
    print("-" * 55)

    for ckpt, r in sorted(all_results.items()):
        step = int(ckpt.split("-")[1]) if ckpt.startswith("step-") else 2000000
        print(f"{ckpt:<20} {step:>10,} {r['ri_acc']:>5.0%} {r['pi_acc']:>5.0%} {r['gap']:>+6.0%}")

    # Save
    save_dir = _V3_DIR / "results" / "training_dynamics"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "smollm2_1.7b_dynamics.json"

    output = {
        "model": BASE_MODEL,
        "checkpoint_repo": CHECKPOINT_REPO,
        "config": {"num_keys": NUM_KEYS, "num_updates": NUM_UPDATES, "n_trials": N_TRIALS},
        "results": all_results,
    }
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nSaved: {save_path}")

    # Interpretation
    print("\n=== INTERPRETATION ===")
    steps_sorted = sorted(
        [(int(k.split("-")[1]) if k.startswith("step-") else 2000000, k, v)
         for k, v in all_results.items()]
    )
    if steps_sorted:
        first_step, first_ckpt, first_r = steps_sorted[0]
        last_step, last_ckpt, last_r = steps_sorted[-1]

        if first_r["gap"] > 0.1:
            print(f"PI > RI exists from earliest checkpoint ({first_ckpt}): gap={first_r['gap']:+.0%}")
            print("→ ARCHITECTURAL: bias present before substantial training")
        elif last_r["gap"] > 0.1 and first_r["gap"] < 0.05:
            print("PI > RI emerges during training")
            print("→ LEARNED: model develops primacy bias through data")
        else:
            print("Pattern unclear — check garbage rates")


if __name__ == "__main__":
    main()
