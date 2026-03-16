"""
Quick diagnostic: Why does Qwen2.5-0.5B produce 80% garbage?

Tests multiple prompt formats on the same trials to isolate:
1. Is it instruction following failure (prompt issue)?
2. Is it context length (too many keys/updates)?
3. Does constrained decoding help?

Run: cd v3 && python diagnose_0.5b.py
"""

import sys
import json
import random
import torch
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mechanistic_probing_v2.core.dataset_configs import (
    load_dataset_config,
    get_eligible_categories,
    generate_values_for_trial,
)
from mechanistic_probing_v2.core.model_loader import load_model_hf

DATASET_TYPE = "ARBITRARY_SINGLE"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_TRIALS = 20
SEEDS = list(range(1000, 1000 + N_TRIALS))


def generate_prompt_variants(num_keys, num_updates, condition, seed):
    """Generate the same trial with different prompt formats."""
    rng = random.Random(seed)
    eligible = get_eligible_categories(DATASET_TYPE, min_values=num_updates)
    categories = rng.sample(eligible, min(num_keys, len(eligible)))
    test_category = categories[seed % num_keys]
    values_per_cat = generate_values_for_trial(DATASET_TYPE, categories, num_updates, rng)

    items = []
    for cat in categories:
        for val in values_per_cat[cat]:
            items.append({"category": cat, "value": val})
    # Simple shuffle (no consecutive same-cat)
    rng.shuffle(items)

    stream = "\n".join(f"{it['category']}: {it['value']}" for it in items)
    query_word = "first" if condition == "RI" else "last"
    cat_values = [it["value"] for it in items if it["category"] == test_category]
    expected = cat_values[0] if condition == "RI" else cat_values[-1]

    variants = {}

    # V1: Original (system prompt + user message)
    variants["v1_system_user"] = {
        "messages": [
            {"role": "system", "content": "Answer with ONLY the exact value. No explanation."},
            {"role": "user", "content": (
                f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
                f"{stream}\n\n"
                f"What was the {query_word} value of {test_category}?"
            )},
        ]
    }

    # V2: Stronger system prompt
    variants["v2_strong_system"] = {
        "messages": [
            {"role": "system", "content": (
                "You are a precise data extraction tool. "
                "Output ONLY a single word - the exact value requested. "
                "No other text, no explanation, no punctuation."
            )},
            {"role": "user", "content": (
                f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
                f"{stream}\n\n"
                f"What was the {query_word} value of {test_category}?"
            )},
        ]
    }

    # V3: Few-shot in-context examples
    variants["v3_few_shot"] = {
        "messages": [
            {"role": "system", "content": "Answer with ONLY the exact value. No explanation."},
            {"role": "user", "content": (
                "bird species: robin\nfruit variety: apple\nbird species: eagle\nfruit variety: mango\n\n"
                "What was the first value of bird species?"
            )},
            {"role": "assistant", "content": "robin"},
            {"role": "user", "content": (
                "bird species: robin\nfruit variety: apple\nbird species: eagle\nfruit variety: mango\n\n"
                "What was the last value of bird species?"
            )},
            {"role": "assistant", "content": "eagle"},
            {"role": "user", "content": (
                f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
                f"{stream}\n\n"
                f"What was the {query_word} value of {test_category}?"
            )},
        ]
    }

    # V4: Completion format (no chat template)
    variants["v4_completion"] = {
        "raw_prompt": (
            f"Example:\nbird species: robin\nfruit variety: apple\nbird species: eagle\n"
            f"The first value of bird species was: robin\nThe last value of bird species was: eagle\n\n"
            f"Data:\n{stream}\n"
            f"The {query_word} value of {test_category} was:"
        )
    }

    return variants, expected, cat_values


def run_one(model, tokenizer, prompt_text, device, max_new_tokens=10):
    """Run a single prompt and return the generated text."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    print(f"Loading {MODEL}...")
    model, tokenizer, model_info = load_model_hf(MODEL, device="mps")
    device = next(model.parameters()).device
    print(f"Device: {device}")

    configs = [
        # (num_keys, num_updates, condition)
        (2, 5, "RI"),
        (2, 5, "PI"),
        (3, 5, "RI"),
        (3, 5, "PI"),
        (5, 5, "RI"),
        (5, 5, "PI"),
    ]

    results = {}
    for num_keys, num_updates, condition in configs:
        config_key = f"{num_keys}k_{num_updates}u_{condition}"
        print(f"\n{'='*60}")
        print(f"Config: {config_key}")
        print(f"{'='*60}")

        variant_scores = {}
        for seed in SEEDS[:10]:  # 10 trials per config per variant
            variants, expected, cat_values = generate_prompt_variants(
                num_keys, num_updates, condition, seed
            )

            for vname, vdata in variants.items():
                if "messages" in vdata:
                    prompt_text = tokenizer.apply_chat_template(
                        vdata["messages"], tokenize=False, add_generation_prompt=True
                    )
                else:
                    prompt_text = vdata["raw_prompt"]

                answer = run_one(model, tokenizer, prompt_text, device)
                answer_clean = answer.lower().strip().split()[0] if answer.strip() else ""

                correct = answer_clean == expected.lower()
                is_value = answer_clean in [v.lower() for v in cat_values]
                is_garbage = not is_value

                if vname not in variant_scores:
                    variant_scores[vname] = {"correct": 0, "value_hit": 0, "garbage": 0, "total": 0, "examples": []}

                variant_scores[vname]["total"] += 1
                if correct:
                    variant_scores[vname]["correct"] += 1
                if is_value:
                    variant_scores[vname]["value_hit"] += 1
                else:
                    variant_scores[vname]["garbage"] += 1
                    if len(variant_scores[vname]["examples"]) < 3:
                        variant_scores[vname]["examples"].append(f"exp='{expected}' got='{answer[:60]}'")

        for vname, scores in variant_scores.items():
            n = scores["total"]
            acc = scores["correct"] / n if n else 0
            garb = scores["garbage"] / n if n else 0
            val_hit = scores["value_hit"] / n if n else 0
            print(f"  {vname:25s}: acc={acc:.0%}  value_hit={val_hit:.0%}  garbage={garb:.0%}  ({n} trials)")
            for ex in scores["examples"]:
                print(f"    garbage ex: {ex}")

        results[config_key] = variant_scores

    # Save
    out_path = _SCRIPT_DIR / "results" / "diagnostic_0.5B.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable
    serializable = {}
    for ck, vs in results.items():
        serializable[ck] = {}
        for vn, sc in vs.items():
            serializable[ck][vn] = {k: v for k, v in sc.items()}

    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
