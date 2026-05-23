"""
Local SEM OOD validation of the main LoRA adapter on a few held-out cells.
Compares base Qwen2.5-3B-Instruct vs base+LoRA on SEMANTIC_MULTI.

Runs on MPS (Apple Silicon).  Small grid + small trial count for feasibility.
"""

import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mechanistic_probing_v2.core.dataset_configs import (
    get_eligible_categories,
    generate_values_for_trial,
)

MODEL_ID    = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER     = "/tmp/lora_intervention/checkpoints/main/final"
OUT_PATH    = "lora_sem_validation_results.json"
DEVICE      = "mps" if torch.backends.mps.is_available() else "cpu"

# Held-out cells on SEMANTIC_MULTI.  K and N values are absent from LoRA
# training (training was K in {2,3,5,10} x N in {5,10,15,20}).
SEM_CELLS = [
    (7, 20),   # mild stress
    (15, 20),  # reversal entry in baseline
    (15, 30),  # solid reversal in baseline
    (20, 30),  # deep reversal in baseline
]
TRIALS_PER_CELL = 20  # 20 FVQ + 20 CVQ per cell

SYSTEM_PROMPT = (
    "You are a precise data extraction tool. "
    "Output ONLY a single word - the exact value requested. "
    "No other text, no explanation, no punctuation."
)


def shuffle_no_consecutive(items, rng):
    for _ in range(100):
        c = items.copy(); rng.shuffle(c)
        if all(c[i]["category"] != c[i - 1]["category"] for i in range(1, len(c))):
            return c
    result, remaining = [], items.copy(); rng.shuffle(remaining); last = None
    while remaining:
        valid = [i for i, x in enumerate(remaining) if x["category"] != last]
        if not valid:
            result.extend(remaining); break
        idx = rng.choice(valid)
        result.append(remaining.pop(idx)); last = result[-1]["category"]
    return result


def build_prompt(tokenizer, num_keys, num_updates, condition, seed):
    rng = random.Random(seed)
    eligible = get_eligible_categories("SEMANTIC_MULTI", min_values=num_updates)
    if num_keys > len(eligible):
        return None, None
    cats = rng.sample(eligible, num_keys)
    try:
        values_per_cat = generate_values_for_trial(
            "SEMANTIC_MULTI", cats, num_updates, rng
        )
    except Exception:
        return None, None
    test_cat = cats[seed % num_keys]
    items = [{"category": c, "value": v} for c in cats for v in values_per_cat[c]]
    items = shuffle_no_consecutive(items, rng)

    stream = "\n".join(f"{it['category']}: {it['value']}" for it in items)
    cat_values = [it["value"] for it in items if it["category"] == test_cat]
    expected = cat_values[0] if condition == "FVQ" else cat_values[-1]
    query_word = "first" if condition == "FVQ" else "last"
    user_text = (
        f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{stream}\n\n"
        f"What was the {query_word} value of {test_cat}?"
    )
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return prompt, expected


def is_correct(pred, expected):
    p = pred.lower().strip()
    e = expected.lower().strip()
    return p == e or e in p or p.startswith(e)


def generate_one(model, tokenizer, prompt, max_new=16):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(DEVICE)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=1.0,  # ignored when do_sample=False, but required to silence warning
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.0,
        )
    gen = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def eval_model(model, tokenizer, tag, cells, trials):
    results = {}
    t0 = time.time()
    for K, N in cells:
        cell_results = {"FVQ": [], "CVQ": []}
        cell_t0 = time.time()
        for cond in ("FVQ", "CVQ"):
            for t in range(trials):
                seed = hash((tag, K, N, cond, t)) % (2**31)
                prompt, expected = build_prompt(tokenizer, K, N, cond, seed)
                if prompt is None:
                    continue
                try:
                    pred = generate_one(model, tokenizer, prompt)
                except Exception as e:
                    print(f"    [{tag} K={K} N={N} {cond} t={t}] ERR {type(e).__name__}: {e}")
                    continue
                cell_results[cond].append((1 if is_correct(pred, expected) else 0, pred, expected))
        ri = sum(r[0] for r in cell_results["FVQ"])
        pi = sum(r[0] for r in cell_results["CVQ"])
        n_ri = len(cell_results["FVQ"]); n_pi = len(cell_results["CVQ"])
        print(
            f"  [{tag}] K={K:>2} N={N:>2}  "
            f"FVQ {ri}/{n_ri} = {ri / max(n_ri, 1):.0%}  "
            f"CVQ {pi}/{n_pi} = {pi / max(n_pi, 1):.0%}  "
            f"gap {(ri - pi) / max(n_ri, 1):+.0%}  "
            f"({time.time() - cell_t0:.0f}s)",
            flush=True,
        )
        results[f"{K}_{N}"] = {
            "num_keys": K, "num_updates": N,
            "fvq_correct": ri, "fvq_n": n_ri, "fvq_acc": ri / max(n_ri, 1),
            "cvq_correct": pi, "cvq_n": n_pi, "cvq_acc": pi / max(n_pi, 1),
            "gap": (ri - pi) / max(n_ri, 1),
            "fvq_details": [{"correct": c, "pred": p, "expected": e} for c, p, e in cell_results["FVQ"][:3]],
            "cvq_details": [{"correct": c, "pred": p, "expected": e} for c, p, e in cell_results["CVQ"][:3]],
        }
    print(f"  [{tag}] total time: {time.time() - t0:.0f}s")
    return results


def main():
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_ID}")
    print(f"Adapter: {ADAPTER}")
    print(f"Cells: {SEM_CELLS}")
    print(f"Trials per (cell, condition): {TRIALS_PER_CELL}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    print("Loading base model on MPS (bf16)...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True
    )
    base.eval()

    print("\n=== Baseline (no adapter) ===", flush=True)
    base_results = eval_model(base, tokenizer, "base", SEM_CELLS, TRIALS_PER_CELL)

    print("\nApplying LoRA adapter...", flush=True)
    lora = PeftModel.from_pretrained(base, ADAPTER)
    lora.eval()

    print("\n=== LoRA-tuned ===", flush=True)
    lora_results = eval_model(lora, tokenizer, "lora", SEM_CELLS, TRIALS_PER_CELL)

    summary = {
        "model": MODEL_ID,
        "adapter": ADAPTER,
        "device": DEVICE,
        "dataset": "SEMANTIC_MULTI",
        "cells": [list(c) for c in SEM_CELLS],
        "trials_per_condition": TRIALS_PER_CELL,
        "baseline": base_results,
        "lora": lora_results,
    }
    Path(OUT_PATH).write_text(json.dumps(summary, indent=2))
    print(f"\nSaved → {OUT_PATH}")

    print("\n=== Summary (FVQ / CVQ / gap) ===")
    print(f"{'cell':>10}  {'baseline':>22}  {'lora':>22}  verdict")
    for K, N in SEM_CELLS:
        b = base_results[f"{K}_{N}"]; l = lora_results[f"{K}_{N}"]
        b_str = f"{b['fvq_acc']:>4.0%} / {b['cvq_acc']:>4.0%} / {b['gap']:+5.0%}"
        l_str = f"{l['fvq_acc']:>4.0%} / {l['cvq_acc']:>4.0%} / {l['gap']:+5.0%}"
        verdict = "FIXED" if (l['cvq_acc'] > b['cvq_acc'] + 0.2 and l['fvq_acc'] > 0.6) else "..."
        print(f"  K={K:>2}, N={N:>2}  {b_str:>22}  {l_str:>22}  {verdict}")


if __name__ == "__main__":
    main()
