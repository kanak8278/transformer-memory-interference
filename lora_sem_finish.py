"""Finish the two missing LoRA cells (K=15,N=30 and K=20,N=30) at 10 trials."""

import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lora_sem_validation import (
    MODEL_ID, ADAPTER, build_prompt, is_correct, generate_one, DEVICE,
)

CELLS = [(15, 30), (20, 30)]
TRIALS = 10  # per condition


def main():
    print(f"Device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    print("Loading base + LoRA...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()

    results = {}
    t0 = time.time()
    for K, N in CELLS:
        cell_t0 = time.time()
        cell = {"FVQ": [], "CVQ": []}
        for cond in ("FVQ", "CVQ"):
            for t in range(TRIALS):
                seed = hash(("lora", K, N, cond, t)) % (2**31)
                prompt, expected = build_prompt(tokenizer, K, N, cond, seed)
                if prompt is None: continue
                try:
                    pred = generate_one(model, tokenizer, prompt)
                except Exception as e:
                    print(f"    ERR K={K} N={N} {cond} t={t}: {e}", flush=True)
                    continue
                cell[cond].append(1 if is_correct(pred, expected) else 0)
        ri = sum(cell["FVQ"]); pi = sum(cell["CVQ"])
        n_ri = len(cell["FVQ"]); n_pi = len(cell["CVQ"])
        print(
            f"  [lora] K={K:>2} N={N:>2}  "
            f"FVQ {ri}/{n_ri} = {ri/max(n_ri,1):.0%}  "
            f"CVQ {pi}/{n_pi} = {pi/max(n_pi,1):.0%}  "
            f"gap {(ri-pi)/max(n_ri,1):+.0%}  "
            f"({time.time()-cell_t0:.0f}s)",
            flush=True,
        )
        results[f"{K}_{N}"] = {
            "num_keys": K, "num_updates": N,
            "fvq_correct": ri, "fvq_n": n_ri, "fvq_acc": ri / max(n_ri, 1),
            "cvq_correct": pi, "cvq_n": n_pi, "cvq_acc": pi / max(n_pi, 1),
            "gap": (ri - pi) / max(n_ri, 1),
        }
    print(f"\nTotal: {time.time()-t0:.0f}s")

    Path("lora_sem_finish_results.json").write_text(json.dumps({
        "model": MODEL_ID, "adapter": ADAPTER, "dataset": "SEMANTIC_MULTI",
        "cells": [list(c) for c in CELLS], "trials_per_condition": TRIALS,
        "lora": results,
    }, indent=2))
    print("Saved → lora_sem_finish_results.json")


if __name__ == "__main__":
    main()
