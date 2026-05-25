"""
Post-LoRA stage-2 logit lens — HF-only path.

For each trial at given K/N/condition, runs a forward pass with
output_hidden_states=True, then for every layer projects the
last-token residual through the unembedding (model.lm_head) to compute
P(v_first), P(v_last), and per-position P(v_i) across all values
in the test category.

Matches the metrics of v3/scripts/experiments/stage2_logit_lens.py
(the TL-based stage-2) without the TL dependency. Output JSON layout
is compatible with the existing reader code.

Usage:
    python lora_intervention/run_logit_lens_lora_hf.py \\
        --merged_path lora_intervention/checkpoints/gemma_merged \\
        --base google/gemma-3-4b-it \\
        --out_name gemma-3-4b-it-LoRA \\
        --points "2,5;2,10;2,50" --trials 100
"""
import argparse
import importlib.metadata as _imeta
import json
import os
import random
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_orig = _imeta.version
def _safe(pkg):
    try:
        v = _orig(pkg); return v if v is not None else "0.0.0"
    except Exception:
        return "0.0.0"
_imeta.version = _safe

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "v3" / "scripts"))
sys.path.insert(0, str(_ROOT))

from mechanistic_probing_v2.core.dataset_configs import (
    format_for_chat, get_value_pool, get_eligible_categories,
    generate_values_for_trial, FIXED_COMPLETION_DEMOS,
)
from mechanistic_probing_v2.core.model_loader import verify_single_token, is_instruct_model

BASE_MODEL_DEFAULT = "Qwen/Qwen2.5-3B-Instruct"
DATASET_TYPE = "ARBITRARY_SINGLE"


def _model_class_for(model_id: str):
    if "gemma-3" in model_id.lower():
        try:
            from transformers import Gemma3ForCausalLM
            return Gemma3ForCausalLM
        except ImportError:
            pass
    return AutoModelForCausalLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--merged_path", required=True)
    p.add_argument("--base", default=BASE_MODEL_DEFAULT)
    p.add_argument("--points", default="2,5",
                   help="Semicolon-separated 'K,N' points, e.g. '2,5;2,10;2,50'")
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--out_name", default="Qwen2.5-3B-Instruct-LoRA")
    return p.parse_args()


def load_model(merged_path, base_model, device, dtype):
    print(f"Loading merged HF model from {merged_path} ({device}, {dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True)
    model_cls = _model_class_for(base_model)
    model = model_cls.from_pretrained(
        merged_path, dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True, device_map=device,
    )
    model.eval()
    return model, tokenizer


def generate_trial(num_keys, num_updates, condition, seed, value_pool, tokenizer, model_name):
    """Single trial. condition in {'RI', 'PI'}. Returns dict with prompt, all_values, expected, etc."""
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
        raw = (
            f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
            f"{stream}\n\n"
            f"What was the {query_word} value of {test_category}?"
        )
        formatted = format_for_chat(raw, tokenizer, model_name=model_name)
    else:
        formatted = f"{FIXED_COMPLETION_DEMOS}{stream}\nThe {query_word} value of {test_category} was:"

    return {
        "prompt": formatted, "condition": condition, "expected": expected,
        "all_values": cat_values, "test_category": test_category,
        "num_keys": num_keys, "num_updates": num_updates, "seed": seed,
    }


def value_token_ids(value, tokenizer):
    """Get candidate token ids for a value (bare + space-prefixed; first token if multi-token)."""
    tids = set()
    sp = tokenizer.encode(f" {value}", add_special_tokens=False)
    if sp:
        tids.add(sp[0])
    bare = tokenizer.encode(value, add_special_tokens=False)
    if bare:
        tids.add(bare[0])
    return list(tids)


@torch.no_grad()
def run_logit_lens_cell(model, tokenizer, base_model, nk, nu, trials, device):
    """For one (K, N) cell, run RI + PI trials and aggregate per-layer per-value probabilities.

    Returns dict keyed by condition with per-layer mean P(v_first), P(v_last),
    P(v_i) per intermediate position, plus per-trial behavioral correctness.
    """
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    # lm_head: (vocab, hidden) — we project resid through it
    lm_head = model.get_output_embeddings()  # typically Linear(hidden, vocab)
    # Final RMSNorm — fold it in for the intermediate-layer logit lens.
    # Without this, intermediate residual streams are unnormalized and
    # softmax over lm_head(resid) gives near-uniform distributions.
    final_norm = None
    for attr in ("norm", "final_layer_norm"):
        # Gemma3 keeps it at `model.model.norm`; Qwen at `model.model.norm`; etc.
        cand = getattr(getattr(model, "model", None), attr, None) if hasattr(model, "model") else None
        if cand is not None:
            final_norm = cand
            break
    print(f"final_norm: {type(final_norm).__name__ if final_norm is not None else 'NONE — using raw projections'}")

    result = {}

    for condition in ["RI", "PI"]:
        # accumulate per-layer prob trajectories
        p_first_per_layer = np.zeros((n_layers,), dtype=np.float64)
        p_last_per_layer  = np.zeros((n_layers,), dtype=np.float64)
        # per-position averages — index by value position 0..N-1
        p_pos_per_layer   = np.zeros((nu, n_layers), dtype=np.float64)
        pos_count         = np.zeros((nu,), dtype=np.int64)
        n_correct = 0
        n_total = 0
        # Per-trial trajectories for bootstrap CIs (P(v_first/v_last) per layer)
        per_trial_first = []  # list of [n_layers] arrays
        per_trial_last  = []

        for t_idx in range(trials):
            seed = hash((nk, nu, condition, t_idx, "logit_lens")) % (2**31)
            trial = generate_trial(nk, nu, condition, seed, None, tokenizer, base_model)
            all_values = trial["all_values"]
            n_vals = len(all_values)
            if n_vals < 2:
                continue

            # Get candidate token IDs per value (use first token for multi-token values)
            value_tids = [value_token_ids(v, tokenizer) for v in all_values]

            inputs = tokenizer(trial["prompt"], return_tensors="pt").to(device)
            out = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)

            # hidden_states tuple: (n_layers + 1) of [1, T, H]; index L+1 = after layer L
            trial_first = np.zeros(n_layers, dtype=np.float64)
            trial_last  = np.zeros(n_layers, dtype=np.float64)
            for L in range(n_layers):
                resid = out.hidden_states[L + 1][0, -1, :]  # [H]
                # Apply final RMSNorm to intermediate layers ONLY. HF's
                # last hidden_state is already post-final-norm; applying
                # final_norm again zeros out the projection.
                if final_norm is not None and L < n_layers - 1:
                    resid = final_norm(resid.unsqueeze(0)).squeeze(0)
                layer_logits = lm_head(resid)  # [vocab]
                probs = torch.softmax(layer_logits.float(), dim=-1)
                # P(v_i) = max prob across that value's candidate tids
                for vi, tids in enumerate(value_tids):
                    if not tids:
                        continue
                    p_vi = max(probs[t].item() for t in tids)
                    p_pos_per_layer[vi, L] += p_vi
                    if vi == 0:
                        p_first_per_layer[L] += p_vi
                        trial_first[L] = p_vi
                    if vi == n_vals - 1:
                        p_last_per_layer[L] += p_vi
                        trial_last[L] = p_vi
            per_trial_first.append(trial_first.tolist())
            per_trial_last.append(trial_last.tolist())

            # behavioral correctness from final logits
            final_logits = out.logits[0, -1, :]
            pred_tid = int(final_logits.argmax().item())
            pred_text = tokenizer.decode([pred_tid]).strip().lower()
            if pred_text == trial["expected"].lower().strip():
                n_correct += 1
            n_total += 1

            for vi in range(n_vals):
                pos_count[vi] += 1

            del out, inputs
            if (t_idx + 1) % 25 == 0:
                if device == "cuda":
                    torch.cuda.empty_cache()

        # averaging
        if n_total > 0:
            p_first_per_layer /= n_total
            p_last_per_layer  /= n_total
        for vi in range(nu):
            if pos_count[vi] > 0:
                p_pos_per_layer[vi] /= pos_count[vi]

        result[condition] = {
            "n_trials": n_total,
            "behavioral_accuracy": (n_correct / n_total) if n_total else 0.0,
            "p_first_per_layer": p_first_per_layer.tolist(),
            "p_last_per_layer":  p_last_per_layer.tolist(),
            "p_pos_per_layer":   p_pos_per_layer.tolist(),  # [N, n_layers]
            # Per-trial trajectories for bootstrap CIs — [n_trials, n_layers]
            "per_trial_p_first_per_layer": per_trial_first,
            "per_trial_p_last_per_layer":  per_trial_last,
        }
    return result


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    points = []
    for p in args.points.split(";"):
        k, n = map(int, p.strip().split(","))
        points.append((k, n))
    print(f"Points: {points}, trials: {args.trials}")

    model, tokenizer = load_model(args.merged_path, args.base, args.device, dtype)
    n_layers = model.config.num_hidden_layers
    print(f"Model: {n_layers} layers, hidden={model.config.hidden_size}")

    cells = {}
    t_total = time.time()
    for (nk, nu) in points:
        print(f"\n=== Cell K={nk}, N={nu}, trials={args.trials} ===")
        t0 = time.time()
        cells[f"{nk}_{nu}"] = run_logit_lens_cell(
            model, tokenizer, args.base, nk, nu, args.trials, args.device,
        )
        t_cell = time.time() - t0
        ri = cells[f"{nk}_{nu}"]["RI"]
        pi = cells[f"{nk}_{nu}"]["PI"]
        # print P(v_last) at last layer for PI (key headline metric)
        print(f"  Behavioral: RI={ri['behavioral_accuracy']:.0%}, PI={pi['behavioral_accuracy']:.0%}")
        print(f"  PI P(v_last) at L{n_layers-1}: {pi['p_last_per_layer'][-1]:.3f}")
        print(f"  RI P(v_first) at L{n_layers-1}: {ri['p_first_per_layer'][-1]:.3f}")
        print(f"  Cell elapsed: {t_cell:.1f}s")

    save_dir = _ROOT / "v3" / "results_vllm" / "logit_lens" / args.out_name
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    save_path = save_dir / f"stage2_logit_lens_{ts}.json"

    output = {
        "model": args.out_name,
        "base_model": args.base,
        "lora_merged_from": str(args.merged_path),
        "n_layers": n_layers,
        "trials": args.trials,
        "points": [{"keys": k, "updates": n} for k, n in points],
        "cells": cells,
        "elapsed_sec": time.time() - t_total,
        "backend": "hf_hidden_states",
        "dtype": str(dtype),
    }
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
