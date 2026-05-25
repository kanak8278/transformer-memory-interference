"""
Post-LoRA probing classifier — HF-only path.

Avoids transformer_lens entirely (which OOMs Gemma-3-4b on a 23 GB L4 even
with from_pretrained_no_processing because it duplicates weights into its
layer-wise structures during construction). Uses HF's
`output_hidden_states=True` for residual-stream collection at the last
token position; passes results to the unchanged sklearn `train_probes`.

Output format and downstream consumers identical to run_probing_lora.py:
    v3/results_vllm/probing/probing_<out_name>_<K>k_<N>u.json

Usage:
    python lora_intervention/run_probing_lora_hf.py \\
        --merged_path lora_intervention/checkpoints/gemma_merged \\
        --base google/gemma-3-4b-it \\
        --out_name gemma-3-4b-it-LoRA \\
        --point 2,5 --trials 200 --device cuda --dtype bfloat16
"""
import argparse
import importlib.metadata as _imeta
import json
import os
import random
import sys
import time
from pathlib import Path

# Match the lazy-import patches used elsewhere
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

from experiments.probing_classifier import (
    train_probes, SYSTEM_PROMPT, DATASET_TYPE, generate_trial,
)
from mechanistic_probing_v2.core.model_loader import verify_single_token
from mechanistic_probing_v2.core.dataset_configs import (
    get_value_pool, get_eligible_categories,
)

BASE_MODEL_DEFAULT = "Qwen/Qwen2.5-3B-Instruct"


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
    p.add_argument("--point", default="2,5")
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--out_name", default="Qwen2.5-3B-Instruct-LoRA")
    return p.parse_args()


def load_model(merged_path, base_model, device, dtype):
    print(f"Loading merged HF model from {merged_path} (device={device}, dtype={dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True)
    model_cls = _model_class_for(base_model)
    model = model_cls.from_pretrained(
        merged_path, dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True, device_map=device,
    )
    model.eval()
    return model, tokenizer


@torch.no_grad()
def collect_representations_hf(model, tokenizer, value_pool, model_name,
                                num_keys, num_updates, trials, value_to_tid,
                                device):
    """Collect residual stream representations using HF's output_hidden_states.

    HF hidden_states tuple: (n_layers + 1) tensors of shape [B, T, H], where
    index 0 is the embedding output and index L is the output AFTER layer L
    (i.e., the residual stream after that layer's writes). This matches TL's
    `resid_post` semantics — same data, different access path.
    """
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    d_model = cfg.hidden_size

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

            inputs = tokenizer(trial["prompt"], return_tensors="pt").to(device)

            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

            hidden = outputs.hidden_states  # tuple of (n_layers + 1)
            # Skip index 0 (embedding); store layers 1..n_layers in 0-indexed slots
            reps = np.zeros((n_layers, d_model), dtype=np.float32)
            for L in range(n_layers):
                reps[L] = hidden[L + 1][0, -1, :].float().cpu().numpy()

            # Prediction from logits at last position
            logits = outputs.logits[0, -1, :]
            pred_tid = int(logits.argmax().item())
            pred_text = tokenizer.decode([pred_tid]).strip().lower()
            correct = pred_text == trial["expected"].lower()

            # Find which value index the model predicted (lenient match)
            pred_value_idx = None
            for vi, val in enumerate(all_values):
                if val.lower() in pred_text or pred_text.startswith(val.lower()):
                    pred_value_idx = vi
                    break

            data[condition].append({
                "reps": reps,
                "expected_idx": expected_idx,
                "pred_value_idx": pred_value_idx,
                "correct": correct,
                "n_values": n_values,
            })

            del outputs, hidden, logits, inputs
            if (t_idx + 1) % 50 == 0:
                acc = sum(d["correct"] for d in data[condition]) / len(data[condition])
                print(f"    [{t_idx + 1}/{trials}] acc={acc:.0%}")
                if device == "cuda":
                    torch.cuda.empty_cache()

    return data


def main():
    args = parse_args()
    nk, nu = map(int, args.point.split(","))
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    print(f"Device: {args.device}  |  dtype: {dtype}")

    model, tokenizer = load_model(args.merged_path, args.base, args.device, dtype)
    n_layers = model.config.num_hidden_layers
    print(f"Model: {model.config.num_hidden_layers} layers, hidden={model.config.hidden_size}, "
          f"n_heads={getattr(model.config, 'num_attention_heads', '?')}")

    candidate_pool = get_value_pool(DATASET_TYPE)
    value_to_tid = verify_single_token(tokenizer, values=candidate_pool)
    value_pool = list(value_to_tid.keys())
    print(f"Value pool: {len(value_pool)} verified single-token values")

    print(f"\nOperating point: {nk}k_{nu}u  |  {args.trials} trials per condition\n")

    t0 = time.time()
    data = collect_representations_hf(model, tokenizer, value_pool, args.base,
                                       nk, nu, args.trials, value_to_tid,
                                       args.device)
    probe_results = train_probes(data, n_layers)
    elapsed = time.time() - t0

    print(f"\n{'='*60}\nPROBING RESULTS — {args.out_name}  (post-LoRA)\n"
          f"  Point: {nk}k_{nu}u, {args.trials} trials\n{'='*60}")

    cond_probe = probe_results.get("condition_probe", {})
    print("\nCondition discrimination (RI vs PI) — last 8 layers:")
    for L in range(max(0, n_layers - 8), n_layers):
        acc = cond_probe.get(L, {}).get("accuracy", 0.5)
        print(f"  L{L:>4}: {acc:.0%}")
    for condition in ["RI", "PI"]:
        probe = probe_results.get(f"{condition}_correct_probe", {})
        if probe:
            print(f"\n{condition} correct vs incorrect — last 8 layers:")
            for L in range(max(0, n_layers - 8), n_layers):
                acc = probe.get(L, {}).get("accuracy", 0.5)
                print(f"  L{L:>4}: {acc:.0%}")
    print(f"\n  Elapsed: {elapsed:.1f}s")

    save_dir = _ROOT / "v3" / "results_vllm" / "probing"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"probing_{args.out_name}_{nk}k_{nu}u.json"

    output = {
        "model": args.out_name,
        "base_model": args.base,
        "lora_merged_from": str(args.merged_path),
        "point": {"keys": nk, "updates": nu},
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
        "device": args.device,
        "dtype": str(dtype),
        "backend": "hf_hidden_states",
    }
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()
