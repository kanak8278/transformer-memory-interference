"""
Baseline probing on Qwen2.5-3B-Instruct (no LoRA) at any operating point,
forcing bf16 on MPS to fit K=15/N=20 in 36 GB unified memory.

Mirrors run_probing_lora.py but loads the unmodified base model.

Usage:
    .venv/bin/python lora_intervention/run_probing_baseline.py \
        --point 15,20 --trials 200
"""
import argparse
import importlib.metadata as _imeta
import json
import sys
import time
from pathlib import Path

_orig = _imeta.version
def _safe(pkg):
    try:
        v = _orig(pkg); return v if v is not None else "0.0.0"
    except Exception:
        return "0.0.0"
_imeta.version = _safe

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "v3" / "scripts"))
sys.path.insert(0, str(_ROOT))

from experiments.probing_classifier import collect_representations, train_probes
from mechanistic_probing_v2.core.model_loader import detect_device, verify_single_token
from mechanistic_probing_v2.core.dataset_configs import get_value_pool

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--point", default="15,20")
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--out_name", default="Qwen2.5-3B-Instruct")
    return p.parse_args()


def load_base_into_tl(device, dtype):
    print(f"Loading base {BASE_MODEL} on {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True,
    )
    hf_model.eval()
    print(f"Wrapping in HookedTransformer...")
    model = HookedTransformer.from_pretrained(
        BASE_MODEL, hf_model=hf_model, tokenizer=tokenizer,
        device=device, dtype=dtype,
        fold_ln=True, center_writing_weights=True, center_unembed=True,
    )
    model.eval()
    del hf_model
    return model, tokenizer


def main():
    args = parse_args()
    nk, nu = map(int, args.point.split(","))
    device = args.device or detect_device()[0]
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    print(f"Device: {device}  dtype: {dtype}\nPoint: {nk}k_{nu}u  Trials: {args.trials}\n")

    model, tokenizer = load_base_into_tl(device, dtype)

    candidate_pool = get_value_pool("ARBITRARY_SINGLE")
    value_to_tid = verify_single_token(tokenizer, values=candidate_pool)
    value_pool = list(value_to_tid.keys())
    print(f"Value pool: {len(value_pool)} verified tokens\n")

    t0 = time.time()
    data = collect_representations(model, tokenizer, value_pool, BASE_MODEL,
                                   nk, nu, args.trials, value_to_tid)
    probe_results = train_probes(data, model.cfg.n_layers)
    elapsed = time.time() - t0

    save_dir = _ROOT / "v3" / "results_vllm" / "probing"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"probing_{args.out_name}_{nk}k_{nu}u.json"

    output = {
        "model": args.out_name,
        "point": {"keys": nk, "updates": nu},
        "trials": args.trials,
        "n_layers": model.cfg.n_layers,
        "probe_results": probe_results,
        "behavioral": {
            "RI": {"accuracy": sum(d["correct"] for d in data["RI"]) / len(data["RI"]),
                   "n": len(data["RI"])},
            "PI": {"accuracy": sum(d["correct"] for d in data["PI"]) / len(data["PI"]),
                   "n": len(data["PI"])},
        },
        "elapsed_sec": elapsed,
        "device": device, "dtype": str(dtype),
    }
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Saved: {save_path}  (elapsed {elapsed:.0f}s)")


if __name__ == "__main__":
    main()
