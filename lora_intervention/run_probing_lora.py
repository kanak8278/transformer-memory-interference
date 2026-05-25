"""
Post-LoRA probing classifier — run probing_classifier on the merged
(Qwen2.5-3B-Instruct + LoRA) model and save alongside the baseline.

Baseline file: v3/results_vllm/probing/probing_Qwen2.5-3B-Instruct_2k_5u.json
This script writes:  v3/results_vllm/probing/probing_Qwen2.5-3B-Instruct-LoRA_2k_5u.json

Usage:
    .venv/bin/python lora_intervention/run_probing_lora.py \
        --merged_path lora_intervention/checkpoints/merged \
        --point 2,5 --trials 200
"""
import argparse
import importlib.metadata as _imeta
import json
import sys
import time
from pathlib import Path

# Patch missing version lookups
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

# Import probing functions from the shared script
from experiments.probing_classifier import (
    collect_representations, train_probes,
)
from mechanistic_probing_v2.core.model_loader import detect_device
from mechanistic_probing_v2.core.dataset_configs import (
    get_value_pool,
)
from mechanistic_probing_v2.core.model_loader import verify_single_token

# We pretend the merged model is still Qwen2.5-3B-Instruct (architecturally it is —
# LoRA was merged into the weights). This is the model_name string used for
# chat-template formatting in probing_classifier.
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--merged_path", required=True,
                   help="Path to merged (base + LoRA) HF model dir")
    p.add_argument("--point", default="2,5", help="Operating point 'keys,updates'")
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--device", default=None, help="mps | cpu | cuda. Default: auto")
    p.add_argument("--out_name", default="Qwen2.5-3B-Instruct-LoRA",
                   help="model_short_name used in output filename")
    return p.parse_args()


def load_merged_into_tl(merged_path, device, dtype):
    """Load merged HF model and inject into HookedTransformer."""
    print(f"Loading merged HF model from {merged_path}...")
    tokenizer = AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        merged_path, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True,
    )
    hf_model.eval()

    print(f"Wrapping in HookedTransformer (base name={BASE_MODEL}, device={device}, dtype={dtype})...")
    model = HookedTransformer.from_pretrained(
        BASE_MODEL, hf_model=hf_model, tokenizer=tokenizer,
        device=device, dtype=dtype,
        fold_ln=True, center_writing_weights=True, center_unembed=True,
    )
    model.eval()

    # Free the HF reference once HT has copied weights
    del hf_model
    if device == "cuda":
        torch.cuda.empty_cache()
    return model, tokenizer


def main():
    args = parse_args()
    nk, nu = map(int, args.point.split(","))

    if args.device:
        device = args.device
    else:
        device, _ = detect_device()

    dtype = torch.float32 if device in ("cpu", "mps") else torch.float16
    print(f"Device: {device}  |  dtype: {dtype}")

    model, tokenizer = load_merged_into_tl(args.merged_path, device, dtype)

    candidate_pool = get_value_pool("ARBITRARY_SINGLE")
    value_to_tid = verify_single_token(tokenizer, values=candidate_pool)
    value_pool = list(value_to_tid.keys())
    print(f"Value pool: {len(value_pool)} verified tokens")

    print(f"\nOperating point: {nk}k_{nu}u  |  {args.trials} trials per condition\n")

    t0 = time.time()
    data = collect_representations(model, tokenizer, value_pool, BASE_MODEL,
                                   nk, nu, args.trials, value_to_tid)
    probe_results = train_probes(data, model.cfg.n_layers)
    elapsed = time.time() - t0

    n_layers = model.cfg.n_layers
    print(f"\n{'='*60}\nPROBING RESULTS — {args.out_name}  (post-LoRA)\n"
          f"  Point: {nk}k_{nu}u, {args.trials} trials\n{'='*60}")

    cond_probe = probe_results.get("condition_probe", {})
    print("\nCondition discrimination (RI vs PI) — last 6 layers:")
    for L in range(max(0, n_layers - 6), n_layers):
        acc = cond_probe.get(L, {}).get("accuracy", 0.5)
        print(f"  L{L:>4}: {acc:.0%}")
    for condition in ["RI", "PI"]:
        probe = probe_results.get(f"{condition}_correct_probe", {})
        if probe:
            print(f"\n{condition} correct vs incorrect — last 6 layers:")
            for L in range(max(0, n_layers - 6), n_layers):
                acc = probe.get(L, {}).get("accuracy", 0.5)
                print(f"  L{L:>4}: {acc:.0%}")
    print(f"\n  Elapsed: {elapsed:.1f}s")

    save_dir = _ROOT / "v3" / "results_vllm" / "probing"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"probing_{args.out_name}_{nk}k_{nu}u.json"

    output = {
        "model": args.out_name,
        "base_model": BASE_MODEL,
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
        "device": device,
        "dtype": str(dtype),
    }
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()
