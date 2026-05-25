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

# The merged model is architecturally the base (LoRA folded into weights).
# This is the model_name string used for chat-template formatting in
# probing_classifier and for transformer_lens architecture loading.
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
    p.add_argument("--merged_path", required=True,
                   help="Path to merged (base + LoRA) HF model dir")
    p.add_argument("--base", default=BASE_MODEL_DEFAULT,
                   help=f"HF model id of base architecture (default: {BASE_MODEL_DEFAULT})")
    p.add_argument("--point", default="2,5", help="Operating point 'keys,updates'")
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--device", default=None, help="mps | cpu | cuda. Default: auto")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                   help="float32 is default; bf16 needed for K=15/N=20 to fit in 36GB MPS")
    p.add_argument("--out_name", default="Qwen2.5-3B-Instruct-LoRA",
                   help="model_short_name used in output filename")
    return p.parse_args()


def load_merged_into_tl(merged_path, base_model, device, dtype):
    """Load merged HF model and inject into HookedTransformer.

    Two-step load to avoid OOM on small GPUs:
    1. Load HF model on CPU and wrap TL on CPU (~16GB host RAM for Gemma-3-4b bf16)
    2. Move TL model to target device (8GB GPU); free hf_model first
    """
    print(f"Loading merged HF model from {merged_path} (CPU)...")
    tokenizer = AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True)
    model_cls = _model_class_for(base_model)
    hf_model = model_cls.from_pretrained(
        merged_path, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True,
    )
    hf_model.eval()

    use_no_processing = dtype in (torch.float16, torch.bfloat16)
    if use_no_processing:
        print(f"Wrapping in HookedTransformer.from_pretrained_no_processing (base={base_model}, CPU, dtype={dtype})...")
        model = HookedTransformer.from_pretrained_no_processing(
            base_model, hf_model=hf_model, tokenizer=tokenizer,
            device="cpu", dtype=dtype,
        )
    else:
        print(f"Wrapping in HookedTransformer.from_pretrained (base={base_model}, CPU, dtype={dtype})...")
        model = HookedTransformer.from_pretrained(
            base_model, hf_model=hf_model, tokenizer=tokenizer,
            device="cpu", dtype=dtype,
            fold_ln=True, center_writing_weights=True, center_unembed=True,
        )
    model.eval()

    # Free HF reference BEFORE moving TL to GPU — keeps peak GPU memory low
    del hf_model
    import gc; gc.collect()
    if device != "cpu":
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"Moving HookedTransformer to {device}...")
        model = model.to(device)
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

    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif device in ("cpu", "mps"):
        dtype = torch.float32
    else:
        dtype = torch.float16
    print(f"Device: {device}  |  dtype: {dtype}")

    model, tokenizer = load_merged_into_tl(args.merged_path, args.base, device, dtype)

    candidate_pool = get_value_pool("ARBITRARY_SINGLE")
    value_to_tid = verify_single_token(tokenizer, values=candidate_pool)
    value_pool = list(value_to_tid.keys())
    print(f"Value pool: {len(value_pool)} verified tokens")

    print(f"\nOperating point: {nk}k_{nu}u  |  {args.trials} trials per condition\n")

    t0 = time.time()
    data = collect_representations(model, tokenizer, value_pool, args.base,
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
        "device": device,
        "dtype": str(dtype),
    }
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()
