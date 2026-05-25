"""
Post-LoRA causal mechanism (Stage 3) — run stage3_causal on the merged
(Qwen2.5-3B-Instruct + LoRA) model.

Baseline: v3/results_vllm/causal/Qwen2.5-3B-Instruct/stage3_causal_20260409_055011.json
          (top heads pre-LoRA: L26H5, L26H7, L26H0, L27H3, L27H2, L30H3, L29H0, L29H4)

Saves to: v3/scripts/experiments/results/Qwen2.5-3B-Instruct-LoRA/stage3_causal_<ts>.json

Uses the post-LoRA stage2 logit lens (from our prior run) for operating-point
metadata. Stage 3 itself runs against the merged model.

Usage:
    .venv/bin/python lora_intervention/run_stage3_lora.py \
        --merged_path lora_intervention/checkpoints/merged \
        --stage2 v3/scripts/experiments/results/Qwen2.5-3B-Instruct-LoRA/stage2_logit_lens_20260524_194921.json \
        --point 2,5 --trials 50 --experiments 3A,3B,3C
"""
import argparse
import importlib.metadata as _imeta
import sys
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

from experiments.stage3_causal import run_stage3
from mechanistic_probing_v2.core.model_loader import detect_device, ModelInfo

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--merged_path", required=True)
    p.add_argument("--stage2", required=True,
                   help="Path to a stage2 logit lens JSON (provides operating-point metadata)")
    p.add_argument("--point", default=None,
                   help="Operating point 'k,u' (default: auto-pick from stage2)")
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--top_k", type=int, default=11,
                   help="Number of top heads to use in 3B/3C (default 11 to match the L27-L32 cluster)")
    p.add_argument("--experiments", default="3A,3B,3C")
    p.add_argument("--device", default=None)
    p.add_argument("--out_name", default="Qwen2.5-3B-Instruct-LoRA")
    return p.parse_args()


def load_merged_into_tl(merged_path, device, dtype):
    print(f"Loading merged HF model from {merged_path}...")
    tokenizer = AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        merged_path, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True,
    )
    hf_model.eval()
    print(f"Wrapping in HookedTransformer (device={device}, dtype={dtype})...")
    model = HookedTransformer.from_pretrained(
        BASE_MODEL, hf_model=hf_model, tokenizer=tokenizer,
        device=device, dtype=dtype,
        fold_ln=True, center_writing_weights=True, center_unembed=True,
    )
    model.eval()
    del hf_model
    info = ModelInfo(
        name=BASE_MODEL,
        n_layers=model.cfg.n_layers,
        n_heads=model.cfg.n_heads,
        d_model=model.cfg.d_model,
        n_ctx=model.cfg.n_ctx,
        device=str(device),
        device_name=str(device),
        backend="transformer_lens",
        dtype=str(model.cfg.dtype),
    )
    return model, tokenizer, info


def main():
    args = parse_args()
    device = args.device or detect_device()[0]
    dtype = torch.float32 if device in ("cpu", "mps") else torch.float16

    model, tokenizer, info = load_merged_into_tl(args.merged_path, device, dtype)
    info_lora = info._replace(name=args.out_name) if hasattr(info, "_replace") else info

    point = None
    if args.point:
        point = tuple(map(int, args.point.split(",")))

    config = {
        "model": args.out_name,
        "stage2_path": str(Path(args.stage2).resolve()),
        "point": point,
        "trials": args.trials,
        "top_k": args.top_k,
        "experiments": [e.strip() for e in args.experiments.split(",")],
        "gpu": None,
    }
    results = run_stage3(config, model=model, tokenizer=tokenizer, info=info_lora)
    print(f"\n✓ Done. Top-level keys: {list(results.keys())}")


if __name__ == "__main__":
    main()
