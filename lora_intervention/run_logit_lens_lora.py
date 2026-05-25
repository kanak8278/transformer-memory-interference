"""
Post-LoRA logit lens — run stage2_logit_lens on the merged
(Qwen2.5-3B-Instruct + LoRA) model.

Baseline: v3/results_vllm/logit_lens/Qwen2.5-3B-Instruct/stage2_logit_lens_20260409_054632.json

Saves to: v3/results_vllm/logit_lens/Qwen2.5-3B-Instruct-LoRA/stage2_logit_lens_<ts>.json

Usage:
    .venv/bin/python lora_intervention/run_logit_lens_lora.py \
        --merged_path lora_intervention/checkpoints/merged \
        --point 2,5 --trials 100
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

from experiments.stage2_logit_lens import run_stage2
from mechanistic_probing_v2.core.model_loader import detect_device, ModelInfo

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--merged_path", required=True)
    p.add_argument("--point", default="2,5", help="Single point 'k,u' OR ignored if --points used")
    p.add_argument("--points", default=None,
                   help="Multi-point spec like '2,5;2,10;2,50' to match baseline")
    p.add_argument("--trials", type=int, default=100)
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
    if device == "cuda":
        torch.cuda.empty_cache()

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
    if args.points:
        points = [tuple(map(int, p.split(","))) for p in args.points.split(";")]
    else:
        points = [tuple(map(int, args.point.split(",")))]
    device = args.device or detect_device()[0]
    dtype = torch.float32 if device in ("cpu", "mps") else torch.float16

    model, tokenizer, info = load_merged_into_tl(args.merged_path, device, dtype)

    # Override the model_name embedded in info so the saved file is tagged as the LoRA variant
    info_lora = info._replace(name=args.out_name) if hasattr(info, "_replace") else info

    config = {
        "model": args.out_name,
        "points": points,
        "trials": args.trials,
        "gpu": None,
        "n_ctx": model.cfg.n_ctx,
    }

    results = run_stage2(config, model=model, tokenizer=tokenizer, info=info_lora)
    print(f"\n✓ Done. Results dict keys: {list(results.keys())}")


if __name__ == "__main__":
    main()
