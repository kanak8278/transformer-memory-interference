"""
Post-LoRA Stage 3 — TARGETED variant.

Why this exists: stock Stage 3 starts with attribution patching (3A) to identify
top heads, then ablates them in 3B/3C. On the LoRA model 3A is degenerate
(gradient saturates at P(v_last)≈1.0, all attribution scores → 0).

This wrapper skips 3A and directly runs 3B + 3C using BASELINE's top heads as
the ablation set. The scientific question is:

   If we knock out the heads that baseline used to suppress v_last,
   does the LoRA model also lose its v_last (i.e. did LoRA refine the
   same circuitry) — or does it remain confident (i.e. LoRA built
   alternative paths)?

Baseline top heads (from
v3/results_vllm/causal/Qwen2.5-3B-Instruct/stage3_causal_20260409_055011.json,
top 8 by 3A attribution):
   L26H5, L26H7, L26H0, L27H3, L27H2, L30H3, L29H0, L29H4

Operating point: K=2, N=5  (matches baseline stage3 operating point).

Usage:
    .venv/bin/python lora_intervention/run_stage3_targeted_lora.py \
        --merged_path lora_intervention/checkpoints/merged \
        --stage2 v3/results_vllm/logit_lens/Qwen2.5-3B-Instruct/stage2_logit_lens_20260409_054632.json \
        --trials 50
"""
import argparse
import importlib.metadata as _imeta
import json
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

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

from experiments.stage3_causal import (
    analyze_stage2, run_targeted_patching, run_ablation_logit_lens,
    save_results,
)
from mechanistic_probing_v2.core.model_loader import detect_device, ModelInfo
from mechanistic_probing_v2.core.dataset_configs import get_value_pool
from mechanistic_probing_v2.core.model_loader import verify_single_token

BASE_MODEL_DEFAULT = "Qwen/Qwen2.5-3B-Instruct"

# Default Qwen suppressor heads. Source:
# v3/results_vllm/causal/Qwen2.5-3B-Instruct/stage3_causal_20260409_055011.json
# field experiments.3C.heads_ablated
QWEN_baseline_heads = [
    {"layer": 26, "head": 3, "label": "L26H3"},
    {"layer": 27, "head": 3, "label": "L27H3"},
    {"layer": 30, "head": 3, "label": "L30H3"},
    {"layer": 29, "head": 3, "label": "L29H3"},
    {"layer": 29, "head": 4, "label": "L29H4"},
]

# Gemma-3-4b-it suppressor heads identified by baseline Stage 3.
# Source: v3/results_vllm/causal/gemma-3-4b-it/stage3_causal_20260409_150736.json
# field experiments.3C.heads_ablated
GEMMA_baseline_heads = [
    {"layer": 17, "head": 0, "label": "L17H0"},
    {"layer": 17, "head": 1, "label": "L17H1"},
    {"layer": 17, "head": 3, "label": "L17H3"},
    {"layer": 19, "head": 4, "label": "L19H4"},
    {"layer": 30, "head": 6, "label": "L30H6"},
]


def _model_class_for(model_id: str):
    if "gemma-3" in model_id.lower():
        try:
            from transformers import Gemma3ForCausalLM
            return Gemma3ForCausalLM
        except ImportError:
            pass
    return AutoModelForCausalLM


def _default_heads_for(base_model: str):
    if "gemma-3" in base_model.lower():
        return GEMMA_baseline_heads
    return QWEN_baseline_heads


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--merged_path", required=True)
    p.add_argument("--base", default=BASE_MODEL_DEFAULT,
                   help=f"HF model id of base architecture (default: {BASE_MODEL_DEFAULT})")
    p.add_argument("--stage2", required=True)
    p.add_argument("--point", default="2,5")
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--experiments", default="3B,3C")
    p.add_argument("--device", default=None)
    p.add_argument("--out_name", default="Qwen2.5-3B-Instruct-LoRA-targeted")
    return p.parse_args()


def load_merged_into_tl(merged_path, base_model, device, dtype):
    print(f"Loading merged HF model from {merged_path}...")
    tokenizer = AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True)
    model_cls = _model_class_for(base_model)
    hf_model = model_cls.from_pretrained(
        merged_path, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True,
    )
    hf_model.eval()
    print(f"Wrapping in HookedTransformer (base={base_model}, device={device}, dtype={dtype})...")
    model = HookedTransformer.from_pretrained(
        base_model, hf_model=hf_model, tokenizer=tokenizer,
        device=device, dtype=dtype,
        fold_ln=True, center_writing_weights=True, center_unembed=True,
    )
    model.eval()
    del hf_model
    info = ModelInfo(
        name=base_model,
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
    nk, nu = map(int, args.point.split(","))
    device = args.device or detect_device()[0]
    dtype = torch.float32 if device in ("cpu", "mps") else torch.float16
    experiments = [e.strip() for e in args.experiments.split(",")]

    model, tokenizer, _info = load_merged_into_tl(args.merged_path, args.base, device, dtype)
    baseline_heads = _default_heads_for(args.base)

    # Reuse the stage2 metadata for operating-point context
    s2_info = analyze_stage2(args.stage2)
    pt_info = s2_info["point_info"].get(f"{nk},{nu}", {})

    candidate_pool = get_value_pool("ARBITRARY_SINGLE")
    value_to_tid = verify_single_token(tokenizer, values=candidate_pool)
    value_pool = list(value_to_tid.keys())

    print(f"\n{'='*70}")
    print(f"V3 STAGE 3 — TARGETED (baseline-heads ablation on LoRA)")
    print(f"  Model:           {args.out_name}")
    print(f"  Operating point: {nk}k_{nu}u")
    print(f"  Target heads:    {[h['label'] for h in baseline_heads]}")
    print(f"  Trials:          {args.trials}")
    print(f"  Experiments:     {experiments}")
    print(f"{'='*70}\n")

    results = {
        "model": args.out_name,
        "n_layers": model.cfg.n_layers,
        "n_heads": model.cfg.n_heads,
        "operating_point": {"keys": nk, "updates": nu},
        "stage2_info": pt_info,
        "config": {
            "stage2_path": args.stage2,
            "trials": args.trials,
            "experiments": experiments,
            "target_heads_source": f"baseline 3C heads for base={args.base}",
        },
        "target_heads": baseline_heads,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "experiments": {},
    }

    if "3B" in experiments:
        t0 = time.time()
        result_3b = run_targeted_patching(
            model, tokenizer, value_to_tid, value_pool,
            args.out_name, nk, nu, args.trials,
            baseline_heads, results["config"],
        )
        result_3b["elapsed_sec"] = round(time.time() - t0, 1)
        results["experiments"]["3B"] = result_3b
        save_results(results, args.out_name, partial=True)

    if "3C" in experiments:
        t0 = time.time()
        result_3c = run_ablation_logit_lens(
            model, tokenizer, value_to_tid, value_pool,
            args.out_name, nk, nu, args.trials,
            baseline_heads, results["config"],
        )
        result_3c["elapsed_sec"] = round(time.time() - t0, 1)
        results["experiments"]["3C"] = result_3c
        save_results(results, args.out_name, partial=True)

    results["end_time"] = datetime.now(timezone.utc).isoformat()
    save_path = save_results(results, args.out_name, partial=False)
    print(f"\n✓ Saved: {save_path}")


if __name__ == "__main__":
    main()
