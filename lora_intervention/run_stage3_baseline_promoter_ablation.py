"""
Stage 3C — BASELINE promoter-head ablation control.

Ablates the SAME 8 LoRA-installed promoter heads (L32H3, L32H7, L31H12,
L31H15, L32H10, L30H11, L32H0, L32H14) in the FROZEN BASE model (no LoRA),
to answer the reviewer question:

  "Did LoRA install the v_last-attending function in these heads, or did
  they already carry it and LoRA just amplified?"

Predicted result: ablation has ~0 effect on baseline P(v_last) at L32,
because in baseline these heads attend to v_last only at P ≈ 0.06-0.23
(per attention_routing_comparison.txt). If true, LoRA installed the
function; if a large effect appears, we re-frame as amplification.

Operating point: K=2, N=5 (matches the LoRA-side promoter ablation run
v3/scripts/experiments/results/Qwen2.5-3B-Instruct-LoRA-promoters/stage3_causal_20260525_112134.json
so the trajectories are directly comparable).

Usage:
    .venv/bin/python lora_intervention/run_stage3_baseline_promoter_ablation.py \
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

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# Top 8 LoRA-side promoter heads (largest Δ in P(attend to v_last_round) under
# CVQ between baseline and LoRA, K=2/N=30 attention routing comparison).
# Source: lora_intervention/results/attention_routing_comparison.txt
BASELINE_TOP_HEADS = [
    {"layer": 32, "head": 3,  "label": "L32H3"},   # 0.108 → 0.789 (+0.682)
    {"layer": 32, "head": 7,  "label": "L32H7"},   # 0.126 → 0.777 (+0.651)
    {"layer": 31, "head": 12, "label": "L31H12"},  # 0.166 → 0.814 (+0.648)
    {"layer": 31, "head": 15, "label": "L31H15"},  # 0.233 → 0.863 (+0.630)
    {"layer": 32, "head": 10, "label": "L32H10"},  # 0.063 → 0.669 (+0.606)
    {"layer": 30, "head": 11, "label": "L30H11"},  # 0.190 → 0.781 (+0.590)
    {"layer": 32, "head": 0,  "label": "L32H0"},   # 0.085 → 0.675 (+0.590)
    {"layer": 32, "head": 14, "label": "L32H14"},  # 0.077 → 0.661 (+0.584)
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage2", required=True)
    p.add_argument("--point", default="2,5")
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--experiments", default="3C")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--out_name", default="Qwen2.5-3B-Instruct-baseline-promoters")
    return p.parse_args()


def load_base_into_tl(device, dtype):
    print(f"Loading base {BASE_MODEL} (no LoRA) on {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True,
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
    nk, nu = map(int, args.point.split(","))
    device = args.device or detect_device()[0]
    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif device in ("cpu", "mps"):
        dtype = torch.float32
    else:
        dtype = torch.float16
    experiments = [e.strip() for e in args.experiments.split(",")]

    model, tokenizer, _info = load_base_into_tl(device, dtype)

    # Reuse the stage2 metadata for operating-point context
    s2_info = analyze_stage2(args.stage2)
    pt_info = s2_info["point_info"].get(f"{nk},{nu}", {})

    candidate_pool = get_value_pool("ARBITRARY_SINGLE")
    value_to_tid = verify_single_token(tokenizer, values=candidate_pool)
    value_pool = list(value_to_tid.keys())

    print(f"\n{'='*70}")
    print(f"V3 STAGE 3 — CONTROL (8 LoRA-promoter heads ablated in BASELINE model)")
    print(f"  Model:           {args.out_name}")
    print(f"  Operating point: {nk}k_{nu}u")
    print(f"  Target heads:    {[h['label'] for h in BASELINE_TOP_HEADS]}")
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
            "target_heads_source": "BASELINE top-8 from stage3_causal_20260409_055011.json",
        },
        "target_heads": BASELINE_TOP_HEADS,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "experiments": {},
    }

    if "3B" in experiments:
        t0 = time.time()
        result_3b = run_targeted_patching(
            model, tokenizer, value_to_tid, value_pool,
            args.out_name, nk, nu, args.trials,
            BASELINE_TOP_HEADS, results["config"],
        )
        result_3b["elapsed_sec"] = round(time.time() - t0, 1)
        results["experiments"]["3B"] = result_3b
        save_results(results, args.out_name, partial=True)

    if "3C" in experiments:
        t0 = time.time()
        result_3c = run_ablation_logit_lens(
            model, tokenizer, value_to_tid, value_pool,
            args.out_name, nk, nu, args.trials,
            BASELINE_TOP_HEADS, results["config"],
        )
        result_3c["elapsed_sec"] = round(time.time() - t0, 1)
        results["experiments"]["3C"] = result_3c
        save_results(results, args.out_name, partial=True)

    results["end_time"] = datetime.now(timezone.utc).isoformat()
    save_path = save_results(results, args.out_name, partial=False)
    print(f"\n✓ Saved: {save_path}")


if __name__ == "__main__":
    main()
