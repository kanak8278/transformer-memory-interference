"""
Merge the LoRA adapter into the base Qwen2.5-3B-Instruct weights and save the
resulting standalone HF model. Downstream mechanistic scripts (probing,
logit lens, transformer_lens) load the merged model and don't need PEFT.

Validates by comparing logits from (base + PeftModel) against logits from the
merged model on a small prompt. Aborts if they diverge.

Usage:
    .venv/bin/python lora_intervention/merge_lora.py \
        --adapter /tmp/lora_intervention/checkpoints/main/final \
        --out lora_intervention/checkpoints/merged
"""
import argparse
import os
import sys
import shutil
from pathlib import Path

# Patch importlib.metadata version lookup (some torch/peft versions choke
# on missing tensorflow version string).
import importlib.metadata as _imeta
_orig = _imeta.version
def _safe(pkg):
    try:
        v = _orig(pkg); return v if v is not None else "0.0.0"
    except Exception:
        return "0.0.0"
_imeta.version = _safe

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DTYPE = torch.float32  # MPS: float32 for safest merge; downstream can cast
SMOKE_PROMPT = "fruit: apple\nfruit: banana\nfruit: cherry\nWhat was the first 'fruit'?"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True, help="Path to PEFT adapter dir")
    p.add_argument("--out", required=True, help="Where to save merged HF model")
    p.add_argument("--device", default=None, help="cpu | mps. Default: cpu (safest for merge)")
    p.add_argument("--no-validate", action="store_true", help="Skip pre/post smoke check")
    return p.parse_args()


def load_base(device):
    print(f"[1/4] Loading base model {BASE_MODEL} on {device} ({DTYPE})...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=DTYPE, low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    base.eval()
    return base, tok


def attach_adapter(base, adapter_path, device):
    print(f"[2/4] Attaching LoRA adapter from {adapter_path}...")
    peft_model = PeftModel.from_pretrained(base, adapter_path)
    peft_model.eval()
    peft_model.to(device)
    return peft_model


@torch.no_grad()
def smoke_logits(model, tok, device):
    inputs = tok(SMOKE_PROMPT, return_tensors="pt").to(device)
    out = model(**inputs)
    # Last-token logits, top-5 ids + values
    last = out.logits[0, -1]
    topv, topi = torch.topk(last.float(), 5)
    return last, topv.cpu(), topi.cpu()


def main():
    args = parse_args()
    out = Path(args.out).resolve()
    adapter = Path(args.adapter).resolve()
    assert adapter.exists(), f"Adapter not found at {adapter}"

    device = args.device or "cpu"  # CPU is safest + has enough RAM (36GB unified)
    print(f"Device: {device}\nOut:    {out}\n")

    base, tok = load_base(device)
    peft_model = attach_adapter(base, str(adapter), device)

    # ── Validate: forward through PEFT model ──
    if not args.no_validate:
        print("[3/4] Smoke-test: running forward through PEFT model...")
        _, peft_topv, peft_topi = smoke_logits(peft_model, tok, device)
        print("  PEFT top-5 token IDs :", peft_topi.tolist())
        print("  PEFT top-5 logits    :", [f"{v:.4f}" for v in peft_topv.tolist()])
    else:
        peft_topv = peft_topi = None

    # ── Merge ──
    print("[3/4] Merging LoRA into base weights (merge_and_unload)...")
    merged = peft_model.merge_and_unload()
    merged.eval()
    merged.to(device)

    # ── Validate: forward through merged model ──
    if not args.no_validate:
        print("[4/4] Smoke-test: running forward through merged model...")
        _, m_topv, m_topi = smoke_logits(merged, tok, device)
        print("  MRG  top-5 token IDs :", m_topi.tolist())
        print("  MRG  top-5 logits    :", [f"{v:.4f}" for v in m_topv.tolist()])

        ids_match = (peft_topi.tolist() == m_topi.tolist())
        max_diff = (peft_topv - m_topv).abs().max().item()
        print(f"  Top-5 IDs match     : {ids_match}")
        print(f"  Max logit |diff|    : {max_diff:.2e}")
        if not ids_match or max_diff > 1e-2:
            print("\n!! MERGE VALIDATION FAILED — aborting save. Top-5 IDs or logits diverge.")
            sys.exit(2)
        print("  ✓ merged model matches PEFT forward (within tolerance)")

    # ── Save ──
    print(f"\n[4/4] Saving merged model to {out}...")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True)
    tok.save_pretrained(str(out))

    # Size
    total_mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / (1024 ** 2)
    print(f"\n✓ Saved merged model: {out}  ({total_mb:.0f} MB)")
    print("\nNext: load it with AutoModelForCausalLM.from_pretrained(out) or")
    print("HookedTransformer.from_pretrained('Qwen/Qwen2.5-3B-Instruct', hf_model=merged_hf, ...)")


if __name__ == "__main__":
    main()
