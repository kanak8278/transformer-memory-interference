"""
LoRA fine-tuning for the interference intervention experiment.

Uses PEFT + TRL SFTTrainer. Trains on lora_intervention/data/train.jsonl,
validates on val.jsonl. Saves adapter to checkpoints/<run_name>/.

Usage:
    cd /home/sagemaker-user/transformer-memory-interference

    # Main experiment (40/40/20 FVQ/CVQ/IVQ mix)
    python lora_intervention/train.py --run-name main

    # Endpoint-only comparison (50/50 FVQ/CVQ, no intermediate)
    python lora_intervention/train.py --run-name endpoint_only \\
        --train-file lora_intervention/data/train_endpoint_only.jsonl

    # Control: arithmetic (null condition)
    python lora_intervention/train.py --run-name control_arithmetic \\
        --train-file lora_intervention/data/train_arithmetic.jsonl \\
        --val-file   lora_intervention/data/val_arithmetic.jsonl

Adapter checkpoints saved to:
    lora_intervention/checkpoints/<run_name>/
    lora_intervention/checkpoints/<run_name>/final/   ← merged-ready adapter
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")

# Patch broken tensorflow version string before datasets imports it
import importlib.metadata as _imeta
_imeta_orig = _imeta.version
def _safe_version(pkg):
    try:
        v = _imeta_orig(pkg)
        return v if v is not None else "0.0.0"
    except Exception:
        return "0.0.0"
_imeta.version = _safe_version

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
try:
    from transformers import Gemma3ForCausalLM
except ImportError:
    Gemma3ForCausalLM = None
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# ─── defaults ─────────────────────────────────────────────────────────────────
MODEL_ID   = "Qwen/Qwen2.5-3B-Instruct"
LORA_R     = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET  = ["q_proj", "k_proj", "v_proj", "o_proj"]

LR           = 2e-4
EPOCHS       = 2
BATCH_SIZE   = 8
GRAD_ACCUM   = 8      # effective batch = 64
MAX_SEQ_LEN  = 2048
WARMUP_STEPS = 100

DATA_DIR  = Path(__file__).parent / "data"
CKPT_DIR  = Path(__file__).parent / "checkpoints"


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LoRA intervention fine-tuning")
    p.add_argument("--run-name",   default="main", help="Name for this run (default: main)")
    p.add_argument("--model",      default=MODEL_ID)
    p.add_argument("--train-file", default=None, help="Override train JSONL path")
    p.add_argument("--val-file",   default=None, help="Override val JSONL path")
    p.add_argument("--epochs",     type=int,   default=EPOCHS)
    p.add_argument("--lr",         type=float, default=LR)
    p.add_argument("--rank",       type=int,   default=LORA_R)
    p.add_argument("--batch-size", type=int,   default=BATCH_SIZE)
    p.add_argument("--grad-accum", type=int,   default=GRAD_ACCUM)
    p.add_argument("--max-seq-len",type=int,   default=MAX_SEQ_LEN)
    p.add_argument("--target-modules", default=None,
                   help="Comma-separated PEFT target module names. "
                        "Default: q_proj,k_proj,v_proj,o_proj (attention). "
                        "For MLP-only: gate_proj,up_proj,down_proj.")
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Override num_train_epochs with hard step cap (default: -1 = use epochs)")
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--smoke",      action="store_true", help="Quick 10-step test")
    return p.parse_args()


# ─── data loading ─────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    train_path = Path(args.train_file) if args.train_file else DATA_DIR / "train.jsonl"
    val_path   = Path(args.val_file)   if args.val_file   else DATA_DIR / "val.jsonl"
    ckpt_path  = CKPT_DIR / args.run_name

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = CKPT_DIR / args.run_name / f"train_log_{ts}.txt"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LoRA Intervention — Training")
    print(f"  Run:         {args.run_name}")
    print(f"  Model:       {args.model}")
    print(f"  Train file:  {train_path}")
    print(f"  Val file:    {val_path}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  LR:          {args.lr}")
    print(f"  Rank:        {args.rank}  (alpha={args.rank * 2})")
    print(f"  Batch:       {args.batch_size} × grad_accum={args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"  Max seq len: {args.max_seq_len}")
    print(f"  Output:      {ckpt_path}")
    print("=" * 60)

    # ── Load datasets ─────────────────────────────────────────────────────────
    print("\nLoading data...")
    train_ds = load_jsonl(train_path)
    val_ds   = load_jsonl(val_path)
    if args.smoke:
        train_ds = train_ds.select(range(min(80, len(train_ds))))
        val_ds   = val_ds.select(range(min(20, len(val_ds))))
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    # ── Load model + tokenizer ────────────────────────────────────────────────
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # SFTTrainer needs right-padding

    # Gemma-3 checkpoints ship as vision-language; load text-only causal head.
    model_cls = (
        Gemma3ForCausalLM
        if (Gemma3ForCausalLM is not None and "gemma-3" in args.model.lower())
        else AutoModelForCausalLM
    )
    model = model_cls.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.enable_input_require_grads()  # needed for gradient checkpointing with PEFT

    # ── LoRA config ───────────────────────────────────────────────────────────
    target_modules = (
        [m.strip() for m in args.target_modules.split(",")]
        if args.target_modules else LORA_TARGET
    )
    print(f"  LoRA targets: {target_modules}")
    lora_cfg = LoraConfig(
        task_type     = TaskType.CAUSAL_LM,
        r             = args.rank,
        lora_alpha    = args.rank * 2,
        lora_dropout  = LORA_DROPOUT,
        target_modules= target_modules,
        bias          = "none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Training config (TRL 1.x SFTConfig) ──────────────────────────────────
    if args.smoke:
        max_steps = 10
    elif args.max_steps > 0:
        max_steps = args.max_steps
    else:
        max_steps = -1

    sft_cfg = SFTConfig(
        output_dir              = str(ckpt_path),
        num_train_epochs        = args.epochs,
        max_steps               = max_steps,
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        per_device_eval_batch_size  = args.batch_size,
        learning_rate           = args.lr,
        lr_scheduler_type       = "cosine",
        warmup_steps            = WARMUP_STEPS,
        bf16                    = True,
        logging_steps           = 10,
        eval_strategy           = "steps",
        eval_steps              = args.eval_steps,
        save_strategy           = "steps",
        save_steps              = args.save_steps,
        save_total_limit        = 3,
        load_best_model_at_end  = True,
        metric_for_best_model   = "eval_loss",
        greater_is_better       = False,
        report_to               = "none",
        dataloader_num_workers  = 0,
        gradient_checkpointing  = True,
        remove_unused_columns   = False,
        # SFT-specific: mask prompt tokens, compute loss only on assistant response
        dataset_text_field      = "text",
        max_length              = args.max_seq_len,
        assistant_only_loss     = True,
        packing                 = False,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model        = model,
        train_dataset= train_ds,
        eval_dataset = val_ds,
        args         = sft_cfg,
        processing_class = tokenizer,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nStarting training...")
    trainer.train()

    # ── Save final adapter ────────────────────────────────────────────────────
    final_path = ckpt_path / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nAdapter saved → {final_path}")

    # Save training config alongside adapter for reproducibility
    config = {
        "run_name":    args.run_name,
        "model":       args.model,
        "train_file":  str(train_path),
        "val_file":    str(val_path),
        "epochs":      args.epochs,
        "lr":          args.lr,
        "rank":        args.rank,
        "lora_alpha":  args.rank * 2,
        "target_modules": LORA_TARGET,
        "batch_size":  args.batch_size,
        "grad_accum":  args.grad_accum,
        "max_seq_len": args.max_seq_len,
        "timestamp":   ts,
    }
    with open(final_path / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
