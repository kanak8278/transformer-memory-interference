"""
End-to-end GSM8K task accuracy eval.

For the §5.3 arithmetic-control LoRA: confirms the model actually learned
to solve GSM8K (not just fit the training distribution under teacher
forcing). Compares base Qwen2.5-3B-Instruct vs +arithmetic-LoRA.

Procedure: for each test example,
  - format prompt as system + user(question), open assistant turn
  - vLLM-generate up to 384 tokens (covers CoT + final answer)
  - extract last '#### <num>' or trailing number; compare to gold

Output: lora_intervention/results/gsm8k_task_eval_<run_name>_<ts>.json

Usage:
    python lora_intervention/eval_gsm8k_task.py \\
        --model Qwen/Qwen2.5-3B-Instruct \\
        --adapter lora_intervention/checkpoints/qwen_arith_control/checkpoint-100 \\
        --run-name qwen_arith_control \\
        --n 250

    # Baseline (no adapter)
    python lora_intervention/eval_gsm8k_task.py \\
        --model Qwen/Qwen2.5-3B-Instruct --adapter none \\
        --run-name qwen_base --n 250
"""
import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timezone

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoTokenizer
from datasets import load_dataset

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Same system prompt as the training data (data_gen_arithmetic.py).
SYSTEM_PROMPT = (
    "You are a precise data extraction tool. "
    "Output ONLY a single word - the exact value requested. "
    "No other text, no explanation, no punctuation."
)

RESULTS_DIR = Path(__file__).parent / "results"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Base HF model id")
    p.add_argument("--adapter", default=None,
                   help="Path to adapter, or 'none' for base eval")
    p.add_argument("--run-name", required=True)
    p.add_argument("--n", type=int, default=250,
                   help="Number of GSM8K test examples to eval (full test=1319)")
    p.add_argument("--max-new-tokens", type=int, default=384)
    return p.parse_args()


# ─── answer extraction ───────────────────────────────────────────────────────
NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)*")


def normalize_num(s: str) -> str:
    """Strip commas, dollar signs, trailing periods; canonicalize."""
    s = s.strip()
    if s.endswith("."):
        s = s[:-1]
    s = s.replace(",", "").replace("$", "").strip()
    return s


def extract_answer(text: str) -> str | None:
    """Extract a numeric answer from a generation.

    Order of preference:
    1. Text after the last '####' marker (GSM8K canonical)
    2. The last numeric token in the text
    """
    if "####" in text:
        tail = text.rsplit("####", 1)[-1].strip()
        m = NUM_RE.search(tail)
        if m:
            return normalize_num(m.group(0))
    matches = NUM_RE.findall(text)
    if matches:
        return normalize_num(matches[-1])
    return None


def extract_gold(answer_field: str) -> str:
    """GSM8K gold answers are after '####' in the answer field."""
    tail = answer_field.rsplit("####", 1)[-1].strip()
    return normalize_num(tail)


# ─── LoRA merge (reused from evaluate.py pattern) ────────────────────────────
def merge_and_save(adapter_path: Path, tmp_dir: str, base_model_id: str):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def _cls(model_id):
        if "gemma-3" in model_id.lower():
            try:
                from transformers import Gemma3ForCausalLM
                return Gemma3ForCausalLM
            except ImportError:
                pass
        return AutoModelForCausalLM

    print(f"  Loading base model {base_model_id}...", flush=True)
    model = _cls(base_model_id).from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    print(f"  Loading adapter from {adapter_path}...", flush=True)
    model = PeftModel.from_pretrained(model, str(adapter_path))
    print("  Merging weights...", flush=True)
    model = model.merge_and_unload()
    print(f"  Saving merged model to {tmp_dir}...", flush=True)
    model.save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)
    print("  Merge complete.", flush=True)


def main():
    args = parse_args()
    use_adapter = args.adapter is not None and args.adapter != "none"

    print("=" * 60)
    print(f"GSM8K task-accuracy eval — {args.run_name}")
    print(f"  Base model: {args.model}")
    print(f"  Adapter:    {args.adapter if use_adapter else 'NONE (baseline)'}")
    print(f"  N samples:  {args.n}")
    print("=" * 60)

    # Load GSM8K test
    print("\nLoading GSM8K test split...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if args.n < len(ds):
        ds = ds.select(range(args.n))
    print(f"  Using {len(ds):,} examples")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Build prompts
    print("\nBuilding prompts...")
    prompts = []
    golds = []
    for r in ds:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": r["question"].strip()},
        ]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        golds.append(extract_gold(r["answer"]))

    # Engine
    tmp = None
    if use_adapter:
        tmp = tempfile.TemporaryDirectory()
        merge_and_save(Path(args.adapter), tmp.name, args.model)
        model_path = tmp.name
    else:
        model_path = args.model

    from vllm import LLM, SamplingParams
    print(f"\nLoading vLLM engine from {model_path}...")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=2048,
        trust_remote_code=True,
        gpu_memory_utilization=0.92,
        enable_prefix_caching=True,
    )
    print("  ✓ vLLM ready")

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    print(f"\nGenerating answers for {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sp)

    n_correct = 0
    rows = []
    for i, o in enumerate(outputs):
        gen = o.outputs[0].text
        pred = extract_answer(gen)
        gold = golds[i]
        correct = pred is not None and pred == gold
        n_correct += int(correct)
        rows.append({
            "idx": i,
            "gold": gold,
            "pred": pred,
            "correct": correct,
            "gen": gen[:500],  # truncate for storage
        })

    acc = n_correct / len(prompts)
    print(f"\n{'='*60}\nGSM8K task accuracy — {args.run_name}\n{'='*60}")
    print(f"  Correct: {n_correct} / {len(prompts)}  ({acc:.1%})")

    if tmp is not None:
        tmp.cleanup()

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"gsm8k_task_eval_{args.run_name}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "adapter": str(args.adapter) if use_adapter else None,
            "run_name": args.run_name,
            "n": len(prompts),
            "accuracy": acc,
            "n_correct": n_correct,
            "timestamp": ts,
            "rows": rows,
        }, f, indent=2)
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
