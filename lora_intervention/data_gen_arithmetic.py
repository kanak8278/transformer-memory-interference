"""
Generate the §5.3 arithmetic-control LoRA training data.

Downloads GSM8K (openai/gsm8k, 'main' config) train split via HF
datasets and formats as conversational JSONL — same structure as
data_gen.py output. **No filtering**: trains on the full ~7.5k
examples with chain-of-thought reasoning + numeric answer.

Why no filter: pre-registered control in PLAN.md Decision 7 suggests
single-token answers for format identity. We deviate to a stronger
control — even arithmetic *reasoning* training shouldn't close the
FVQ-CVQ gap. The trade-off (mismatch with eval's single-word output
format) is acknowledged in GEMMA_RUN_LOG.md and is mitigated by
evaluate.py's lenient `is_correct` (`exp in pred or pred.startswith(exp)`).

Output:
    lora_intervention/data_arithmetic/train.jsonl  (~7k examples after train/val split)
    lora_intervention/data_arithmetic/val.jsonl    (~10% held out)

Usage:
    python lora_intervention/data_gen_arithmetic.py
    python lora_intervention/data_gen_arithmetic.py --smoke   # 50 examples
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")

from datasets import load_dataset

SYSTEM_PROMPT = (
    "You are a precise data extraction tool. "
    "Output ONLY a single word - the exact value requested. "
    "No other text, no explanation, no punctuation."
)
# Note: SYSTEM_PROMPT is intentionally the SAME as data_gen.py — we want the
# trained model to learn arithmetic UNDER the same system prompt as the main
# run, so any system-prompt-driven behavior is shared.

VAL_FRAC = 0.10
TRAIN_SEED = 42

OUT_DIR = Path(__file__).parent / "data_arithmetic"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="Quick test, 50 examples")
    return p.parse_args()


def make_example(question: str, answer: str) -> dict:
    """Format a GSM8K (question, answer) pair as our conversational JSONL.

    The answer field in GSM8K includes the chain-of-thought reasoning AND
    the final numeric answer after '####'. We train the model on the full
    answer (reasoning + answer), so it learns arithmetic reasoning end-to-end.
    """
    prompt_msgs = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": question.strip()},
    ]
    completion_msgs = [
        {"role": "assistant", "content": answer.strip()},
    ]
    # Pull out the final numeric answer for inspection (used by evaluate-like tools)
    # GSM8K answer format ends with '#### <num>'
    final = answer.split("####")[-1].strip() if "####" in answer else ""
    return {
        "prompt":      prompt_msgs,
        "completion":  completion_msgs,
        "label":       final,
        "condition":   "ARITH",
        "dataset":     "gsm8k",
    }


def main():
    args = parse_args()
    print("=" * 60)
    print("§5.3 Arithmetic-control data generation (GSM8K, no filter)")
    print(f"  Out dir:    {OUT_DIR}")
    print(f"  Val frac:   {VAL_FRAC}")
    print(f"  Seed:       {TRAIN_SEED}")
    print(f"  Smoke:      {args.smoke}")
    print("=" * 60)

    print("\nDownloading GSM8K train split...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    print(f"  Loaded {len(ds):,} examples")

    records = [make_example(r["question"], r["answer"]) for r in ds]

    if args.smoke:
        records = records[:50]
        print(f"  Smoke mode: keeping {len(records)} examples")

    # Deterministic shuffle + val split
    random.Random(TRAIN_SEED).shuffle(records)
    val_n = max(1, int(len(records) * VAL_FRAC))
    val_set, trn_set = records[:val_n], records[val_n:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "train.jsonl"
    val_path = OUT_DIR / "val.jsonl"
    for split, recs, path in [("train", trn_set, train_path), ("val", val_set, val_path)]:
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"  {split:5s}: {len(recs):>5,} records → {path}")

    # Sanity: show a sample
    print("\nSample record (first train):")
    s = trn_set[0]
    print(f"  user: {s['prompt'][1]['content'][:200]}...")
    print(f"  assistant: {s['completion'][0]['content'][:200]}...")
    print(f"  final answer: {s['label']!r}")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
