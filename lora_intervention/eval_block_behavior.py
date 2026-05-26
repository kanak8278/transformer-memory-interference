"""
Behavioral check: does Block format close the FVQ/CVQ gap on a base model
at a specific (K, N) cell?

Required by PLAN_BLOCK_LOCUS_PROBE.md before the mechanism probe: if Block
doesn't behaviorally close the gap at K=2/N=30 on Qwen2.5-3B-Instruct,
then the mechanism question (does Block activate L30-L33 heads?) is
moot.

Runs vLLM end-to-end generation with 4 formats × {FVQ, CVQ} × n_trials.

Usage:
    python lora_intervention/eval_block_behavior.py \\
        --model Qwen/Qwen2.5-3B-Instruct \\
        --K 2 --N 30 --trials 100
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path
from datetime import datetime, timezone

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoTokenizer

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from mechanistic_probing_v2.core.dataset_configs import (
    get_eligible_categories, get_value_pool, format_for_chat,
)
from mechanistic_probing_v2.core.model_loader import (
    verify_single_token, is_instruct_model,
)

SYSTEM_PROMPT = (
    "You are a precise data extraction tool. "
    "Output ONLY a single word - the exact value requested. "
    "No other text, no explanation, no punctuation."
)
DATASET_TYPE = "ARBITRARY_SINGLE"
RESULTS_DIR = Path(__file__).parent / "results"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--N", type=int, default=30)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--seed-start", type=int, default=1000)
    return p.parse_args()


def build_rounded_stream(K, N, value_pool, rng):
    eligible = get_eligible_categories(DATASET_TYPE, min_values=N)
    categories = rng.sample(eligible, K)
    values_per_cat = {}
    used = set()
    for cat in categories:
        chosen = []
        attempts = 0
        while len(chosen) < N and attempts < N * 20:
            attempts += 1
            v = rng.choice(value_pool)
            if v in used:
                continue
            used.add(v)
            chosen.append(v)
        values_per_cat[cat] = chosen
    round_groups = []
    flat = []
    for r in range(N):
        round_items = [
            {"category": cat, "value": values_per_cat[cat][r], "round": r}
            for cat in categories
        ]
        rng.shuffle(round_items)
        round_groups.append(round_items)
        flat.extend(round_items)
    return flat, round_groups, categories, values_per_cat


def render(format_name, flat_items, round_groups, test_category, condition):
    query_word = "first" if condition == "FVQ" else "last"
    if format_name == "plain":
        stream = "\n".join(f"{it['category']}: {it['value']}" for it in flat_items)
        return (
            f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
            f"{stream}\n\n"
            f"What was the {query_word} value of {test_category}?"
        )
    if format_name == "labeled":
        # Per-entry "(update j)" numbering, interleaved order
        stream = "\n".join(
            f"{it['category']} (update {it['round'] + 1}): {it['value']}"
            for it in flat_items
        )
        return (
            f"Read the following key-value stream. Each key is updated multiple times. "
            f"The number in parentheses after each key name indicates which update of that "
            f"key it is — for example, 'color (update 3): green' means the 3rd time "
            f"color was updated, its value was 'green'.\n\n"
            f"{stream}\n\n"
            f"What was the {query_word} value of {test_category}?"
        )
    if format_name == "landmark":
        rounds = []
        for items in round_groups:
            rounds.append("\n".join(f"{it['category']}: {it['value']}" for it in items))
        stream = "\n---\n".join(rounds)
        return (
            f"Read the following key-value stream. Each key is updated multiple times. "
            f"The '---' marker separates consecutive update rounds.\n\n"
            f"{stream}\n\n"
            f"What was the {query_word} value of {test_category}?"
        )
    if format_name == "block":
        lines = []
        for r_idx, items in enumerate(round_groups):
            lines.append(f"[Round {r_idx + 1}]")
            for it in items:
                lines.append(f"  {it['category']}: {it['value']}")
        stream = "\n".join(lines)
        return (
            f"Read the following key-value stream. Each key is updated multiple times, "
            f"grouped by update round.\n\n"
            f"{stream}\n\n"
            f"What was the {query_word} value of {test_category}?"
        )
    raise ValueError(f"Unknown format {format_name}")


def is_correct(pred, exp):
    pred = pred.lower().strip()
    exp = exp.lower().strip()
    return pred == exp or exp in pred or pred.startswith(exp)


def main():
    args = parse_args()
    print(f"Eval — {args.model} @ K={args.K}/N={args.N}, 4 formats × FVQ/CVQ × {args.trials} trials")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    candidate_pool = get_value_pool(DATASET_TYPE)
    value_to_tid = verify_single_token(tokenizer, values=candidate_pool)
    value_pool = list(value_to_tid.keys())
    print(f"  value pool: {len(value_pool)}")

    # Build all prompts up front, then batch through vLLM
    formats = ["plain", "labeled", "landmark", "block"]
    prompts = []
    meta = []  # parallel list of {format, condition, expected, seed}
    for fmt in formats:
        for condition in ["FVQ", "CVQ"]:
            for i in range(args.trials):
                seed = args.seed_start + i
                rng = random.Random(seed)
                flat, groups, cats, vals_per_cat = build_rounded_stream(args.K, args.N, value_pool, rng)
                tcat = cats[seed % args.K]
                cat_vals = vals_per_cat[tcat]
                exp = cat_vals[0] if condition == "FVQ" else cat_vals[-1]
                raw = render(fmt, flat, groups, tcat, condition)
                use_chat = is_instruct_model(args.model)
                formatted = format_for_chat(raw, tokenizer, model_name=args.model) if use_chat else raw
                # Add system prompt for chat-templated:
                if use_chat:
                    msgs = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": raw},
                    ]
                    formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                prompts.append(formatted)
                meta.append({"format": fmt, "condition": condition, "expected": exp, "seed": seed})

    # vLLM batch generate
    from vllm import LLM, SamplingParams
    print(f"\nLoading vLLM engine from {args.model}...")
    llm = LLM(
        model=args.model, dtype="bfloat16", max_model_len=4096,
        trust_remote_code=True, gpu_memory_utilization=0.92,
        enable_prefix_caching=True,
    )
    print("  ✓ ready")

    sp = SamplingParams(temperature=0.0, max_tokens=12, stop=["<|im_end|>", "<|endoftext|>", "\n"])
    print(f"\nGenerating {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sp)

    # Aggregate
    results = {}
    for o, m in zip(outputs, meta):
        text = o.outputs[0].text.strip()
        correct = is_correct(text, m["expected"])
        key = (m["format"], m["condition"])
        results.setdefault(key, []).append(correct)

    print(f"\n{'='*60}\nResults — K={args.K}, N={args.N}, n={args.trials} per cell\n{'='*60}")
    print(f"  {'Format':>10}  {'FVQ':>8}  {'CVQ':>8}  {'gap (FVQ-CVQ)':>14}")
    summary = {}
    for fmt in formats:
        fvq = sum(results[(fmt, "FVQ")]) / args.trials
        cvq = sum(results[(fmt, "CVQ")]) / args.trials
        gap = fvq - cvq
        print(f"  {fmt:>10}  {fvq:.2%}    {cvq:.2%}    {gap:+.2%}")
        summary[fmt] = {"fvq": fvq, "cvq": cvq, "gap": gap, "n": args.trials}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short = args.model.replace("/", "_")
    out_path = RESULTS_DIR / f"block_behavior_{short}_K{args.K}N{args.N}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model, "K": args.K, "N": args.N,
            "trials": args.trials, "timestamp": ts,
            "summary": summary,
        }, f, indent=2)
    print(f"\n✓ Saved → {out_path}")


if __name__ == "__main__":
    main()
