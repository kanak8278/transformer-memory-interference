"""
Behavioral sweep on the narrative interference dataset.

Tests whether PI > RI holds in naturalistic story-based settings
(not just key-value pairs). Runs a model on all 44 narrative stories,
asking both RI (first value) and PI (last value) questions.

Usage:
    .venv/bin/python mechanistic_probing_v2/experiments/50_narrative_behavioral_sweep.py \
        --model Qwen/Qwen2.5-1.5B-Instruct

    # Or with a smaller model:
    .venv/bin/python mechanistic_probing_v2/experiments/50_narrative_behavioral_sweep.py \
        --model Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Helpers ──────────────────────────────────────────────────────────────────

def timestamp_str():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def model_short(name: str) -> str:
    return name.split("/")[-1]


def load_dataset(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_prompt(narrative: str, question: str) -> str:
    """Build a prompt: narrative + question."""
    return (
        f"Read the following passage carefully.\n\n"
        f"{narrative}\n\n"
        f"Based ONLY on the passage above, answer the following question "
        f"with a short, exact answer (just the value, no explanation).\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def format_chat(prompt: str, tokenizer) -> str:
    """Wrap in chat template if available."""
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": "Answer with ONLY the exact value. No explanation."},
            {"role": "user", "content": prompt},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt


def classify_answer(answer: str, expected: str, all_values: list[str] = None) -> str:
    """Classify the model's answer."""
    answer_lower = answer.lower().strip()
    expected_lower = expected.lower().strip()

    if expected_lower in answer_lower or answer_lower in expected_lower:
        return "correct"

    # Check if it's any tracked value
    if all_values:
        for v in all_values:
            if v.lower() in answer_lower or answer_lower in v.lower():
                return "wrong_value"

    return "garbage"


# ── Main ─────────────────────────────────────────────────────────────────────

def run_sweep(args):
    ts = timestamp_str()
    dataset_path = Path("data/narrative_interference/narrative_interference_dataset.json")
    dataset = load_dataset(str(dataset_path))
    stories = dataset["stories"]

    print(f"Model: {args.model}")
    print(f"Stories: {len(stories)}")
    print(f"Timestamp: {ts}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print()

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device != "cpu" else torch.float32,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded on {args.device}")

    # Check context limit
    ctx_limit = getattr(model.config, "max_position_embeddings", 8192)
    print(f"Context limit: {ctx_limit}")
    print()

    # Run on all stories
    results = []
    skipped = 0
    total_correct = {"RI": 0, "PI": 0}
    total_count = {"RI": 0, "PI": 0}

    for i, story in enumerate(stories):
        sid = story["id"]
        domain = story["domain"]
        nk = story["num_keys"]
        nu = story["num_updates"]
        narrative = story["narrative"]

        # Get all tracked values for the target entity (for error classification)
        target_entity = story["questions"]["RI"]["target_entity"]
        target_attr = story["questions"]["RI"]["target_attribute"]
        all_values = []
        for key, vals in story["entity_tracking"].items():
            if any(part.lower() in key.lower() for part in target_entity.split()[:2]):
                all_values = vals
                break

        for condition in ["RI", "PI"]:
            q_data = story["questions"][condition]
            question = q_data["question"]
            expected = q_data["expected_answer"]

            prompt = build_prompt(narrative, question)
            formatted = format_chat(prompt, tokenizer)

            # Tokenize and check context
            input_ids = tokenizer.encode(formatted, return_tensors="pt")
            n_tokens = input_ids.shape[1]

            if n_tokens > ctx_limit * 0.95:
                print(f"  [{i+1}/{len(stories)}] {sid} {condition}: SKIP ({n_tokens} tokens > context)")
                skipped += 1
                results.append({
                    "story_id": sid,
                    "domain": domain,
                    "num_keys": nk,
                    "num_updates": nu,
                    "condition": condition,
                    "expected": expected,
                    "answer": None,
                    "correct": None,
                    "error_type": "skipped_context",
                    "input_tokens": n_tokens,
                })
                continue

            input_ids = input_ids.to(model.device)

            # Generate
            with torch.no_grad():
                gen_ids = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )

            new_ids = gen_ids[0, input_ids.shape[1]:]
            answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            # Take first line only
            answer = answer.split("\n")[0].strip()

            error_type = classify_answer(answer, expected, all_values)
            correct = error_type == "correct"

            total_count[condition] += 1
            if correct:
                total_correct[condition] += 1

            results.append({
                "story_id": sid,
                "domain": domain,
                "num_keys": nk,
                "num_updates": nu,
                "condition": condition,
                "expected": expected,
                "answer": answer,
                "correct": correct,
                "error_type": error_type,
                "input_tokens": n_tokens,
            })

            ri_acc = total_correct["RI"] / max(total_count["RI"], 1)
            pi_acc = total_correct["PI"] / max(total_count["PI"], 1)
            mark = "✓" if correct else "✗"
            print(
                f"  [{i+1}/{len(stories)}] {sid:40s} {condition}: {mark} "
                f"(expected={expected[:30]}, got={answer[:30]}) "
                f"[RI={ri_acc:.0%} PI={pi_acc:.0%}]"
            )

    # Compute summary stats
    ri_results = [r for r in results if r["condition"] == "RI" and r["correct"] is not None]
    pi_results = [r for r in results if r["condition"] == "PI" and r["correct"] is not None]

    ri_acc = sum(r["correct"] for r in ri_results) / max(len(ri_results), 1)
    pi_acc = sum(r["correct"] for r in pi_results) / max(len(pi_results), 1)

    # By domain
    domain_stats = {}
    for r in results:
        if r["correct"] is None:
            continue
        d = r["domain"]
        c = r["condition"]
        if d not in domain_stats:
            domain_stats[d] = {"RI": {"correct": 0, "total": 0}, "PI": {"correct": 0, "total": 0}}
        domain_stats[d][c]["total"] += 1
        if r["correct"]:
            domain_stats[d][c]["correct"] += 1

    # By grid cell (num_keys, num_updates)
    cell_stats = {}
    for r in results:
        if r["correct"] is None:
            continue
        cell = f"{r['num_keys']}k_{r['num_updates']}u"
        c = r["condition"]
        if cell not in cell_stats:
            cell_stats[cell] = {"RI": {"correct": 0, "total": 0}, "PI": {"correct": 0, "total": 0}}
        cell_stats[cell][c]["total"] += 1
        if r["correct"]:
            cell_stats[cell][c]["correct"] += 1

    # Error type distribution
    error_dist = {"RI": {}, "PI": {}}
    for r in results:
        if r["error_type"]:
            c = r["condition"]
            et = r["error_type"]
            error_dist[c][et] = error_dist[c].get(et, 0) + 1

    # Print summary
    print("\n" + "=" * 60)
    print(f"NARRATIVE INTERFERENCE SWEEP — {model_short(args.model)}")
    print("=" * 60)
    print(f"  RI accuracy: {ri_acc:.1%} ({sum(r['correct'] for r in ri_results)}/{len(ri_results)})")
    print(f"  PI accuracy: {pi_acc:.1%} ({sum(r['correct'] for r in pi_results)}/{len(pi_results)})")
    print(f"  PI > RI: {pi_acc > ri_acc} (diff = {pi_acc - ri_acc:+.1%})")
    print(f"  Skipped: {skipped}")
    print()

    print("By domain:")
    for d in sorted(domain_stats):
        ds = domain_stats[d]
        ri_a = ds["RI"]["correct"] / max(ds["RI"]["total"], 1)
        pi_a = ds["PI"]["correct"] / max(ds["PI"]["total"], 1)
        print(f"  {d:25s}  RI={ri_a:.0%} ({ds['RI']['correct']}/{ds['RI']['total']})  "
              f"PI={pi_a:.0%} ({ds['PI']['correct']}/{ds['PI']['total']})")

    print("\nBy grid cell:")
    for cell in sorted(cell_stats, key=lambda x: (int(x.split("k")[0]), int(x.split("_")[1].rstrip("u")))):
        cs = cell_stats[cell]
        ri_a = cs["RI"]["correct"] / max(cs["RI"]["total"], 1)
        pi_a = cs["PI"]["correct"] / max(cs["PI"]["total"], 1)
        print(f"  {cell:10s}  RI={ri_a:.0%}  PI={pi_a:.0%}")

    print("\nError distribution:")
    for cond in ["RI", "PI"]:
        print(f"  {cond}: {error_dist[cond]}")

    # Save results
    output_dir = Path("mechanistic_probing_v2/results") / model_short(args.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"narrative_sweep_{ts}.json"

    output_data = {
        "metadata": {
            "model": args.model,
            "model_short": model_short(args.model),
            "dataset": str(dataset_path),
            "dataset_stories": len(stories),
            "timestamp": ts,
            "max_new_tokens": args.max_new_tokens,
            "device": args.device,
            "context_limit": ctx_limit,
        },
        "summary": {
            "ri_accuracy": ri_acc,
            "pi_accuracy": pi_acc,
            "pi_gt_ri": pi_acc > ri_acc,
            "diff": pi_acc - ri_acc,
            "ri_n": len(ri_results),
            "pi_n": len(pi_results),
            "skipped": skipped,
        },
        "domain_stats": {
            d: {
                "RI_accuracy": ds["RI"]["correct"] / max(ds["RI"]["total"], 1),
                "PI_accuracy": ds["PI"]["correct"] / max(ds["PI"]["total"], 1),
                "RI_n": ds["RI"]["total"],
                "PI_n": ds["PI"]["total"],
            }
            for d, ds in domain_stats.items()
        },
        "cell_stats": {
            cell: {
                "RI_accuracy": cs["RI"]["correct"] / max(cs["RI"]["total"], 1),
                "PI_accuracy": cs["PI"]["correct"] / max(cs["PI"]["total"], 1),
                "RI_n": cs["RI"]["total"],
                "PI_n": cs["PI"]["total"],
            }
            for cell, cs in cell_stats.items()
        },
        "error_distribution": error_dist,
        "trials": results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")
    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Narrative interference behavioral sweep")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detect if not specified)")
    parser.add_argument("--max-new-tokens", type=int, default=30,
                        help="Max tokens to generate per answer")
    args = parser.parse_args()

    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    run_sweep(args)
