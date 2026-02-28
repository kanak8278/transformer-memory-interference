"""
Exp 17b: Instruction sensitivity for causally-identified heads.

Same analysis as exp 17 Part B, but takes heads via CLI args instead of
loading from exp 16's attention-based classification.

Use heads identified by exp 25a (per-head causal knockout).

Tests: Do the causally-identified primacy heads attend to the query word
"first"/"last"? Do they change behavior between RI and PI conditions?

Usage:
    cd mechanistic_probing_v2

    # Single head:
    uv run python experiments/17b_instruction_sensitivity_causal.py \
        --model google/gemma-3-1b-it --keys 2 --updates 2 --trials 100 \
        --heads "14,2"

    # Multiple heads:
    uv run python experiments/17b_instruction_sensitivity_causal.py \
        --model Qwen/Qwen2.5-1.5B-Instruct --keys 1 --updates 3 --trials 100 \
        --heads "8,3 0,7 0,3"
"""

import sys
import time
import random
import argparse
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_loader import load_model, verify_single_token
from core.dataset_configs import (
    format_for_chat, ORIGINAL_CATEGORIES,
    get_value_pool, build_interleaved_sequence,
)
from core.output import save_results


def parse_heads(heads_str):
    """Parse '14,2 8,3 0,7' into [(14,2), (8,3), (0,7)]."""
    heads = []
    for h in heads_str.strip().split():
        parts = h.split(",")
        heads.append((int(parts[0]), int(parts[1])))
    return heads


def build_matched_ri_pi(num_keys, num_updates, seed, value_pool):
    """Build IDENTICAL prompts differing ONLY in 'first' vs 'last'."""
    rng = random.Random(seed)
    categories = rng.sample(ORIGINAL_CATEGORIES, min(num_keys, len(ORIGINAL_CATEGORIES)))
    total = num_keys * num_updates
    selected = rng.sample(value_pool, total)

    values_per_cat = {}
    idx = 0
    for cat in categories:
        values_per_cat[cat] = selected[idx:idx + num_updates]
        idx += num_updates

    test_cat = categories[seed % num_keys]

    sequence = build_interleaved_sequence(categories, values_per_cat, rng)
    stream = "\n".join(f"{it['category']}: {it['value']}" for it in sequence)
    cat_values = [it["value"] for it in sequence if it["category"] == test_cat]

    ri_prompt = (
        f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{stream}\n\n"
        f"What was the first value of {test_cat}?"
    )
    pi_prompt = (
        f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{stream}\n\n"
        f"What was the last value of {test_cat}?"
    )

    return {
        "ri_prompt": ri_prompt, "pi_prompt": pi_prompt,
        "initial_value": cat_values[0], "final_value": cat_values[-1],
        "all_values": cat_values,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--keys", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--heads", required=True,
                        help="Heads to analyze as 'layer,head' pairs. E.g., '14,2 8,3'")
    args = parser.parse_args()

    target_heads = parse_heads(args.heads)

    print("=" * 70)
    print("INSTRUCTION SENSITIVITY — Causally-identified heads")
    print(f"  keys={args.keys}, updates={args.updates}, trials={args.trials}")
    print(f"  Heads: {target_heads}")
    print(f"  Source: exp 25a per-head causal knockout")
    print("=" * 70)

    model, tokenizer, info = load_model(args.model, n_ctx=args.n_ctx)
    candidate_pool = get_value_pool("ARBITRARY_SINGLE")
    value_to_tid = verify_single_token(tokenizer, values=candidate_pool)
    value_pool = list(value_to_tid.keys())

    # Per-head accumulators
    head_results = {}
    for layer, head in target_heads:
        head_results[(layer, head)] = {
            "RI": {"instruction": [], "initial": [], "final": [],
                   "values": [], "query_word": [], "query_rest": []},
            "PI": {"instruction": [], "initial": [], "final": [],
                   "values": [], "query_word": [], "query_rest": []},
        }

    ri_correct = 0
    pi_correct = 0
    t_start = time.time()

    for t_idx in range(args.trials):
        seed = hash(("17b", t_idx, args.updates, args.keys)) % (2**31)
        pair = build_matched_ri_pi(args.keys, args.updates, seed, value_pool)

        init_tid = value_to_tid.get(pair["initial_value"], -1)
        final_tid = value_to_tid.get(pair["final_value"], -1)
        inter_tids = set()
        for v in pair["all_values"][1:-1]:
            tid = value_to_tid.get(v, -1)
            if tid >= 0:
                inter_tids.add(tid)

        for cond, prompt in [("RI", pair["ri_prompt"]), ("PI", pair["pi_prompt"])]:
            formatted = format_for_chat(prompt, tokenizer)
            tokens = model.to_tokens(formatted)
            token_ids = tokens[0].tolist()
            str_tokens = model.to_str_tokens(formatted)

            with torch.no_grad():
                logits, cache = model.run_with_cache(tokens)

            # Check correctness
            pred = tokenizer.decode([logits[0, -1].argmax().item()]).strip()
            expected = pair["initial_value"] if cond == "RI" else pair["final_value"]
            if pred.lower() == expected.lower():
                if cond == "RI":
                    ri_correct += 1
                else:
                    pi_correct += 1

            # Find positions
            init_pos = [i for i, t in enumerate(token_ids) if t == init_tid] if init_tid >= 0 else []
            final_pos = [i for i, t in enumerate(token_ids) if t == final_tid] if final_tid >= 0 else []
            inter_pos = [i for i, t in enumerate(token_ids) if t in inter_tids]
            value_pos = sorted(set(init_pos + final_pos + inter_pos))

            # Find query word position
            query_word_pos = []
            query_rest_pos = []
            in_query = False
            for i, st in enumerate(str_tokens):
                if "What" in st:
                    in_query = True
                if in_query:
                    if "first" in st.lower() or "last" in st.lower():
                        query_word_pos.append(i)
                    else:
                        query_rest_pos.append(i)

            query_start = query_rest_pos[0] if query_rest_pos else len(token_ids)
            instr_pos = [i for i in range(query_start) if i not in value_pos]

            # Extract attention for each target head
            for layer, head in target_heads:
                pattern = cache["pattern", layer]
                head_attn = pattern[0, head, -1, :]  # from answer position

                hr = head_results[(layer, head)][cond]
                hr["instruction"].append(head_attn[instr_pos].sum().item() if instr_pos else 0)
                hr["initial"].append(head_attn[init_pos].sum().item() if init_pos else 0)
                hr["final"].append(head_attn[final_pos].sum().item() if final_pos else 0)
                hr["values"].append(head_attn[value_pos].sum().item() if value_pos else 0)
                hr["query_word"].append(head_attn[query_word_pos].sum().item() if query_word_pos else 0)
                hr["query_rest"].append(head_attn[query_rest_pos].sum().item() if query_rest_pos else 0)

            del cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        if (t_idx + 1) % 25 == 0:
            print(f"  {t_idx + 1}/{args.trials} done ({time.time() - t_start:.0f}s)")

    total_time = time.time() - t_start

    # ── Results ──
    print(f"\n{'=' * 70}")
    print(f"RESULTS ({total_time:.0f}s)")
    print(f"{'=' * 70}")
    print(f"  Accuracy: RI={ri_correct}/{args.trials}, PI={pi_correct}/{args.trials}")

    save_head_data = []

    for layer, head in target_heads:
        print(f"\n  ── L{layer}H{head} ──")
        print(f"  {'Region':<20} {'RI':>10} {'PI':>10} {'Diff':>10}")
        print(f"  {'─' * 50}")

        head_save = {"layer": layer, "head": head, "attention_by_region": {}}

        for region in ["instruction", "initial", "final", "values", "query_word", "query_rest"]:
            ri_avg = np.mean(head_results[(layer, head)]["RI"][region])
            pi_avg = np.mean(head_results[(layer, head)]["PI"][region])
            diff = pi_avg - ri_avg
            marker = " ***" if abs(diff) > 0.05 else ""
            print(f"  {region:<20} {ri_avg:>10.4f} {pi_avg:>10.4f} {diff:>+10.4f}{marker}")

            head_save["attention_by_region"][region] = {
                "RI": float(ri_avg), "PI": float(pi_avg), "diff": float(diff),
            }

        # Interpretation
        ri_qw = np.mean(head_results[(layer, head)]["RI"]["query_word"])
        pi_qw = np.mean(head_results[(layer, head)]["PI"]["query_word"])
        ri_init = np.mean(head_results[(layer, head)]["RI"]["initial"])
        pi_init = np.mean(head_results[(layer, head)]["PI"]["initial"])
        ri_final = np.mean(head_results[(layer, head)]["RI"]["final"])
        pi_final = np.mean(head_results[(layer, head)]["PI"]["final"])

        findings = []
        if abs(ri_qw - pi_qw) < 0.01:
            findings.append(f"IGNORES query word (attn≈{(ri_qw+pi_qw)/2:.4f} in both)")
        else:
            findings.append(f"READS query word (RI={ri_qw:.4f}, PI={pi_qw:.4f})")

        if abs(ri_init - pi_init) < 0.05:
            findings.append(f"Same attn to init in both conditions ({(ri_init+pi_init)/2:.4f})")
        else:
            findings.append(f"Attn to init shifts: RI={ri_init:.4f}, PI={pi_init:.4f}")

        if abs(ri_final - pi_final) < 0.05:
            findings.append(f"Same attn to final in both conditions ({(ri_final+pi_final)/2:.4f})")
        else:
            findings.append(f"Attn to final shifts: RI={ri_final:.4f}, PI={pi_final:.4f}")

        for f in findings:
            print(f"  → {f}")

        head_save["findings"] = findings
        save_head_data.append(head_save)

    # ── Save ──
    save_data = {
        "model": args.model,
        "config": {"keys": args.keys, "updates": args.updates, "trials": args.trials},
        "heads_tested": [{"layer": l, "head": h} for l, h in target_heads],
        "head_source": "exp_25a_per_head_knockout",
        "accuracy": {"RI": ri_correct / args.trials, "PI": pi_correct / args.trials},
        "per_head": save_head_data,
        "total_time_sec": round(total_time, 1),
    }
    save_results(save_data, args.model, args.keys, args.updates, "instruction_sensitivity_causal")


if __name__ == "__main__":
    main()
