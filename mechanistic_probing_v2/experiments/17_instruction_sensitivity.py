"""
Step 2.9: Instruction sensitivity probe.

We proved patching at query position fixes PI. But WHY does the query fail?

Two analyses:
  A) Representation divergence: measure ||resid_RI - resid_PI|| at each layer
     for IDENTICAL prompts differing only in "first" vs "last".
     - If near-identical → model can't distinguish the instructions
     - If they diverge → model understands but can't route

  B) L16H3 QK analysis: what does the primacy-biased head actually attend to?
     Does it look at the query word at all, or ignore it entirely?
     Compare its attention pattern in RI vs PI.

Usage:
    cd mechanistic_probing_v2
    uv run python experiments/17_instruction_sensitivity.py [--trials 20]
"""

import sys
import json
import time
import random
import argparse
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_loader import load_model
from core.dataset import format_for_chat, ORIGINAL_CATEGORIES
from core.model_loader import verify_single_token


def build_matched_ri_pi(num_keys, num_updates, seed, value_pool, categories):
    """Build IDENTICAL prompts differing ONLY in 'first' vs 'last'."""
    rng = random.Random(seed)
    cats = categories[:num_keys]
    total = num_keys * num_updates
    selected = rng.sample(value_pool, total)

    values_per_cat = {}
    idx = 0
    for cat in cats:
        values_per_cat[cat] = selected[idx:idx + num_updates]
        idx += num_updates

    test_cat = cats[seed % num_keys]

    items = []
    for cat in cats:
        for val in values_per_cat[cat]:
            items.append({"category": cat, "value": val})

    rng.shuffle(items)
    for _ in range(100):
        ok = all(items[i]["category"] != items[i-1]["category"] for i in range(1, len(items)))
        if ok:
            break
        rng.shuffle(items)

    stream = "\n".join(f"{it['category']}: {it['value']}" for it in items)
    cat_values = [it["value"] for it in items if it["category"] == test_cat]

    # Two prompts: identical except "first" vs "last"
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
        "ri_prompt": ri_prompt,
        "pi_prompt": pi_prompt,
        "test_category": test_cat,
        "initial_value": cat_values[0],
        "final_value": cat_values[-1],
        "all_values": cat_values,
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--keys", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=2048)
    args = parser.parse_args()

    print("=" * 70)
    print("INSTRUCTION SENSITIVITY PROBE")
    print(f"  keys={args.keys}, updates={args.updates}, trials={args.trials}")
    print("=" * 70)

    model, tokenizer, info = load_model(args.model, n_ctx=args.n_ctx)
    value_to_tid = verify_single_token(tokenizer)
    value_pool = list(value_to_tid.keys())
    categories = ORIGINAL_CATEGORIES
    n_layers = info.n_layers
    n_heads = info.n_heads

    # ══════════════════════════════════════════════════════════════════════
    # PART A: Representation divergence ||resid_RI - resid_PI||
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print("PART A: Representation divergence between RI and PI")
    print(f"{'='*70}")

    # Per-layer divergence at different token positions
    # [n_layers] — averaged across trials
    divergence_at_answer = np.zeros(n_layers)
    divergence_at_query_start = np.zeros(n_layers)
    # Cosine similarity at answer position
    cosine_at_answer = np.zeros(n_layers)

    # Also track: does the model get each condition right?
    ri_correct_count = 0
    pi_correct_count = 0

    for t_idx in range(args.trials):
        seed = hash(("instr_sense", t_idx, args.updates)) % (2**31)
        pair = build_matched_ri_pi(args.keys, args.updates, seed, value_pool, categories)

        ri_formatted = format_for_chat(pair["ri_prompt"], tokenizer)
        pi_formatted = format_for_chat(pair["pi_prompt"], tokenizer)

        ri_tokens = model.to_tokens(ri_formatted)
        pi_tokens = model.to_tokens(pi_formatted)

        # The prompts should be the same length (only "first"/"last" differs)
        ri_len = ri_tokens.shape[1]
        pi_len = pi_tokens.shape[1]

        with torch.no_grad():
            ri_logits, ri_cache = model.run_with_cache(ri_tokens)
            pi_logits, pi_cache = model.run_with_cache(pi_tokens)

        # Check correctness
        ri_pred = tokenizer.decode([ri_logits[0, -1].argmax().item()]).strip()
        pi_pred = tokenizer.decode([pi_logits[0, -1].argmax().item()]).strip()
        ri_ok = ri_pred.lower() == pair["initial_value"].lower()
        pi_ok = pi_pred.lower() == pair["final_value"].lower()
        ri_correct_count += ri_ok
        pi_correct_count += pi_ok

        # Find where "first"/"last" token is (query start)
        ri_str_tokens = model.to_str_tokens(ri_formatted)
        query_start = -1
        for i in range(len(ri_str_tokens) - 1, -1, -1):
            if "What" in ri_str_tokens[i]:
                query_start = i
                break

        # Compare residual streams at each layer
        for layer in range(n_layers):
            ri_resid = ri_cache["resid_post", layer][0]  # [seq, d_model]
            pi_resid = pi_cache["resid_post", layer][0]

            # At answer position (last token)
            # Use min length in case they differ by 1 token
            ans_pos = min(ri_len, pi_len) - 1
            ri_ans = ri_resid[ans_pos]
            pi_ans = pi_resid[ans_pos]

            diff = torch.norm(ri_ans - pi_ans).item()
            ri_norm = torch.norm(ri_ans).item()
            pi_norm = torch.norm(pi_ans).item()
            # Normalized divergence
            divergence_at_answer[layer] += diff / ((ri_norm + pi_norm) / 2 + 1e-10)

            # Cosine similarity
            cos = torch.nn.functional.cosine_similarity(
                ri_ans.unsqueeze(0), pi_ans.unsqueeze(0)
            ).item()
            cosine_at_answer[layer] += cos

            # At query start position
            if query_start >= 0 and query_start < min(ri_len, pi_len):
                ri_q = ri_resid[query_start]
                pi_q = pi_resid[query_start]
                q_diff = torch.norm(ri_q - pi_q).item()
                q_norm = (torch.norm(ri_q).item() + torch.norm(pi_q).item()) / 2
                divergence_at_query_start[layer] += q_diff / (q_norm + 1e-10)

        del ri_cache, pi_cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        if t_idx < 3:
            print(f"  Trial {t_idx}: RI='{ri_pred}'({'OK' if ri_ok else 'WRONG'}) "
                  f"PI='{pi_pred}'({'OK' if pi_ok else 'WRONG'})")

    divergence_at_answer /= args.trials
    cosine_at_answer /= args.trials
    divergence_at_query_start /= args.trials

    print(f"\n  Accuracy: RI={ri_correct_count}/{args.trials}, PI={pi_correct_count}/{args.trials}")
    print(f"\n  Layer-by-layer divergence (||resid_RI - resid_PI|| / ||resid||):")
    print(f"  {'Layer':>5} {'Answer div':>12} {'Query div':>12} {'Cosine@ans':>12}")
    print(f"  {'-'*5} {'-'*12} {'-'*12} {'-'*12}")
    for layer in range(n_layers):
        print(f"  {layer:>5} {divergence_at_answer[layer]:>12.6f} "
              f"{divergence_at_query_start[layer]:>12.6f} "
              f"{cosine_at_answer[layer]:>12.6f}")

    # Summary: where does divergence peak?
    peak_answer_layer = int(np.argmax(divergence_at_answer))
    peak_query_layer = int(np.argmax(divergence_at_query_start))
    print(f"\n  Peak divergence at answer position: layer {peak_answer_layer} "
          f"({divergence_at_answer[peak_answer_layer]:.6f})")
    print(f"  Peak divergence at query position: layer {peak_query_layer} "
          f"({divergence_at_query_start[peak_query_layer]:.6f})")
    print(f"  Min cosine similarity at answer: layer {int(np.argmin(cosine_at_answer))} "
          f"({cosine_at_answer[int(np.argmin(cosine_at_answer))]:.6f})")

    # ══════════════════════════════════════════════════════════════════════
    # PART B: L16H3 QK analysis — what does it attend to?
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print("PART B: Primacy-biased head (L16H3) attention analysis")
    print(f"{'='*70}")

    # Load head identification results to find primacy-biased heads
    from core.output import load_head_identification, get_output_path
    try:
        primacy_heads = load_head_identification(args.model, args.keys, args.updates)
        print(f"  Loaded {len(primacy_heads)} primacy-biased heads")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No head_identification results for {args.model} at {args.keys}k_{args.updates}u. "
            f"Run exp 16 first."
        )

    # For each primacy-biased head, analyze its attention pattern in RI vs PI
    # Focus on: does it attend to the "first"/"last" query word differently?

    for target_layer, target_head in primacy_heads[:5]:  # top 5
        print(f"\n  --- L{target_layer}H{target_head} ---")

        # Collect attention patterns across trials
        ri_attn_to_regions = {"instruction": [], "values": [], "query_word": [],
                              "query_rest": [], "initial": [], "final": []}
        pi_attn_to_regions = {"instruction": [], "values": [], "query_word": [],
                              "query_rest": [], "initial": [], "final": []}

        for t_idx in range(min(args.trials, 15)):
            seed = hash(("head_qk", t_idx, args.updates)) % (2**31)
            pair = build_matched_ri_pi(args.keys, args.updates, seed, value_pool, categories)

            init_tid = value_to_tid.get(pair["initial_value"], -1)
            final_tid = value_to_tid.get(pair["final_value"], -1)

            for cond, prompt in [("RI", pair["ri_prompt"]), ("PI", pair["pi_prompt"])]:
                formatted = format_for_chat(prompt, tokenizer)
                tokens = model.to_tokens(formatted)
                token_ids = tokens[0].tolist()
                str_tokens = model.to_str_tokens(formatted)
                seq_len = tokens.shape[1]

                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens)

                pattern = cache["pattern", target_layer]  # [batch, heads, seq, seq]
                head_attn = pattern[0, target_head, -1, :]  # [seq] — from answer pos

                # Classify positions
                init_pos = [i for i, t in enumerate(token_ids) if t == init_tid]
                final_pos = [i for i, t in enumerate(token_ids) if t == final_tid]

                # Find query word position ("first" or "last")
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

                # Everything before query is either instruction or values
                query_start = query_rest_pos[0] if query_rest_pos else seq_len
                value_pos = init_pos + final_pos
                for v in pair["all_values"][1:-1]:
                    vtid = value_to_tid.get(v, -1)
                    if vtid >= 0:
                        value_pos.extend(i for i, t in enumerate(token_ids) if t == vtid)
                value_pos = sorted(set(value_pos))
                instr_pos = [i for i in range(query_start)
                             if i not in value_pos and i not in init_pos and i not in final_pos]

                target = ri_attn_to_regions if cond == "RI" else pi_attn_to_regions
                target["instruction"].append(head_attn[instr_pos].sum().item() if instr_pos else 0)
                target["values"].append(head_attn[value_pos].sum().item() if value_pos else 0)
                target["query_word"].append(head_attn[query_word_pos].sum().item() if query_word_pos else 0)
                target["query_rest"].append(head_attn[query_rest_pos].sum().item() if query_rest_pos else 0)
                target["initial"].append(head_attn[init_pos].sum().item() if init_pos else 0)
                target["final"].append(head_attn[final_pos].sum().item() if final_pos else 0)

                del cache
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        # Print comparison
        print(f"  Attention from answer position (averaged over {min(args.trials, 15)} trials):")
        print(f"  {'Region':<20} {'RI':>10} {'PI':>10} {'Diff':>10}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
        for region in ["instruction", "initial", "final", "values", "query_word", "query_rest"]:
            ri_avg = np.mean(ri_attn_to_regions[region])
            pi_avg = np.mean(pi_attn_to_regions[region])
            diff = pi_avg - ri_avg
            marker = " ***" if abs(diff) > 0.05 else ""
            print(f"  {region:<20} {ri_avg:>10.4f} {pi_avg:>10.4f} {diff:>+10.4f}{marker}")

        # Key question: does this head attend to "first"/"last" differently?
        ri_qw = np.mean(ri_attn_to_regions["query_word"])
        pi_qw = np.mean(pi_attn_to_regions["query_word"])
        if abs(ri_qw - pi_qw) < 0.01:
            print(f"  → Head IGNORES the query word (attn to 'first'/'last' is ~{ri_qw:.4f} in both)")
        else:
            print(f"  → Head SEES the query word differently (RI={ri_qw:.4f}, PI={pi_qw:.4f})")

        ri_init = np.mean(ri_attn_to_regions["initial"])
        pi_init = np.mean(pi_attn_to_regions["initial"])
        if abs(ri_init - pi_init) < 0.05:
            print(f"  → Attention to INITIAL value is the same regardless of instruction ({ri_init:.4f})")
        else:
            print(f"  → Attention to initial shifts: RI={ri_init:.4f}, PI={pi_init:.4f}")

    # Save
    save_data = {
        "model": args.model,
        "config": {"keys": args.keys, "updates": args.updates, "trials": args.trials},
        "accuracy": {"RI": ri_correct_count / args.trials, "PI": pi_correct_count / args.trials},
        "divergence_at_answer": divergence_at_answer.tolist(),
        "divergence_at_query_start": divergence_at_query_start.tolist(),
        "cosine_at_answer": cosine_at_answer.tolist(),
        "peak_divergence_layer_answer": peak_answer_layer,
        "peak_divergence_layer_query": peak_query_layer,
    }
    from core.output import save_results
    out_path = save_results(save_data, args.model, args.keys, args.updates, "instruction_sensitivity")


if __name__ == "__main__":
    main()
