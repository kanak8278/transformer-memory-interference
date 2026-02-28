"""
Experiment 24: OV Theory Disambiguation Tests

Three competing theories for why forced attention helps 0.5B but hurts 1.5B:
  T1: OV circuit is genuinely broken at scale
  T2: Residual stream at final value position is contaminated with initial value
  T3: Forced attention is an invalid intervention

Four tests:
  A: Residual stream probe at value positions — P(init) at final value position
  B: OV output decomposition — what does the head output promote when forced?
  C: Value position patching — patch clean residual at final value position
  D: Forced attention control — force to random non-value position

Usage:
    cd mechanistic_probing_v2
    uv run python experiments/24_ov_theory_tests.py --model Qwen/Qwen2.5-0.5B-Instruct --keys 1 --updates 5
    uv run python experiments/24_ov_theory_tests.py --model Qwen/Qwen2.5-1.5B-Instruct --keys 1 --updates 3
"""

import sys
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
from core.output import save_results


def build_trial(num_keys, num_updates, condition, seed, value_pool):
    """Build a trial, return prompt + metadata about value positions."""
    rng = random.Random(seed)
    categories = rng.sample(ORIGINAL_CATEGORIES, min(num_keys, len(ORIGINAL_CATEGORIES)))

    total_needed = num_keys * num_updates
    selected = rng.sample(value_pool, total_needed)
    values_per_cat = {}
    idx = 0
    for cat in categories:
        values_per_cat[cat] = selected[idx:idx + num_updates]
        idx += num_updates

    test_cat = categories[seed % num_keys]

    items = []
    for cat in categories:
        for val in values_per_cat[cat]:
            items.append({"category": cat, "value": val})

    rng.shuffle(items)
    for _ in range(100):
        ok = all(items[i]["category"] != items[i-1]["category"] for i in range(1, len(items)))
        if ok:
            break
        rng.shuffle(items)

    stream_lines = [f"{it['category']}: {it['value']}" for it in items]
    stream_text = "\n".join(stream_lines)
    query_word = "first" if condition == "RI" else "last"
    cat_values = [it["value"] for it in items if it["category"] == test_cat]
    expected = cat_values[0] if condition == "RI" else cat_values[-1]

    prompt = (
        f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{stream_text}\n\n"
        f"What was the {query_word} value of {test_cat}?"
    )

    return {
        "prompt": prompt,
        "expected": expected,
        "condition": condition,
        "test_category": test_cat,
        "initial_value": cat_values[0],
        "final_value": cat_values[-1],
        "all_values": cat_values,
        "items": items,
    }


def build_clean_trial(test_cat, value, seed, value_pool):
    """Build a clean trial with just 1 value for the test category."""
    rng = random.Random(seed)
    prompt = (
        f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{test_cat}: {value}\n\n"
        f"What was the first value of {test_cat}?"
    )
    return {"prompt": prompt, "expected": value}


def find_token_position(tokens, token_id, tokenizer):
    """Find position of token_id in token sequence."""
    token_list = tokens[0].tolist()
    positions = [i for i, t in enumerate(token_list) if t == token_id]
    return positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--keys", type=int, default=1)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--n-ctx", type=int, default=2048)
    args = parser.parse_args()

    model, tokenizer, info = load_model(args.model, n_ctx=args.n_ctx)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    value_to_tid = verify_single_token(tokenizer)
    value_pool = list(value_to_tid.keys())

    model_short = args.model.split("/")[-1]
    print("=" * 70)
    print(f"OV THEORY TESTS — {model_short}")
    print(f"  keys={args.keys}, updates={args.updates}, trials={args.trials}")
    print(f"  {n_layers} layers, {n_heads} heads")
    print("=" * 70)

    # ── TEST A: Residual stream probe at value positions ──
    print("\n" + "=" * 70)
    print("TEST A: P(initial_value) at final value's token position")
    print("  If high on 1.5B but low on 0.5B → residual contamination (Theory 2)")
    print("=" * 70)

    test_a_results = {"RI": [], "PI": []}

    for t_idx in range(args.trials):
        for condition in ["RI", "PI"]:
            seed = hash((condition, t_idx, "testA")) % (2**31)
            trial = build_trial(args.keys, args.updates, condition, seed, value_pool)

            init_val = trial["initial_value"]
            final_val = trial["final_value"]

            init_tid = value_to_tid.get(init_val)
            final_tid = value_to_tid.get(final_val)
            if init_tid is None or final_tid is None:
                continue

            formatted = format_for_chat(trial["prompt"], tokenizer)
            tokens = model.to_tokens(formatted)

            # Find positions of initial and final values in token stream
            init_positions = find_token_position(tokens, init_tid, tokenizer)
            final_positions = find_token_position(tokens, final_tid, tokenizer)

            if not init_positions or not final_positions:
                continue

            with torch.no_grad():
                logits, cache = model.run_with_cache(tokens)

            # Check logit lens at FINAL value's position (last occurrence)
            final_pos = final_positions[-1]

            # Check multiple layers
            layer_probes = {}
            for layer in range(n_layers):
                resid = cache[f"blocks.{layer}.hook_resid_post"][0, final_pos]
                logit_at_layer = resid @ model.W_U + model.b_U
                p_init = torch.softmax(logit_at_layer, dim=-1)[init_tid].item()
                p_final = torch.softmax(logit_at_layer, dim=-1)[final_tid].item()
                layer_probes[layer] = {"p_init": p_init, "p_final": p_final}

            # Also check at answer position
            ans_pos = tokens.shape[1] - 1
            resid_ans = cache[f"blocks.{n_layers-1}.hook_resid_post"][0, ans_pos]
            logit_ans = resid_ans @ model.W_U + model.b_U
            p_init_ans = torch.softmax(logit_ans, dim=-1)[init_tid].item()
            p_final_ans = torch.softmax(logit_ans, dim=-1)[final_tid].item()

            test_a_results[condition].append({
                "init_val": init_val, "final_val": final_val,
                "p_init_at_final_pos_last_layer": layer_probes[n_layers-1]["p_init"],
                "p_final_at_final_pos_last_layer": layer_probes[n_layers-1]["p_final"],
                "p_init_at_final_pos_mid_layer": layer_probes[n_layers//2]["p_init"],
                "p_init_at_ans_pos": p_init_ans,
                "p_final_at_ans_pos": p_final_ans,
            })

            del cache
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        if (t_idx + 1) % 10 == 0:
            print(f"  {t_idx+1}/{args.trials} done")

    # Summarize Test A
    for cond in ["RI", "PI"]:
        data = test_a_results[cond]
        if not data:
            continue
        avg_pinit_at_final = np.mean([d["p_init_at_final_pos_last_layer"] for d in data])
        avg_pfinal_at_final = np.mean([d["p_final_at_final_pos_last_layer"] for d in data])
        avg_pinit_mid = np.mean([d["p_init_at_final_pos_mid_layer"] for d in data])
        print(f"\n  {cond} (n={len(data)}):")
        print(f"    At FINAL value's position (last layer): P(init)={avg_pinit_at_final:.4f}, P(final)={avg_pfinal_at_final:.4f}")
        print(f"    At FINAL value's position (mid layer):  P(init)={avg_pinit_mid:.6f}")
        if avg_pinit_at_final > 0.05:
            print(f"    → CONTAMINATION DETECTED: initial value present at final position")
        else:
            print(f"    → Clean: initial value NOT present at final position")

    # ── TEST B: OV output decomposition ──
    print("\n" + "=" * 70)
    print("TEST B: What does head OV output promote when reading from final pos?")
    print("  If promotes init → T2 (contamination). If garbage → T3 (invalid).")
    print("=" * 70)

    # Find top heads by DLA from our data
    # We'll compute OV output for top 5 heads by attention weight
    test_b_results = []

    for t_idx in range(min(args.trials, 30)):
        seed = hash(("PI", t_idx, "testB")) % (2**31)
        trial = build_trial(args.keys, args.updates, "PI", seed, value_pool)

        init_val = trial["initial_value"]
        final_val = trial["final_value"]
        init_tid = value_to_tid.get(init_val)
        final_tid = value_to_tid.get(final_val)
        if init_tid is None or final_tid is None:
            continue

        formatted = format_for_chat(trial["prompt"], tokenizer)
        tokens = model.to_tokens(formatted)

        init_positions = find_token_position(tokens, init_tid, tokenizer)
        final_positions = find_token_position(tokens, final_tid, tokenizer)
        if not init_positions or not final_positions:
            continue

        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens)

        ans_pos = tokens.shape[1] - 1
        final_pos = final_positions[-1]
        init_pos = init_positions[0]

        trial_result = {"heads": []}

        # For each layer's top heads, compute OV output from different positions
        for layer in range(n_layers):
            z = cache["z", layer][0, ans_pos, :, :]  # [n_heads, d_head]
            W_O = model.W_O[layer]
            attn_pattern = cache["pattern", layer][0, :, ans_pos, :]  # [n_heads, seq]

            for head in range(n_heads):
                attn_to_final = attn_pattern[head, final_pos].item()
                attn_to_init = attn_pattern[head, init_pos].item()

                # Only analyze heads with significant attention to values
                if attn_to_final + attn_to_init < 0.05:
                    continue

                # OV output: z[head] @ W_O[head] projected through W_U
                head_out = z[head] @ W_O[head]  # [d_model]
                logit_contribution = head_out @ model.W_U  # [vocab]

                # What does this head promote?
                p_init_contribution = logit_contribution[init_tid].item()
                p_final_contribution = logit_contribution[final_tid].item()

                trial_result["heads"].append({
                    "layer": layer, "head": head,
                    "attn_to_final": attn_to_final,
                    "attn_to_init": attn_to_init,
                    "ov_promotes_init": p_init_contribution,
                    "ov_promotes_final": p_final_contribution,
                    "ov_diff": p_init_contribution - p_final_contribution,
                })

        test_b_results.append(trial_result)
        del cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Summarize Test B: for heads attending to final, does OV promote init or final?
    heads_attending_final = []
    for tr in test_b_results:
        for h in tr["heads"]:
            if h["attn_to_final"] > 0.1:
                heads_attending_final.append(h)

    if heads_attending_final:
        avg_ov_diff = np.mean([h["ov_diff"] for h in heads_attending_final])
        promotes_init = sum(1 for h in heads_attending_final if h["ov_diff"] > 0)
        promotes_final = sum(1 for h in heads_attending_final if h["ov_diff"] <= 0)
        print(f"\n  Heads attending >10% to final value position (n={len(heads_attending_final)}):")
        print(f"    Avg OV diff (init-final): {avg_ov_diff:+.3f}")
        print(f"    Promotes init: {promotes_init} ({promotes_init/len(heads_attending_final):.0%})")
        print(f"    Promotes final: {promotes_final} ({promotes_final/len(heads_attending_final):.0%})")
        if avg_ov_diff > 0.5:
            print(f"    → SUPPORTS T1/T2: OV output promotes init even from final position")
        elif avg_ov_diff < -0.5:
            print(f"    → OV works correctly: promotes final from final position")
        else:
            print(f"    → Ambiguous: OV output is weak or mixed")

    # ── TEST D: Forced attention to random non-value position ──
    print("\n" + "=" * 70)
    print("TEST D: Force heads to random NON-VALUE position")
    print("  If output is garbage (not init or final) → T3 weakened")
    print("  If output biased toward init → contamination everywhere")
    print("=" * 70)

    test_d_pi_correct = 0
    test_d_pi_outputs_init = 0
    test_d_pi_outputs_final = 0
    test_d_pi_outputs_other = 0
    test_d_total = 0

    # Identify primacy-biased heads (top 5 by attention to init in PI)
    # Use a quick scan
    primacy_heads = []
    for t_idx in range(5):
        seed = hash(("PI", t_idx, "testD_scan")) % (2**31)
        trial = build_trial(args.keys, args.updates, "PI", seed, value_pool)
        formatted = format_for_chat(trial["prompt"], tokenizer)
        tokens = model.to_tokens(formatted)
        init_tid = value_to_tid.get(trial["initial_value"])
        init_positions = find_token_position(tokens, init_tid, tokenizer)
        if not init_positions:
            continue
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        ans_pos = tokens.shape[1] - 1
        init_pos = init_positions[0]
        for layer in range(n_layers):
            attn = cache["pattern", layer][0, :, ans_pos, init_pos]
            for head in range(n_heads):
                if attn[head].item() > 0.3:
                    primacy_heads.append((layer, head))
        del cache

    from collections import Counter
    primacy_counter = Counter(primacy_heads)
    top_primacy = [lh for lh, _ in primacy_counter.most_common(5)]
    print(f"  Primacy heads identified: {['L{}H{}'.format(l,h) for l,h in top_primacy]}")

    for t_idx in range(min(args.trials, 30)):
        seed = hash(("PI", t_idx, "testD")) % (2**31)
        trial = build_trial(args.keys, args.updates, "PI", seed, value_pool)

        init_val = trial["initial_value"]
        final_val = trial["final_value"]
        init_tid = value_to_tid.get(init_val)
        final_tid = value_to_tid.get(final_val)
        if init_tid is None or final_tid is None:
            continue

        formatted = format_for_chat(trial["prompt"], tokenizer)
        tokens = model.to_tokens(formatted)
        seq_len = tokens.shape[1]

        # Find a non-value position (instruction tokens, early in prompt)
        random_pos = min(5, seq_len - 2)  # Near start of prompt

        def make_force_hook(head_idx, target_pos):
            def hook_fn(pattern, hook):
                pattern[0, head_idx, -1, :] = 0
                pattern[0, head_idx, -1, target_pos] = 1.0
                return pattern
            return hook_fn

        # Run with forced attention to random position
        hooks = []
        for layer, head in top_primacy:
            hooks.append((
                f"blocks.{layer}.attn.hook_pattern",
                make_force_hook(head, random_pos)
            ))

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

        pred_tid = logits[0, -1].argmax().item()
        pred_text = tokenizer.decode([pred_tid]).strip().lower()

        if pred_text == final_val.lower():
            test_d_pi_correct += 1
            test_d_pi_outputs_final += 1
        elif pred_text == init_val.lower():
            test_d_pi_outputs_init += 1
        else:
            test_d_pi_outputs_other += 1
        test_d_total += 1

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    if test_d_total > 0:
        print(f"\n  PI with forced attention to NON-VALUE position (n={test_d_total}):")
        print(f"    Outputs initial value: {test_d_pi_outputs_init} ({test_d_pi_outputs_init/test_d_total:.0%})")
        print(f"    Outputs final value:   {test_d_pi_outputs_final} ({test_d_pi_outputs_final/test_d_total:.0%})")
        print(f"    Outputs other:         {test_d_pi_outputs_other} ({test_d_pi_outputs_other/test_d_total:.0%})")
        if test_d_pi_outputs_other / test_d_total > 0.7:
            print(f"    → T3 weakened: forced attention produces garbage, not biased output")
        elif test_d_pi_outputs_init / test_d_total > 0.3:
            print(f"    → Init bias persists even at non-value positions")
        else:
            print(f"    → Mixed results")

    # ── VERDICT ──
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Save
    all_results = {
        "model": args.model,
        "config": {"keys": args.keys, "updates": args.updates, "trials": args.trials},
        "test_a": test_a_results,
        "test_b_summary": {
            "n_heads_attending_final": len(heads_attending_final) if heads_attending_final else 0,
            "avg_ov_diff": float(avg_ov_diff) if heads_attending_final else None,
        },
        "test_d": {
            "total": test_d_total,
            "outputs_init": test_d_pi_outputs_init,
            "outputs_final": test_d_pi_outputs_final,
            "outputs_other": test_d_pi_outputs_other,
        },
    }
    save_results(all_results, args.model, args.keys, args.updates, "ov_theory_tests")


if __name__ == "__main__":
    main()
