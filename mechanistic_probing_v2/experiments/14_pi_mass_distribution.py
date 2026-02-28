"""
Where does the probability mass land in PI (recall last)?

Runs at a single (keys, updates) operating point and tracks:
  - Which position index gets the highest P?
  - Is it the literal last (v_{N-1})? Second-to-last? Somewhere in the middle?
  - How concentrated vs spread is the distribution?

Usage:
    cd mechanistic_probing_v2
    uv run python experiments/14_pi_mass_distribution.py --keys 2 --updates 5 [--trials 100]
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


def build_trial(num_keys, num_updates, condition, seed, value_pool, categories):
    rng = random.Random(seed)
    total_needed = num_keys * num_updates
    if total_needed > len(value_pool):
        raise ValueError(f"Need {total_needed} values, have {len(value_pool)}")

    selected = rng.sample(value_pool, total_needed)
    values_per_cat = {}
    idx = 0
    for cat in categories[:num_keys]:
        values_per_cat[cat] = selected[idx:idx + num_updates]
        idx += num_updates

    test_cat = categories[seed % num_keys]

    items = []
    for cat in categories[:num_keys]:
        for ui, val in enumerate(values_per_cat[cat]):
            items.append({"category": cat, "value": val, "update_idx": ui})

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

    prompt = (
        f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{stream_text}\n\n"
        f"What was the {query_word} value of {test_cat}?"
    )

    return {
        "prompt": prompt,
        "condition": condition,
        "test_category": test_cat,
        "all_values": cat_values,
        "num_updates": num_updates,
        "seed": seed,
    }


def run_analysis(model, tokenizer, trial, value_to_tid):
    all_values = trial["all_values"]
    n_values = len(all_values)

    # Get token IDs (space-prefixed + bare)
    value_tids_sp = []
    value_tids_bare = []
    for v in all_values:
        sp = tokenizer.encode(f" {v}", add_special_tokens=False)
        bare = tokenizer.encode(v, add_special_tokens=False)
        value_tids_sp.append(sp[0] if len(sp) == 1 else -1)
        value_tids_bare.append(bare[0] if len(bare) == 1 else -1)

    formatted = format_for_chat(trial["prompt"], tokenizer)
    tokens = model.to_tokens(formatted)
    token_ids = tokens[0].tolist()
    seq_len = tokens.shape[1]

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)

    pred_tid = logits[0, -1].argmax().item()
    pred_text = tokenizer.decode([pred_tid]).strip()

    n_layers = model.cfg.n_layers

    # P(v_i) at last 3 layers
    prob_last_layers = {}
    for layer in [n_layers - 3, n_layers - 2, n_layers - 1]:
        resid = cache["resid_post", layer]
        h = resid[0, -1, :]
        layer_logits = h @ model.W_U + model.b_U
        probs = torch.softmax(layer_logits, dim=-1)

        p_row = []
        for vi in range(n_values):
            tids = set()
            if value_tids_sp[vi] >= 0:
                tids.add(value_tids_sp[vi])
            if value_tids_bare[vi] >= 0:
                tids.add(value_tids_bare[vi])
            p_row.append(max(probs[t].item() for t in tids) if tids else 0.0)

        prob_last_layers[layer] = p_row

    # Attention from answer position to each value position (last layer)
    value_positions = []
    for vi in range(n_values):
        tid = value_tids_sp[vi]
        if tid >= 0:
            positions = [i for i, t in enumerate(token_ids) if t == tid]
            value_positions.append(positions[0] if positions else -1)
        else:
            value_positions.append(-1)

    pattern = cache["pattern", n_layers - 1]
    attn_from_answer = pattern[0, :, -1, :]  # [n_heads, seq]
    attn_per_value = []
    for vi in range(n_values):
        pos = value_positions[vi]
        if pos >= 0:
            attn_per_value.append(attn_from_answer[:, pos].sum().item())
        else:
            attn_per_value.append(0.0)

    del cache
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    probs_final = prob_last_layers[n_layers - 1]
    argmax_pos = int(np.argmax(probs_final))
    max_prob = probs_final[argmax_pos]

    # Concentration: what fraction of total value prob is in top-1 vs top-3?
    total_p = sum(probs_final)
    top1_frac = max_prob / total_p if total_p > 0 else 0
    sorted_p = sorted(probs_final, reverse=True)
    top3_frac = sum(sorted_p[:3]) / total_p if total_p > 0 else 0

    return {
        "condition": trial["condition"],
        "num_updates": trial["num_updates"],
        "seed": trial["seed"],
        "predicted": pred_text,
        "expected": trial["all_values"][-1] if trial["condition"] == "PI" else trial["all_values"][0],
        "correct": pred_text.lower() == (trial["all_values"][-1] if trial["condition"] == "PI" else trial["all_values"][0]).lower(),
        "all_values": trial["all_values"],
        "probs_final_layer": probs_final,
        "attn_final_layer": attn_per_value,
        "argmax_position": argmax_pos,
        "argmax_value": trial["all_values"][argmax_pos],
        "max_prob": max_prob,
        "top1_concentration": top1_frac,
        "top3_concentration": top3_frac,
        # Relative position: 0 = first, 1 = last
        "argmax_relative_pos": argmax_pos / (n_values - 1) if n_values > 1 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--keys", type=int, default=2)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--n-ctx", type=int, default=2048)
    args = parser.parse_args()

    n_updates = args.updates

    print("=" * 70)
    print("PI MASS DISTRIBUTION ANALYSIS")
    print(f"  keys={args.keys}, updates={n_updates}, trials={args.trials}")
    print("=" * 70)

    model, tokenizer, info = load_model(args.model, n_ctx=args.n_ctx)
    value_to_tid = verify_single_token(tokenizer)
    value_pool = list(value_to_tid.keys())

    categories = ORIGINAL_CATEGORIES

    all_results = {
        "model": args.model,
        "config": {"keys": args.keys, "trials": args.trials, "updates": n_updates},
        "analyses": [],
    }

    t_start = time.time()

    for condition in ["RI", "PI"]:
        print(f"\n--- {condition} ---")
        for t_idx in range(args.trials):
            seed = hash((condition, t_idx, args.updates, "pi_mass")) % (2**31)
            trial = build_trial(args.keys, n_updates, condition, seed, value_pool, categories)

            result = run_analysis(model, tokenizer, trial, value_to_tid)
            result["elapsed_sec"] = 0
            all_results["analyses"].append(result)

            status = "OK" if result["correct"] else "WRONG"
            print(f"  t{t_idx}: pred='{result['predicted']}' exp='{result['expected']}' "
                  f"{status} | peak@v{result['argmax_position']}/{n_updates-1} "
                  f"(rel={result['argmax_relative_pos']:.2f}) "
                  f"P={result['max_prob']:.4f} "
                  f"top1={result['top1_concentration']:.0%} top3={result['top3_concentration']:.0%}")

    total = time.time() - t_start
    all_results["total_elapsed_sec"] = round(total, 1)

    save_results(all_results, args.model, args.keys, n_updates, "pi_mass_distribution")

    # ── Aggregate ──
    print(f"\n{'='*70}")
    print("AGGREGATE: Where does peak probability land?")
    print(f"{'='*70}")

    print(f"\n{'Cond':>4} {'Acc':>5} | {'Peak@':>6} {'RelPos':>7} | "
          f"{'MaxP':>6} {'Top1%':>6} {'Top3%':>6}")
    print("-" * 60)

    for condition in ["RI", "PI"]:
        results = [r for r in all_results["analyses"] if r["condition"] == condition]
        if not results:
            continue

        n_correct = sum(1 for r in results if r["correct"])
        acc = n_correct / len(results)
        avg_peak = np.mean([r["argmax_position"] for r in results])
        avg_rel = np.mean([r["argmax_relative_pos"] for r in results])
        avg_maxp = np.mean([r["max_prob"] for r in results])
        avg_top1 = np.mean([r["top1_concentration"] for r in results])
        avg_top3 = np.mean([r["top3_concentration"] for r in results])

        print(f"{condition:>4} {acc:>5.0%} | "
              f"v{avg_peak:>4.1f} {avg_rel:>7.2f} | "
              f"{avg_maxp:>6.4f} {avg_top1:>5.0%} {avg_top3:>5.0%}")

    # ── PI-specific: peak position histogram ──
    pi_results = [r for r in all_results["analyses"] if r["condition"] == "PI"]
    if pi_results:
        print(f"\nPI peak positions:")
        rel_peaks = [r["argmax_relative_pos"] for r in pi_results]
        at_last = sum(1 for r in pi_results if r["argmax_position"] == n_updates - 1)
        in_first_quarter = sum(1 for r in pi_results if r["argmax_relative_pos"] <= 0.25)
        n = len(pi_results)
        print(f"  @last={at_last}/{n} first_25%={in_first_quarter}/{n} "
              f"avg_rel_pos={np.mean(rel_peaks):.2f}")


if __name__ == "__main__":
    main()
