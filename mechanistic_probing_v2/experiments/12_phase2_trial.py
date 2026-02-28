"""
Logit lens + attention + DLA analysis at a single operating point.

Uses SINGLE-TOKEN values so that:
  - Logit lens can track P(initial) vs P(final) as distinct tokens
  - DLA can measure per-component contribution to each token's logit
  - Attention analysis can identify which positions heads attend to

Run once per operating point. Use --keys and --updates to specify the point.

Usage:
    cd mechanistic_probing_v2
    uv run python experiments/12_phase2_trial.py --keys 2 --updates 5 [--trials 100]
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
from core.dataset_configs import format_for_chat, SYSTEM_PROMPT, ORIGINAL_CATEGORIES
from core.model_loader import get_single_token_pool, verify_single_token
from core.token_tracker import build_token_map, summarize_token_map
from core.analysis_utils import (
    logit_lens_all_layers, logit_lens_to_dict,
    extract_attention_stats, attention_stats_to_arrays,
    compute_dla, dla_to_dict,
    get_positions_by_role,
)
from core.output import save_results


def build_single_token_trial(
    num_keys: int,
    num_updates: int,
    condition: str,
    seed: int,
    value_pool: list[str],
    categories: list[str] = None,
) -> dict:
    """Build a trial using single-token values."""
    rng = random.Random(seed)

    if categories is None:
        categories = ORIGINAL_CATEGORIES[:num_keys]

    total_values_needed = num_keys * num_updates
    if total_values_needed > len(value_pool):
        raise ValueError(f"Need {total_values_needed} unique values, pool has {len(value_pool)}")

    selected_values = rng.sample(value_pool, total_values_needed)

    values_per_cat = {}
    idx = 0
    for cat in categories:
        values_per_cat[cat] = selected_values[idx:idx + num_updates]
        idx += num_updates

    test_cat = categories[seed % num_keys]

    items = []
    for cat in categories:
        for ui, val in enumerate(values_per_cat[cat]):
            items.append({"category": cat, "value": val, "update_idx": ui})

    rng.shuffle(items)
    for attempt in range(100):
        ok = True
        for i in range(1, len(items)):
            if items[i]["category"] == items[i-1]["category"]:
                ok = False
                break
        if ok:
            break
        rng.shuffle(items)

    stream_lines = [f"{it['category']}: {it['value']}" for it in items]
    stream_text = "\n".join(stream_lines)
    query_word = "first" if condition == "RI" else "last"

    cat_values = [it["value"] for it in items if it["category"] == test_cat]
    expected = cat_values[0] if condition == "RI" else cat_values[-1]
    initial_value = cat_values[0]
    final_value = cat_values[-1]
    intermediate_values = cat_values[1:-1] if len(cat_values) > 2 else []

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
        "initial_value": initial_value,
        "final_value": final_value,
        "intermediate_values": intermediate_values,
        "all_values": cat_values,
        "num_keys": num_keys,
        "num_updates": num_updates,
        "seed": seed,
    }


def run_analysis(model, tokenizer, trial: dict, value_to_tid: dict) -> dict:
    """Run logit lens + attention + DLA on one trial."""

    init_tid = value_to_tid[trial["initial_value"]]
    final_tid = value_to_tid[trial["final_value"]]
    inter_tids = [value_to_tid[v] for v in trial["intermediate_values"]]

    init_bare_tid = tokenizer.encode(trial["initial_value"], add_special_tokens=False)[0]
    final_bare_tid = tokenizer.encode(trial["final_value"], add_special_tokens=False)[0]
    init_tids = list(set([init_tid, init_bare_tid]))
    final_tids = list(set([final_tid, final_bare_tid]))

    print(f"    Values: init='{trial['initial_value']}' (tids={init_tids}), "
          f"final='{trial['final_value']}' (tids={final_tids})")

    if init_tid == final_tid:
        print(f"    WARNING: init and final share same token ID!")

    formatted = format_for_chat(trial["prompt"], tokenizer)
    tokens = model.to_tokens(formatted)
    token_ids = tokens[0].tolist()
    str_tokens = model.to_str_tokens(formatted)
    seq_len = tokens.shape[1]

    print(f"    Tokens: {seq_len}")

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)

    pred_tid = logits[0, -1].argmax().item()
    pred_text = tokenizer.decode([pred_tid]).strip()
    correct = pred_text.lower() == trial["expected"].lower()
    print(f"    Predicts: '{pred_text}' (expected: '{trial['expected']}') -> {'CORRECT' if correct else 'WRONG'}")

    tmap = build_token_map(str_tokens, token_ids, init_tid, final_tid, inter_tids)
    tmap_summary = summarize_token_map(tmap)
    print(f"    Token map: init@{tmap.initial_value_positions}, "
          f"final@{tmap.final_value_positions}, "
          f"intermediate: {len(tmap.intermediate_value_positions)} positions")

    # ── Logit Lens ──
    ll_results = logit_lens_all_layers(model, cache, init_tids, final_tids, answer_position=-1)

    sample_layers = list(range(0, model.cfg.n_layers, max(1, model.cfg.n_layers // 6)))
    for r in [ll_results[i] for i in sample_layers]:
        winner = "INIT" if r.prob_initial > r.prob_final else "FINAL"
        print(f"      L{r.layer:>2}: P(init)={r.prob_initial:.4f} P(final)={r.prob_final:.4f} "
              f"rank_i={r.rank_initial:>5} rank_f={r.rank_final:>5} -> {winner}")

    # ── Attention ──
    attn_stats = extract_attention_stats(
        cache, model.cfg.n_layers, model.cfg.n_heads,
        answer_position=seq_len - 1,
        initial_positions=tmap.initial_value_positions,
        final_positions=tmap.final_value_positions,
        intermediate_positions=tmap.intermediate_value_positions,
        query_positions=tmap.query_positions,
        instruction_positions=tmap.instruction_positions,
    )
    attn_arrays = attention_stats_to_arrays(attn_stats, model.cfg.n_layers, model.cfg.n_heads)

    retr = attn_arrays["retrieval_score"]
    prim = attn_arrays["primacy_score"]
    flat_top5 = np.argsort(retr.ravel())[-5:][::-1]
    top_idx = np.unravel_index(flat_top5, retr.shape)
    print(f"    Top 5 retrieval heads:")
    for l, h in zip(top_idx[0], top_idx[1]):
        print(f"      L{l}H{h}: retr={retr[l,h]:.4f} primacy={prim[l,h]:.4f}")

    # ── DLA ──
    dla_result = compute_dla(model, cache, init_tid, final_tid, answer_position=-1)
    hd = dla_result.head_logit_diff
    flat_top5_dla = np.argsort(np.abs(hd).ravel())[-5:][::-1]
    top_dla = np.unravel_index(flat_top5_dla, hd.shape)
    print(f"    Top 5 heads by |DLA|:")
    for l, h in zip(top_dla[0], top_dla[1]):
        print(f"      L{l}H{h}: logit_diff={hd[l,h]:+.4f}")
    print(f"    MLP total: {dla_result.mlp_logit_diff.sum():+.4f}")

    del cache
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "condition": trial["condition"],
        "seed": trial["seed"],
        "initial_value": trial["initial_value"],
        "final_value": trial["final_value"],
        "expected": trial["expected"],
        "predicted": pred_text,
        "correct": correct,
        "seq_len": seq_len,
        "token_map": tmap_summary,
        "logit_lens": logit_lens_to_dict(ll_results),
        "attention": {
            "retrieval_score": attn_arrays["retrieval_score"].tolist(),
            "primacy_score": attn_arrays["primacy_score"].tolist(),
            "attn_to_initial": attn_arrays["attn_to_initial"].tolist(),
            "attn_to_final": attn_arrays["attn_to_final"].tolist(),
        },
        "dla": dla_to_dict(dla_result),
    }


def main():
    parser = argparse.ArgumentParser(description="Logit lens + attention + DLA at a single operating point")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--keys", type=int, default=2)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--n-ctx", type=int, default=2048)
    args = parser.parse_args()

    print("=" * 70)
    print(f"LOGIT LENS — keys={args.keys}, updates={args.updates}")
    print("=" * 70)

    model, tokenizer, info = load_model(args.model, n_ctx=args.n_ctx)

    value_to_tid = verify_single_token(tokenizer)
    value_pool = list(value_to_tid.keys())
    print(f"\nVerified {len(value_pool)} single-token values")

    categories = ORIGINAL_CATEGORIES

    all_results = {
        "model": args.model,
        "n_layers": info.n_layers,
        "n_heads": info.n_heads,
        "d_model": info.d_model,
        "keys": args.keys,
        "updates": args.updates,
        "trials_per_condition": args.trials,
        "analyses": [],
    }

    t_start = time.time()

    for condition in ["RI", "PI"]:
        print(f"\n  --- {condition} ---")
        for t_idx in range(args.trials):
            seed = hash((condition, t_idx, args.updates, "logit_lens")) % (2**31)
            trial = build_single_token_trial(
                num_keys=args.keys,
                num_updates=args.updates,
                condition=condition,
                seed=seed,
                value_pool=value_pool,
                categories=categories[:args.keys],
            )
            print(f"\n  Trial {t_idx+1}/{args.trials} (seed={seed})")

            t0 = time.time()
            result = run_analysis(model, tokenizer, trial, value_to_tid)
            result["elapsed_sec"] = round(time.time() - t0, 1)
            all_results["analyses"].append(result)

    total = time.time() - t_start
    all_results["total_elapsed_sec"] = round(total, 1)

    save_results(all_results, args.model, args.keys, args.updates, "logit_lens")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY ({total:.1f}s total)")
    print(f"{'='*70}")

    for cond in ["RI", "PI"]:
        cond_res = [r for r in all_results["analyses"] if r["condition"] == cond]
        n_correct = sum(1 for r in cond_res if r["correct"])
        print(f"  {cond}: {n_correct}/{len(cond_res)} correct")

        if cond_res:
            avg_pi = np.mean([r["logit_lens"]["prob_initial"][-1] for r in cond_res])
            avg_pf = np.mean([r["logit_lens"]["prob_final"][-1] for r in cond_res])
            print(f"    Last layer avg: P(init)={avg_pi:.4f}, P(final)={avg_pf:.4f}")


if __name__ == "__main__":
    main()
