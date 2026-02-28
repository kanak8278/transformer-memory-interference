"""
Exp 25c: Observational head metrics — 5 methods from a single forward pass.

Computes per-head scores using 5 established metrics:
  1. DLA (Direct Logit Attribution) — Wang et al. 2022 (IOI)
  2. Attention primacy score — attn_to_init / (attn_to_init + attn_to_final) in PI
  3. Attention entropy — low entropy = concentrated attention (Xiao et al. 2023)
  4. Copy score — does max-attended token match output? (Wu et al. 2024)
  5. Condition sensitivity — |primacy_RI - primacy_PI|

All computed from a single forward pass with cache per trial.
No thresholds — outputs continuous scores for all heads.

Usage:
    cd mechanistic_probing_v2
    uv run python experiments/25c_observational_head_metrics.py \
        --model google/gemma-3-1b-it --keys 2 --updates 2 --trials 100
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
    get_value_pool, build_interleaved_sequence, build_prompt,
)
from core.output import save_results


def build_trial(num_keys, num_updates, condition, seed, value_pool):
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
    sequence = build_interleaved_sequence(categories, values_per_cat, rng)
    prompt, expected = build_prompt(sequence, condition, test_cat)
    cat_values = [it["value"] for it in sequence if it["category"] == test_cat]

    return {
        "prompt": prompt, "condition": condition, "expected": expected,
        "initial_value": cat_values[0], "final_value": cat_values[-1],
    }


def compute_all_metrics(model, tokenizer, trial, value_to_tid):
    """Run one trial, compute all 5 metrics for every head.

    Returns dict with arrays of shape [n_layers, n_heads] for each metric.
    """
    init_val = trial["initial_value"]
    final_val = trial["final_value"]
    expected = trial["expected"]
    condition = trial["condition"]

    init_tid = value_to_tid.get(init_val, -1)
    final_tid = value_to_tid.get(final_val, -1)

    formatted = format_for_chat(trial["prompt"], tokenizer)
    tokens = model.to_tokens(formatted)
    token_ids = tokens[0].tolist()

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)

    pred_tid = logits[0, -1].argmax().item()
    pred_text = tokenizer.decode([pred_tid]).strip()
    correct = pred_text.lower() == expected.lower()

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Find value positions
    init_pos = [i for i, t in enumerate(token_ids) if t == init_tid] if init_tid >= 0 else []
    final_pos = [i for i, t in enumerate(token_ids) if t == final_tid] if final_tid >= 0 else []

    # ── Metric 1: DLA per head ──
    # logit_diff direction: logit(init) - logit(final)
    init_tid_bare = tokenizer.encode(init_val, add_special_tokens=False)[0]
    final_tid_bare = tokenizer.encode(final_val, add_special_tokens=False)[0]
    logit_dir = model.W_U[:, init_tid_bare] - model.W_U[:, final_tid_bare]

    dla = np.zeros((n_layers, n_heads))
    for layer in range(n_layers):
        z = cache["z", layer][0, -1, :, :]  # [n_heads, d_head]
        W_O = model.W_O[layer]  # [n_heads, d_head, d_model]
        for head in range(n_heads):
            head_out = z[head] @ W_O[head]  # [d_model]
            dla[layer, head] = (head_out @ logit_dir).item()

    # ── Metric 2: Attention primacy score ──
    primacy = np.full((n_layers, n_heads), 0.5)
    attn_to_init_arr = np.zeros((n_layers, n_heads))
    attn_to_final_arr = np.zeros((n_layers, n_heads))
    retrieval_score = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        pattern = cache["pattern", layer]  # [batch, n_heads, seq, seq]
        attn = pattern[0, :, -1, :]  # [n_heads, seq]

        for head in range(n_heads):
            h_attn = attn[head]
            a_init = h_attn[init_pos].sum().item() if init_pos else 0.0
            a_final = h_attn[final_pos].sum().item() if final_pos else 0.0
            attn_to_init_arr[layer, head] = a_init
            attn_to_final_arr[layer, head] = a_final
            retrieval_score[layer, head] = a_init + a_final
            denom = a_init + a_final
            if denom > 1e-8:
                primacy[layer, head] = a_init / denom

    # ── Metric 3: Attention entropy at answer position ──
    entropy = np.zeros((n_layers, n_heads))
    for layer in range(n_layers):
        pattern = cache["pattern", layer]
        attn = pattern[0, :, -1, :]  # [n_heads, seq]
        for head in range(n_heads):
            p = attn[head].clamp(min=1e-10)
            entropy[layer, head] = -(p * p.log()).sum().item()

    # ── Metric 4: Copy score ──
    # Does the token at the max-attended position match the predicted output token?
    copy_score = np.zeros((n_layers, n_heads))
    for layer in range(n_layers):
        pattern = cache["pattern", layer]
        attn = pattern[0, :, -1, :]  # [n_heads, seq]
        for head in range(n_heads):
            max_pos = attn[head].argmax().item()
            attended_tid = token_ids[max_pos] if max_pos < len(token_ids) else -1
            copy_score[layer, head] = 1.0 if attended_tid == pred_tid else 0.0

    del cache
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "correct": correct,
        "dla": dla,
        "primacy": primacy,
        "entropy": entropy,
        "copy_score": copy_score,
        "retrieval_score": retrieval_score,
        "attn_to_init": attn_to_init_arr,
        "attn_to_final": attn_to_final_arr,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--keys", type=int, default=2)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--n-ctx", type=int, default=2048)
    args = parser.parse_args()

    print("=" * 70)
    print("OBSERVATIONAL HEAD METRICS — 5 methods, 1 forward pass")
    print(f"  keys={args.keys}, updates={args.updates}, trials={args.trials}")
    print("=" * 70)

    model, tokenizer, info = load_model(args.model, n_ctx=args.n_ctx)
    candidate_pool = get_value_pool("ARBITRARY_SINGLE")
    value_to_tid = verify_single_token(tokenizer, values=candidate_pool)
    value_pool = list(value_to_tid.keys())

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Accumulators per condition
    accum = {}
    for cond in ["RI", "PI"]:
        accum[cond] = {
            "dla": np.zeros((n_layers, n_heads)),
            "primacy": np.zeros((n_layers, n_heads)),
            "entropy": np.zeros((n_layers, n_heads)),
            "copy_score": np.zeros((n_layers, n_heads)),
            "retrieval_score": np.zeros((n_layers, n_heads)),
            "n": 0,
            "n_correct": 0,
        }

    t_start = time.time()

    for t_idx in range(args.trials):
        for condition in ["RI", "PI"]:
            seed = hash((condition, t_idx, args.updates, args.keys, "25c")) % (2**31)
            trial = build_trial(args.keys, args.updates, condition, seed, value_pool)
            metrics = compute_all_metrics(model, tokenizer, trial, value_to_tid)

            a = accum[condition]
            a["dla"] += metrics["dla"]
            a["primacy"] += metrics["primacy"]
            a["entropy"] += metrics["entropy"]
            a["copy_score"] += metrics["copy_score"]
            a["retrieval_score"] += metrics["retrieval_score"]
            a["n"] += 1
            a["n_correct"] += int(metrics["correct"])

        if (t_idx + 1) % 25 == 0:
            elapsed = time.time() - t_start
            print(f"  {t_idx + 1}/{args.trials} done ({elapsed:.0f}s)")

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Average
    for cond in ["RI", "PI"]:
        n = accum[cond]["n"]
        for key in ["dla", "primacy", "entropy", "copy_score", "retrieval_score"]:
            accum[cond][key] /= n

    # ── Metric 5: Condition sensitivity ──
    condition_sensitivity = np.abs(accum["RI"]["primacy"] - accum["PI"]["primacy"])

    # ── Print results ──
    total_time = time.time() - t_start
    ri_acc = accum["RI"]["n_correct"] / accum["RI"]["n"]
    pi_acc = accum["PI"]["n_correct"] / accum["PI"]["n"]

    print(f"\n{'=' * 70}")
    print(f"RESULTS ({total_time:.0f}s)")
    print(f"{'=' * 70}")
    print(f"  RI accuracy: {ri_acc:.0%}, PI accuracy: {pi_acc:.0%}")

    # Rank heads by each metric for PI condition
    print(f"\n  Top 10 heads by each metric (PI condition):")
    print(f"  {'─' * 65}")

    metrics_to_rank = [
        ("DLA (promotes init in PI)", accum["PI"]["dla"], True),       # high = primacy
        ("Primacy score (PI)", accum["PI"]["primacy"], True),          # high = primacy
        ("Low entropy (PI)", -accum["PI"]["entropy"], True),           # low entropy = concentrated
        ("Copy score (PI)", accum["PI"]["copy_score"], True),          # high = copies
        ("Low cond. sensitivity", -condition_sensitivity, True),       # low = ignores instruction
        ("Retrieval score (PI)", accum["PI"]["retrieval_score"], True), # high = retrieves values
    ]

    for name, scores, descending in metrics_to_rank:
        flat = [(scores[l, h], l, h) for l in range(n_layers) for h in range(n_heads)]
        flat.sort(reverse=descending)
        print(f"\n  {name}:")
        for rank, (score, l, h) in enumerate(flat[:10]):
            extra = ""
            if name.startswith("DLA"):
                extra = f"  primacy={accum['PI']['primacy'][l, h]:.3f}"
            elif name.startswith("Primacy"):
                extra = f"  retr={accum['PI']['retrieval_score'][l, h]:.3f}"
            print(f"    {rank + 1:>2}. L{l}H{h}  score={score:>+8.4f}{extra}")

    # ── Rank correlation between methods ──
    from scipy.stats import spearmanr

    flat_dla = accum["PI"]["dla"].flatten()
    flat_prim = accum["PI"]["primacy"].flatten()
    flat_ent = (-accum["PI"]["entropy"]).flatten()
    flat_copy = accum["PI"]["copy_score"].flatten()
    flat_sens = (-condition_sensitivity).flatten()
    flat_retr = accum["PI"]["retrieval_score"].flatten()

    method_names = ["DLA", "Primacy", "-Entropy", "CopyScore", "-Sensitivity", "Retrieval"]
    method_arrays = [flat_dla, flat_prim, flat_ent, flat_copy, flat_sens, flat_retr]

    print(f"\n  Spearman rank correlations between methods:")
    print(f"  {'':>14}", end="")
    for name in method_names:
        print(f"  {name:>10}", end="")
    print()

    corr_matrix = {}
    for i, (n1, a1) in enumerate(zip(method_names, method_arrays)):
        print(f"  {n1:>14}", end="")
        for j, (n2, a2) in enumerate(zip(method_names, method_arrays)):
            rho, _ = spearmanr(a1, a2)
            corr_matrix[f"{n1}_vs_{n2}"] = float(rho)
            print(f"  {rho:>+10.3f}", end="")
        print()

    # ── Save ──
    save_data = {
        "model": args.model,
        "config": {"keys": args.keys, "updates": args.updates, "trials": args.trials},
        "n_layers": n_layers,
        "n_heads": n_heads,
        "accuracy": {"RI": ri_acc, "PI": pi_acc},
        "methods": {
            "dla_pi": accum["PI"]["dla"].tolist(),
            "dla_ri": accum["RI"]["dla"].tolist(),
            "primacy_pi": accum["PI"]["primacy"].tolist(),
            "primacy_ri": accum["RI"]["primacy"].tolist(),
            "entropy_pi": accum["PI"]["entropy"].tolist(),
            "entropy_ri": accum["RI"]["entropy"].tolist(),
            "copy_score_pi": accum["PI"]["copy_score"].tolist(),
            "copy_score_ri": accum["RI"]["copy_score"].tolist(),
            "retrieval_score_pi": accum["PI"]["retrieval_score"].tolist(),
            "retrieval_score_ri": accum["RI"]["retrieval_score"].tolist(),
            "condition_sensitivity": condition_sensitivity.tolist(),
        },
        "rank_correlations": corr_matrix,
        "total_time_sec": round(total_time, 1),
    }
    save_results(save_data, args.model, args.keys, args.updates, "observational_head_metrics")


if __name__ == "__main__":
    main()
