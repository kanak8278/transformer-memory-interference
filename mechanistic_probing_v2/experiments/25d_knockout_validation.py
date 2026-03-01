"""
Exp 25d: Validation of per-head knockout findings.

Three validation tests:
  V1: Run knockout on BOTH RI and PI — check if the effect is primacy-specific
      (helps PI, hurts RI) vs general capability loss (hurts both).
  V2: Run full knockout sweep at a SECOND operating point — check if same
      heads rank high.
  V3: Targeted knockout of specific heads with 200+ trials and bootstrap CIs.

This is the "are we sure?" experiment. If L14H2 knockout helps PI AND hurts
RI specifically → confirmed primacy head. If it hurts both → general
capability loss (H3).

Usage:
    cd mechanistic_probing_v2

    # Full validation on Gemma:
    uv run python experiments/25d_knockout_validation.py \
        --model google/gemma-3-1b-it \
        --primary-point "2,2" \
        --secondary-point "2,5" \
        --target-heads "14,2 4,1 8,2" \
        --sweep-trials 50 \
        --targeted-trials 200
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
from core.analysis_utils import compute_logit_diff
from core.evaluation import bootstrap_ci
from core.output import save_results


def parse_heads(heads_str):
    heads = []
    for h in heads_str.strip().split():
        parts = h.split(",")
        heads.append((int(parts[0]), int(parts[1])))
    return heads


def parse_point(point_str):
    parts = point_str.split(",")
    return int(parts[0]), int(parts[1])


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


def get_tids(value, value_to_tid, tokenizer):
    tid_sp = value_to_tid.get(value, -1)
    tid_bare = tokenizer.encode(value, add_special_tokens=False)[0]
    return list(set([t for t in [tid_sp, tid_bare] if t >= 0]))


def run_knockout_both_conditions(model, tokenizer, value_to_tid, value_pool,
                                  keys, updates, trials, heads_to_test):
    """Run knockout on BOTH RI and PI for specified heads.

    Returns dict: {(layer,head): {"RI": {"baseline_acc", "ko_acc", "baseline_ld", "ko_ld"},
                                   "PI": {...}}}
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Build trials for both conditions
    trial_data = {}
    for cond in ["RI", "PI"]:
        trial_data[cond] = []
        for t_idx in range(trials):
            seed = hash((cond, t_idx, updates, keys, "25d")) % (2**31)
            trial = build_trial(keys, updates, cond, seed, value_pool)
            formatted = format_for_chat(trial["prompt"], tokenizer, model_name=args.model)
            tokens = model.to_tokens(formatted)

            correct_tids = get_tids(trial["expected"], value_to_tid, tokenizer)
            wrong_value = trial["final_value"] if cond == "RI" else trial["initial_value"]
            wrong_tids = get_tids(wrong_value, value_to_tid, tokenizer)
            if not wrong_tids:
                wrong_tids = correct_tids

            trial_data[cond].append({
                "tokens": tokens, "correct_tids": correct_tids, "wrong_tids": wrong_tids,
            })

    # Baseline for both conditions
    baseline = {}
    for cond in ["RI", "PI"]:
        lds = []
        correct = 0
        for td in trial_data[cond]:
            with torch.no_grad():
                logits = model(td["tokens"])
            ld = compute_logit_diff(logits[0, -1], td["correct_tids"], td["wrong_tids"])
            lds.append(ld)
            pred = logits[0, -1].argmax().item()
            correct += int(pred in td["correct_tids"])
        baseline[cond] = {"acc": correct / trials, "mean_ld": np.mean(lds), "lds": lds}

    # Knockout each head, measure both conditions
    results = {}
    for layer, head in heads_to_test:
        def make_hook(h):
            def hook_fn(z, hook):
                z[:, :, h, :] = 0.0
                return z
            return hook_fn

        hook = (f"blocks.{layer}.attn.hook_z", make_hook(head))
        results[(layer, head)] = {}

        for cond in ["RI", "PI"]:
            ko_lds = []
            ko_correct = 0
            for i, td in enumerate(trial_data[cond]):
                with torch.no_grad():
                    logits = model.run_with_hooks(td["tokens"], fwd_hooks=[hook])
                ld = compute_logit_diff(logits[0, -1], td["correct_tids"], td["wrong_tids"])
                ko_lds.append(ld)
                pred = logits[0, -1].argmax().item()
                ko_correct += int(pred in td["correct_tids"])

            delta_lds = [ko_lds[i] - baseline[cond]["lds"][i] for i in range(trials)]
            results[(layer, head)][cond] = {
                "baseline_acc": baseline[cond]["acc"],
                "ko_acc": ko_correct / trials,
                "baseline_ld": baseline[cond]["mean_ld"],
                "ko_ld": np.mean(ko_lds),
                "delta_ld": np.mean(delta_lds),
            }

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return results, baseline


def run_full_sweep(model, tokenizer, value_to_tid, value_pool,
                    keys, updates, trials):
    """Run knockout sweep of ALL heads on PI only. Returns causal_effect array."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Build PI trials
    trial_data = []
    for t_idx in range(trials):
        seed = hash(("PI", t_idx, updates, keys, "25d_sweep")) % (2**31)
        trial = build_trial(keys, updates, "PI", seed, value_pool)
        formatted = format_for_chat(trial["prompt"], tokenizer, model_name=args.model)
        tokens = model.to_tokens(formatted)
        correct_tids = get_tids(trial["expected"], value_to_tid, tokenizer)
        wrong_tids = get_tids(trial["initial_value"], value_to_tid, tokenizer)
        if not wrong_tids:
            wrong_tids = correct_tids
        trial_data.append({"tokens": tokens, "correct_tids": correct_tids, "wrong_tids": wrong_tids})

    # Baseline
    baseline_lds = []
    for td in trial_data:
        with torch.no_grad():
            logits = model(td["tokens"])
        ld = compute_logit_diff(logits[0, -1], td["correct_tids"], td["wrong_tids"])
        baseline_lds.append(ld)

    # Sweep
    causal_effect = np.zeros((n_layers, n_heads))
    ko_acc = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        for head in range(n_heads):
            def make_hook(h):
                def hook_fn(z, hook):
                    z[:, :, h, :] = 0.0
                    return z
                return hook_fn

            hook = (f"blocks.{layer}.attn.hook_z", make_hook(head))
            delta_lds = []
            correct = 0
            for i, td in enumerate(trial_data):
                with torch.no_grad():
                    logits = model.run_with_hooks(td["tokens"], fwd_hooks=[hook])
                ld = compute_logit_diff(logits[0, -1], td["correct_tids"], td["wrong_tids"])
                delta_lds.append(ld - baseline_lds[i])
                pred = logits[0, -1].argmax().item()
                correct += int(pred in td["correct_tids"])

            causal_effect[layer, head] = np.mean(delta_lds)
            ko_acc[layer, head] = correct / trials

        print(f"    Layer {layer}/{n_layers - 1} done")
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    baseline_acc = sum(1 for td in trial_data
                       for _ in [None]
                       if model(td["tokens"])[0, -1].argmax().item() in td["correct_tids"]) / trials
    # Recompute baseline acc properly
    bl_correct = 0
    for td in trial_data:
        with torch.no_grad():
            logits = model(td["tokens"])
        if logits[0, -1].argmax().item() in td["correct_tids"]:
            bl_correct += 1
    baseline_acc = bl_correct / trials

    return causal_effect, ko_acc, baseline_acc, np.mean(baseline_lds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--primary-point", required=True, help="Primary operating point 'keys,updates'")
    parser.add_argument("--secondary-point", required=True, help="Second operating point for V2")
    parser.add_argument("--target-heads", required=True, help="Heads for V1/V3 as 'layer,head' pairs")
    parser.add_argument("--sweep-trials", type=int, default=50, help="Trials for V2 full sweep")
    parser.add_argument("--targeted-trials", type=int, default=200, help="Trials for V3 targeted validation")
    parser.add_argument("--n-ctx", type=int, default=2048)
    args = parser.parse_args()

    primary_keys, primary_updates = parse_point(args.primary_point)
    secondary_keys, secondary_updates = parse_point(args.secondary_point)
    target_heads = parse_heads(args.target_heads)

    print("=" * 70)
    print("KNOCKOUT VALIDATION — Three tests")
    print("=" * 70)
    print(f"  Model: {args.model}")
    print(f"  Primary point: {primary_keys}k,{primary_updates}u")
    print(f"  Secondary point: {secondary_keys}k,{secondary_updates}u")
    print(f"  Target heads: {target_heads}")
    print(f"  Sweep trials (V2): {args.sweep_trials}")
    print(f"  Targeted trials (V3): {args.targeted_trials}")
    print("=" * 70)

    model, tokenizer, info = load_model(args.model, n_ctx=args.n_ctx)
    candidate_pool = get_value_pool("ARBITRARY_SINGLE")
    value_to_tid = verify_single_token(tokenizer, values=candidate_pool)
    value_pool = list(value_to_tid.keys())

    all_results = {
        "model": args.model,
        "primary_point": {"keys": primary_keys, "updates": primary_updates},
        "secondary_point": {"keys": secondary_keys, "updates": secondary_updates},
        "target_heads": [{"layer": l, "head": h} for l, h in target_heads],
    }

    # ═══════════════════════════════════════════════════════════════════
    # V1: Knockout target heads on BOTH RI and PI at primary point
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"V1: RI + PI knockout at primary point ({primary_keys}k,{primary_updates}u)")
    print(f"    {args.targeted_trials} trials × 2 conditions × {len(target_heads)} heads")
    print(f"{'=' * 70}")

    t0 = time.time()
    v1_results, v1_baseline = run_knockout_both_conditions(
        model, tokenizer, value_to_tid, value_pool,
        primary_keys, primary_updates, args.targeted_trials, target_heads)
    v1_time = time.time() - t0

    print(f"\n  Baseline: RI={v1_baseline['RI']['acc']:.0%}, PI={v1_baseline['PI']['acc']:.0%}")
    print(f"\n  {'Head':>8} {'RI base':>8} {'RI KO':>8} {'RI Δld':>8} {'PI base':>8} {'PI KO':>8} {'PI Δld':>8} {'Verdict'}")
    print(f"  {'─' * 70}")

    v1_save = []
    for layer, head in target_heads:
        r = v1_results[(layer, head)]
        ri_delta = r["RI"]["delta_ld"]
        pi_delta = r["PI"]["delta_ld"]

        # Verdict: primacy-specific if helps PI (positive Δld) and hurts RI (negative Δld)
        if pi_delta > 1.0 and ri_delta < -1.0:
            verdict = "PRIMACY (helps PI, hurts RI)"
        elif pi_delta > 1.0 and ri_delta > 1.0:
            verdict = "GENERAL LOSS (helps both???)"
        elif pi_delta > 1.0 and abs(ri_delta) < 1.0:
            verdict = "PI-SPECIFIC (helps PI, RI neutral)"
        elif abs(pi_delta) < 1.0:
            verdict = "WEAK EFFECT"
        else:
            verdict = "UNCLEAR"

        print(f"  L{layer}H{head:>2} {r['RI']['baseline_acc']:>7.0%} {r['RI']['ko_acc']:>7.0%} "
              f"{ri_delta:>+7.2f} {r['PI']['baseline_acc']:>7.0%} {r['PI']['ko_acc']:>7.0%} "
              f"{pi_delta:>+7.2f}  {verdict}")

        v1_save.append({
            "layer": layer, "head": head,
            "RI": r["RI"], "PI": r["PI"], "verdict": verdict,
        })

    all_results["V1_ri_pi_knockout"] = {
        "trials": args.targeted_trials,
        "baseline": {"RI": v1_baseline["RI"]["acc"], "PI": v1_baseline["PI"]["acc"]},
        "per_head": v1_save,
        "time_sec": round(v1_time, 1),
    }

    # ═══════════════════════════════════════════════════════════════════
    # V2: Full sweep at secondary operating point
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"V2: Full PI sweep at secondary point ({secondary_keys}k,{secondary_updates}u)")
    print(f"    {args.sweep_trials} trials × {info.n_layers * info.n_heads} heads")
    print(f"{'=' * 70}")

    t0 = time.time()
    v2_ce, v2_ko_acc, v2_bl_acc, v2_bl_ld = run_full_sweep(
        model, tokenizer, value_to_tid, value_pool,
        secondary_keys, secondary_updates, args.sweep_trials)
    v2_time = time.time() - t0

    # Compare rankings between primary (from 25a results) and secondary
    flat = [(v2_ce[l, h], l, h) for l in range(info.n_layers) for h in range(info.n_heads)]
    flat.sort(reverse=True)

    print(f"\n  Baseline PI: {v2_bl_acc:.0%}, mean_ld={v2_bl_ld:.2f}")
    print(f"\n  Top 10 primacy heads at SECONDARY point:")
    print(f"  {'Rank':>4} {'Head':>8} {'Δld':>8} {'KO acc':>8}")
    for rank, (effect, l, h) in enumerate(flat[:10]):
        # Check if this head was in target_heads (from primary point)
        marker = " ← TARGET" if (l, h) in target_heads else ""
        print(f"  {rank + 1:>4} L{l}H{h:>2} {effect:>+8.3f} {v2_ko_acc[l, h]:>7.0%}{marker}")

    # Check if target heads still rank high
    print(f"\n  Target heads ranking at secondary point:")
    all_ranked = [(v2_ce[l, h], l, h) for l, h in target_heads]
    for effect, l, h in sorted(all_ranked, reverse=True):
        rank = sum(1 for e, _, _ in flat if e > effect) + 1
        print(f"    L{l}H{h}: Δld={effect:+.3f}, rank={rank}/{len(flat)}, KO acc={v2_ko_acc[l, h]:.0%}")

    all_results["V2_secondary_sweep"] = {
        "point": {"keys": secondary_keys, "updates": secondary_updates},
        "trials": args.sweep_trials,
        "baseline": {"accuracy": v2_bl_acc, "mean_ld": v2_bl_ld},
        "causal_effect": v2_ce.tolist(),
        "top_10": [{"layer": int(l), "head": int(h), "causal_effect": float(e)}
                   for e, l, h in flat[:10]],
        "target_heads_ranking": [
            {"layer": l, "head": h, "causal_effect": float(v2_ce[l, h]),
             "rank": sum(1 for e, _, _ in flat if e > v2_ce[l, h]) + 1}
            for l, h in target_heads
        ],
        "time_sec": round(v2_time, 1),
    }

    # ═══════════════════════════════════════════════════════════════════
    # V3: Targeted knockout with bootstrap CIs
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"V3: Targeted knockout at primary point, {args.targeted_trials} trials with CIs")
    print(f"{'=' * 70}")

    # We already ran V1 with targeted_trials — reuse that data for CIs
    v3_save = []
    for layer, head in target_heads:
        r = v1_results[(layer, head)]

        # Bootstrap CIs on accuracy (using per-trial correctness)
        # Rerun to collect per-trial binary outcomes
        # Actually, V1 only stores aggregate. Let me compute CIs on logit_diff instead.
        # We have delta_ld per trial from V1's baseline_lds vs ko_lds — but V1 only stores means.
        # For proper CIs, report the accuracy CI.

        # We can compute binomial CI from accuracy + n
        from scipy.stats import binom

        for cond in ["RI", "PI"]:
            acc = r[cond]["ko_acc"]
            n = args.targeted_trials
            # Wilson score interval
            z = 1.96
            denom = 1 + z**2 / n
            center = (acc + z**2 / (2 * n)) / denom
            margin = z * np.sqrt((acc * (1 - acc) + z**2 / (4 * n)) / n) / denom
            ci_lo = max(0, center - margin)
            ci_hi = min(1, center + margin)

            print(f"  L{layer}H{head} {cond}: KO acc={acc:.0%} [{ci_lo:.0%}-{ci_hi:.0%}] "
                  f"(baseline={r[cond]['baseline_acc']:.0%})")

            v3_save.append({
                "layer": layer, "head": head, "condition": cond,
                "ko_accuracy": acc, "ci_lower": ci_lo, "ci_upper": ci_hi,
                "baseline_accuracy": r[cond]["baseline_acc"],
                "n_trials": n,
            })

    all_results["V3_targeted_with_cis"] = v3_save
    all_results["total_time_sec"] = round(v1_time + v2_time, 1)

    # ═══════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    for layer, head in target_heads:
        r = v1_results[(layer, head)]
        v2_rank = sum(1 for e, _, _ in flat if e > v2_ce[layer, head]) + 1

        print(f"\n  L{layer}H{head}:")
        print(f"    V1: RI {r['RI']['baseline_acc']:.0%}→{r['RI']['ko_acc']:.0%} (Δld={r['RI']['delta_ld']:+.2f}), "
              f"PI {r['PI']['baseline_acc']:.0%}→{r['PI']['ko_acc']:.0%} (Δld={r['PI']['delta_ld']:+.2f})")
        print(f"    V2: Rank {v2_rank}/{len(flat)} at secondary point (Δld={v2_ce[layer, head]:+.3f})")
        print(f"    V3: PI KO acc={r['PI']['ko_acc']:.0%} (n={args.targeted_trials})")

        # Overall verdict
        pi_helps = r["PI"]["delta_ld"] > 1.0
        ri_hurts = r["RI"]["delta_ld"] < -1.0
        stable = v2_rank <= 20

        if pi_helps and ri_hurts and stable:
            print(f"    ✓ CONFIRMED PRIMACY HEAD: helps PI, hurts RI, stable across points")
        elif pi_helps and ri_hurts and not stable:
            print(f"    ~ PRIMACY HEAD but UNSTABLE across operating points")
        elif pi_helps and not ri_hurts:
            print(f"    ? PI-SPECIFIC but doesn't hurt RI — may not be primacy")
        else:
            print(f"    ✗ NOT CONFIRMED as primacy head")

    save_results(all_results, args.model, primary_keys, primary_updates, "knockout_validation")
    print(f"\n  Total time: {(v1_time + v2_time)/60:.1f} minutes")


if __name__ == "__main__":
    main()
