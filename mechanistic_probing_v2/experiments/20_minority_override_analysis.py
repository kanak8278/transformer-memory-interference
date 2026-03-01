"""
Exp 20: Why does the minority (primacy-biased heads) override
the majority (recency-responsive heads)?

Three hypotheses tested:
  H1: Disproportionate DLA magnitude — primacy heads push harder per head
  H2: OV circuit gain — primacy heads have larger W_OV norms
  H3: Layer position — primacy heads get the "last word" in later layers

Inputs come from exp 25a (per_head_knockout.json):
  --primacy-heads   : top_primacy_heads  (positive causal_effect = knocking out HELPS PI)
  --recency-heads   : top_retrieval_heads (negative causal_effect = knocking out HURTS PI)
  --causal-effects  : causal_effect values for each primacy head (in same order as --primacy-heads)

How to extract from exp 25a results:
    import json
    d = json.load(open("results/{model}/{keys}k_{updates}u/per_head_knockout.json"))

    # Primacy heads (help PI when knocked out):
    primacy = d["top_primacy_heads"][:5]  # take top N
    --primacy-heads  = " ".join(f"{h['layer']},{h['head']}" for h in primacy)
    --causal-effects = " ".join(f"{h['causal_effect']:.3f}" for h in primacy)

    # Recency heads (hurt PI when knocked out):
    recency = d["top_retrieval_heads"][:10]  # take top N
    --recency-heads  = " ".join(f"{h['layer']},{h['head']}" for h in recency)

Example:
    python experiments/20_minority_override_analysis.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct --keys 1 --updates 3 --trials 30 \\
        --primacy-heads "8,3 0,7 0,3" \\
        --recency-heads "12,5 16,2 22,1 8,9 3,4" \\
        --causal-effects "8.3 2.1 1.7"
"""

import sys
import json
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
from core.analysis_utils import compute_dla


def build_trial(num_keys, num_updates, condition, seed, value_pool, categories):
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

    sequence = build_interleaved_sequence(cats, values_per_cat, rng)
    prompt, expected = build_prompt(sequence, condition, test_cat)
    cat_values = [it["value"] for it in sequence if it["category"] == test_cat]

    return {
        "prompt": prompt,
        "condition": condition, "expected": expected,
        "initial_value": cat_values[0], "final_value": cat_values[-1],
        "all_values": cat_values, "seed": seed,
    }


def run_with_knockout(model, tokenizer, trial, heads_to_knock, model_name):
    """Zero out specific heads and return accuracy."""
    formatted = format_for_chat(trial["prompt"], tokenizer, model_name=model_name)
    tokens = model.to_tokens(formatted)

    hooks = []
    for layer, head in heads_to_knock:
        def make_hook(h):
            def hook_fn(z, hook):
                z[:, :, h, :] = 0.0
                return z
            return hook_fn
        hooks.append((f"blocks.{layer}.attn.hook_z", make_hook(head)))

    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

    pred_tid = logits[0, -1].argmax().item()
    pred_text = tokenizer.decode([pred_tid]).strip()
    correct = pred_text.lower() == trial["expected"].lower()
    return pred_text, correct


def parse_heads(heads_str):
    """Parse '14,2 8,3' into [(14,2), (8,3)]."""
    heads = []
    for h in heads_str.strip().split():
        parts = h.split(",")
        heads.append((int(parts[0]), int(parts[1])))
    return heads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--keys", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--primacy-heads", required=True,
                        help="Primacy heads from exp 25a top_primacy_heads. E.g., '12,0 8,8'")
    parser.add_argument("--recency-heads", required=True,
                        help="Recency heads from exp 25a top_retrieval_heads. E.g., '8,3 0,7'")
    parser.add_argument("--causal-effects", default=None,
                        help="Causal effect scores from exp 25a for each primacy head "
                             "(space-separated floats, same order as --primacy-heads). "
                             "E.g., '8.3 2.1 1.7'. If omitted, shown as N/A.")
    args = parser.parse_args()

    print("=" * 70)
    print("MINORITY OVERRIDE ANALYSIS: Why do 8 heads beat 53?")
    print(f"  keys={args.keys}, updates={args.updates}, trials={args.trials}")
    print("=" * 70)

    model, tokenizer, info = load_model(args.model, n_ctx=args.n_ctx)
    candidate_pool = get_value_pool("ARBITRARY_SINGLE")
    value_to_tid = verify_single_token(tokenizer, values=candidate_pool)
    value_pool = list(value_to_tid.keys())
    categories = ORIGINAL_CATEGORIES

    primacy_heads = parse_heads(args.primacy_heads)
    recency_heads = parse_heads(args.recency_heads)
    causal_effects = (
        [float(x) for x in args.causal_effects.strip().split()]
        if args.causal_effects else [None] * len(primacy_heads)
    )
    if len(causal_effects) != len(primacy_heads):
        raise ValueError(
            f"--causal-effects has {len(causal_effects)} values but "
            f"--primacy-heads has {len(primacy_heads)} heads"
        )

    print(f"Primacy heads (from exp 25a top_primacy_heads): {len(primacy_heads)}")
    for (l, h), ce in zip(primacy_heads, causal_effects):
        ce_str = f"  causal_effect={ce:+.3f}" if ce is not None else ""
        print(f"  L{l}H{h}{ce_str}")
    print(f"Recency heads (from exp 25a top_retrieval_heads): {len(recency_heads)}")
    for l, h in recency_heads:
        print(f"  L{l}H{h}")

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # ═══════════════════════════════════════════════════════════════════
    # H2: OV circuit gain (static weight analysis — no trials needed)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("H2: OV CIRCUIT GAIN (static weight analysis)")
    print(f"{'='*70}")

    wov_norms = np.zeros((n_layers, n_heads))
    for layer in range(n_layers):
        W_V = model.W_V[layer]   # [n_heads, d_model, d_head]
        W_O = model.W_O[layer]   # [n_heads, d_head, d_model]
        for head in range(n_heads):
            # W_OV = W_V[head] @ W_O[head]  → [d_model, d_model]
            # But computing full d_model×d_model is expensive, use Frobenius via composition
            # ||W_V @ W_O||_F = ||W_OV||_F
            # Efficient: ||A @ B||_F^2 = trace(B^T A^T A B) = ||A^T A B||_F... just do it directly
            W_OV = W_V[head] @ W_O[head]  # [d_model, d_model]
            wov_norms[layer, head] = torch.norm(W_OV, p='fro').item()

    primacy_wov = [wov_norms[l, h] for l, h in primacy_heads]
    recency_wov = [wov_norms[l, h] for l, h in recency_heads]
    all_wov = wov_norms.flatten()

    print(f"\n  W_OV Frobenius norms:")
    print(f"  {'Group':<25} {'Mean':>8} {'Median':>8} {'Max':>8}")
    print(f"  {'-'*55}")
    print(f"  {'Primacy-biased (8)':<25} {np.mean(primacy_wov):>8.2f} {np.median(primacy_wov):>8.2f} {np.max(primacy_wov):>8.2f}")
    print(f"  {'Recency-responsive (53)':<25} {np.mean(recency_wov):>8.2f} {np.median(recency_wov):>8.2f} {np.max(recency_wov):>8.2f}")
    print(f"  {'All heads (336)':<25} {np.mean(all_wov):>8.2f} {np.median(all_wov):>8.2f} {np.max(all_wov):>8.2f}")
    print(f"\n  Ratio (primacy/recency mean): {np.mean(primacy_wov)/np.mean(recency_wov):.2f}x")

    print(f"\n  Per primacy head:")
    for (l, h), norm in zip(primacy_heads, primacy_wov):
        pctl = 100 * np.mean(all_wov <= norm)
        print(f"    L{l}H{h}: ||W_OV||={norm:.2f}  (percentile: {pctl:.0f}%)")

    # ═══════════════════════════════════════════════════════════════════
    # H1: DLA magnitude comparison (needs trial runs)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("H1: DLA MAGNITUDE COMPARISON")
    print(f"{'='*70}")

    # Accumulate per-head DLA across trials
    # For PI condition specifically — that's where the override matters
    pi_dla_accum = np.zeros((n_layers, n_heads))
    ri_dla_accum = np.zeros((n_layers, n_heads))
    pi_mlp_accum = np.zeros(n_layers)
    ri_mlp_accum = np.zeros(n_layers)
    n_pi = 0
    n_ri = 0

    print(f"\nRunning {args.trials} trials × 2 conditions for DLA...")

    for t_idx in range(args.trials):
        for condition in ["RI", "PI"]:
            seed = hash((condition, t_idx, args.updates, "override")) % (2**31)
            trial = build_trial(args.keys, args.updates, condition, seed,
                                value_pool, categories)

            # Get token IDs for initial and final values
            init_val = trial["initial_value"]
            final_val = trial["final_value"]
            init_tid = value_to_tid.get(init_val, -1)
            final_tid = value_to_tid.get(final_val, -1)

            if init_tid < 0 or final_tid < 0 or init_tid == final_tid:
                continue

            formatted = format_for_chat(trial["prompt"], tokenizer, model_name=args.model)
            tokens = model.to_tokens(formatted)

            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)

            dla = compute_dla(model, cache, init_tid, final_tid, answer_position=-1)

            if condition == "PI":
                pi_dla_accum += dla.head_logit_diff
                pi_mlp_accum += dla.mlp_logit_diff
                n_pi += 1
            else:
                ri_dla_accum += dla.head_logit_diff
                ri_mlp_accum += dla.mlp_logit_diff
                n_ri += 1

            del cache

        if (t_idx + 1) % 10 == 0:
            print(f"  {t_idx + 1}/{args.trials} done")

    # Average DLA
    pi_dla_mean = pi_dla_accum / max(n_pi, 1)
    ri_dla_mean = ri_dla_accum / max(n_ri, 1)
    pi_mlp_mean = pi_mlp_accum / max(n_pi, 1)
    ri_mlp_mean = ri_mlp_accum / max(n_ri, 1)

    # Extract per-group DLA for PI condition
    # Positive logit_diff = pushes toward initial value
    # In PI: positive = HURTING (promoting wrong answer)
    primacy_dla_pi = [pi_dla_mean[l, h] for l, h in primacy_heads]
    recency_dla_pi = [pi_dla_mean[l, h] for l, h in recency_heads]

    primacy_dla_ri = [ri_dla_mean[l, h] for l, h in primacy_heads]
    recency_dla_ri = [ri_dla_mean[l, h] for l, h in recency_heads]

    print(f"\n  PI condition — DLA (logit_diff = contribution to logit(initial) - logit(final)):")
    print(f"  Positive = pushing toward initial value (WRONG for PI)")
    print(f"  {'Group':<25} {'Mean':>8} {'Sum':>8} {'|Mean|':>8} {'|Sum|':>8}")
    print(f"  {'-'*60}")
    print(f"  {'Primacy-biased (8)':<25} {np.mean(primacy_dla_pi):>+8.2f} {np.sum(primacy_dla_pi):>+8.2f} {np.mean(np.abs(primacy_dla_pi)):>8.2f} {np.sum(np.abs(primacy_dla_pi)):>8.2f}")
    print(f"  {'Recency-responsive (53)':<25} {np.mean(recency_dla_pi):>+8.2f} {np.sum(recency_dla_pi):>+8.2f} {np.mean(np.abs(recency_dla_pi)):>8.2f} {np.sum(np.abs(recency_dla_pi)):>8.2f}")
    print(f"  {'MLP total':<25} {'':>8} {np.sum(pi_mlp_mean):>+8.2f}")

    print(f"\n  Per-head ratio: primacy heads push {np.mean(np.abs(primacy_dla_pi))/max(np.mean(np.abs(recency_dla_pi)), 0.001):.1f}x harder per head")
    print(f"  Aggregate: primacy sum={np.sum(primacy_dla_pi):+.2f}, recency sum={np.sum(recency_dla_pi):+.2f}")

    # Tug of war
    total_primacy_push = np.sum(primacy_dla_pi)
    total_recency_push = np.sum(recency_dla_pi)
    net = total_primacy_push + total_recency_push
    print(f"\n  TUG OF WAR (PI condition):")
    print(f"    Primacy team pushes: {total_primacy_push:+.2f} toward initial")
    print(f"    Recency team pushes: {total_recency_push:+.2f} toward initial (negative = toward final)")
    print(f"    Net attention push:  {net:+.2f}")
    print(f"    MLP total push:      {np.sum(pi_mlp_mean):+.2f}")
    print(f"    Grand total:         {net + np.sum(pi_mlp_mean):+.2f}")

    if net > 0:
        print(f"    → Primacy team WINS the tug of war among attention heads")
    else:
        print(f"    → Recency team wins, but MLP may tip the balance")

    # Per-head breakdown for primacy heads
    print(f"\n  Per primacy head DLA (PI condition):")
    print(f"  {'Head':<10} {'DLA(PI)':>10} {'DLA(RI)':>10} {'||W_OV||':>10} {'KO Effect':>10}")
    print(f"  {'-'*55}")
    for i, (l, h) in enumerate(primacy_heads):
        ce = causal_effects[i]
        ce_str = f"{ce:>+10.3f}" if ce is not None else f"{'N/A':>10}"
        print(f"  L{l}H{h:<5} {primacy_dla_pi[i]:>+10.2f} {primacy_dla_ri[i]:>+10.2f} {primacy_wov[i]:>10.2f} {ce_str}")

    # ═══════════════════════════════════════════════════════════════════
    # H3: Per-head ablation (layer position / causal impact)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("H3: PER-HEAD ABLATION (causal impact)")
    print(f"{'='*70}")

    ablation_trials = min(args.trials, 20)
    print(f"\nRunning {ablation_trials} trials × 2 conditions × {len(primacy_heads)+1} configs...")

    ablation_results = {"baseline": {"RI": 0, "PI": 0, "n": 0}}
    for l, h in primacy_heads:
        ablation_results[f"knockout_L{l}H{h}"] = {"RI": 0, "PI": 0, "n": 0}

    for t_idx in range(ablation_trials):
        for condition in ["RI", "PI"]:
            seed = hash((condition, t_idx, args.updates, "ablation2")) % (2**31)
            trial = build_trial(args.keys, args.updates, condition, seed,
                                value_pool, categories)

            # Baseline
            formatted = format_for_chat(trial["prompt"], tokenizer, model_name=args.model)
            tokens = model.to_tokens(formatted)
            with torch.no_grad():
                logits = model(tokens)
            pred = tokenizer.decode([logits[0, -1].argmax().item()]).strip()
            correct = pred.lower() == trial["expected"].lower()
            ablation_results["baseline"][condition] += correct
            ablation_results["baseline"]["n"] += 1

            # Knockout each primacy head individually
            for l, h in primacy_heads:
                _, correct = run_with_knockout(model, tokenizer, trial, [(l, h)], args.model)
                key = f"knockout_L{l}H{h}"
                ablation_results[key][condition] += correct
                ablation_results[key]["n"] += 1

        if (t_idx + 1) % 5 == 0:
            print(f"  {t_idx + 1}/{ablation_trials} done")

    # Normalize
    n_abl = ablation_trials
    print(f"\n  {'Config':<25} {'RI':>6} {'PI':>6} {'Gap':>8} {'PI change':>10} {'Layer':>6}")
    print(f"  {'-'*65}")

    base_ri = ablation_results["baseline"]["RI"] / n_abl
    base_pi = ablation_results["baseline"]["PI"] / n_abl
    print(f"  {'baseline':<25} {base_ri:>5.0%} {base_pi:>5.0%} {base_ri-base_pi:>+7.0%} {'':>10} {'':>6}")

    head_impacts = []
    for l, h in primacy_heads:
        key = f"knockout_L{l}H{h}"
        ri = ablation_results[key]["RI"] / n_abl
        pi = ablation_results[key]["PI"] / n_abl
        pi_change = pi - base_pi
        head_impacts.append({"layer": l, "head": h, "ri": ri, "pi": pi,
                             "pi_change": pi_change, "gap": ri - pi})
        print(f"  {key:<25} {ri:>5.0%} {pi:>5.0%} {ri-pi:>+7.0%} {pi_change:>+9.0%} {l:>6}")

    # Sort by PI impact
    head_impacts.sort(key=lambda x: x["pi_change"], reverse=True)
    print(f"\n  Ranked by PI improvement when knocked out:")
    for hi in head_impacts:
        print(f"    L{hi['layer']}H{hi['head']}: PI {hi['pi_change']:+.0%} (layer {hi['layer']})")

    # ═══════════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("SYNTHESIS")
    print(f"{'='*70}")

    print(f"\nH1 (DLA magnitude):")
    ratio = np.mean(np.abs(primacy_dla_pi)) / max(np.mean(np.abs(recency_dla_pi)), 0.001)
    if ratio > 2:
        print(f"  CONFIRMED: Primacy heads push {ratio:.1f}x harder per head.")
        print(f"  Even outnumbered 8:53, their aggregate DLA ({np.sum(primacy_dla_pi):+.1f}) competes with recency ({np.sum(recency_dla_pi):+.1f}).")
    else:
        print(f"  WEAK: Primacy heads are only {ratio:.1f}x stronger per head — not a full explanation.")

    print(f"\nH2 (OV gain):")
    wov_ratio = np.mean(primacy_wov) / max(np.mean(recency_wov), 0.001)
    if wov_ratio > 1.3:
        print(f"  CONFIRMED: Primacy heads have {wov_ratio:.2f}x larger W_OV norms.")
    else:
        print(f"  NOT CONFIRMED: W_OV norms are similar ({wov_ratio:.2f}x). Gain is not the mechanism.")

    print(f"\nH3 (Layer position):")
    if head_impacts:
        last_layer = max(hi["layer"] for hi in head_impacts)
        last_layer_impact = next((hi for hi in head_impacts if hi["layer"] == last_layer), None)
        other_impacts = [hi for hi in head_impacts if hi["layer"] != last_layer]
        if last_layer_impact and other_impacts:
            last_change = last_layer_impact["pi_change"]
            other_mean = np.mean([hi["pi_change"] for hi in other_impacts])
            last_head = f"L{last_layer}H{last_layer_impact['head']}"
            if last_change > other_mean + 0.05:
                print(f"  CONFIRMED: {last_head} has outsized impact (PI {last_change:+.0%}) vs others (mean {other_mean:+.0%}).")
                print(f"  Being in the final layer gives it the 'last word' on the residual stream.")
            else:
                print(f"  NOT CONFIRMED: {last_head} impact ({last_change:+.0%}) is similar to others ({other_mean:+.0%}).")

    # ── Save ──
    save_data = {
        "model": args.model,
        "config": {"keys": args.keys, "updates": args.updates, "trials": args.trials},
        "primacy_heads": [{"layer": l, "head": h} for l, h in primacy_heads],
        "recency_heads_count": len(recency_heads),
        "H1_dla": {
            "n_pi_trials": n_pi,
            "n_ri_trials": n_ri,
            "primacy_dla_pi_per_head": primacy_dla_pi,
            "recency_dla_pi_per_head": recency_dla_pi,
            "primacy_dla_pi_sum": float(np.sum(primacy_dla_pi)),
            "recency_dla_pi_sum": float(np.sum(recency_dla_pi)),
            "primacy_dla_pi_mean_abs": float(np.mean(np.abs(primacy_dla_pi))),
            "recency_dla_pi_mean_abs": float(np.mean(np.abs(recency_dla_pi))),
            "per_head_ratio": float(ratio),
            "mlp_total_pi": float(np.sum(pi_mlp_mean)),
            "full_dla_pi": pi_dla_mean.tolist(),
            "full_mlp_pi": pi_mlp_mean.tolist(),
        },
        "H2_wov_norms": {
            "primacy_norms": primacy_wov,
            "recency_mean_norm": float(np.mean(recency_wov)),
            "primacy_mean_norm": float(np.mean(primacy_wov)),
            "ratio": float(wov_ratio),
        },
        "H3_ablation": {
            "n_trials": ablation_trials,
            "baseline_ri": base_ri,
            "baseline_pi": base_pi,
            "per_head": head_impacts,
        },
    }
    from core.output import save_results
    out_path = save_results(save_data, args.model, args.keys, args.updates, "minority_override")


if __name__ == "__main__":
    main()
