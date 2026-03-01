"""
Exp 30: Copy-Suppression Test — V1-promotion vs V2-suppression.

Answers: is the primacy mechanism implemented via:
  (A) V1-PROMOTION: primacy head's output actively promotes the initial value logit
  (B) V2-SUPPRESSION: primacy head suppresses the final value via negative OV eigenvalues
  (C) BOTH

Three tests:
  Test 1: OV eigenspectrum — negative dominant eigenvalue → copy-suppression
          NOTE: Requires MHA architecture (Pythia, GPT-2).
                GQA models (Qwen, Gemma) collapse eigenspectrum to ~0 — auto-detected.
  Test 2: Directional suppression score — static weight analysis, works on all architectures
  Test 3: DLA decomposed into V1-promotion vs V2-suppression — forward passes, all architectures

Works on any model:
  - Instruct (Qwen, Gemma): chat template applied automatically
  - Base (Pythia, GPT-2):   fixed few-shot demos applied automatically
  - MHA (Pythia, GPT-2):    all 3 tests run
  - GQA (Qwen, Gemma):      Test 1 skipped with warning, Tests 2&3 run

Heads come from exp 25a (per_head_knockout.json → top_primacy_heads).
How to extract:
    import json
    d = json.load(open("results/{model}/{keys}k_{updates}u/per_head_knockout.json"))
    primacy = d["top_primacy_heads"][:5]
    --heads = " ".join(f"{h['layer']},{h['head']}" for h in primacy)

Usage:
    # Pythia (MHA, base model) — all 3 tests
    python experiments/30_copy_suppression_test.py \\
        --model EleutherAI/pythia-410m --keys 2 --updates 3 --trials 50 \\
        --heads "5,2 8,6"

    # Qwen (GQA, instruct) — Tests 2 & 3 only
    python experiments/30_copy_suppression_test.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct --keys 1 --updates 3 --trials 50 \\
        --heads "8,3 0,7"

    # Gemma (GQA, instruct) — Tests 2 & 3 only
    python experiments/30_copy_suppression_test.py \\
        --model google/gemma-3-1b-it --keys 2 --updates 2 --trials 50 \\
        --heads "14,2 4,1"
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
    get_value_pool, build_interleaved_sequence, build_prompt, format_for_chat,
)
from core.output import save_results


def parse_heads(heads_str):
    """Parse '14,2 8,3' into [(14,2), (8,3)]."""
    heads = []
    for h in heads_str.strip().split():
        parts = h.split(",")
        heads.append((int(parts[0]), int(parts[1])))
    return heads


def detect_gqa(model):
    """Check if model uses GQA (n_key_value_heads < n_heads).

    Returns (is_gqa, n_kv_heads). MHA models return (False, n_heads).
    """
    cfg = model.cfg
    n_heads = cfg.n_heads
    n_kv = getattr(cfg, "n_key_value_heads", None)
    if n_kv is None or n_kv == 0:
        n_kv = n_heads  # MHA — all heads have own KV
    return n_kv < n_heads, n_kv


def build_trial(num_keys, num_updates, condition, seed, value_pool, model_name):
    """Build a trial using dataset_configs building blocks."""
    rng = random.Random(seed)
    from core.dataset_configs import ORIGINAL_CATEGORIES
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
        "test_category": test_cat,
        "initial_value": cat_values[0], "final_value": cat_values[-1],
        "all_values": cat_values,
    }


# ─── TEST 1: OV Eigenspectrum ─────────────────────────────────────────────────

def test_ov_eigenspectrum(model, info, primacy_heads, is_gqa, n_kv_heads):
    """OV eigenspectrum analysis.

    MHA: eigenvalues are nonzero → interpretable.
    GQA: eigenspectrum collapses to 0 → unreliable, skipped.
    """
    print("\n" + "=" * 70)
    print("TEST 1: OV Eigenspectrum")
    print("=" * 70)

    if is_gqa:
        print(f"  SKIPPED: Model uses GQA ({n_kv_heads} KV heads vs {info.n_heads} Q heads).")
        print(f"  The OV eigenspectrum collapses to ~0 under GQA — not interpretable.")
        print(f"  See Tests 2 & 3 for architecture-agnostic results.")
        return {"skipped": True, "reason": "GQA architecture — eigenspectrum not interpretable"}

    n_layers = info.n_layers
    n_heads = info.n_heads
    results = []

    for layer in range(n_layers):
        W_V = model.W_V[layer]  # [n_heads, d_model, d_head]
        W_O = model.W_O[layer]  # [n_heads, d_head, d_model]

        for head in range(n_heads):
            W_OV = W_V[head] @ W_O[head]        # [d_model, d_model]
            W_OV_head = W_O[head] @ W_V[head]   # [d_head, d_head]

            try:
                eigvals = torch.linalg.eigvals(W_OV_head.cpu())
                eigvals_real = eigvals.real.tolist()
                max_real = max(eigvals_real)
                min_real = min(eigvals_real)
                neg_frac = sum(1 for e in eigvals_real if e < 0) / len(eigvals_real)
                abs_eigvals = [abs(e) for e in eigvals_real]
                dominant_eigval = eigvals_real[abs_eigvals.index(max(abs_eigvals))]
                dominant_sign = "negative" if dominant_eigval < 0 else "positive"
            except Exception as ex:
                eigvals_real = []
                max_real = min_real = neg_frac = 0.0
                dominant_eigval = 0.0
                dominant_sign = "error"

            try:
                _, S, _ = torch.linalg.svd(W_OV, full_matrices=False)
                top_singular = S[:5].cpu().tolist()
            except Exception:
                top_singular = []

            is_primacy = (layer, head) in primacy_heads
            results.append({
                "layer": layer, "head": head, "is_primacy": is_primacy,
                "dominant_eigval": dominant_eigval, "dominant_sign": dominant_sign,
                "neg_fraction": neg_frac,
                "max_eigval_real": max_real, "min_eigval_real": min_real,
                "top_singular_values": top_singular,
            })

    primacy_r = [r for r in results if r["is_primacy"]]
    other_r = [r for r in results if not r["is_primacy"]]

    print(f"\n  Primacy heads ({len(primacy_r)}):")
    for r in primacy_r:
        print(f"    L{r['layer']}H{r['head']}: dominant={r['dominant_eigval']:+.4f} "
              f"({r['dominant_sign']}), neg_frac={r['neg_fraction']:.2f}")

    if other_r:
        p_neg = np.mean([r["neg_fraction"] for r in primacy_r]) if primacy_r else 0
        o_neg = np.mean([r["neg_fraction"] for r in other_r])
        print(f"\n  neg_frac: primacy={p_neg:.3f}, other={o_neg:.3f}")

        all_dom = [abs(r["dominant_eigval"]) for r in results]
        if max(all_dom) < 0.001:
            print(f"  WARNING: All eigenvalues near zero — unexpected for MHA")
        else:
            print(f"  ✓ MHA eigenspectrum is valid (max |dominant| = {max(all_dom):.4f})")

    return results


# ─── TEST 2: Directional Suppression Score ────────────────────────────────────

def test_suppression_score(model, info, primacy_heads, value_to_tid, n_tokens=100):
    """Directional suppression score — works on all architectures."""
    print("\n" + "=" * 70)
    print("TEST 2: Directional Suppression Score (all architectures)")
    print("  Negative score = head suppresses that value (V2-suppression)")
    print("=" * 70)

    n_layers = info.n_layers
    n_heads = info.n_heads
    W_U = model.W_U

    all_values = list(value_to_tid.items())
    rng = random.Random(42)
    rng.shuffle(all_values)
    sample_tokens = all_values[:n_tokens]

    results = []
    for layer in range(n_layers):
        W_V = model.W_V[layer]
        W_O = model.W_O[layer]
        for head in range(n_heads):
            W_OV = W_V[head] @ W_O[head]
            scores = [(W_U[:, tid] @ W_OV @ W_U[:, tid]).item()
                      for _, tid in sample_tokens]
            is_primacy = (layer, head) in primacy_heads
            results.append({
                "layer": layer, "head": head, "is_primacy": is_primacy,
                "mean_suppression_score": np.mean(scores),
                "std_suppression_score": np.std(scores),
                "negative_fraction": sum(1 for s in scores if s < 0) / len(scores),
            })

    primacy_r = [r for r in results if r["is_primacy"]]
    other_r = [r for r in results if not r["is_primacy"]]

    print(f"\n  Primacy heads ({len(primacy_r)}):")
    for r in primacy_r:
        print(f"    L{r['layer']}H{r['head']}: score={r['mean_suppression_score']:+.4f} "
              f"neg_frac={r['negative_fraction']:.2f}")

    if primacy_r and other_r:
        p_s = np.mean([r["mean_suppression_score"] for r in primacy_r])
        o_s = np.mean([r["mean_suppression_score"] for r in other_r])
        direction = "MORE" if p_s < o_s else "LESS"
        print(f"\n  Primacy={p_s:+.4f} vs Other={o_s:+.4f} → primacy heads {direction} suppressive")

    return results


# ─── TEST 3: DLA Decomposed ───────────────────────────────────────────────────

def test_dla_decomposed(model, info, primacy_heads, tokenizer, value_to_tid, value_pool,
                        num_keys, num_updates, n_trials, model_name):
    """DLA decomposition into V1-promotion vs V2-suppression. Works on all architectures."""
    print("\n" + "=" * 70)
    print("TEST 3: DLA Decomposed — V1-promotion vs V2-suppression (all architectures)")
    print(f"  {n_trials} PI trials, keys={num_keys}, updates={num_updates}")
    print("=" * 70)

    n_layers = info.n_layers
    n_heads = info.n_heads
    W_U = model.W_U

    head_dla_v1 = np.zeros((n_layers, n_heads))
    head_dla_v2 = np.zeros((n_layers, n_heads))
    n_correct = 0
    n_valid = 0

    for t_idx in range(n_trials):
        seed = hash(("PI_copy_supp", t_idx, model_name)) % (2**31)
        trial = build_trial(num_keys, num_updates, "PI", seed, value_pool, model_name)

        v1_tid = value_to_tid.get(trial["initial_value"])
        v2_tid = value_to_tid.get(trial["final_value"])
        if v1_tid is None or v2_tid is None or v1_tid == v2_tid:
            continue

        formatted = format_for_chat(trial["prompt"], tokenizer, model_name=model_name)
        tokens = model.to_tokens(formatted)

        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens)

        ans_pos = tokens.shape[1] - 1
        pred_tid = logits[0, ans_pos].argmax().item()
        is_correct = pred_tid in [v2_tid,
                                   *tokenizer.encode(" " + trial["final_value"],
                                                     add_special_tokens=False)[:1]]

        u_v1 = W_U[:, v1_tid]
        u_v2 = W_U[:, v2_tid]
        sp_v1 = tokenizer.encode(" " + trial["initial_value"], add_special_tokens=False)
        sp_v2 = tokenizer.encode(" " + trial["final_value"], add_special_tokens=False)
        if len(sp_v1) == 1:
            u_v1 = u_v1 + W_U[:, sp_v1[0]]
        if len(sp_v2) == 1:
            u_v2 = u_v2 + W_U[:, sp_v2[0]]

        for layer in range(n_layers):
            hook_name = f"blocks.{layer}.attn.hook_result"
            if hook_name in cache:
                head_outputs = cache[hook_name][0, ans_pos]
            else:
                z = cache[f"blocks.{layer}.attn.hook_z"][0, ans_pos]
                head_outputs = torch.einsum("hd,hdm->hm", z, model.W_O[layer])

            for head in range(n_heads):
                h_out = head_outputs[head]
                head_dla_v1[layer, head] += (h_out @ u_v1).item()
                head_dla_v2[layer, head] += (h_out @ u_v2).item()

        if is_correct:
            n_correct += 1
        n_valid += 1

        del cache
        from core.model_loader import clear_accelerator_cache
        clear_accelerator_cache(str(next(model.parameters()).device))

        if (t_idx + 1) % 20 == 0:
            print(f"  {t_idx + 1}/{n_trials} done ({n_correct}/{n_valid} correct)")

    if n_valid == 0:
        return {}

    head_dla_v1 /= n_valid
    head_dla_v2 /= n_valid

    print(f"\n  {n_valid} valid trials, {n_correct} correct ({n_correct/n_valid:.1%})")
    print(f"\n  Primacy heads:")
    print(f"  {'Head':<8} {'DLA_v1':>10} {'DLA_v2':>10} {'logit_diff':>12} {'mechanism':>20}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*12} {'─'*20}")

    primacy_summary = []
    for (l, h) in sorted(primacy_heads):
        if l >= n_layers or h >= n_heads:
            continue
        v1 = head_dla_v1[l, h]
        v2 = head_dla_v2[l, h]
        ld = v1 - v2
        if v2 < -0.1 and abs(v1) < abs(v2):
            mechanism = "V2-SUPPRESSION"
        elif v1 > 0.1 and abs(v1) > abs(v2):
            mechanism = "V1-PROMOTION"
        elif v2 < -0.1 and v1 > 0.1:
            mechanism = "BOTH"
        else:
            mechanism = "WEAK"
        print(f"  L{l}H{h:<4} {v1:>+10.4f} {v2:>+10.4f} {ld:>+12.4f} {mechanism:>20}")
        primacy_summary.append({
            "layer": l, "head": h,
            "dla_v1": v1, "dla_v2": v2, "logit_diff": ld, "mechanism": mechanism,
        })

    # Top 10 all heads
    all_heads = sorted(
        [(l, h, head_dla_v1[l, h], head_dla_v2[l, h],
          head_dla_v1[l, h] - head_dla_v2[l, h], (l, h) in primacy_heads)
         for l in range(n_layers) for h in range(n_heads)],
        key=lambda x: abs(x[4]), reverse=True
    )
    print(f"\n  Top 10 by |logit_diff|:")
    for l, h, v1, v2, ld, is_p in all_heads[:10]:
        tag = "***" if is_p else ""
        print(f"  L{l}H{h:<4} {v1:>+10.4f} {v2:>+10.4f} {ld:>+12.4f} {tag}")

    return {
        "n_valid": n_valid, "n_correct": n_correct,
        "pi_accuracy": n_correct / n_valid,
        "primacy_head_summary": primacy_summary,
        "head_dla_v1": head_dla_v1.tolist(),
        "head_dla_v2": head_dla_v2.tolist(),
    }


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Copy-Suppression Test (any model)")
    parser.add_argument("--model", default="EleutherAI/pythia-410m")
    parser.add_argument("--keys", type=int, default=2)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--heads", required=True,
                        help="Primacy heads from exp 25a top_primacy_heads. E.g., '5,2 8,6'")
    args = parser.parse_args()

    model, tokenizer, info = load_model(args.model, n_ctx=args.n_ctx)

    is_gqa, n_kv_heads = detect_gqa(model)
    arch_str = f"GQA ({n_kv_heads} KV heads)" if is_gqa else "MHA"

    print(f"\nModel: {info.name}")
    print(f"  {info.n_layers}L × {info.n_heads}H, d_model={info.d_model}")
    print(f"  Architecture: {arch_str}")
    if is_gqa:
        print(f"  → Test 1 (eigenspectrum) will be skipped (GQA confound)")
        print(f"  → Tests 2 & 3 will run normally")

    candidate_pool = get_value_pool("ARBITRARY_SINGLE")
    value_to_tid = verify_single_token(tokenizer, values=candidate_pool)
    value_pool = list(value_to_tid.keys())
    print(f"\n  {len(value_pool)} single-token values verified")

    primacy_heads = set(parse_heads(args.heads))
    print(f"  Primacy heads (from exp 25a): {sorted(primacy_heads)}")

    print("\n" + "=" * 70)
    print(f"COPY-SUPPRESSION TEST — {info.name.split('/')[-1]} ({arch_str})")
    print(f"  keys={args.keys}, updates={args.updates}, trials={args.trials}")
    print("=" * 70)

    t0 = time.time()

    eigen_results = test_ov_eigenspectrum(model, info, primacy_heads, is_gqa, n_kv_heads)
    supp_results = test_suppression_score(model, info, primacy_heads, value_to_tid)
    dla_results = test_dla_decomposed(
        model, info, primacy_heads, tokenizer, value_to_tid, value_pool,
        args.keys, args.updates, args.trials, args.model,
    )

    elapsed = time.time() - t0

    # ── VERDICT ──
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if not is_gqa and isinstance(eigen_results, list):
        primacy_eigen = [r for r in eigen_results if r["is_primacy"]]
        if primacy_eigen:
            p_neg = np.mean([r["neg_fraction"] for r in primacy_eigen])
            print(f"Test 1 (eigenspectrum): primacy neg_frac={p_neg:.2f}")
            print(f"  → {'V2-SUPPRESSION signal' if p_neg > 0.5 else 'V1-PROMOTION signal'}")
    else:
        print("Test 1: SKIPPED (GQA)")

    if isinstance(supp_results, list) and supp_results:
        primacy_supp = [r for r in supp_results if r["is_primacy"]]
        other_supp = [r for r in supp_results if not r["is_primacy"]]
        if primacy_supp:
            p_s = np.mean([r["mean_suppression_score"] for r in primacy_supp])
            o_s = np.mean([r["mean_suppression_score"] for r in other_supp]) if other_supp else 0
            print(f"Test 2 (suppression score): primacy={p_s:+.4f}, other={o_s:+.4f}")
            print(f"  → {'V2-SUPPRESSION' if p_s < o_s else 'V1-PROMOTION'} (primacy {'more' if p_s < o_s else 'less'} suppressive)")

    if dla_results and "primacy_head_summary" in dla_results:
        mechs = [r["mechanism"] for r in dla_results["primacy_head_summary"]]
        from collections import Counter
        print(f"Test 3 (DLA): {Counter(mechs)}")
        dominant = Counter(mechs).most_common(1)[0][0]
        print(f"  → Dominant mechanism: {dominant}")

    print(f"\n  Total time: {elapsed:.0f}s")

    save_results({
        "model": args.model,
        "architecture": arch_str,
        "is_gqa": is_gqa,
        "config": {"keys": args.keys, "updates": args.updates, "trials": args.trials},
        "primacy_heads": list(sorted(primacy_heads)),
        "test1_eigenspectrum": eigen_results,
        "test2_suppression": supp_results,
        "test3_dla": dla_results,
    }, args.model, args.keys, args.updates, "copy_suppression_test")


if __name__ == "__main__":
    main()
