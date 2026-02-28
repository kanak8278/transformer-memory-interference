"""
Experiment 30: Copy-Suppression Test on Pythia-160M (MHA Architecture)

Resolves the GQA confound from Experiment 28 (Qwen 0.5B):
  - Qwen uses GQA → eigenspectrum collapsed to 0.0 → Test 1 was inconclusive
  - Pythia uses standard MHA → eigenspectrum should work cleanly

This experiment answers: Is the mechanism V1-promotion or V2-suppression?
If Pythia also shows V1-promotion, the finding is architecture-independent.
If Pythia shows V2-suppression, the mechanism depends on MHA vs GQA.

Three tests (same as Exp 28):
  Test 1: OV eigenspectrum — should actually produce nonzero eigenvalues on MHA
  Test 2: Directional suppression score — static weight analysis
  Test 3: DLA decomposed into V1-promotion vs V2-suppression — forward passes

Since Pythia has no prior head_identification results, this script includes
an inline head identification step: run N trials, compute per-head attention
bias toward initial vs final value positions, and classify primacy heads.

Usage:
    cd mechanistic_probing_v2
    python experiments/30_pythia_copy_suppression.py
    python experiments/30_pythia_copy_suppression.py --model EleutherAI/pythia-410m --keys 2 --updates 5
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
from core.dataset import (
    generate_completion_trial, COMPLETION_CATEGORIES,
    build_completion_prompt, build_interleaved_sequence,
    generate_few_shot_demo,
)
from core.model_loader import verify_single_token
from core.output import save_results


# ─── INLINE HEAD IDENTIFICATION ──────────────────────────────────────────────

def identify_primacy_heads(model, info, tokenizer, value_pool, num_keys, num_updates,
                           n_trials=50, threshold=0.15):
    """Identify primacy-biased heads by comparing attention to initial vs final value positions.

    For each RI trial (where we know the model attends to some value positions),
    measure per-head attention weight on initial_value_position vs final_value_position.

    Heads where mean(attn_to_initial) >> mean(attn_to_final) are primacy-biased.

    Returns list of (layer, head) tuples and full attention stats.
    """
    print("\n" + "=" * 70)
    print("HEAD IDENTIFICATION (inline)")
    print(f"  {n_trials} RI trials, keys={num_keys}, updates={num_updates}")
    print(f"  threshold={threshold} (init_attn - final_attn > threshold)")
    print("=" * 70)

    n_layers = info.n_layers
    n_heads = info.n_heads

    # Accumulate attention to init vs final value positions per head
    attn_to_init = np.zeros((n_layers, n_heads))
    attn_to_final = np.zeros((n_layers, n_heads))
    n_valid = 0

    for t_idx in range(n_trials):
        seed = hash(("head_id", t_idx)) % (2**31)
        trial = generate_completion_trial(
            num_keys, num_updates, "RI", seed, value_pool,
        )

        tokens = model.to_tokens(trial.prompt)
        str_tokens = model.to_str_tokens(trial.prompt)

        # Find positions of initial and final values of test category
        init_val = trial.initial_value
        final_val = trial.final_value

        init_pos = None
        final_pos = None
        for i, tok in enumerate(str_tokens):
            tok_clean = tok.strip()
            if tok_clean == init_val and init_pos is None:
                init_pos = i
            if tok_clean == final_val:
                final_pos = i

        if init_pos is None or final_pos is None:
            # Try space-prefixed matching
            for i, tok in enumerate(str_tokens):
                tok_clean = tok.strip()
                if init_val in tok_clean and init_pos is None:
                    init_pos = i
                if final_val in tok_clean:
                    final_pos = i

        if init_pos is None or final_pos is None or init_pos == final_pos:
            continue

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        ans_pos = tokens.shape[1] - 1  # last token position

        for layer in range(n_layers):
            pattern = cache["pattern", layer]  # [batch, heads, seq, seq]
            attn = pattern[0, :, ans_pos, :]  # [heads, seq]
            attn_to_init[:, :][layer] += attn[:, init_pos].cpu().numpy()
            attn_to_final[:, :][layer] += attn[:, final_pos].cpu().numpy()

        n_valid += 1
        del cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        if (t_idx + 1) % 10 == 0:
            print(f"  {t_idx + 1}/{n_trials} trials processed ({n_valid} valid)")

    if n_valid == 0:
        print("  No valid trials for head identification!")
        return [], {}

    attn_to_init /= n_valid
    attn_to_final /= n_valid
    bias = attn_to_init - attn_to_final

    # Classify primacy heads
    primacy_heads = []
    print(f"\n  Heads with init_attn - final_attn > {threshold}:")
    print(f"  {'Head':<8} {'attn_init':>10} {'attn_final':>10} {'bias':>10}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10}")

    for layer in range(n_layers):
        for head in range(n_heads):
            if bias[layer, head] > threshold:
                primacy_heads.append((layer, head))
                print(f"  L{layer}H{head:<4} {attn_to_init[layer, head]:>10.4f} "
                      f"{attn_to_final[layer, head]:>10.4f} {bias[layer, head]:>+10.4f}")

    print(f"\n  Found {len(primacy_heads)} primacy heads (threshold={threshold})")

    # If no heads found, lower threshold
    if not primacy_heads:
        lower = threshold / 2
        print(f"  Trying lower threshold={lower}...")
        for layer in range(n_layers):
            for head in range(n_heads):
                if bias[layer, head] > lower:
                    primacy_heads.append((layer, head))
        print(f"  Found {len(primacy_heads)} at threshold={lower}")

    # If still none, take top 3 by bias
    if not primacy_heads:
        flat = [(bias[l, h], l, h) for l in range(n_layers) for h in range(n_heads)]
        flat.sort(reverse=True)
        primacy_heads = [(l, h) for _, l, h in flat[:3]]
        print(f"  Fallback: using top 3 by bias: {primacy_heads}")

    stats = {
        "n_valid": n_valid,
        "threshold": threshold,
        "attn_to_init": attn_to_init.tolist(),
        "attn_to_final": attn_to_final.tolist(),
        "bias": bias.tolist(),
        "primacy_heads": [(l, h) for l, h in primacy_heads],
    }

    return primacy_heads, stats


# ─── TEST 1: OV Eigenspectrum (MHA — should actually work) ──────────────────

def test_ov_eigenspectrum(model, info, primacy_heads):
    """OV eigenspectrum analysis. With MHA, eigenvalues should be nonzero."""
    n_layers = info.n_layers
    n_heads = info.n_heads

    print("\n" + "=" * 70)
    print("TEST 1: OV Eigenspectrum (MHA — should produce real eigenvalues)")
    print("  Copy-suppression heads have dominant NEGATIVE eigenvalues")
    print("=" * 70)

    results = []

    for layer in range(n_layers):
        W_V = model.W_V[layer]  # [n_heads, d_model, d_head]
        W_O = model.W_O[layer]  # [n_heads, d_head, d_model]

        for head in range(n_heads):
            # Full OV in residual space: x @ W_V[head] @ W_O[head]
            # W_OV = W_V[head] @ W_O[head]: [d_model, d_model]
            W_OV = W_V[head] @ W_O[head]

            # OV in head-space: W_O[head] @ W_V[head] = [d_head, d_head]
            W_OV_head = W_O[head] @ W_V[head]

            try:
                # MPS doesn't support linalg_eig — move to CPU for eigenvalues
                eigvals = torch.linalg.eigvals(W_OV_head.cpu())
                eigvals_real = eigvals.real.tolist()

                max_real = max(eigvals_real)
                min_real = min(eigvals_real)
                mean_real = np.mean(eigvals_real)
                neg_frac = sum(1 for e in eigvals_real if e < 0) / len(eigvals_real)

                abs_eigvals = [abs(e) for e in eigvals_real]
                dominant_idx = abs_eigvals.index(max(abs_eigvals))
                dominant_eigval = eigvals_real[dominant_idx]
                dominant_sign = "negative" if dominant_eigval < 0 else "positive"
            except Exception as ex:
                print(f"  WARNING: eigvals failed for L{layer}H{head}: {ex}")
                eigvals_real = []
                max_real = min_real = mean_real = neg_frac = 0.0
                dominant_eigval = 0.0
                dominant_sign = "error"

            # SVD of full OV
            try:
                U, S, Vh = torch.linalg.svd(W_OV, full_matrices=False)
                top_singular = S[:5].cpu().tolist()
            except Exception:
                top_singular = []

            is_primacy = (layer, head) in primacy_heads

            results.append({
                "layer": layer,
                "head": head,
                "is_primacy": is_primacy,
                "dominant_eigval": dominant_eigval,
                "dominant_sign": dominant_sign,
                "neg_fraction": neg_frac,
                "max_eigval_real": max_real,
                "min_eigval_real": min_real,
                "mean_eigval_real": mean_real,
                "top_singular_values": top_singular,
            })

    # Print results
    primacy_results = [r for r in results if r["is_primacy"]]
    other_results = [r for r in results if not r["is_primacy"]]

    print(f"\n  Primacy heads ({len(primacy_results)}):")
    for r in primacy_results:
        print(f"    L{r['layer']}H{r['head']}: dominant={r['dominant_eigval']:+.4f} ({r['dominant_sign']}), "
              f"neg_frac={r['neg_fraction']:.2f}, range=[{r['min_eigval_real']:.4f}, {r['max_eigval_real']:.4f}]")
        if r['top_singular_values']:
            print(f"      top SVs: {[f'{s:.3f}' for s in r['top_singular_values']]}")

    if other_results:
        other_dominant = [r["dominant_eigval"] for r in other_results]
        other_neg_frac = [r["neg_fraction"] for r in other_results]
        primacy_dominant = [r["dominant_eigval"] for r in primacy_results] if primacy_results else [0]
        primacy_neg_frac = [r["neg_fraction"] for r in primacy_results] if primacy_results else [0]

        print(f"\n  Summary:")
        print(f"    Primacy: dominant={np.mean(primacy_dominant):+.4f}, neg_frac={np.mean(primacy_neg_frac):.3f}")
        print(f"    Other:   dominant={np.mean(other_dominant):+.4f}, neg_frac={np.mean(other_neg_frac):.3f}")

        # Key check: are eigenvalues actually nonzero? (GQA confound check)
        all_dominant = [abs(r["dominant_eigval"]) for r in results]
        print(f"    |dominant| range: [{min(all_dominant):.4f}, {max(all_dominant):.4f}]")
        if max(all_dominant) < 0.001:
            print(f"    WARNING: All eigenvalues near zero — same GQA collapse as Qwen?")
        else:
            print(f"    ✓ Eigenvalues are nonzero — MHA eigenspectrum is valid")

    return results


# ─── TEST 2: Directional Suppression Score ───────────────────────────────────

def test_suppression_score(model, info, primacy_heads, value_to_tid, n_tokens=100):
    """Same as Exp 28 Test 2."""
    print("\n" + "=" * 70)
    print("TEST 2: Directional Suppression Score")
    print("  Negative score = head suppresses that value when attending near it")
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

            scores = []
            for val_word, tid in sample_tokens:
                u = W_U[:, tid]
                score = (u @ W_OV @ u).item()
                scores.append(score)

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
              f"(±{r['std_suppression_score']:.4f}), neg_frac={r['negative_fraction']:.2f}")

    if other_r:
        p_score = np.mean([r["mean_suppression_score"] for r in primacy_r]) if primacy_r else 0
        o_score = np.mean([r["mean_suppression_score"] for r in other_r])
        print(f"\n  Primacy mean: {p_score:+.4f}, Other mean: {o_score:+.4f}")
        if p_score < o_score:
            print(f"    → Primacy heads MORE suppressive (Δ={p_score - o_score:+.4f})")
        else:
            print(f"    → Primacy heads LESS suppressive (Δ={p_score - o_score:+.4f})")

    return results


# ─── TEST 3: DLA Decomposed ──────────────────────────────────────────────────

def test_dla_decomposed(model, info, primacy_heads, tokenizer, value_to_tid, value_pool,
                        num_keys, num_updates, n_trials):
    """DLA decomposition into V1-promotion vs V2-suppression."""
    print("\n" + "=" * 70)
    print("TEST 3: DLA Decomposed — V1-promotion vs V2-suppression")
    print(f"  {n_trials} PI trials, keys={num_keys}, updates={num_updates}")
    print("=" * 70)

    n_layers = info.n_layers
    n_heads = info.n_heads
    W_U = model.W_U

    head_dla_v1 = np.zeros((n_layers, n_heads))
    head_dla_v2 = np.zeros((n_layers, n_heads))
    n_correct = 0
    n_incorrect = 0
    n_valid = 0

    for t_idx in range(n_trials):
        seed = hash(("PI_copy_supp_pythia", t_idx)) % (2**31)
        trial = generate_completion_trial(
            num_keys, num_updates, "PI", seed, value_pool,
        )

        init_val = trial.initial_value
        final_val = trial.final_value
        v1_tid = value_to_tid.get(init_val)
        v2_tid = value_to_tid.get(final_val)

        if v1_tid is None or v2_tid is None or v1_tid == v2_tid:
            continue

        tokens = model.to_tokens(trial.prompt)

        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens)

        ans_pos = tokens.shape[1] - 1
        pred_tid = logits[0, ans_pos].argmax().item()
        is_correct = pred_tid == v2_tid

        # Check space-prefixed
        if not is_correct:
            space_v2 = tokenizer.encode(" " + final_val, add_special_tokens=False)
            if len(space_v2) == 1 and pred_tid == space_v2[0]:
                is_correct = True

        u_v1 = W_U[:, v1_tid]
        u_v2 = W_U[:, v2_tid]

        # Also check space-prefixed unembedding
        space_v1_tids = tokenizer.encode(" " + init_val, add_special_tokens=False)
        space_v2_tids = tokenizer.encode(" " + final_val, add_special_tokens=False)
        if len(space_v1_tids) == 1:
            u_v1 = u_v1 + W_U[:, space_v1_tids[0]]
        if len(space_v2_tids) == 1:
            u_v2 = u_v2 + W_U[:, space_v2_tids[0]]

        for layer in range(n_layers):
            hook_name = f"blocks.{layer}.attn.hook_result"
            if hook_name in cache:
                head_outputs = cache[hook_name][0, ans_pos]
            else:
                z = cache[f"blocks.{layer}.attn.hook_z"][0, ans_pos]
                W_O = model.W_O[layer]
                head_outputs = torch.einsum("hd,hdm->hm", z, W_O)

            for head in range(n_heads):
                h_out = head_outputs[head]
                head_dla_v1[layer, head] += (h_out @ u_v1).item()
                head_dla_v2[layer, head] += (h_out @ u_v2).item()

        if is_correct:
            n_correct += 1
        else:
            n_incorrect += 1
        n_valid += 1

        del cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        if (t_idx + 1) % 20 == 0:
            print(f"  {t_idx + 1}/{n_trials} done (correct: {n_correct}/{n_valid})")

    if n_valid == 0:
        print("  No valid trials!")
        return {}

    head_dla_v1 /= n_valid
    head_dla_v2 /= n_valid

    print(f"\n  Trials: {n_valid} valid, {n_correct} correct ({n_correct/n_valid:.1%}), {n_incorrect} incorrect")

    # Print primacy head results
    print(f"\n  Primacy heads (averaged over {n_valid} PI trials):")
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
            "dla_v1": v1, "dla_v2": v2, "logit_diff": ld,
            "mechanism": mechanism,
        })

    # Top 20 by |logit_diff|
    all_heads = []
    for l in range(n_layers):
        for h in range(n_heads):
            v1 = head_dla_v1[l, h]
            v2 = head_dla_v2[l, h]
            all_heads.append((l, h, v1, v2, v1 - v2, (l, h) in primacy_heads))

    all_heads.sort(key=lambda x: abs(x[4]), reverse=True)
    print(f"\n  Top 20 heads by |logit_diff| (PI condition):")
    print(f"  {'Head':<8} {'DLA_v1':>10} {'DLA_v2':>10} {'logit_diff':>12} {'primacy?':>10}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*12} {'─'*10}")
    for l, h, v1, v2, ld, is_p in all_heads[:20]:
        tag = "*** YES" if is_p else ""
        print(f"  L{l}H{h:<4} {v1:>+10.4f} {v2:>+10.4f} {ld:>+12.4f} {tag:>10}")

    return {
        "n_valid": n_valid,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "pi_accuracy": n_correct / n_valid if n_valid > 0 else 0,
        "primacy_head_summary": primacy_summary,
        "head_dla_v1": head_dla_v1.tolist(),
        "head_dla_v2": head_dla_v2.tolist(),
    }


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pythia Copy-Suppression Test (MHA)")
    parser.add_argument("--model", default="EleutherAI/pythia-410m")
    parser.add_argument("--keys", type=int, default=2)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--head-id-trials", type=int, default=50)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--head-threshold", type=float, default=0.10,
                        help="Attention bias threshold for primacy head classification")
    args = parser.parse_args()

    model, tokenizer, info = load_model(args.model, n_ctx=args.n_ctx)

    print(f"\nModel: {info.name}")
    print(f"  {info.n_layers} layers, {info.n_heads} heads (MHA), d_model={info.d_model}")

    value_to_tid = verify_single_token(tokenizer)
    value_pool = list(value_to_tid.keys())
    print(f"\n  {len(value_pool)} single-token values verified")

    model_short = args.model.split("/")[-1]

    print("\n" + "=" * 70)
    print(f"COPY-SUPPRESSION TEST — {model_short} (MHA)")
    print(f"  keys={args.keys}, updates={args.updates}, trials={args.trials}")
    print("=" * 70)

    t0 = time.time()

    # Step 0: Identify primacy heads inline
    primacy_heads, head_id_stats = identify_primacy_heads(
        model, info, tokenizer, value_pool,
        args.keys, args.updates,
        n_trials=args.head_id_trials,
        threshold=args.head_threshold,
    )
    primacy_set = set(primacy_heads)

    if not primacy_set:
        print("  No primacy heads found. Cannot proceed.")
        return

    # Test 1: OV Eigenspectrum
    eigen_results = test_ov_eigenspectrum(model, info, primacy_set)

    # Test 2: Suppression Score
    supp_results = test_suppression_score(model, info, primacy_set, value_to_tid)

    # Test 3: DLA Decomposed
    dla_results = test_dla_decomposed(
        model, info, primacy_set, tokenizer, value_to_tid, value_pool,
        args.keys, args.updates, args.trials,
    )

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s")

    # ── VERDICT ──
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Test 1 check
    primacy_eigen = [r for r in eigen_results if r["is_primacy"]]
    other_eigen = [r for r in eigen_results if not r["is_primacy"]]
    if primacy_eigen:
        p_neg = np.mean([r["neg_fraction"] for r in primacy_eigen])
        o_neg = np.mean([r["neg_fraction"] for r in other_eigen]) if other_eigen else 0
        all_abs = [abs(r["dominant_eigval"]) for r in eigen_results]
        print(f"  Test 1 (eigenspectrum): primacy neg_frac={p_neg:.3f}, other={o_neg:.3f}")
        print(f"    |dominant| range: [{min(all_abs):.4f}, {max(all_abs):.4f}]")
        if max(all_abs) < 0.001:
            print(f"    → INCONCLUSIVE (eigenvalues near zero — unexpected for MHA!)")
        elif p_neg > o_neg + 0.1:
            print(f"    → SUPPORTS copy-suppression")
        else:
            print(f"    → DOES NOT SUPPORT copy-suppression")

    # Test 2 check
    primacy_supp = [r for r in supp_results if r["is_primacy"]]
    other_supp = [r for r in supp_results if not r["is_primacy"]]
    if primacy_supp:
        p_score = np.mean([r["mean_suppression_score"] for r in primacy_supp])
        o_score = np.mean([r["mean_suppression_score"] for r in other_supp]) if other_supp else 0
        print(f"  Test 2 (suppression): primacy={p_score:+.4f}, other={o_score:+.4f}")
        if p_score < o_score:
            print(f"    → SUPPORTS copy-suppression")
        else:
            print(f"    → DOES NOT SUPPORT")

    # Test 3 check
    if dla_results and "primacy_head_summary" in dla_results:
        v1_promo = sum(1 for h in dla_results["primacy_head_summary"] if h["mechanism"] == "V1-PROMOTION")
        v2_supp = sum(1 for h in dla_results["primacy_head_summary"] if h["mechanism"] == "V2-SUPPRESSION")
        total_h = len(dla_results["primacy_head_summary"])
        print(f"  Test 3 (DLA): {v1_promo}/{total_h} V1-PROMOTION, {v2_supp}/{total_h} V2-SUPPRESSION")
        for hs in dla_results["primacy_head_summary"]:
            print(f"    L{hs['layer']}H{hs['head']}: DLA_v1={hs['dla_v1']:+.4f}, "
                  f"DLA_v2={hs['dla_v2']:+.4f} → {hs['mechanism']}")

    # Cross-architecture comparison summary
    print(f"\n  ═══ CROSS-ARCHITECTURE COMPARISON ═══")
    print(f"  Qwen 0.5B (GQA):  Test 1=INCONCLUSIVE (eigenvalues=0), Test 3=V1-PROMOTION")
    print(f"  Pythia-160M (MHA): Results above ↑↑↑")
    if dla_results and "primacy_head_summary" in dla_results:
        if v1_promo > v2_supp:
            print(f"  → V1-PROMOTION confirmed across MHA and GQA → architecture-independent finding")
        elif v2_supp > v1_promo:
            print(f"  → V2-SUPPRESSION in MHA vs V1-PROMOTION in GQA → architecture-dependent mechanism")
        else:
            print(f"  → Mixed results — needs more investigation")

    # Save
    all_results = {
        "config": {
            "model": args.model,
            "architecture": "MHA",
            "keys": args.keys,
            "updates": args.updates,
            "trials": args.trials,
            "head_id_trials": args.head_id_trials,
            "primacy_heads": sorted(list(primacy_set)),
            "elapsed_seconds": elapsed,
        },
        "head_identification": head_id_stats,
        "test1_eigenspectrum": {
            "primacy_heads": [r for r in eigen_results if r["is_primacy"]],
            "summary": {
                "primacy_mean_neg_frac": np.mean([r["neg_fraction"] for r in primacy_eigen]) if primacy_eigen else None,
                "other_mean_neg_frac": np.mean([r["neg_fraction"] for r in other_eigen]) if other_eigen else None,
                "primacy_mean_dominant": np.mean([r["dominant_eigval"] for r in primacy_eigen]) if primacy_eigen else None,
                "other_mean_dominant": np.mean([r["dominant_eigval"] for r in other_eigen]) if other_eigen else None,
                "all_eigenvalues_nonzero": max(abs(r["dominant_eigval"]) for r in eigen_results) > 0.001,
            },
        },
        "test2_suppression_score": {
            "primacy_heads": [r for r in supp_results if r["is_primacy"]],
            "summary": {
                "primacy_mean": np.mean([r["mean_suppression_score"] for r in primacy_supp]) if primacy_supp else None,
                "other_mean": np.mean([r["mean_suppression_score"] for r in other_supp]) if other_supp else None,
            },
        },
        "test3_dla_decomposed": dla_results,
    }

    save_results(all_results, args.model, args.keys, args.updates, "copy_suppression_test")


if __name__ == "__main__":
    main()
