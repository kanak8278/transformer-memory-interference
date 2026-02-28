"""
Experiment 28: Copy-Suppression Hypothesis Test

Tests whether primacy-biased heads implement copy suppression (V2 negation)
rather than V1 promotion. This resolves the DLA paradox: recency heads win
the logit-space tug of war (sum=-11.00 vs +0.83) yet PI accuracy is only 55%.

Hypothesis (from McDougall et al., BlackboxNLP 2024):
  Primacy heads suppress V2 via negative OV eigenvalues, rather than promoting V1.
  The recency signal is actively cancelled, leaving V1 as the default winner.

Three tests:
  Test 1: OV eigenspectrum (static weight analysis, zero forward passes)
  Test 2: Directional suppression score (static, per token across trials)
  Test 3: DLA decomposed into V1-promotion vs V2-suppression (forward passes)

Usage:
    cd mechanistic_probing_v2
    uv run python experiments/28_copy_suppression_test.py --model Qwen/Qwen2.5-0.5B-Instruct --keys 1 --updates 5
    uv run python experiments/28_copy_suppression_test.py --model Qwen/Qwen2.5-1.5B-Instruct --keys 1 --updates 3
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
from core.dataset_configs import format_for_chat, ORIGINAL_CATEGORIES
from core.model_loader import verify_single_token
from core.output import save_results, load_head_identification


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
        ok = all(items[i]["category"] != items[i - 1]["category"] for i in range(1, len(items)))
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


# ─── TEST 1: OV Eigenspectrum (static weight analysis) ───────────────────────

def test_ov_eigenspectrum(model, info, primacy_heads):
    """Compute OV eigenspectrum for all heads. Compare primacy vs rest."""
    n_layers = info.n_layers
    n_heads = info.n_heads

    print("\n" + "=" * 70)
    print("TEST 1: OV Eigenspectrum")
    print("  Copy-suppression heads have dominant NEGATIVE eigenvalues")
    print("=" * 70)

    results = []

    for layer in range(n_layers):
        # W_V: [n_heads, d_model, d_head], W_O: [n_heads, d_head, d_model]
        W_V = model.W_V[layer]  # [n_heads, d_model, d_head]
        W_O = model.W_O[layer]  # [n_heads, d_head, d_model]

        for head in range(n_heads):
            # OV circuit: d_head x d_head (in head-space)
            # W_V[head] is [d_model, d_head], W_O[head] is [d_head, d_model]
            # Full OV in model space: [d_model, d_model] = W_V^T @ W_O^T...
            # Actually for eigenspectrum, use head-space: W_O[head] @ W_V[head]
            # This is [d_head, d_model] @ [d_model, d_head] = [d_head, d_head]
            W_OV_head = W_O[head] @ W_V[head]  # [d_head, d_head] — but this might be wrong

            # TransformerLens convention:
            # The OV circuit maps: input [d_model] -> W_V -> [d_head] -> W_O -> [d_model]
            # So full OV = W_V[head].T @ W_O[head].T ... no.
            # W_V: [d_model, d_head] maps residual to value space
            # W_O: [d_head, d_model] maps value space back to residual
            # Full OV in residual space: W_V.T is [d_head, d_model], but we want:
            # x -> W_V -> v -> W_O -> output
            # output = W_O @ W_V @ x ... no.
            # x is [d_model], W_V is [d_model, d_head], so v = x @ W_V = [d_head]
            # then output = v @ W_O = [d_model], where W_O is [d_head, d_model]
            # So the full map: x -> (x @ W_V) @ W_O = x @ (W_V @ W_O)
            # W_OV = W_V @ W_O: [d_model, d_head] @ [d_head, d_model] = [d_model, d_model]
            W_OV = W_V[head] @ W_O[head]  # [d_model, d_model]

            # Eigenvalues of [d_model, d_model] matrix
            # This is expensive for large d_model. Use SVD instead for stability.
            # For copy-suppression: look at singular values and the sign structure.
            # McDougall et al. actually check the projection onto specific directions,
            # not raw eigenvalues. Raw eigenvalues of non-symmetric matrices can be complex.

            # More practical: compute top singular values/vectors
            # and check if the dominant component flips sign
            try:
                U, S, Vh = torch.linalg.svd(W_OV, full_matrices=False)
                top_k = min(10, len(S))
                top_singular = S[:top_k].cpu().tolist()
            except Exception:
                top_singular = []

            # For eigenvalue analysis, use a smaller projection
            # Check eigenvalues in head-space instead: [d_head, d_head]
            W_OV_headspace = W_O[head] @ W_V[head]  # [d_head, d_head] — but wrong dim
            # Actually: W_O is [d_head, d_model], W_V is [d_model, d_head]
            # W_O @ W_V = [d_head, d_head] ✓
            W_OV_small = W_O[head] @ W_V[head]  # [d_head, d_head]

            try:
                eigvals = torch.linalg.eigvals(W_OV_small)
                eigvals_real = eigvals.real.cpu().tolist()
                eigvals_imag = eigvals.imag.cpu().tolist()

                # Summary stats
                max_real = max(eigvals_real)
                min_real = min(eigvals_real)
                mean_real = np.mean(eigvals_real)
                neg_frac = sum(1 for e in eigvals_real if e < 0) / len(eigvals_real)

                # Dominant eigenvalue: largest by absolute value
                abs_eigvals = [abs(e) for e in eigvals_real]
                dominant_idx = abs_eigvals.index(max(abs_eigvals))
                dominant_eigval = eigvals_real[dominant_idx]
                dominant_sign = "negative" if dominant_eigval < 0 else "positive"
            except Exception:
                eigvals_real = []
                max_real = min_real = mean_real = neg_frac = 0.0
                dominant_eigval = 0.0
                dominant_sign = "unknown"

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
                "top_singular_values": top_singular[:5],
                "eigenvalues_real": eigvals_real,
            })

    # Print summary
    primacy_results = [r for r in results if r["is_primacy"]]
    other_results = [r for r in results if not r["is_primacy"]]

    print(f"\n  Primacy heads ({len(primacy_results)}):")
    for r in primacy_results:
        print(f"    L{r['layer']}H{r['head']}: dominant={r['dominant_eigval']:+.4f} ({r['dominant_sign']}), "
              f"neg_frac={r['neg_fraction']:.2f}, range=[{r['min_eigval_real']:.4f}, {r['max_eigval_real']:.4f}]")

    if other_results:
        # Summary stats for non-primacy heads
        other_dominant = [r["dominant_eigval"] for r in other_results]
        other_neg_frac = [r["neg_fraction"] for r in other_results]
        primacy_dominant = [r["dominant_eigval"] for r in primacy_results]
        primacy_neg_frac = [r["neg_fraction"] for r in primacy_results]

        print(f"\n  Non-primacy heads ({len(other_results)}) summary:")
        print(f"    Dominant eigval: mean={np.mean(other_dominant):+.4f}, "
              f"std={np.std(other_dominant):.4f}")
        print(f"    Neg fraction:    mean={np.mean(other_neg_frac):.3f}")

        if primacy_results:
            print(f"\n  Primacy heads ({len(primacy_results)}) summary:")
            print(f"    Dominant eigval: mean={np.mean(primacy_dominant):+.4f}, "
                  f"std={np.std(primacy_dominant):.4f}")
            print(f"    Neg fraction:    mean={np.mean(primacy_neg_frac):.3f}")

    return results


# ─── TEST 2: Directional Suppression Score ────────────────────────────────────

def test_suppression_score(model, info, primacy_heads, tokenizer, value_to_tid, n_tokens=100):
    """
    For each head, project OV circuit onto value token unembedding directions.
    Negative projection = head suppresses that token.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Directional Suppression Score")
    print("  Negative score = head suppresses that value when attending near it")
    print("=" * 70)

    n_layers = info.n_layers
    n_heads = info.n_heads
    W_U = model.W_U  # [d_model, vocab_size]

    # Sample a set of value tokens to average over
    all_values = list(value_to_tid.items())
    rng = random.Random(42)
    rng.shuffle(all_values)
    sample_tokens = all_values[:n_tokens]

    results = []

    for layer in range(n_layers):
        W_V = model.W_V[layer]  # [n_heads, d_model, d_head]
        W_O = model.W_O[layer]  # [n_heads, d_head, d_model]

        for head in range(n_heads):
            # Full OV in model space: x @ W_V[head] @ W_O[head] = x @ W_OV
            # W_OV: [d_model, d_model]
            W_OV = W_V[head] @ W_O[head]  # [d_model, d_model]

            scores = []
            for val_word, tid in sample_tokens:
                # Get unembedding direction for this token
                u = W_U[:, tid]  # [d_model]

                # Suppression score: u^T @ W_OV @ u
                # If negative: when this head processes input aligned with token u,
                # it produces output that REDUCES the logit for token u
                score = (u @ W_OV @ u).item()
                scores.append(score)

            mean_score = np.mean(scores)
            std_score = np.std(scores)
            neg_count = sum(1 for s in scores if s < 0)
            is_primacy = (layer, head) in primacy_heads

            results.append({
                "layer": layer,
                "head": head,
                "is_primacy": is_primacy,
                "mean_suppression_score": mean_score,
                "std_suppression_score": std_score,
                "negative_fraction": neg_count / len(scores),
                "n_tokens_tested": len(scores),
            })

    # Print summary
    primacy_results = [r for r in results if r["is_primacy"]]
    other_results = [r for r in results if not r["is_primacy"]]

    print(f"\n  Primacy heads ({len(primacy_results)}):")
    for r in primacy_results:
        print(f"    L{r['layer']}H{r['head']}: suppression_score={r['mean_suppression_score']:+.4f} "
              f"(±{r['std_suppression_score']:.4f}), neg_frac={r['negative_fraction']:.2f}")

    if other_results:
        other_scores = [r["mean_suppression_score"] for r in other_results]
        primacy_scores = [r["mean_suppression_score"] for r in primacy_results] if primacy_results else []

        print(f"\n  Non-primacy heads ({len(other_results)}) summary:")
        print(f"    Suppression score: mean={np.mean(other_scores):+.4f}, "
              f"std={np.std(other_scores):.4f}")
        print(f"    Range: [{min(other_scores):+.4f}, {max(other_scores):+.4f}]")

        if primacy_scores:
            print(f"\n  Primacy heads ({len(primacy_results)}) summary:")
            print(f"    Suppression score: mean={np.mean(primacy_scores):+.4f}, "
                  f"std={np.std(primacy_scores):.4f}")

            # Are primacy heads more negative than average?
            other_mean = np.mean(other_scores)
            primacy_mean = np.mean(primacy_scores)
            if primacy_mean < other_mean:
                print(f"    → Primacy heads are MORE suppressive (Δ={primacy_mean - other_mean:+.4f})")
            else:
                print(f"    → Primacy heads are LESS suppressive (Δ={primacy_mean - other_mean:+.4f})")

    return results


# ─── TEST 3: DLA Decomposed (V1-promotion vs V2-suppression) ─────────────────

def test_dla_decomposed(model, info, primacy_heads, tokenizer, value_to_tid, value_pool,
                        num_keys, num_updates, n_trials):
    """
    During PI trials, decompose each head's DLA into:
      - dla_v1: contribution to V1 logit (positive = promoting V1)
      - dla_v2: contribution to V2 logit (negative = suppressing V2)
    """
    print("\n" + "=" * 70)
    print("TEST 3: DLA Decomposed — V1-promotion vs V2-suppression")
    print(f"  {n_trials} PI trials, keys={num_keys}, updates={num_updates}")
    print("=" * 70)

    n_layers = info.n_layers
    n_heads = info.n_heads
    W_U = model.W_U  # [d_model, vocab_size]

    # Accumulate per-head DLA_v1 and DLA_v2 across trials
    # Also track by trial correctness
    head_dla_v1 = np.zeros((n_layers, n_heads))
    head_dla_v2 = np.zeros((n_layers, n_heads))
    head_dla_v1_correct = np.zeros((n_layers, n_heads))
    head_dla_v2_correct = np.zeros((n_layers, n_heads))
    head_dla_v1_incorrect = np.zeros((n_layers, n_heads))
    head_dla_v2_incorrect = np.zeros((n_layers, n_heads))
    n_correct = 0
    n_incorrect = 0
    n_valid = 0

    trial_details = []

    for t_idx in range(n_trials):
        seed = hash(("PI_copy_supp", t_idx)) % (2**31)
        trial = build_trial(num_keys, num_updates, "PI", seed, value_pool)

        init_val = trial["initial_value"]
        final_val = trial["final_value"]
        v1_tid = value_to_tid.get(init_val)
        v2_tid = value_to_tid.get(final_val)

        if v1_tid is None or v2_tid is None or v1_tid == v2_tid:
            continue

        formatted = format_for_chat(trial["prompt"], tokenizer)
        tokens = model.to_tokens(formatted)

        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens)

        ans_pos = tokens.shape[1] - 1

        # Check correctness
        pred_tid = logits[0, ans_pos].argmax().item()
        # Check both bare and space-prefixed variants
        is_correct = pred_tid == v2_tid
        if not is_correct:
            # Check space-prefixed
            space_v2 = tokenizer.encode(" " + final_val, add_special_tokens=False)
            if len(space_v2) == 1 and pred_tid == space_v2[0]:
                is_correct = True

        # Get unembedding directions
        u_v1 = W_U[:, v1_tid]  # [d_model]
        u_v2 = W_U[:, v2_tid]  # [d_model]

        # Also check space-prefixed unembedding
        space_v1_tids = tokenizer.encode(" " + init_val, add_special_tokens=False)
        space_v2_tids = tokenizer.encode(" " + final_val, add_special_tokens=False)
        if len(space_v1_tids) == 1:
            u_v1 = u_v1 + W_U[:, space_v1_tids[0]]
        if len(space_v2_tids) == 1:
            u_v2 = u_v2 + W_U[:, space_v2_tids[0]]

        per_trial_v1 = np.zeros((n_layers, n_heads))
        per_trial_v2 = np.zeros((n_layers, n_heads))

        for layer in range(n_layers):
            # Per-head output at answer position
            # cache["result", layer] has shape [batch, seq, n_heads, d_model]
            # but TransformerLens stores attn output differently.
            # Use: cache["blocks.{layer}.attn.hook_result"] → [batch, seq, n_heads, d_model]
            hook_name = f"blocks.{layer}.attn.hook_result"
            if hook_name in cache:
                head_outputs = cache[hook_name][0, ans_pos]  # [n_heads, d_model]
            else:
                # Fallback: compute from z and W_O
                z = cache[f"blocks.{layer}.attn.hook_z"][0, ans_pos]  # [n_heads, d_head]
                W_O = model.W_O[layer]  # [n_heads, d_head, d_model]
                head_outputs = torch.einsum("hd,hdm->hm", z, W_O)  # [n_heads, d_model]

            for head in range(n_heads):
                h_out = head_outputs[head]  # [d_model]
                dla_v1_val = (h_out @ u_v1).item()
                dla_v2_val = (h_out @ u_v2).item()

                per_trial_v1[layer, head] = dla_v1_val
                per_trial_v2[layer, head] = dla_v2_val

        head_dla_v1 += per_trial_v1
        head_dla_v2 += per_trial_v2
        n_valid += 1

        if is_correct:
            head_dla_v1_correct += per_trial_v1
            head_dla_v2_correct += per_trial_v2
            n_correct += 1
        else:
            head_dla_v1_incorrect += per_trial_v1
            head_dla_v2_incorrect += per_trial_v2
            n_incorrect += 1

        # Store per-trial summary for primacy heads
        primacy_trial = {}
        for (l, h) in primacy_heads:
            if l < n_layers and h < n_heads:
                primacy_trial[f"L{l}H{h}"] = {
                    "dla_v1": per_trial_v1[l, h],
                    "dla_v2": per_trial_v2[l, h],
                }
        trial_details.append({
            "trial": t_idx,
            "correct": is_correct,
            "init_val": init_val,
            "final_val": final_val,
            "primacy_heads": primacy_trial,
        })

        del cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        if (t_idx + 1) % 20 == 0:
            print(f"  {t_idx + 1}/{n_trials} done (correct: {n_correct}/{n_valid})")

    if n_valid == 0:
        print("  No valid trials!")
        return {}

    # Average
    head_dla_v1 /= n_valid
    head_dla_v2 /= n_valid
    if n_correct > 0:
        head_dla_v1_correct /= n_correct
        head_dla_v2_correct /= n_correct
    if n_incorrect > 0:
        head_dla_v1_incorrect /= n_incorrect
        head_dla_v2_incorrect /= n_incorrect

    print(f"\n  Trials: {n_valid} valid, {n_correct} correct, {n_incorrect} incorrect")
    print(f"  PI accuracy: {n_correct/n_valid:.1%}")

    # Print primacy head results
    print(f"\n  Primacy heads (PI condition, averaged over {n_valid} trials):")
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

    # Print split by outcome
    if n_correct > 0 and n_incorrect > 0:
        print(f"\n  Primacy heads — split by PI outcome:")
        print(f"  {'Head':<8} {'Correct V1':>10} {'Correct V2':>10} │ {'Wrong V1':>10} {'Wrong V2':>10} │ {'Δ V2':>8}")
        print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*1} {'─'*10} {'─'*10} {'─'*1} {'─'*8}")
        for (l, h) in sorted(primacy_heads):
            if l >= n_layers or h >= n_heads:
                continue
            cv1 = head_dla_v1_correct[l, h]
            cv2 = head_dla_v2_correct[l, h]
            iv1 = head_dla_v1_incorrect[l, h]
            iv2 = head_dla_v2_incorrect[l, h]
            delta_v2 = iv2 - cv2
            print(f"  L{l}H{h:<4} {cv1:>+10.4f} {cv2:>+10.4f} │ {iv1:>+10.4f} {iv2:>+10.4f} │ {delta_v2:>+8.4f}")

    # Non-primacy retrieval head comparison (top 10 by |logit_diff|)
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
        "head_dla_v1_correct": head_dla_v1_correct.tolist() if n_correct > 0 else None,
        "head_dla_v2_correct": head_dla_v2_correct.tolist() if n_correct > 0 else None,
        "head_dla_v1_incorrect": head_dla_v1_incorrect.tolist() if n_incorrect > 0 else None,
        "head_dla_v2_incorrect": head_dla_v2_incorrect.tolist() if n_incorrect > 0 else None,
        "trial_details": trial_details,
    }


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Copy-Suppression Hypothesis Test")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--keys", type=int, default=1)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--suppression-tokens", type=int, default=100,
                        help="Number of value tokens to average over for Test 2")
    args = parser.parse_args()

    model, tokenizer, info = load_model(args.model, n_ctx=args.n_ctx)
    value_to_tid = verify_single_token(tokenizer)
    value_pool = list(value_to_tid.keys())

    model_short = args.model.split("/")[-1]
    print("=" * 70)
    print(f"COPY-SUPPRESSION HYPOTHESIS TEST — {model_short}")
    print(f"  keys={args.keys}, updates={args.updates}, trials={args.trials}")
    print(f"  {info.n_layers} layers, {info.n_heads} heads, d_model={info.d_model}")
    print("=" * 70)

    # Load primacy heads from head_identification results
    try:
        primacy_head_list = load_head_identification(args.model, args.keys, args.updates)
        primacy_heads = set(primacy_head_list)
        print(f"\n  Loaded {len(primacy_heads)} primacy heads from head_identification: "
              f"{sorted(primacy_heads)}")
    except FileNotFoundError:
        # Fallback: use known heads from strict classification
        known_strict = {
            "Qwen2.5-0.5B-Instruct": [(16, 3)],
            "Qwen2.5-1.5B-Instruct": [(19, 1)],
        }
        fallback = known_strict.get(model_short, [])
        if fallback:
            primacy_heads = set(fallback)
            print(f"\n  No head_identification found, using known strict heads: {sorted(primacy_heads)}")
        else:
            primacy_heads = set()
            print(f"\n  WARNING: No primacy heads identified for {model_short}. "
                  f"Run head_identification first or add known heads.")

    if not primacy_heads:
        print("  Cannot run without primacy heads. Exiting.")
        return

    t0 = time.time()

    # ── TEST 1: OV Eigenspectrum ──
    eigen_results = test_ov_eigenspectrum(model, info, primacy_heads)

    # ── TEST 2: Directional Suppression Score ──
    supp_results = test_suppression_score(
        model, info, primacy_heads, tokenizer, value_to_tid,
        n_tokens=args.suppression_tokens
    )

    # ── TEST 3: DLA Decomposed ──
    dla_results = test_dla_decomposed(
        model, info, primacy_heads, tokenizer, value_to_tid, value_pool,
        args.keys, args.updates, args.trials
    )

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s")

    # ── VERDICT ──
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Check Test 1: Do primacy heads have more negative eigenvalues?
    primacy_eigen = [r for r in eigen_results if r["is_primacy"]]
    other_eigen = [r for r in eigen_results if not r["is_primacy"]]
    if primacy_eigen:
        p_neg = np.mean([r["neg_fraction"] for r in primacy_eigen])
        o_neg = np.mean([r["neg_fraction"] for r in other_eigen]) if other_eigen else 0
        print(f"  Test 1 (eigenspectrum): primacy neg_frac={p_neg:.3f}, other={o_neg:.3f}")
        if p_neg > o_neg + 0.1:
            print(f"    → SUPPORTS copy-suppression (primacy heads have more negative eigenvalues)")
        else:
            print(f"    → DOES NOT SUPPORT (no clear difference)")

    # Check Test 2: Do primacy heads have more negative suppression scores?
    primacy_supp = [r for r in supp_results if r["is_primacy"]]
    other_supp = [r for r in supp_results if not r["is_primacy"]]
    if primacy_supp:
        p_score = np.mean([r["mean_suppression_score"] for r in primacy_supp])
        o_score = np.mean([r["mean_suppression_score"] for r in other_supp]) if other_supp else 0
        print(f"  Test 2 (suppression score): primacy={p_score:+.4f}, other={o_score:+.4f}")
        if p_score < o_score:
            print(f"    → SUPPORTS copy-suppression (primacy heads more suppressive)")
        else:
            print(f"    → DOES NOT SUPPORT")

    # Check Test 3: Do primacy heads have negative DLA_v2?
    if dla_results and "primacy_head_summary" in dla_results:
        for hs in dla_results["primacy_head_summary"]:
            print(f"  Test 3 (DLA decomp): L{hs['layer']}H{hs['head']}: "
                  f"DLA_v1={hs['dla_v1']:+.4f}, DLA_v2={hs['dla_v2']:+.4f} → {hs['mechanism']}")

    # ── SAVE ──
    all_results = {
        "config": {
            "model": args.model,
            "keys": args.keys,
            "updates": args.updates,
            "trials": args.trials,
            "primacy_heads": sorted(list(primacy_heads)),
            "elapsed_seconds": elapsed,
        },
        "test1_eigenspectrum": {
            "primacy_heads": [r for r in eigen_results if r["is_primacy"]],
            "summary": {
                "primacy_mean_neg_frac": np.mean([r["neg_fraction"] for r in primacy_eigen]) if primacy_eigen else None,
                "other_mean_neg_frac": np.mean([r["neg_fraction"] for r in other_eigen]) if other_eigen else None,
                "primacy_mean_dominant": np.mean([r["dominant_eigval"] for r in primacy_eigen]) if primacy_eigen else None,
                "other_mean_dominant": np.mean([r["dominant_eigval"] for r in other_eigen]) if other_eigen else None,
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

    # Strip large eigenvalue arrays from saved results to keep file manageable
    for r in all_results["test1_eigenspectrum"]["primacy_heads"]:
        if "eigenvalues_real" in r:
            r["eigenvalues_real"] = r["eigenvalues_real"][:20]  # keep top 20 only

    save_results(all_results, args.model, args.keys, args.updates, "copy_suppression_test")


if __name__ == "__main__":
    main()
