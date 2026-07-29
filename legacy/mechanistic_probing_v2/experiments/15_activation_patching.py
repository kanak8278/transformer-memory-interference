"""
Phase 2: Activation patching — causal evidence for PI > RI.

The idea:
  Clean run:     model sees 1 value per category (no interference) → gets answer trivially
  Corrupted run: model sees N updates per category (full interference) → may fail (especially PI)
  Patched run:   corrupted input, but at (layer, position), swap in the clean activation

If patching recovers accuracy → the information was there but the model couldn't use it.
This is CAUSAL evidence, not just correlation.

We sweep: which (layer, position) patches recover PI accuracy?
  - If patching at the final value's position in late layers works → info present but inaccessible
  - If patching at early layers works → the representation gets corrupted early
  - If RI and PI need patches at DIFFERENT positions → dual-process evidence

Design:
  1. For each trial, build MATCHED clean/corrupted prompts (same categories, same test)
  2. Run clean with cache, run corrupted with cache
  3. For a grid of (layer, position_type), patch clean → corrupted and measure:
     - Does P(correct_answer) increase?
     - Does the model's argmax prediction change to correct?

Position types patched:
  - "initial_value": position of the first value of test category
  - "final_value": position of the last value of test category
  - "query": the question tokens
  - "all_values": ALL value positions of test category at once

Usage:
    cd mechanistic_probing_v2
    uv run python experiments/15_activation_patching.py [--trials 5] [--updates 10]
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

from core.model_loader import load_model, verify_single_token
from core.dataset_configs import (
    format_for_chat, ORIGINAL_CATEGORIES,
    get_value_pool, build_interleaved_sequence,
)
from core.analysis_utils import compute_logit_diff, compute_recovery, aggregate_recovery


def build_matched_pair(
    num_keys: int,
    num_updates: int,
    condition: str,
    seed: int,
    value_pool: list[str],
    categories: list[str],
) -> tuple[dict, dict]:
    """Build a matched clean/corrupted trial pair.

    Clean: 1 update per key (no interference).
    Corrupted: num_updates per key (full interference).
    Same categories, same test category, same initial and final values.

    Returns (clean_trial, corrupted_trial).
    """
    rng = random.Random(seed)
    cats = categories[:num_keys]
    test_cat = cats[seed % num_keys]

    # Pick values: we need num_updates per category
    total_needed = num_keys * num_updates
    if total_needed > len(value_pool):
        raise ValueError(f"Need {total_needed} values, have {len(value_pool)}")

    selected = rng.sample(value_pool, total_needed)
    values_per_cat = {}
    idx = 0
    for cat in cats:
        values_per_cat[cat] = selected[idx:idx + num_updates]
        idx += num_updates

    test_values = values_per_cat[test_cat]
    initial_value = test_values[0]
    final_value = test_values[-1]

    expected = initial_value if condition == "RI" else final_value
    query_word = "first" if condition == "RI" else "last"

    # ── Clean prompt: 1 value per category (the correct answer) ──
    clean_items = []
    for cat in cats:
        if cat == test_cat:
            # Use the expected answer as the single value
            clean_items.append({"category": cat, "value": expected})
        else:
            clean_items.append({"category": cat, "value": values_per_cat[cat][0]})

    rng2 = random.Random(seed + 1000)
    rng2.shuffle(clean_items)
    clean_stream = "\n".join(f"{it['category']}: {it['value']}" for it in clean_items)
    clean_prompt = (
        f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{clean_stream}\n\n"
        f"What was the {query_word} value of {test_cat}?"
    )

    # ── Corrupted prompt: all updates, interleaved ──
    rng3 = random.Random(seed + 2000)
    corr_items = build_interleaved_sequence(cats, values_per_cat, rng3)

    corr_stream = "\n".join(f"{it['category']}: {it['value']}" for it in corr_items)
    corr_prompt = (
        f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{corr_stream}\n\n"
        f"What was the {query_word} value of {test_cat}?"
    )

    cat_values_in_corr = [it["value"] for it in corr_items if it["category"] == test_cat]

    clean_trial = {
        "prompt": clean_prompt,
        "condition": condition,
        "test_category": test_cat,
        "expected": expected,
        "initial_value": initial_value,
        "final_value": final_value,
    }
    corrupted_trial = {
        "prompt": corr_prompt,
        "condition": condition,
        "test_category": test_cat,
        "expected": expected,
        "initial_value": initial_value,
        "final_value": final_value,
        "all_values": cat_values_in_corr,
        "num_updates": num_updates,
    }

    return clean_trial, corrupted_trial


def find_positions(token_ids: list[int], target_tid: int) -> list[int]:
    """Find all positions of a token ID in the sequence."""
    return [i for i, t in enumerate(token_ids) if t == target_tid]


def run_patching(model, tokenizer, clean_trial, corrupted_trial, value_to_tid):
    """Run activation patching experiment for one matched pair."""

    expected_value = corrupted_trial["expected"]
    expected_tid_sp = value_to_tid.get(expected_value, -1)
    expected_tid_bare = tokenizer.encode(expected_value, add_special_tokens=False)[0]
    expected_tids = list(set([t for t in [expected_tid_sp, expected_tid_bare] if t >= 0]))

    # Identify the incorrect token for logit_diff (initial for PI, final for RI)
    condition = corrupted_trial["condition"]
    if condition == "RI":
        wrong_value = corrupted_trial["final_value"]
    else:
        wrong_value = corrupted_trial["initial_value"]
    wrong_tid_sp = value_to_tid.get(wrong_value, -1)
    wrong_tid_bare = tokenizer.encode(wrong_value, add_special_tokens=False)[0]
    wrong_tids = list(set([t for t in [wrong_tid_sp, wrong_tid_bare] if t >= 0]))

    if not wrong_tids:
        wrong_tids = expected_tids  # fallback — recovery will be 0

    # ── Run clean ──
    clean_formatted = format_for_chat(clean_trial["prompt"], tokenizer)
    clean_tokens = model.to_tokens(clean_formatted)
    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)

    clean_pred_tid = clean_logits[0, -1].argmax().item()
    clean_pred = tokenizer.decode([clean_pred_tid]).strip()
    clean_correct = clean_pred.lower() == expected_value.lower()

    # ── Run corrupted ──
    corr_formatted = format_for_chat(corrupted_trial["prompt"], tokenizer)
    corr_tokens = model.to_tokens(corr_formatted)
    corr_token_ids = corr_tokens[0].tolist()
    corr_seq_len = corr_tokens.shape[1]

    with torch.no_grad():
        corr_logits, corr_cache = model.run_with_cache(corr_tokens)

    corr_pred_tid = corr_logits[0, -1].argmax().item()
    corr_pred = tokenizer.decode([corr_pred_tid]).strip()
    corr_correct = corr_pred.lower() == expected_value.lower()

    # Logit difference metric (Wang et al. 2022 IOI, Heimersheim & Nanda 2024)
    # logit_diff = logit(correct) - logit(incorrect)
    clean_ld = compute_logit_diff(clean_logits[0, -1], expected_tids, wrong_tids)
    baseline_ld = compute_logit_diff(corr_logits[0, -1], expected_tids, wrong_tids)

    # Also track P(correct) for display
    corr_probs = torch.softmax(corr_logits[0, -1], dim=-1)
    baseline_p = max(corr_probs[t].item() for t in expected_tids)
    clean_probs = torch.softmax(clean_logits[0, -1], dim=-1)
    clean_p = max(clean_probs[t].item() for t in expected_tids)

    print(f"    Clean: '{clean_pred}' ld={clean_ld:.2f} P={clean_p:.4f} ({'OK' if clean_correct else 'WRONG'})")
    print(f"    Corrupted: '{corr_pred}' ld={baseline_ld:.2f} P={baseline_p:.4f} ({'OK' if corr_correct else 'WRONG'})")

    # ── Find positions to patch ──
    init_tid = value_to_tid.get(corrupted_trial["initial_value"], -1)
    final_tid = value_to_tid.get(corrupted_trial["final_value"], -1)
    all_value_tids = [value_to_tid.get(v, -1) for v in corrupted_trial["all_values"]]

    init_positions = find_positions(corr_token_ids, init_tid) if init_tid >= 0 else []
    final_positions = find_positions(corr_token_ids, final_tid) if final_tid >= 0 else []
    all_value_positions = []
    for tid in all_value_tids:
        if tid >= 0:
            all_value_positions.extend(find_positions(corr_token_ids, tid))
    all_value_positions = sorted(set(all_value_positions))

    # Query position: find "What" near the end
    str_tokens = model.to_str_tokens(corr_formatted)
    query_positions = []
    for i in range(len(str_tokens) - 1, -1, -1):
        if "What" in str_tokens[i]:
            query_positions = list(range(i, corr_seq_len))
            break

    print(f"    Positions: init={init_positions}, final={final_positions}, "
          f"all_values({len(all_value_positions)}), query({len(query_positions)})")

    n_layers = model.cfg.n_layers

    # ── Patch at each layer × position_type ──
    # We can't patch clean→corrupted at exact positions because the sequences
    # have different lengths. Instead, we patch the CORRUPTED cache with
    # the corrupted run's own activations replaced by a targeted intervention.
    #
    # Better approach: "noising" — run the corrupted input but at specific
    # (layer, position), replace with the CLEAN run's activation at the
    # answer position. This tests if the clean "answer representation"
    # can override the corrupted computation.
    #
    # Actually, the standard approach:
    # Patch clean residual stream at position P into the corrupted run at position P.
    # But positions differ between clean and corrupted. So we patch at the
    # ANSWER POSITION (last token) — which is the same concept in both.
    #
    # Most useful: patch at each layer at the ANSWER position.
    # This tells us: at which layer does the answer representation diverge?

    patching_results = {}

    # Sweep 1: Patch residual stream at answer position, each layer
    answer_pos = corr_seq_len - 1
    clean_answer_pos = clean_tokens.shape[1] - 1

    layer_recovery = []
    for layer in range(n_layers):
        # Get clean activation at answer position
        clean_resid = clean_cache["resid_post", layer][0, clean_answer_pos, :].clone()

        def make_hook(clean_act):
            def hook_fn(activation, hook):
                activation[0, answer_pos, :] = clean_act
                return activation
            return hook_fn

        with torch.no_grad():
            patched_logits = model.run_with_hooks(
                corr_tokens,
                fwd_hooks=[(f"blocks.{layer}.hook_resid_post", make_hook(clean_resid))],
            )

        patched_probs = torch.softmax(patched_logits[0, -1], dim=-1)
        patched_p = max(patched_probs[t].item() for t in expected_tids)
        patched_pred = tokenizer.decode([patched_logits[0, -1].argmax().item()]).strip()
        recovered = patched_pred.lower() == expected_value.lower()

        patched_ld = compute_logit_diff(patched_logits[0, -1], expected_tids, wrong_tids)
        recovery = compute_recovery(patched_ld, baseline_ld, clean_ld)
        layer_recovery.append({
            "layer": layer,
            "p_correct": patched_p,
            "logit_diff": patched_ld,
            "recovery": recovery,
            "predicted": patched_pred,
            "correct": recovered,
        })

    patching_results["resid_at_answer"] = layer_recovery

    # Sweep 2: Patch at specific VALUE positions (initial, final, all) at the best layer
    # Find the layer with highest recovery from sweep 1 (skip None)
    valid_layers = [x for x in layer_recovery if x["recovery"] is not None]
    best_layer = max(valid_layers, key=lambda x: x["recovery"])["layer"] if valid_layers else n_layers - 1

    position_type_results = {}
    for pos_type, positions in [
        ("initial_value", init_positions[:1]),  # first occurrence
        ("final_value", final_positions[-1:]),   # last occurrence
        ("all_test_values", all_value_positions),
        ("query", query_positions),
    ]:
        if not positions:
            position_type_results[pos_type] = {"p_correct": baseline_p, "recovery": 0.0}
            continue

        # Patch clean resid at these positions in the corrupted run
        # Since clean/corrupted have different sequences, we patch the
        # corrupted resid with clean resid at the SAME layer but at
        # the corrupted sequence's positions. This is a "zeroing" intervention.
        #
        # Actually: standard approach is to patch at the answer position
        # from the clean run. For position-specific patching, we use
        # a different strategy: "mean ablation" or "resample" at those positions.
        #
        # Simpler: at the best_layer, for each position in the corrupted run,
        # try patching the residual stream at that position (using clean answer pos).
        # This measures: does injecting clean signal at this position help?

        # We'll do something cleaner: patch ALL listed positions at once at best_layer
        clean_resid_answer = clean_cache["resid_post", best_layer][0, clean_answer_pos, :].clone()

        def make_multi_hook(positions, clean_act, layer):
            def hook_fn(activation, hook):
                for pos in positions:
                    if pos < activation.shape[1]:
                        activation[0, pos, :] = clean_act
                return activation
            return hook_fn

        with torch.no_grad():
            patched_logits = model.run_with_hooks(
                corr_tokens,
                fwd_hooks=[(
                    f"blocks.{best_layer}.hook_resid_post",
                    make_multi_hook(positions, clean_resid_answer, best_layer),
                )],
            )

        patched_probs = torch.softmax(patched_logits[0, -1], dim=-1)
        patched_p = max(patched_probs[t].item() for t in expected_tids)
        patched_ld = compute_logit_diff(patched_logits[0, -1], expected_tids, wrong_tids)
        recovery = compute_recovery(patched_ld, baseline_ld, clean_ld)

        position_type_results[pos_type] = {
            "positions": positions,
            "p_correct": patched_p,
            "logit_diff": patched_ld,
            "recovery": recovery,
            "layer": best_layer,
        }

    patching_results["position_types_at_best_layer"] = position_type_results

    # Cleanup
    del clean_cache, corr_cache
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "condition": corrupted_trial["condition"],
        "num_updates": corrupted_trial["num_updates"],
        "seed": hash((corrupted_trial["condition"], corrupted_trial["num_updates"])),
        "expected": expected_value,
        "clean_pred": clean_pred,
        "clean_correct": clean_correct,
        "clean_p": clean_p,
        "clean_logit_diff": clean_ld,
        "corrupted_pred": corr_pred,
        "corrupted_correct": corr_correct,
        "baseline_p": baseline_p,
        "baseline_logit_diff": baseline_ld,
        "best_recovery_layer": best_layer,
        "patching": {
            "resid_at_answer": [
                {"layer": r["layer"], "p_correct": r["p_correct"],
                 "recovery": r["recovery"], "correct": r["correct"]}
                for r in layer_recovery
            ],
            "position_types": {
                k: {kk: vv for kk, vv in v.items() if kk != "positions"}
                for k, v in position_type_results.items()
            },
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--keys", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--gpu", type=int, default=None,
                        help="Physical GPU index to use (e.g. 3 for the 4th GPU). Default: auto-detect.")
    args = parser.parse_args()

    print("=" * 70)
    print("ACTIVATION PATCHING — Causal Evidence")
    print(f"  keys={args.keys}, updates={args.updates}, trials={args.trials}")
    print("=" * 70)

    model, tokenizer, info = load_model(args.model, n_ctx=args.n_ctx, gpu_idx=args.gpu)
    candidate_pool = get_value_pool("ARBITRARY_SINGLE")
    value_to_tid = verify_single_token(tokenizer, values=candidate_pool)
    value_pool = list(value_to_tid.keys())
    categories = ORIGINAL_CATEGORIES

    all_results = {
        "model": args.model,
        "config": {"keys": args.keys, "updates": args.updates, "trials": args.trials},
        "n_layers": info.n_layers,
        "analyses": [],
    }

    t_start = time.time()

    for condition in ["RI", "PI"]:
        print(f"\n{'='*70}")
        print(f"Condition: {condition}")
        print(f"{'='*70}")

        for t_idx in range(args.trials):
            seed = hash((condition, t_idx, args.updates, "patch")) % (2**31)

            clean, corrupted = build_matched_pair(
                args.keys, args.updates, condition, seed, value_pool, categories,
            )

            print(f"\n  Trial {t_idx+1}/{args.trials} ({condition}, "
                  f"init='{corrupted['initial_value']}', final='{corrupted['final_value']}')")

            t0 = time.time()
            result = run_patching(model, tokenizer, clean, corrupted, value_to_tid)
            elapsed = time.time() - t0
            result["elapsed_sec"] = round(elapsed, 1)
            all_results["analyses"].append(result)

            # Print recovery curve summary
            resid_results = result["patching"]["resid_at_answer"]
            valid_recoveries = [r["recovery"] for r in resid_results if r["recovery"] is not None]
            best_layer = result["best_recovery_layer"]
            best_rec = max(valid_recoveries) if valid_recoveries else 0.0
            n_filt = sum(1 for r in resid_results if r["recovery"] is None)
            filt_msg = f" ({n_filt} layers filtered)" if n_filt > 0 else ""
            print(f"    Best recovery: layer {best_layer}, recovery={best_rec:.2%}{filt_msg}")

            pos_results = result["patching"]["position_types"]
            for ptype, pdata in pos_results.items():
                print(f"    Patch {ptype}: P(correct)={pdata['p_correct']:.4f} "
                      f"recovery={pdata['recovery']:.2%}")
            print(f"    ({elapsed:.1f}s)")

    total = time.time() - t_start
    all_results["total_elapsed_sec"] = round(total, 1)

    # Save
    from core.output import save_results
    out_path = save_results(all_results, args.model, args.keys, args.updates, "activation_patching")

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"SUMMARY ({total:.1f}s)")
    print(f"{'='*70}")

    for condition in ["RI", "PI"]:
        cond_results = [r for r in all_results["analyses"] if r["condition"] == condition]
        if not cond_results:
            continue

        n_corr_correct = sum(1 for r in cond_results if r["corrupted_correct"])
        n_clean_correct = sum(1 for r in cond_results if r["clean_correct"])
        avg_baseline_p = np.mean([r["baseline_p"] for r in cond_results])
        avg_clean_p = np.mean([r["clean_p"] for r in cond_results])

        # Aggregate recovery per layer, filtering degenerate trials
        n_layers = info.n_layers
        layer_agg = {}
        for layer in range(n_layers):
            raw = [lr["recovery"] for r in cond_results
                   for lr in r["patching"]["resid_at_answer"]
                   if lr["layer"] == layer]
            layer_agg[layer] = aggregate_recovery(raw)

        avg_recovery = np.array([layer_agg[l]["mean"] for l in range(n_layers)])
        best_avg_layer = int(np.argmax(avg_recovery))
        best_avg_rec = avg_recovery[best_avg_layer]
        total_filtered = sum(layer_agg[l]["n_filtered"] for l in range(n_layers))

        print(f"\n{condition}:")
        print(f"  Clean accuracy: {n_clean_correct}/{len(cond_results)}, "
              f"avg P(correct)={avg_clean_p:.4f}")
        print(f"  Corrupted accuracy: {n_corr_correct}/{len(cond_results)}, "
              f"avg P(correct)={avg_baseline_p:.4f}")
        print(f"  Best avg recovery: layer {best_avg_layer} ({best_avg_rec:.2%})")
        if total_filtered > 0:
            print(f"  Filtered {total_filtered} degenerate trials (|clean_ld - corrupted_ld| < 0.01)")
        print(f"  Recovery by layer (sampled):")
        for l in range(0, n_layers, max(1, n_layers // 8)):
            agg = layer_agg[l]
            bar = "█" * int(avg_recovery[l] * 20) if avg_recovery[l] > 0 else ""
            filt = f" ({agg['n_filtered']} filtered)" if agg["n_filtered"] > 0 else ""
            print(f"    L{l:>2}: {avg_recovery[l]:>7.2%} {bar}{filt}")


if __name__ == "__main__":
    main()
