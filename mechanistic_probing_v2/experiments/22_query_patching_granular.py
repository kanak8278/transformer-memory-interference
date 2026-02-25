"""
Step 2.15: Granular query-position patching.

Disambiguate between:
  (A) Query representation failure: model can't decode "last" → wrong retrieval
  (B) Value routing through query position: accumulated value info at query pos is wrong

Tests:
  1. Layer-specific patching at query position (early vs late)
  2. Component-specific patching (attn_out vs mlp_out vs full resid)
  3. Combined: layer × component sweep

If early-layer patching recovers PI → problem is in the query representation (A)
If only late-layer patching works → problem is in accumulated info (B)
If attn patching works but MLP doesn't → cross-position value routing issue
If MLP patching works → local query processing issue

Usage:
    cd mechanistic_probing_v2
    uv run python experiments/22_query_patching_granular.py [--trials 10]
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
from core.dataset import format_for_chat, ORIGINAL_CATEGORIES
from core.single_token_values import verify_single_token
from core.output import load_results
from core.analysis_utils import compute_logit_diff, compute_recovery, aggregate_recovery


def get_retrieval_onset_layer(model_name, keys, updates, n_layers, threshold=0.10):
    """Find the layer where retrieval begins, using exp 15 activation patching data.

    Retrieval onset = first layer where mean PI recovery exceeds threshold.
    Falls back to 2/3 * n_layers if exp 15 data not available.
    """
    try:
        data = load_results(model_name, keys, updates, "activation_patching")
    except FileNotFoundError:
        fallback = 2 * n_layers // 3
        print(f"  WARNING: No activation_patching results found. Using fallback: L{fallback} (2/3 of {n_layers})")
        return fallback

    import numpy as np
    pi_trials = [a for a in data["analyses"] if a["condition"] == "PI"]
    if not pi_trials:
        return 2 * n_layers // 3

    layer_recovery = {}
    for a in pi_trials:
        for item in a["patching"]["resid_at_answer"]:
            l = item["layer"]
            layer_recovery.setdefault(l, []).append(item["recovery"])

    for l in sorted(layer_recovery.keys()):
        if np.mean(layer_recovery[l]) > threshold:
            print(f"  Retrieval onset: L{l} ({l/n_layers*100:.0f}% through network, from exp 15)")
            return l

    # No layer exceeded threshold — use last layer
    fallback = n_layers - 1
    print(f"  WARNING: No layer exceeded {threshold:.0%} recovery. Using L{fallback}")
    return fallback


def build_matched_pair(num_keys, num_updates, condition, seed, value_pool, categories):
    """Build matched clean/corrupted trial pair (same as experiment 15)."""
    rng = random.Random(seed)
    cats = categories[:num_keys]
    test_cat = cats[seed % num_keys]

    total_needed = num_keys * num_updates
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

    # Clean: 1 value per category (the correct answer)
    clean_items = []
    for cat in cats:
        if cat == test_cat:
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

    # Corrupted: all updates, interleaved
    corr_items = []
    for cat in cats:
        for val in values_per_cat[cat]:
            corr_items.append({"category": cat, "value": val})

    rng3 = random.Random(seed + 2000)
    rng3.shuffle(corr_items)
    for _ in range(100):
        ok = all(corr_items[i]["category"] != corr_items[i-1]["category"]
                 for i in range(1, len(corr_items)))
        if ok:
            break
        rng3.shuffle(corr_items)

    corr_stream = "\n".join(f"{it['category']}: {it['value']}" for it in corr_items)
    corr_prompt = (
        f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{corr_stream}\n\n"
        f"What was the {query_word} value of {test_cat}?"
    )

    clean_trial = {
        "prompt": clean_prompt, "condition": condition,
        "expected": expected, "initial_value": initial_value,
        "final_value": final_value,
    }
    corrupted_trial = {
        "prompt": corr_prompt, "condition": condition,
        "expected": expected, "initial_value": initial_value,
        "final_value": final_value,
    }
    return clean_trial, corrupted_trial


def find_query_positions(model, formatted_text):
    """Find the positions of the query (from 'What' to end)."""
    str_tokens = model.to_str_tokens(formatted_text)
    for i in range(len(str_tokens) - 1, -1, -1):
        if "What" in str_tokens[i]:
            return list(range(i, len(str_tokens)))
    return list(range(max(0, len(str_tokens) - 15), len(str_tokens)))


def patch_at_positions(model, corr_tokens, clean_cache, clean_positions, corr_positions,
                       layer, component, answer_pos):
    """Patch clean activations into corrupted run at specified positions.

    Args:
        component: "resid_post", "attn_out", or "mlp_out"
        clean_positions: positions in clean run to source from
        corr_positions: positions in corrupted run to patch into
    """
    # Map clean → corrupted positions (they may differ in length)
    # Strategy: patch each corrupted query position from the corresponding clean position
    n_to_patch = min(len(clean_positions), len(corr_positions))

    if component == "resid_post":
        hook_name = f"blocks.{layer}.hook_resid_post"
    elif component == "attn_out":
        hook_name = f"blocks.{layer}.hook_attn_out"
    elif component == "mlp_out":
        hook_name = f"blocks.{layer}.hook_mlp_out"
    else:
        raise ValueError(f"Unknown component: {component}")

    # Pre-extract clean activations
    cache_key = (component, layer) if component == "resid_post" else \
                ("attn_out", layer) if component == "attn_out" else ("mlp_out", layer)
    clean_acts = []
    for i in range(n_to_patch):
        cp = clean_positions[i]
        clean_acts.append(clean_cache[cache_key][0, cp, :].clone())

    def hook_fn(activation, hook):
        for i in range(n_to_patch):
            corr_pos = corr_positions[i]
            if corr_pos < activation.shape[1]:
                activation[0, corr_pos, :] = clean_acts[i]
        return activation

    with torch.no_grad():
        patched_logits = model.run_with_hooks(
            corr_tokens, fwd_hooks=[(hook_name, hook_fn)])

    return patched_logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--keys", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=2048)
    args = parser.parse_args()

    print("=" * 70)
    print("GRANULAR QUERY-POSITION PATCHING")
    print(f"  keys={args.keys}, updates={args.updates}, trials={args.trials}")
    print("=" * 70)

    model, tokenizer, info = load_model(args.model, n_ctx=args.n_ctx)
    value_to_tid = verify_single_token(tokenizer)
    value_pool = list(value_to_tid.keys())
    categories = ORIGINAL_CATEGORIES
    n_layers = model.cfg.n_layers

    # Layers to test — dynamically spaced based on model depth
    n_test = 13  # ~13 evenly spaced layers
    step = max(1, n_layers // n_test)
    test_layers = sorted(set(list(range(0, n_layers, step)) + [n_layers - 1]))
    components = ["resid_post", "attn_out", "mlp_out"]

    # Accumulate recovery scores: {condition: {component: {layer: [recoveries]}}}
    recovery_data = {}
    for cond in ["RI", "PI"]:
        recovery_data[cond] = {}
        for comp in components:
            recovery_data[cond][comp] = {l: [] for l in test_layers}
        # Also track baseline stats
        recovery_data[cond]["_baseline_correct"] = 0
        recovery_data[cond]["_clean_correct"] = 0
        recovery_data[cond]["_n"] = 0

    # Determine early/late boundary from exp 15 data
    onset_layer = get_retrieval_onset_layer(args.model, args.keys, args.updates, n_layers)

    print(f"\nRunning {args.trials} trials × 2 conditions × {len(test_layers)} layers × {len(components)} components...")
    t_start = time.time()

    for t_idx in range(args.trials):
        for condition in ["RI", "PI"]:
            seed = hash((condition, t_idx, args.updates, "granular_patch")) % (2**31)
            clean_trial, corr_trial = build_matched_pair(
                args.keys, args.updates, condition, seed, value_pool, categories)

            expected = corr_trial["expected"]
            exp_tid_sp = value_to_tid.get(expected, -1)
            exp_tid_bare = tokenizer.encode(expected, add_special_tokens=False)[0]
            expected_tids = list(set([t for t in [exp_tid_sp, exp_tid_bare] if t >= 0]))

            # Incorrect token for logit_diff
            wrong_value = corr_trial["final_value"] if condition == "RI" else corr_trial["initial_value"]
            wrong_tid_sp = value_to_tid.get(wrong_value, -1)
            wrong_tid_bare = tokenizer.encode(wrong_value, add_special_tokens=False)[0]
            wrong_tids = list(set([t for t in [wrong_tid_sp, wrong_tid_bare] if t >= 0]))
            if not wrong_tids:
                wrong_tids = expected_tids

            # Run clean
            clean_fmt = format_for_chat(clean_trial["prompt"], tokenizer)
            clean_tokens = model.to_tokens(clean_fmt)
            with torch.no_grad():
                clean_logits, clean_cache = model.run_with_cache(clean_tokens)

            clean_pred = tokenizer.decode([clean_logits[0, -1].argmax().item()]).strip()
            clean_correct = clean_pred.lower() == expected.lower()
            clean_ld = compute_logit_diff(clean_logits[0, -1], expected_tids, wrong_tids)

            # Run corrupted
            corr_fmt = format_for_chat(corr_trial["prompt"], tokenizer)
            corr_tokens = model.to_tokens(corr_fmt)
            with torch.no_grad():
                corr_logits = model(corr_tokens)

            corr_pred = tokenizer.decode([corr_logits[0, -1].argmax().item()]).strip()
            corr_correct = corr_pred.lower() == expected.lower()
            baseline_ld = compute_logit_diff(corr_logits[0, -1], expected_tids, wrong_tids)

            recovery_data[condition]["_baseline_correct"] += corr_correct
            recovery_data[condition]["_clean_correct"] += clean_correct
            recovery_data[condition]["_n"] += 1

            # Find query positions in both
            clean_query_pos = find_query_positions(model, clean_fmt)
            corr_query_pos = find_query_positions(model, corr_fmt)

            # Sweep: layer × component, patching at query positions
            for layer in test_layers:
                for comp in components:
                    patched_logits = patch_at_positions(
                        model, corr_tokens, clean_cache,
                        clean_query_pos, corr_query_pos,
                        layer, comp, corr_tokens.shape[1] - 1)

                    patched_ld = compute_logit_diff(patched_logits[0, -1], expected_tids, wrong_tids)
                    recovery = compute_recovery(patched_ld, baseline_ld, clean_ld)

                    recovery_data[condition][comp][layer].append(recovery)

            del clean_cache
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        if (t_idx + 1) % 2 == 0:
            print(f"  {t_idx + 1}/{args.trials} done ({time.time() - t_start:.0f}s)")

    # ── Print results ──
    total_time = time.time() - t_start

    for condition in ["PI", "RI"]:
        n = recovery_data[condition]["_n"]
        base_acc = recovery_data[condition]["_baseline_correct"] / n
        clean_acc = recovery_data[condition]["_clean_correct"] / n

        print(f"\n{'='*70}")
        print(f"{condition} — Query Position Patching (clean={clean_acc:.0%}, corrupted={base_acc:.0%})")
        print(f"{'='*70}")
        print(f"\n  {'Layer':>6}", end="")
        for comp in components:
            print(f"  {comp:>12}", end="")
        print("  filtered")
        print(f"  {'-'*60}")

        for layer in test_layers:
            print(f"  L{layer:>4}", end="")
            layer_filtered = 0
            for comp in components:
                vals = recovery_data[condition][comp][layer]
                agg = aggregate_recovery(vals)
                layer_filtered += agg["n_filtered"]
                print(f"  {agg['mean']:>+11.0%}", end="")
            print(f"  {layer_filtered}" if layer_filtered > 0 else "")

        # Summary: which component matters most?
        print(f"\n  Average across all layers:")
        for comp in components:
            all_vals = []
            for layer in test_layers:
                all_vals.extend(recovery_data[condition][comp][layer])
            agg = aggregate_recovery(all_vals)
            filt_msg = f" ({agg['n_filtered']} filtered)" if agg["n_filtered"] > 0 else ""
            print(f"    {comp:<12}: {agg['mean']:+.0%}{filt_msg}")

        # Early vs late — boundary from exp 15 retrieval onset
        early_layers = [l for l in test_layers if l < onset_layer]
        late_layers = [l for l in test_layers if l >= onset_layer]
        for comp in components:
            early_vals = [v for l in early_layers for v in recovery_data[condition][comp][l]]
            late_vals = [v for l in late_layers for v in recovery_data[condition][comp][l]]
            early_agg = aggregate_recovery(early_vals)
            late_agg = aggregate_recovery(late_vals)
            if early_agg["n_valid"] > 0 and late_agg["n_valid"] > 0:
                print(f"    {comp} early (pre-onset): {early_agg['mean']:+.0%}  late (post-onset): {late_agg['mean']:+.0%}")

    # ── Interpretation ──
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")

    # Check PI specifically (filter None values)
    pi_early_resid = [v for l in test_layers if l < onset_layer for v in recovery_data["PI"]["resid_post"].get(l, []) if v is not None]
    pi_late_resid = [v for l in test_layers if l >= onset_layer for v in recovery_data["PI"]["resid_post"].get(l, []) if v is not None]
    pi_attn_all = [v for l in test_layers for v in recovery_data["PI"]["attn_out"][l] if v is not None]
    pi_mlp_all = [v for l in test_layers for v in recovery_data["PI"]["mlp_out"][l] if v is not None]

    early_mean = np.mean(pi_early_resid) if pi_early_resid else 0
    late_mean = np.mean(pi_late_resid) if pi_late_resid else 0
    attn_mean = np.mean(pi_attn_all) if pi_attn_all else 0
    mlp_mean = np.mean(pi_mlp_all) if pi_mlp_all else 0

    print(f"\n  PI condition (onset at L{onset_layer}):")
    print(f"  Early resid patching (L0-{onset_layer-1}):  {early_mean:+.0%}")
    print(f"  Late resid patching (L{onset_layer}-{n_layers-1}): {late_mean:+.0%}")
    print(f"  Attn_out (all layers):        {attn_mean:+.0%}")
    print(f"  MLP_out (all layers):         {mlp_mean:+.0%}")

    if early_mean > 0.3:
        print(f"\n  → SUPPORTS (A): Early-layer query patching recovers PI.")
        print(f"    The query representation is already wrong from the start.")
        print(f"    PI failure is in instruction processing, not accumulated value routing.")
    elif late_mean > 0.3 and early_mean < 0.1:
        print(f"\n  → SUPPORTS (B): Only late-layer patching works.")
        print(f"    The query word is processed correctly, but accumulated")
        print(f"    cross-position information at the query position is corrupted.")
    else:
        print(f"\n  → MIXED: Neither early nor late patching dominates.")
        print(f"    The corruption may build gradually across layers.")

    if attn_mean > mlp_mean + 0.1:
        print(f"  → Attention output at query position is the critical component.")
        print(f"    Cross-position value routing through query position matters more than local processing.")
    elif mlp_mean > attn_mean + 0.1:
        print(f"  → MLP output at query position is the critical component.")
        print(f"    Local query processing (interpreting 'first'/'last') is the bottleneck.")
    else:
        print(f"  → Both attn and MLP contribute roughly equally.")

    # ── Save ──
    save_data = {
        "model": args.model,
        "config": {"keys": args.keys, "updates": args.updates, "trials": args.trials},
        "test_layers": test_layers,
        "retrieval_onset_layer": onset_layer,
        "components": components,
        "recovery": {},
    }
    for cond in ["RI", "PI"]:
        save_data["recovery"][cond] = {
            "baseline_accuracy": recovery_data[cond]["_baseline_correct"] / recovery_data[cond]["_n"],
            "clean_accuracy": recovery_data[cond]["_clean_correct"] / recovery_data[cond]["_n"],
        }
        for comp in components:
            save_data["recovery"][cond][comp] = {
                str(l): aggregate_recovery(recovery_data[cond][comp][l])
                for l in test_layers
            }

    from core.output import save_results
    out_path = save_results(save_data, args.model, args.keys, args.updates, "query_patching_granular")
    print(f"({total_time:.0f}s total)")


if __name__ == "__main__":
    main()
