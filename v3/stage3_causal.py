"""
V3 Stage 3: Causal Mechanistic Experiments

Identifies which heads suppress P(v_last) and validates causally.

Four experiments:
  3A: Attribution patching — gradient-based head ranking (1 fwd + 1 bwd per trial)
  3B: Targeted activation patching — causal validation on top heads
  3C: Logit lens under ablation — how trajectory changes when heads are removed
  3D: Forced attention — QK vs OV test (only if concentrated competitor)

Requires: TransformerLens, Stage 2 results for critical layer identification.

Usage:
    cd v3
    python stage3_causal.py --model Qwen/Qwen2.5-3B-Instruct \
        --stage2 results/Qwen2.5-3B-Instruct/stage2_logit_lens_*.json \
        --point "5,5" --trials 50 --gpu 0

    # Or auto-pick best operating point from stage2
    python stage3_causal.py --model Qwen/Qwen2.5-3B-Instruct \
        --stage2 results/Qwen2.5-3B-Instruct/stage2_logit_lens_*.json \
        --trials 50 --gpu 0
"""

import sys
import json
import time
import random
import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from functools import partial

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mechanistic_probing_v2.core.model_loader import (
    load_model, verify_single_token, model_short_name, clear_accelerator_cache,
    is_instruct_model,
)
from mechanistic_probing_v2.core.dataset_configs import (
    format_for_chat, get_value_pool, get_eligible_categories,
    generate_values_for_trial, FIXED_COMPLETION_DEMOS,
)
from mechanistic_probing_v2.core.evaluation import classify_error

DATASET_TYPE = "ARBITRARY_SINGLE"
SYSTEM_PROMPT = (
    "You are a precise data extraction tool. "
    "Output ONLY a single word - the exact value requested. "
    "No other text, no explanation, no punctuation."
)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 ANALYSIS: extract critical layers and operating point
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_stage2(stage2_path):
    """Extract critical layers, competitor info, and best operating point from Stage 2."""
    data = json.load(open(stage2_path))
    analyses = data["analyses"]
    n_layers = data["n_layers"]
    points = data["points"]

    best_point = None
    best_score = -1
    point_info = {}

    for pt in points:
        k, u = pt["keys"], pt["updates"]
        pi_failures = [a for a in analyses
                       if a["condition"] == "PI" and a["num_keys"] == k
                       and a["num_updates"] == u
                       and not a["correct"] and a["total_value_prob"] > 0.05]

        if len(pi_failures) < 10:
            continue

        n_values = len(pi_failures[0]["all_values"])
        avg = np.mean([a["value_probs_by_layer"] for a in pi_failures], axis=0)
        correct_idx = n_values - 1

        # Find critical layers: where P(v_last) peaks then drops
        p_last = avg[correct_idx, :]
        peak_layer = np.argmax(p_last)
        peak_val = p_last[peak_layer]
        final_val = p_last[-1]
        suppression = peak_val - final_val

        # Critical layers: from peak to end
        critical_start = max(0, peak_layer - 1)
        critical_layers = list(range(critical_start, n_layers))

        # Competitor analysis at final layer
        final_probs = avg[:, -1].copy()
        final_probs[correct_idx] = -1
        dominant_wrong = int(np.argmax(final_probs))
        dominant_prob = final_probs[dominant_wrong]

        # Is it concentrated or diffuse?
        non_correct_probs = np.delete(avg[:, -1], correct_idx)
        top_wrong_share = dominant_prob / (non_correct_probs.sum() + 1e-10)
        concentrated = top_wrong_share > 0.5

        info = {
            "keys": k, "updates": u,
            "n_failures": len(pi_failures),
            "peak_layer": int(peak_layer),
            "peak_p_last": float(peak_val),
            "final_p_last": float(final_val),
            "suppression": float(suppression),
            "critical_layers": critical_layers,
            "dominant_wrong_idx": dominant_wrong,
            "dominant_wrong_prob": float(dominant_prob),
            "concentrated": concentrated,
        }
        point_info[f"{k},{u}"] = info

        # Score: prefer points with high suppression and many failures
        score = suppression * len(pi_failures)
        if score > best_score:
            best_score = score
            best_point = (k, u)

    return {
        "n_layers": n_layers,
        "n_heads": data["n_heads"],
        "best_point": best_point,
        "point_info": point_info,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TRIAL GENERATION (same as stage2)
# ═══════════════════════════════════════════════════════════════════════════════

def shuffle_no_consecutive(items, rng, max_attempts=100):
    for _ in range(max_attempts):
        candidate = items.copy()
        rng.shuffle(candidate)
        ok = all(candidate[i]["category"] != candidate[i - 1]["category"]
                 for i in range(1, len(candidate)))
        if ok:
            return candidate
    remaining = items.copy()
    rng.shuffle(remaining)
    result, last_cat = [], None
    while remaining:
        valid = [i for i, item in enumerate(remaining) if item["category"] != last_cat]
        if not valid:
            result.extend(remaining)
            break
        idx = rng.choice(valid)
        item = remaining.pop(idx)
        result.append(item)
        last_cat = item["category"]
    return result


def generate_trial(num_keys, num_updates, condition, seed, value_pool, tokenizer,
                   model_name):
    rng = random.Random(seed)
    eligible = get_eligible_categories(DATASET_TYPE, min_values=num_updates)
    categories = rng.sample(eligible, min(num_keys, len(eligible)))
    test_category = categories[seed % num_keys]
    values_per_cat = generate_values_for_trial(DATASET_TYPE, categories, num_updates, rng)

    items = []
    for cat in categories:
        for val in values_per_cat[cat]:
            items.append({"category": cat, "value": val})
    items = shuffle_no_consecutive(items, rng)

    stream = "\n".join(f"{it['category']}: {it['value']}" for it in items)
    query_word = "first" if condition == "RI" else "last"
    cat_values = [it["value"] for it in items if it["category"] == test_category]
    expected = cat_values[0] if condition == "RI" else cat_values[-1]

    use_chat = is_instruct_model(model_name) if model_name else True
    if use_chat:
        raw_prompt = (
            f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
            f"{stream}\n\n"
            f"What was the {query_word} value of {test_category}?"
        )
        formatted = format_for_chat(raw_prompt, tokenizer, model_name=model_name)
    else:
        formatted = f"{FIXED_COMPLETION_DEMOS}{stream}\nThe {query_word} value of {test_category} was:"

    return {
        "prompt": formatted, "condition": condition, "expected": expected,
        "initial_value": cat_values[0], "final_value": cat_values[-1],
        "all_values": cat_values, "test_category": test_category,
        "num_keys": num_keys, "num_updates": num_updates, "seed": seed,
    }


def get_value_tids(all_values, tokenizer, value_to_tid):
    """Get token IDs for each value, with fallback to value_to_tid."""
    value_tids = []
    for v in all_values:
        sp = tokenizer.encode(f" {v}", add_special_tokens=False)
        bare = tokenizer.encode(v, add_special_tokens=False)
        tids = set()
        if len(sp) == 1:
            tids.add(sp[0])
        if len(bare) == 1:
            tids.add(bare[0])
        if v in value_to_tid:
            tids.add(value_to_tid[v])
        value_tids.append(list(tids))
    return value_tids


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3A: Attribution Patching
# ═══════════════════════════════════════════════════════════════════════════════

def run_attribution_patching(model, tokenizer, value_to_tid, value_pool,
                             model_name, num_keys, num_updates, trials, config):
    """Gradient-based head importance for P(v_last).

    For each PI trial:
      1. Forward pass with cache (corrupted = PI prompt)
      2. Compute metric: logit(v_last) at final position
      3. Backward to get gradient at each head's output (z)
      4. Attribution = |grad| (or grad · (clean - corrupted) if we have clean)

    Simplified version: just use |grad of metric w.r.t. head output|.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Accumulate attribution scores
    total_attribution = np.zeros((n_layers, n_heads))
    n_valid = 0

    print(f"\n  Exp 3A: Attribution Patching ({trials} PI trials)")

    for t_idx in range(trials):
        seed = hash((num_keys, num_updates, "PI", t_idx, "v3_stage3")) % (2**31)
        trial = generate_trial(num_keys, num_updates, "PI", seed,
                               value_pool, tokenizer, model_name)

        all_values = trial["all_values"]
        n_values = len(all_values)
        correct_idx = n_values - 1
        value_tids = get_value_tids(all_values, tokenizer, value_to_tid)

        if not value_tids[correct_idx]:
            continue

        target_tid = value_tids[correct_idx][0]

        tokens = model.to_tokens(trial["prompt"])

        # We need gradients, so enable them for this pass
        model.zero_grad()

        # Forward pass — need to hook z (pre-projection head outputs) and make them require grad
        z_activations = {}

        def save_z_hook(activation, hook, layer):
            activation.retain_grad()  # Keep gradients for non-leaf tensors
            z_activations[layer] = activation
            return activation

        hooks = []
        for layer in range(n_layers):
            hooks.append((
                f"blocks.{layer}.attn.hook_z",
                partial(save_z_hook, layer=layer),
            ))

        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

        # Metric: logit of v_last at the answer position
        metric = logits[0, -1, target_tid]

        # Check if this is a failure trial (metric should be low)
        pred_tid = logits[0, -1].argmax().item()
        pred_text = tokenizer.decode([pred_tid]).strip()
        is_correct = pred_text.lower() == trial["expected"].lower()

        if is_correct:
            # Skip correct trials — we want to study failures
            continue

        # Backward pass to get gradients at each z
        metric.backward()

        # Extract gradient magnitude per head
        for layer in range(n_layers):
            z = z_activations[layer]
            if z.grad is not None:
                # z shape: [batch, seq, n_heads, d_head]
                # Take gradient at the answer position, sum over d_head
                grad = z.grad[0, -1, :, :]  # [n_heads, d_head]
                total_attribution[layer, :] += grad.abs().sum(dim=-1).cpu().numpy()

        n_valid += 1
        model.zero_grad()

        if (t_idx + 1) % 10 == 0:
            print(f"    [{t_idx + 1}/{trials}] valid={n_valid}", end="")
            if n_valid > 0:
                top_idx = np.unravel_index(np.argmax(total_attribution), total_attribution.shape)
                print(f"  top so far: L{top_idx[0]}H{top_idx[1]}")
            else:
                print()

        # Clean up
        for layer in z_activations:
            if z_activations[layer].grad is not None:
                z_activations[layer].grad = None
        z_activations.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Normalize
    if n_valid > 0:
        avg_attribution = total_attribution / n_valid
    else:
        avg_attribution = total_attribution

    # Rank heads
    flat_idx = np.argsort(avg_attribution.ravel())[::-1]
    top_k = min(config.get("top_k", 20), len(flat_idx))
    top_heads = []
    for i in range(top_k):
        layer, head = np.unravel_index(flat_idx[i], avg_attribution.shape)
        top_heads.append({
            "layer": int(layer), "head": int(head),
            "attribution": float(avg_attribution[layer, head]),
            "label": f"L{layer}H{head}",
        })

    print(f"\n  Top {top_k} heads by attribution:")
    for h in top_heads[:10]:
        print(f"    {h['label']}: {h['attribution']:.4f}")

    return {
        "experiment": "3A_attribution_patching",
        "n_trials": trials,
        "n_valid": n_valid,
        "attribution_scores": avg_attribution.tolist(),
        "top_heads": top_heads,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3B: Targeted Activation Patching
# ═══════════════════════════════════════════════════════════════════════════════

def run_targeted_patching(model, tokenizer, value_to_tid, value_pool,
                          model_name, num_keys, num_updates, trials,
                          target_heads, config):
    """Causal validation: patch each target head from clean→corrupted, measure P(v_last) change."""

    n_layers = model.cfg.n_layers
    print(f"\n  Exp 3B: Targeted Patching ({len(target_heads)} heads × {trials} trials)")

    head_results = {}

    for t_idx in range(trials):
        seed = hash((num_keys, num_updates, "PI", t_idx, "v3_stage3_patch")) % (2**31)

        # Generate paired trials: same sequence, different query
        pi_trial = generate_trial(num_keys, num_updates, "PI", seed,
                                  value_pool, tokenizer, model_name)
        ri_trial = generate_trial(num_keys, num_updates, "RI", seed,
                                  value_pool, tokenizer, model_name)

        all_values = pi_trial["all_values"]
        n_values = len(all_values)
        correct_idx = n_values - 1
        value_tids = get_value_tids(all_values, tokenizer, value_to_tid)

        if not value_tids[correct_idx]:
            continue
        target_tid = value_tids[correct_idx][0]

        # Clean run (RI — model succeeds at retrieving v_0)
        clean_tokens = model.to_tokens(ri_trial["prompt"])
        with torch.no_grad():
            _, clean_cache = model.run_with_cache(clean_tokens)

        # Corrupted run (PI — model fails)
        corr_tokens = model.to_tokens(pi_trial["prompt"])
        with torch.no_grad():
            corr_logits, corr_cache = model.run_with_cache(corr_tokens)

        # Baseline P(v_last) on corrupted
        corr_probs = torch.softmax(corr_logits[0, -1], dim=-1)
        baseline_p_last = corr_probs[target_tid].item()

        # Skip if already correct
        pred_tid = corr_logits[0, -1].argmax().item()
        pred_text = tokenizer.decode([pred_tid]).strip()
        if pred_text.lower() == pi_trial["expected"].lower():
            del clean_cache, corr_cache
            continue

        # Patch each target head
        for h_info in target_heads:
            layer, head = h_info["layer"], h_info["head"]
            label = h_info["label"]

            # Get clean head output at answer position
            clean_z = clean_cache["z", layer][0, -1, head, :].clone()

            def patch_hook(activation, hook, _layer=layer, _head=head, _clean_z=clean_z):
                activation[0, -1, _head, :] = _clean_z
                return activation

            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    corr_tokens,
                    fwd_hooks=[(f"blocks.{layer}.attn.hook_z", patch_hook)],
                )

            patched_probs = torch.softmax(patched_logits[0, -1], dim=-1)
            patched_p_last = patched_probs[target_tid].item()

            delta = patched_p_last - baseline_p_last

            if label not in head_results:
                head_results[label] = {"deltas": [], "layer": layer, "head": head}
            head_results[label]["deltas"].append(delta)

        del clean_cache, corr_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (t_idx + 1) % 10 == 0:
            print(f"    [{t_idx + 1}/{trials}]")

    # Aggregate
    patching_summary = {}
    for label, info in head_results.items():
        deltas = info["deltas"]
        patching_summary[label] = {
            "layer": info["layer"],
            "head": info["head"],
            "mean_delta_p_last": float(np.mean(deltas)),
            "std_delta_p_last": float(np.std(deltas)),
            "n_trials": len(deltas),
            "positive_rate": float(np.mean([d > 0 for d in deltas])),
        }

    # Sort by mean delta
    sorted_heads = sorted(patching_summary.items(), key=lambda x: -x[1]["mean_delta_p_last"])

    print(f"\n  Top heads by patching effect (ΔP(v_last)):")
    for label, info in sorted_heads[:10]:
        print(f"    {label}: ΔP(v_last)={info['mean_delta_p_last']:+.4f} "
              f"±{info['std_delta_p_last']:.4f} "
              f"(positive {info['positive_rate']:.0%} of trials)")

    return {
        "experiment": "3B_targeted_patching",
        "n_trials": trials,
        "head_results": patching_summary,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3C: Logit Lens Under Ablation
# ═══════════════════════════════════════════════════════════════════════════════

def run_ablation_logit_lens(model, tokenizer, value_to_tid, value_pool,
                            model_name, num_keys, num_updates, trials,
                            ablate_heads, config):
    """Run logit lens with and without ablating target heads. Compare P(v_i) trajectories."""

    n_layers = model.cfg.n_layers
    print(f"\n  Exp 3C: Logit Lens Under Ablation ({len(ablate_heads)} heads ablated, {trials} trials)")

    normal_trajectories = []
    ablated_trajectories = []

    for t_idx in range(trials):
        seed = hash((num_keys, num_updates, "PI", t_idx, "v3_stage3_ablate")) % (2**31)
        trial = generate_trial(num_keys, num_updates, "PI", seed,
                               value_pool, tokenizer, model_name)

        all_values = trial["all_values"]
        n_values = len(all_values)
        value_tids = get_value_tids(all_values, tokenizer, value_to_tid)

        tokens = model.to_tokens(trial["prompt"])

        # Normal run
        with torch.no_grad():
            _, normal_cache = model.run_with_cache(tokens)

        normal_probs = np.zeros((n_values, n_layers))
        for layer in range(n_layers):
            resid = normal_cache["resid_post", layer][0, -1, :]
            logits = resid @ model.W_U + model.b_U
            probs = torch.softmax(logits, dim=-1)
            for vi in range(n_values):
                if value_tids[vi]:
                    normal_probs[vi, layer] = max(probs[tid].item() for tid in value_tids[vi])

        del normal_cache

        # Ablated run: zero out target heads
        def ablate_hook(activation, hook, layer, head):
            activation[0, :, head, :] = 0.0
            return activation

        hooks = []
        for h_info in ablate_heads:
            hooks.append((
                f"blocks.{h_info['layer']}.attn.hook_z",
                partial(ablate_hook, layer=h_info["layer"], head=h_info["head"]),
            ))

        with torch.no_grad():
            _, ablated_cache = model.run_with_hooks(
                tokens, fwd_hooks=hooks, return_type="both"
            )

        # TransformerLens run_with_hooks doesn't return cache by default with return_type
        # Use run_with_cache + hooks instead
        del ablated_cache

        # Re-run with hooks and cache
        ablated_probs = np.zeros((n_values, n_layers))

        # Manual approach: run with hooks, collect resid_post at each layer
        resid_hooks = {}

        def save_resid_hook(activation, hook, layer):
            resid_hooks[layer] = activation.detach().clone()
            return activation

        all_hooks = list(hooks)
        for layer in range(n_layers):
            all_hooks.append((
                f"blocks.{layer}.hook_resid_post",
                partial(save_resid_hook, layer=layer),
            ))

        with torch.no_grad():
            model.run_with_hooks(tokens, fwd_hooks=all_hooks)

        for layer in range(n_layers):
            if layer in resid_hooks:
                resid = resid_hooks[layer][0, -1, :]
                logits = resid @ model.W_U + model.b_U
                probs = torch.softmax(logits, dim=-1)
                for vi in range(n_values):
                    if value_tids[vi]:
                        ablated_probs[vi, layer] = max(probs[tid].item() for tid in value_tids[vi])

        resid_hooks.clear()

        normal_trajectories.append(normal_probs.tolist())
        ablated_trajectories.append(ablated_probs.tolist())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (t_idx + 1) % 10 == 0:
            n_v = n_values
            norm_last = np.mean([t[n_v - 1][-1] for t in normal_trajectories])
            abl_last = np.mean([t[n_v - 1][-1] for t in ablated_trajectories])
            print(f"    [{t_idx + 1}/{trials}] P(v_last) final: "
                  f"normal={norm_last:.4f} ablated={abl_last:.4f} Δ={abl_last - norm_last:+.4f}")

    # Aggregate
    avg_normal = np.mean(normal_trajectories, axis=0)
    avg_ablated = np.mean(ablated_trajectories, axis=0)
    n_values = avg_normal.shape[0]

    print(f"\n  P(v_last) at final layer:")
    print(f"    Normal:  {avg_normal[n_values - 1, -1]:.4f}")
    print(f"    Ablated: {avg_ablated[n_values - 1, -1]:.4f}")
    print(f"    Delta:   {avg_ablated[n_values - 1, -1] - avg_normal[n_values - 1, -1]:+.4f}")

    return {
        "experiment": "3C_ablation_logit_lens",
        "n_trials": trials,
        "heads_ablated": [h["label"] for h in ablate_heads],
        "avg_normal_trajectory": avg_normal.tolist(),
        "avg_ablated_trajectory": avg_ablated.tolist(),
        "p_last_normal_final": float(avg_normal[n_values - 1, -1]),
        "p_last_ablated_final": float(avg_ablated[n_values - 1, -1]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════════

def get_save_dir(model_name):
    m_short = model_short_name(model_name)
    return Path(__file__).resolve().parent / "results" / m_short


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types that json.dump can't serialize."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_results(results, model_name, partial=False):
    save_dir = get_save_dir(model_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    if partial:
        path = save_dir / "stage3_checkpoint.json"
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = save_dir / f"stage3_causal_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    label = "Checkpoint" if partial else "Saved"
    print(f"  -> {label}: {path}")
    return str(path)


# ═══════════════════════════════════════════════════════════════════════════════
# RUN STAGE 3
# ═══════════════════════════════════════════════════════════════════════════════

def run_stage3(config, model=None, tokenizer=None, info=None):
    """Run Stage 3 causal experiments.

    Args:
        config: dict with keys:
            model: model name
            stage2_path: path to Stage 2 results JSON
            point: (keys, updates) tuple, or None for auto-pick
            trials: trials per experiment
            top_k: number of top heads from attribution patching (default 20)
            gpu: GPU index
            experiments: list of experiments to run (default: all)
        model, tokenizer, info: pre-loaded TransformerLens model (optional)
    """
    model_name = config["model"]
    stage2_path = config["stage2_path"]
    trials = config.get("trials", 50)
    top_k = config.get("top_k", 20)
    experiments = config.get("experiments", ["3A", "3B", "3C"])

    # Analyze Stage 2
    print("Analyzing Stage 2 results...")
    s2_info = analyze_stage2(stage2_path)
    n_layers = s2_info["n_layers"]
    n_heads = s2_info["n_heads"]

    # Pick operating point
    if config.get("point"):
        num_keys, num_updates = config["point"]
    else:
        num_keys, num_updates = s2_info["best_point"]
    point_key = f"{num_keys},{num_updates}"

    if point_key not in s2_info["point_info"]:
        raise ValueError(f"Point {point_key} not found in Stage 2 results. "
                        f"Available: {list(s2_info['point_info'].keys())}")

    pt_info = s2_info["point_info"][point_key]

    print(f"\n{'='*70}")
    print(f"V3 STAGE 3: CAUSAL MECHANISTIC EXPERIMENTS")
    print(f"  Model:           {model_name}")
    print(f"  Operating point: {num_keys}k_{num_updates}u")
    print(f"  From Stage 2:    pattern={'B (overtaken)' if pt_info['suppression'] > 0.05 else 'unclear'}")
    print(f"    Critical layers: {pt_info['critical_layers'][:5]}...{pt_info['critical_layers'][-1]}")
    print(f"    Peak P(v_last):  {pt_info['peak_p_last']:.4f} at L{pt_info['peak_layer']}")
    print(f"    Final P(v_last): {pt_info['final_p_last']:.4f}")
    print(f"    Suppression:     {pt_info['suppression']:.4f}")
    print(f"    Concentrated:    {pt_info['concentrated']}")
    print(f"  Trials: {trials}")
    print(f"  Experiments: {experiments}")
    print(f"{'='*70}")

    # Load model if not provided
    if model is None:
        model, tokenizer, info = load_model(model_name, gpu_idx=config.get("gpu"))

    # Build value pool
    candidate_pool = get_value_pool(DATASET_TYPE)
    value_to_tid = verify_single_token(tokenizer, values=candidate_pool)
    value_pool = list(value_to_tid.keys())
    print(f"\nValue pool: {len(value_pool)} single-token verified")

    results = {
        "model": model_name,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "operating_point": {"keys": num_keys, "updates": num_updates},
        "stage2_info": pt_info,
        "config": config,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "experiments": {},
    }

    # ── 3A: Attribution Patching ──
    if "3A" in experiments:
        t0 = time.time()
        result_3a = run_attribution_patching(
            model, tokenizer, value_to_tid, value_pool,
            model_name, num_keys, num_updates, trials, config
        )
        result_3a["elapsed_sec"] = round(time.time() - t0, 1)
        results["experiments"]["3A"] = result_3a
        save_results(results, model_name, partial=True)

        # Build Set A: top-K from attribution
        set_a = result_3a["top_heads"][:top_k]
    else:
        set_a = []

    # Build Set B: all heads in critical layers
    critical_layers = pt_info["critical_layers"]
    set_b = []
    for layer in critical_layers:
        for head in range(n_heads):
            label = f"L{layer}H{head}"
            set_b.append({"layer": layer, "head": head, "label": label})

    # Union: A ∪ B
    seen = set()
    target_heads = []
    for h in set_a:
        key = (h["layer"], h["head"])
        if key not in seen:
            seen.add(key)
            h["in_set_A"] = True
            h["in_set_B"] = h["layer"] in critical_layers
            target_heads.append(h)
    for h in set_b:
        key = (h["layer"], h["head"])
        if key not in seen:
            seen.add(key)
            h["in_set_A"] = False
            h["in_set_B"] = True
            h["attribution"] = 0.0
            target_heads.append(h)

    results["target_heads"] = {
        "set_A_size": len(set_a),
        "set_B_size": len(set_b),
        "union_size": len(target_heads),
        "heads": [{"label": h["label"], "in_A": h.get("in_set_A", False),
                   "in_B": h.get("in_set_B", False)} for h in target_heads],
    }

    print(f"\n  Target heads: Set A={len(set_a)}, Set B={len(set_b)}, "
          f"Union={len(target_heads)}")

    # ── 3B: Targeted Activation Patching ──
    if "3B" in experiments:
        t0 = time.time()
        result_3b = run_targeted_patching(
            model, tokenizer, value_to_tid, value_pool,
            model_name, num_keys, num_updates, trials,
            target_heads, config
        )
        result_3b["elapsed_sec"] = round(time.time() - t0, 1)
        results["experiments"]["3B"] = result_3b
        save_results(results, model_name, partial=True)

        # Pick top-5 heads for ablation (by patching delta)
        sorted_by_delta = sorted(result_3b["head_results"].items(),
                                 key=lambda x: -x[1]["mean_delta_p_last"])
        top_ablate = [{"layer": v["layer"], "head": v["head"], "label": k}
                      for k, v in sorted_by_delta[:5]]
    else:
        top_ablate = target_heads[:5]

    # ── 3C: Logit Lens Under Ablation ──
    if "3C" in experiments:
        t0 = time.time()
        result_3c = run_ablation_logit_lens(
            model, tokenizer, value_to_tid, value_pool,
            model_name, num_keys, num_updates, trials,
            top_ablate, config
        )
        result_3c["elapsed_sec"] = round(time.time() - t0, 1)
        results["experiments"]["3C"] = result_3c
        save_results(results, model_name, partial=True)

    results["end_time"] = datetime.now(timezone.utc).isoformat()

    # Final save
    save_results(results, model_name, partial=False)

    # Summary
    print(f"\n{'='*70}")
    print(f"STAGE 3 COMPLETE")
    print(f"{'='*70}")

    if "3A" in results["experiments"]:
        print(f"\n  3A Attribution Patching: top head = "
              f"{results['experiments']['3A']['top_heads'][0]['label']} "
              f"(score={results['experiments']['3A']['top_heads'][0]['attribution']:.4f})")

    if "3B" in results["experiments"]:
        sorted_patch = sorted(results["experiments"]["3B"]["head_results"].items(),
                             key=lambda x: -x[1]["mean_delta_p_last"])
        if sorted_patch:
            print(f"  3B Targeted Patching: top head = {sorted_patch[0][0]} "
                  f"(ΔP(v_last)={sorted_patch[0][1]['mean_delta_p_last']:+.4f})")

    if "3C" in results["experiments"]:
        r3c = results["experiments"]["3C"]
        print(f"  3C Ablation Logit Lens: P(v_last) {r3c['p_last_normal_final']:.4f} → "
              f"{r3c['p_last_ablated_final']:.4f} "
              f"(Δ={r3c['p_last_ablated_final'] - r3c['p_last_normal_final']:+.4f})")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="V3 Stage 3: Causal mechanistic experiments")
    p.add_argument("--model", required=True)
    p.add_argument("--stage2", required=True, help="Path to Stage 2 results JSON")
    p.add_argument("--point", default=None, help='Operating point "keys,updates" (default: auto from stage2)')
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--top-k", type=int, default=20, help="Top K heads from attribution patching")
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--experiments", nargs="+", default=["3A", "3B", "3C"],
                   help="Which experiments to run (default: 3A 3B 3C)")
    return p.parse_args()


def main():
    args = parse_args()
    point = None
    if args.point:
        k, u = args.point.split(",")
        point = (int(k), int(u))

    config = {
        "model": args.model,
        "stage2_path": args.stage2,
        "point": point,
        "trials": args.trials,
        "top_k": args.top_k,
        "gpu": args.gpu,
        "experiments": args.experiments,
    }
    run_stage3(config)


if __name__ == "__main__":
    main()
