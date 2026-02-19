"""
Phase 2: Systematic retrieval head identification.

Runs N trials at a fixed operating point, computes per-head attention stats,
and classifies every head (24 layers × 14 heads = 336 heads) into:
  - retrieval head (attends to value positions) vs non-retrieval
  - primacy-biased (attends to initial value even in PI)
  - recency-biased (correctly shifts to final in PI)
  - condition-sensitive (changes behavior between RI and PI)

Then runs ablation: zero out each identified head and measure accuracy change.

Usage:
    cd mechanistic_probing_v2
    uv run python experiments/16_head_identification.py [--trials 30] [--updates 5]
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
from core.dataset import format_for_chat
from core.single_token_values import verify_single_token


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
    items = []
    for cat in cats:
        for ui, val in enumerate(values_per_cat[cat]):
            items.append({"category": cat, "value": val})

    rng.shuffle(items)
    for _ in range(100):
        ok = all(items[i]["category"] != items[i-1]["category"] for i in range(1, len(items)))
        if ok:
            break
        rng.shuffle(items)

    stream = "\n".join(f"{it['category']}: {it['value']}" for it in items)
    query_word = "first" if condition == "RI" else "last"
    cat_values = [it["value"] for it in items if it["category"] == test_cat]
    expected = cat_values[0] if condition == "RI" else cat_values[-1]

    prompt = (
        f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{stream}\n\n"
        f"What was the {query_word} value of {test_cat}?"
    )
    return {
        "prompt": prompt, "condition": condition, "expected": expected,
        "initial_value": cat_values[0], "final_value": cat_values[-1],
        "all_values": cat_values, "seed": seed,
    }


def collect_attention_stats(model, tokenizer, trial, value_to_tid):
    """Run one trial, return per-head attention stats at answer position."""
    init_tid = value_to_tid.get(trial["initial_value"], -1)
    final_tid = value_to_tid.get(trial["final_value"], -1)
    inter_tids = set(value_to_tid.get(v, -1) for v in trial["all_values"][1:-1]) - {-1}

    formatted = format_for_chat(trial["prompt"], tokenizer)
    tokens = model.to_tokens(formatted)
    token_ids = tokens[0].tolist()
    seq_len = tokens.shape[1]

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)

    pred_tid = logits[0, -1].argmax().item()
    pred_text = tokenizer.decode([pred_tid]).strip()
    correct = pred_text.lower() == trial["expected"].lower()

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Find value positions
    init_pos = [i for i, t in enumerate(token_ids) if t == init_tid] if init_tid >= 0 else []
    final_pos = [i for i, t in enumerate(token_ids) if t == final_tid] if final_tid >= 0 else []
    inter_pos = [i for i, t in enumerate(token_ids) if t in inter_tids]

    # Per-head stats: [n_layers, n_heads]
    attn_to_init = np.zeros((n_layers, n_heads))
    attn_to_final = np.zeros((n_layers, n_heads))
    attn_to_inter = np.zeros((n_layers, n_heads))
    attn_to_values = np.zeros((n_layers, n_heads))  # total value attention

    for layer in range(n_layers):
        pattern = cache["pattern", layer]  # [batch, n_heads, seq, seq]
        attn = pattern[0, :, -1, :]  # [n_heads, seq] — from answer position

        for head in range(n_heads):
            h_attn = attn[head]
            a_init = h_attn[init_pos].sum().item() if init_pos else 0.0
            a_final = h_attn[final_pos].sum().item() if final_pos else 0.0
            a_inter = h_attn[inter_pos].sum().item() if inter_pos else 0.0

            attn_to_init[layer, head] = a_init
            attn_to_final[layer, head] = a_final
            attn_to_inter[layer, head] = a_inter
            attn_to_values[layer, head] = a_init + a_final + a_inter

    del cache
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "correct": correct,
        "attn_to_init": attn_to_init,
        "attn_to_final": attn_to_final,
        "attn_to_inter": attn_to_inter,
        "attn_to_values": attn_to_values,
    }


def run_ablation(model, tokenizer, trial, value_to_tid, heads_to_ablate):
    """Run a trial with specific heads zeroed out. Returns (predicted, correct)."""
    formatted = format_for_chat(trial["prompt"], tokenizer)
    tokens = model.to_tokens(formatted)

    # Build hooks to zero out specific heads
    hooks = []
    for layer, head in heads_to_ablate:
        def make_hook(h):
            def hook_fn(activation, hook):
                # activation shape: [batch, seq, n_heads, d_head]
                activation[:, :, h, :] = 0.0
                return activation
            return hook_fn
        hooks.append((f"blocks.{layer}.attn.hook_z", make_hook(head)))

    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

    pred_tid = logits[0, -1].argmax().item()
    pred_text = tokenizer.decode([pred_tid]).strip()
    correct = pred_text.lower() == trial["expected"].lower()
    return pred_text, correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--keys", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--ablation-trials", type=int, default=20,
                        help="Trials per ablation condition")
    args = parser.parse_args()

    print("=" * 70)
    print("RETRIEVAL HEAD IDENTIFICATION + ABLATION")
    print(f"  keys={args.keys}, updates={args.updates}, trials={args.trials}")
    print("=" * 70)

    model, tokenizer, info = load_model(args.model, n_ctx=args.n_ctx)
    value_to_tid = verify_single_token(tokenizer)
    value_pool = list(value_to_tid.keys())
    categories = ["color", "animal", "material", "weather", "weapon"]

    n_layers = info.n_layers
    n_heads = info.n_heads

    # ══════════════════════════════════════════════════════════════════════
    # PART 1: Collect attention stats across many trials
    # ══════════════════════════════════════════════════════════════════════

    # Accumulators: [n_layers, n_heads] summed across trials
    stats = {}
    for cond in ["RI", "PI"]:
        stats[cond] = {
            "attn_to_init": np.zeros((n_layers, n_heads)),
            "attn_to_final": np.zeros((n_layers, n_heads)),
            "attn_to_values": np.zeros((n_layers, n_heads)),
            "n_correct": 0,
            "n_total": 0,
        }

    print(f"\nCollecting attention stats ({args.trials} trials × 2 conditions)...")
    t_start = time.time()

    for cond in ["RI", "PI"]:
        for t_idx in range(args.trials):
            seed = hash((cond, t_idx, args.updates, "headid")) % (2**31)
            trial = build_trial(args.keys, args.updates, cond, seed, value_pool, categories)
            result = collect_attention_stats(model, tokenizer, trial, value_to_tid)

            stats[cond]["attn_to_init"] += result["attn_to_init"]
            stats[cond]["attn_to_final"] += result["attn_to_final"]
            stats[cond]["attn_to_values"] += result["attn_to_values"]
            stats[cond]["n_correct"] += result["correct"]
            stats[cond]["n_total"] += 1

        acc = stats[cond]["n_correct"] / stats[cond]["n_total"]
        print(f"  {cond}: {stats[cond]['n_correct']}/{stats[cond]['n_total']} = {acc:.0%}")

    collect_time = time.time() - t_start
    print(f"  ({collect_time:.1f}s)")

    # ══════════════════════════════════════════════════════════════════════
    # PART 2: Classify heads
    # ══════════════════════════════════════════════════════════════════════

    n_trials = args.trials

    # Average across trials
    ri_retrieval = stats["RI"]["attn_to_values"] / n_trials
    pi_retrieval = stats["PI"]["attn_to_values"] / n_trials
    mean_retrieval = (ri_retrieval + pi_retrieval) / 2

    ri_init = stats["RI"]["attn_to_init"] / n_trials
    ri_final = stats["RI"]["attn_to_final"] / n_trials
    pi_init = stats["PI"]["attn_to_init"] / n_trials
    pi_final = stats["PI"]["attn_to_final"] / n_trials

    # Primacy score per condition: attn_init / (attn_init + attn_final)
    ri_primacy = ri_init / (ri_init + ri_final + 1e-10)
    pi_primacy = pi_init / (pi_init + pi_final + 1e-10)

    # Condition sensitivity: |primacy_RI - primacy_PI|
    cond_sensitivity = np.abs(ri_primacy - pi_primacy)

    # Classification thresholds
    RETRIEVAL_THRESHOLD = 0.10  # >10% of attention goes to value tokens
    PRIMACY_HIGH = 0.6
    PRIMACY_LOW = 0.4

    classifications = {}
    retrieval_heads = []
    primacy_biased_heads = []
    recency_biased_heads = []
    condition_sensitive_heads = []

    for layer in range(n_layers):
        for head in range(n_heads):
            retr = mean_retrieval[layer, head]
            pi_prim = pi_primacy[layer, head]
            cs = cond_sensitivity[layer, head]

            if retr >= RETRIEVAL_THRESHOLD:
                is_retrieval = True
                retrieval_heads.append((layer, head))

                if pi_prim > PRIMACY_HIGH:
                    label = "retrieval_primacy_biased"
                    primacy_biased_heads.append((layer, head))
                elif pi_prim < PRIMACY_LOW:
                    label = "retrieval_recency_responsive"
                    recency_biased_heads.append((layer, head))
                else:
                    label = "retrieval_neutral"

                if cs > 0.2:
                    condition_sensitive_heads.append((layer, head))
            else:
                label = "non_retrieval"

            classifications[(layer, head)] = {
                "label": label,
                "mean_retrieval": float(retr),
                "ri_primacy": float(ri_primacy[layer, head]),
                "pi_primacy": float(pi_prim),
                "condition_sensitivity": float(cs),
            }

    # Print classification summary
    print(f"\n{'='*70}")
    print("HEAD CLASSIFICATION")
    print(f"{'='*70}")
    print(f"  Total heads: {n_layers * n_heads}")
    print(f"  Retrieval heads (>10% value attention): {len(retrieval_heads)}")
    print(f"    Primacy-biased (PI primacy > 0.6): {len(primacy_biased_heads)}")
    print(f"    Recency-responsive (PI primacy < 0.4): {len(recency_biased_heads)}")
    print(f"    Neutral: {len(retrieval_heads) - len(primacy_biased_heads) - len(recency_biased_heads)}")
    print(f"  Condition-sensitive (|RI-PI primacy| > 0.2): {len(condition_sensitive_heads)}")

    # Top retrieval heads
    print(f"\n  Top 15 retrieval heads by mean attention to values:")
    sorted_retrieval = sorted(retrieval_heads, key=lambda h: mean_retrieval[h[0], h[1]], reverse=True)
    for l, h in sorted_retrieval[:15]:
        c = classifications[(l, h)]
        print(f"    L{l:>2}H{h:>2}: retr={c['mean_retrieval']:.4f} "
              f"RI_prim={c['ri_primacy']:.3f} PI_prim={c['pi_primacy']:.3f} "
              f"sens={c['condition_sensitivity']:.3f} [{c['label']}]")

    # Primacy-biased heads (the "smoking gun")
    if primacy_biased_heads:
        print(f"\n  PRIMACY-BIASED retrieval heads (attend to initial even in PI):")
        for l, h in sorted(primacy_biased_heads, key=lambda x: pi_primacy[x[0], x[1]], reverse=True):
            c = classifications[(l, h)]
            print(f"    L{l:>2}H{h:>2}: PI_primacy={c['pi_primacy']:.3f} retr={c['mean_retrieval']:.4f}")

    if recency_biased_heads:
        print(f"\n  RECENCY-RESPONSIVE heads (shift to final in PI):")
        for l, h in sorted(recency_biased_heads, key=lambda x: pi_primacy[x[0], x[1]]):
            c = classifications[(l, h)]
            print(f"    L{l:>2}H{h:>2}: PI_primacy={c['pi_primacy']:.3f} retr={c['mean_retrieval']:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # PART 3: Ablation study
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print("ABLATION STUDY")
    print(f"{'='*70}")

    ablation_configs = {
        "baseline": [],  # no ablation
    }

    # Ablate top-5 retrieval heads individually
    for l, h in sorted_retrieval[:5]:
        ablation_configs[f"knockout_L{l}H{h}"] = [(l, h)]

    # Ablate ALL primacy-biased heads
    if primacy_biased_heads:
        ablation_configs["knockout_all_primacy_biased"] = primacy_biased_heads

    # Ablate ALL recency-responsive heads
    if recency_biased_heads:
        ablation_configs["knockout_all_recency"] = recency_biased_heads

    # Ablate ALL retrieval heads
    ablation_configs["knockout_all_retrieval"] = retrieval_heads

    ablation_results = {}

    for config_name, heads in ablation_configs.items():
        print(f"\n  Config: {config_name} ({len(heads)} heads)")
        results_per_cond = {}

        for cond in ["RI", "PI"]:
            n_correct = 0
            for t_idx in range(args.ablation_trials):
                seed = hash((cond, t_idx, args.updates, "ablation")) % (2**31)
                trial = build_trial(args.keys, args.updates, cond, seed, value_pool, categories)

                if heads:
                    pred, correct = run_ablation(model, tokenizer, trial, value_to_tid, heads)
                else:
                    # Baseline: run normally
                    formatted = format_for_chat(trial["prompt"], tokenizer)
                    tokens = model.to_tokens(formatted)
                    with torch.no_grad():
                        logits = model(tokens)
                    pred_tid = logits[0, -1].argmax().item()
                    pred = tokenizer.decode([pred_tid]).strip()
                    correct = pred.lower() == trial["expected"].lower()

                n_correct += correct

            acc = n_correct / args.ablation_trials
            results_per_cond[cond] = {"accuracy": acc, "n": args.ablation_trials}
            print(f"    {cond}: {n_correct}/{args.ablation_trials} = {acc:.0%}")

        ablation_results[config_name] = results_per_cond

    # ══════════════════════════════════════════════════════════════════════
    # Summary & Save
    # ══════════════════════════════════════════════════════════════════════

    total_time = time.time() - t_start

    # Print ablation comparison
    print(f"\n{'='*70}")
    print("ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<35} {'RI':>6} {'PI':>6} {'RI-PI gap':>10}")
    print("-" * 60)
    for config_name, res in ablation_results.items():
        ri_acc = res["RI"]["accuracy"]
        pi_acc = res["PI"]["accuracy"]
        gap = ri_acc - pi_acc
        marker = ""
        if config_name == "baseline":
            marker = "  ← baseline"
        elif "primacy" in config_name and gap < ablation_results["baseline"]["RI"]["accuracy"] - ablation_results["baseline"]["PI"]["accuracy"]:
            marker = "  ← gap reduced!"
        print(f"{config_name:<35} {ri_acc:>5.0%} {pi_acc:>5.0%} {gap:>+9.0%}{marker}")

    # Save
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    model_short = args.model.split("/")[-1]

    save_data = {
        "model": args.model,
        "config": {"keys": args.keys, "updates": args.updates, "trials": args.trials,
                    "ablation_trials": args.ablation_trials},
        "n_layers": n_layers, "n_heads": n_heads,
        "behavioral_accuracy": {
            "RI": stats["RI"]["n_correct"] / stats["RI"]["n_total"],
            "PI": stats["PI"]["n_correct"] / stats["PI"]["n_total"],
        },
        "head_classification_summary": {
            "total_heads": n_layers * n_heads,
            "retrieval_heads": len(retrieval_heads),
            "primacy_biased": len(primacy_biased_heads),
            "recency_responsive": len(recency_biased_heads),
            "condition_sensitive": len(condition_sensitive_heads),
        },
        "retrieval_heads": [{"layer": l, "head": h, **classifications[(l, h)]}
                            for l, h in sorted_retrieval],
        "primacy_biased_heads": [{"layer": l, "head": h} for l, h in primacy_biased_heads],
        "recency_responsive_heads": [{"layer": l, "head": h} for l, h in recency_biased_heads],
        "mean_retrieval_score": mean_retrieval.tolist(),
        "ri_primacy_score": ri_primacy.tolist(),
        "pi_primacy_score": pi_primacy.tolist(),
        "condition_sensitivity": cond_sensitivity.tolist(),
        "ablation_results": ablation_results,
        "total_time_sec": round(total_time, 1),
    }

    out_path = results_dir / f"head_identification_{model_short}.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"Total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
