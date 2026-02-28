"""
Exp 31: Per-head causal knockout on Pythia-410M — gold standard head identification.

Adapted from 25a for completion-format prompts. For EVERY head in the model,
zero it out and measure the change in PI logit_diff.

Positive causal_effect = head was hurting PI (promoting init) → primacy head
Negative causal_effect = head was helping PI (promoting final) → retrieval head

This resolves whether Pythia's primacy circuit is in mid/late layers (L11-L22,
as suggested by attention-based ID in Exp 30) or in early layers (L0-L8, as
found by causal knockout on Qwen 1.5B and Gemma 1B).

Usage:
    cd mechanistic_probing_v2
    python experiments/31_pythia_per_head_knockout.py
    python experiments/31_pythia_per_head_knockout.py --keys 3 --updates 5 --trials 50
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
from core.dataset_configs import generate_completion_trial
from core.model_loader import verify_single_token
from core.analysis_utils import compute_logit_diff
from core.output import save_results


def get_token_ids(value, value_to_tid, tokenizer):
    """Get both space-prefixed and bare token IDs for a value."""
    tid_sp = value_to_tid.get(value, -1)
    toks = tokenizer.encode(value, add_special_tokens=False)
    tid_bare = toks[0] if toks else -1
    return list(set([t for t in [tid_sp, tid_bare] if t >= 0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-410m")
    parser.add_argument("--keys", type=int, default=3)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--n-ctx", type=int, default=2048)
    args = parser.parse_args()

    print("=" * 70)
    print("PER-HEAD CAUSAL KNOCKOUT — Pythia (completion format)")
    print(f"  keys={args.keys}, updates={args.updates}, trials={args.trials}")
    print("=" * 70)

    model, tokenizer, info = load_model(args.model, n_ctx=args.n_ctx)
    value_to_tid = verify_single_token(tokenizer)
    value_pool = list(value_to_tid.keys())

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    total_heads = n_layers * n_heads

    print(f"  {n_layers} layers × {n_heads} heads = {total_heads} heads to test")

    # ── Build all PI trials upfront (completion format) ──
    trial_data = []
    skipped = 0
    for t_idx in range(args.trials + 20):  # extra buffer for skipped trials
        if len(trial_data) >= args.trials:
            break
        seed = hash(("PI", t_idx, args.updates, args.keys, "31_ko")) % (2**31)
        trial = generate_completion_trial(
            args.keys, args.updates, "PI", seed, value_pool,
        )

        # Completion format: use prompt directly, no chat template
        tokens = model.to_tokens(trial.prompt)
        correct_tids = get_token_ids(trial.expected_answer, value_to_tid, tokenizer)
        wrong_tids = get_token_ids(trial.initial_value, value_to_tid, tokenizer)

        if not correct_tids or not wrong_tids:
            skipped += 1
            continue

        trial_data.append({
            "tokens": tokens,
            "correct_tids": correct_tids,
            "wrong_tids": wrong_tids,
        })

    print(f"  {len(trial_data)} valid trials ({skipped} skipped)")

    # ── Run baseline (no ablation) ──
    print(f"\n  Running baseline ({len(trial_data)} PI trials)...")
    t_start = time.time()

    baseline_lds = []
    baseline_correct = 0
    for td in trial_data:
        with torch.no_grad():
            logits = model(td["tokens"])
        ld = compute_logit_diff(logits[0, -1], td["correct_tids"], td["wrong_tids"])
        baseline_lds.append(ld)
        pred = logits[0, -1].argmax().item()
        baseline_correct += int(pred in td["correct_tids"])

    baseline_mean_ld = np.mean(baseline_lds)
    baseline_acc = baseline_correct / len(trial_data)
    print(f"  Baseline: PI={baseline_acc:.0%}, mean_ld={baseline_mean_ld:.2f}")

    # ── Sweep all heads ──
    causal_effect = np.zeros((n_layers, n_heads))
    knockout_acc = np.zeros((n_layers, n_heads))

    print(f"\n  Sweeping {total_heads} heads × {len(trial_data)} trials...")

    for layer in range(n_layers):
        for head in range(n_heads):
            def make_hook(h):
                def hook_fn(activation, hook):
                    activation[:, :, h, :] = 0.0
                    return activation
                return hook_fn

            hook = (f"blocks.{layer}.attn.hook_z", make_hook(head))

            ko_lds = []
            ko_correct = 0
            for i, td in enumerate(trial_data):
                with torch.no_grad():
                    logits = model.run_with_hooks(td["tokens"], fwd_hooks=[hook])
                ld = compute_logit_diff(logits[0, -1], td["correct_tids"], td["wrong_tids"])
                ko_lds.append(ld - baseline_lds[i])
                pred = logits[0, -1].argmax().item()
                ko_correct += int(pred in td["correct_tids"])

            causal_effect[layer, head] = np.mean(ko_lds)
            knockout_acc[layer, head] = ko_correct / len(trial_data)

        elapsed = time.time() - t_start
        done = (layer + 1) * n_heads
        eta = (elapsed / done) * (total_heads - done) if done > 0 else 0
        print(f"  Layer {layer}/{n_layers-1} done ({done}/{total_heads}, "
              f"{elapsed:.0f}s, ETA {eta/60:.0f}min)")

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # ── Results ──
    total_time = time.time() - t_start

    print(f"\n{'=' * 70}")
    print(f"RESULTS ({total_time/60:.1f} minutes)")
    print(f"{'=' * 70}")
    print(f"  Baseline PI: {baseline_acc:.0%}, mean_ld={baseline_mean_ld:.2f}")

    flat = [(causal_effect[l, h], l, h) for l in range(n_layers) for h in range(n_heads)]
    flat.sort(reverse=True)

    print(f"\n  Top 15 PRIMACY heads (knockout helps PI → head hurts PI):")
    print(f"  {'Rank':>4} {'Head':>8} {'Δld':>8} {'KO acc':>8} {'Baseline':>8}")
    for rank, (effect, l, h) in enumerate(flat[:15]):
        print(f"  {rank+1:>4} L{l}H{h:>2} {effect:>+8.3f} {knockout_acc[l, h]:>7.0%} {baseline_acc:>7.0%}")

    print(f"\n  Top 15 RETRIEVAL heads (knockout hurts PI → head helps PI):")
    for rank, (effect, l, h) in enumerate(flat[-15:][::-1]):
        print(f"  {rank+1:>4} L{l}H{h:>2} {effect:>+8.3f} {knockout_acc[l, h]:>7.0%} {baseline_acc:>7.0%}")

    # ── Compare to attention-based heads from Exp 30 ──
    exp30_attention_heads = [
        (11,2),(11,6),(11,14),(13,5),(14,0),(15,5),(15,8),(16,1),(16,3),
        (16,5),(16,7),(17,10),(17,11),(18,0),(18,8),(18,14),(19,4),(19,8),
        (19,15),(21,8),(22,13),
    ]
    print(f"\n  ── Exp 30 attention-identified heads vs causal ranking ──")
    print(f"  {'Head':>8} {'Causal Δld':>12} {'Causal rank':>12} {'KO acc':>8}")
    for l, h in sorted(exp30_attention_heads):
        effect = causal_effect[l, h]
        rank = sum(1 for e, _, _ in flat if e > effect) + 1
        print(f"  L{l}H{h:>2} {effect:>+12.3f} {rank:>12}/{total_heads} {knockout_acc[l, h]:>7.0%}")

    # ── Save ──
    save_data = {
        "model": args.model,
        "prompt_format": "completion_few_shot",
        "config": {"keys": args.keys, "updates": args.updates, "trials": len(trial_data)},
        "n_layers": n_layers,
        "n_heads": n_heads,
        "baseline": {"accuracy": baseline_acc, "mean_logit_diff": baseline_mean_ld},
        "causal_effect": causal_effect.tolist(),
        "knockout_accuracy": knockout_acc.tolist(),
        "top_primacy_heads": [
            {"layer": int(l), "head": int(h), "causal_effect": float(e)}
            for e, l, h in flat[:20]
        ],
        "top_retrieval_heads": [
            {"layer": int(l), "head": int(h), "causal_effect": float(e)}
            for e, l, h in flat[-20:][::-1]
        ],
        "exp30_attention_heads_causal_rank": [
            {"layer": l, "head": h, "causal_effect": float(causal_effect[l, h]),
             "causal_rank": int(sum(1 for e, _, _ in flat if e > causal_effect[l, h]) + 1)}
            for l, h in exp30_attention_heads
        ],
        "total_time_sec": round(total_time, 1),
    }
    save_results(save_data, args.model, args.keys, args.updates, "per_head_knockout")


if __name__ == "__main__":
    main()
