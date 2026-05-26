"""
Compare attention routing results across conditions.

Supports both:
  - #3 cross-family replication: Gemma base vs Gemma+LoRA
  - #6 Block-locus probe: Qwen base + Plain vs Qwen base + Block vs Qwen + LoRA + Plain

For each pairwise comparison, computes:
  - Per-head Δ (P(attend to v_last round) mean across trials)
  - Top-K heads by Δ
  - Bootstrap 95% CI on each Δ
  - Aggregation per-layer (mean across heads)

Outputs a human-readable txt + JSON.
"""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
N_BOOT = 2000
SEED = 17


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True,
                   help="Path to baseline attention routing JSON")
    p.add_argument("--lora", required=True,
                   help="Path to LoRA (or alternative condition) JSON")
    p.add_argument("--condition", default="expA_CVQ",
                   help="Which trial sub-dict to compare (expA_CVQ or expA_FVQ)")
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--out", required=True, help="Output txt path")
    p.add_argument("--out-json", default=None)
    p.add_argument("--label-base", default="baseline")
    p.add_argument("--label-lora", default="lora")
    return p.parse_args()


def per_trial_p_last_per_head(rec, condition):
    """Returns array shape (n_trials, L, H) = P(attend to v_last round)."""
    out = []
    for t in rec["trials"]:
        cond_data = t.get(condition)
        if cond_data is None:
            continue
        a = np.asarray(cond_data["attn_from_gen"])  # (L, H, R, K)
        a_sum = a.sum(axis=-1)                       # (L, H, R) — sum over keys
        # Normalize per (L, H) over rounds
        s = a_sum.sum(axis=-1, keepdims=True) + 1e-12
        norm = a_sum / s
        out.append(norm[..., -1])                    # P(attend to last round) per head
    if not out:
        return np.zeros((0, 0, 0))
    return np.stack(out)


def boot_ci_paired(x_b, x_l, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    """Unpaired bootstrap CI for x_l.mean() - x_b.mean()."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        a = x_b[rng.integers(0, len(x_b), len(x_b))].mean()
        b = x_l[rng.integers(0, len(x_l), len(x_l))].mean()
        diffs.append(b - a)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(x_l.mean() - x_b.mean()), float(lo), float(hi)


def boot_ci_single(x, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(x)
    samples = []
    for _ in range(n_boot):
        samples.append(x[rng.integers(0, n, n)].mean())
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def main():
    args = parse_args()
    print(f"Loading {args.base} ...")
    with open(args.base) as f:
        B = json.load(f)
    print(f"Loading {args.lora} ...")
    with open(args.lora) as f:
        L = json.load(f)

    b_arr = per_trial_p_last_per_head(B, args.condition)
    l_arr = per_trial_p_last_per_head(L, args.condition)
    if b_arr.size == 0 or l_arr.size == 0:
        print(f"ERROR: missing condition {args.condition} in one of the files")
        return

    n_trials_b, n_layers, n_heads = b_arr.shape
    n_trials_l = l_arr.shape[0]
    print(f"  base: {n_trials_b} trials, L={n_layers}, H={n_heads}")
    print(f"  lora: {n_trials_l} trials, L={n_layers}, H={n_heads}")

    delta = l_arr.mean(axis=0) - b_arr.mean(axis=0)  # (L, H)
    flat = []
    for La in range(n_layers):
        for Ha in range(n_heads):
            flat.append((La, Ha, float(delta[La, Ha])))
    flat.sort(key=lambda x: -x[2])

    # Top-K heads with bootstrap CIs
    lines = []
    lines.append("=" * 100)
    lines.append(f"Attention routing comparison — condition={args.condition}")
    lines.append(f"  baseline file: {args.base}")
    lines.append(f"  lora file:     {args.lora}")
    lines.append(f"  baseline n trials: {n_trials_b}, lora n trials: {n_trials_l}")
    lines.append(f"  bootstrap n={N_BOOT}, seed={SEED}")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"Top-{args.top_k} promoter heads (by Δ = lora − baseline mean P(attend last round))")
    lines.append("")
    lines.append(f"{'Head':>10}  {'base [95% CI]':>22}  {'lora [95% CI]':>22}  {'Δ [95% CI]':>22}")
    lines.append("-" * 100)
    top_records = []
    for La, Ha, _ in flat[:args.top_k]:
        b_pt, b_lo, b_hi = boot_ci_single(b_arr[:, La, Ha])
        l_pt, l_lo, l_hi = boot_ci_single(l_arr[:, La, Ha])
        d_pt, d_lo, d_hi = boot_ci_paired(b_arr[:, La, Ha], l_arr[:, La, Ha])
        lines.append(
            f"  L{La:>2}H{Ha:<2}  "
            f"{b_pt:.3f} [{b_lo:.3f}, {b_hi:.3f}]   "
            f"{l_pt:.3f} [{l_lo:.3f}, {l_hi:.3f}]   "
            f"{d_pt:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}]"
        )
        top_records.append({
            "layer": La, "head": Ha,
            "base_mean": b_pt, "base_lo": b_lo, "base_hi": b_hi,
            "lora_mean": l_pt, "lora_lo": l_lo, "lora_hi": l_hi,
            "delta_mean": d_pt, "delta_lo": d_lo, "delta_hi": d_hi,
        })

    # Per-layer aggregate
    lines.append("")
    lines.append("─" * 100)
    lines.append("Per-layer mean P(attend last round) — change averaged over all heads")
    lines.append("─" * 100)
    lines.append(f"{'Layer':>5}  {'base':>7}  {'lora':>7}  {'Δ':>9}")
    layer_records = []
    for La in range(n_layers):
        b_per_layer = b_arr[:, La, :].mean()
        l_per_layer = l_arr[:, La, :].mean()
        d = l_per_layer - b_per_layer
        layer_records.append({"layer": La, "base": float(b_per_layer), "lora": float(l_per_layer), "delta": float(d)})
        if La >= n_layers - 10 or abs(d) > 0.05:
            lines.append(f"  L{La:<3}  {b_per_layer:.3f}    {l_per_layer:.3f}    {d:+.3f}")

    text = "\n".join(lines)
    print(text)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(f"\n✓ Saved → {out_path}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({
                "base_file": args.base,
                "lora_file": args.lora,
                "condition": args.condition,
                "top_heads": top_records,
                "per_layer": layer_records,
                "n_trials_base": n_trials_b,
                "n_trials_lora": n_trials_l,
            }, f, indent=2)
        print(f"✓ Saved → {args.out_json}")


if __name__ == "__main__":
    main()
