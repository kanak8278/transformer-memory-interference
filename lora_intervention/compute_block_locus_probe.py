"""
§6 Block-locus probe: does Block-format input activate the L30-L33 promoter
heads that LoRA activates?

Three conditions on Qwen2.5-3B-Instruct, K=2/N=30, expA_CVQ:
  A: Base + Plain   — v3/results_vllm/attention_routing/Qwen2.5-3B-Instruct__normal.json
  B: Base + Block   — v3/results_vllm/attention_routing/Qwen2.5-3B-Instruct-baseline-block__block.json
  C: LoRA + Plain   — v3/results_vllm/attention_routing/Qwen2.5-3B-Instruct-LoRA__normal.json

For the 15 LoRA-promoter heads (identified in compute_cis.py from C - A):
  L32H3, L31H12, L32H7, L31H15, L30H11, L31H8, L32H10, L32H0,
  L33H8, L32H14, L32H9, L32H12, L33H11, L31H1, L33H14

Report:
  - p_A, p_B, p_C with bootstrap 95% CIs
  - Δ_BA = B - A (block lifts head?)
  - Δ_CA = C - A (LoRA lifts head?)
  - Δ_BC = B - C (does block reach LoRA's level?)

Verdict logic:
  - Common locus supported: Δ_BC near zero (B matches C) AND Δ_BA > 0.3
  - Distinct mechanisms:    Δ_BA near zero (B leaves head untouched)
  - Partial overlap:        anywhere in between
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
N_BOOT = 2000
SEED = 17

LORA_PROMOTER_HEADS = [
    (32, 3), (31, 12), (32, 7), (31, 15), (30, 11),
    (31, 8), (32, 10), (32, 0), (33, 8), (32, 14),
    (32, 9), (32, 12), (33, 11), (31, 1), (33, 14),
]


def per_trial_p_last_per_head(rec, condition="expA_CVQ"):
    out = []
    for t in rec["trials"]:
        a = np.asarray(t[condition]["attn_from_gen"])
        a_sum = a.sum(axis=-1)
        s = a_sum.sum(axis=-1, keepdims=True) + 1e-12
        norm = a_sum / s
        out.append(norm[..., -1])
    return np.stack(out)


def boot_ci(x, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(x)
    samples = []
    for _ in range(n_boot):
        samples.append(x[rng.integers(0, n, n)].mean())
    lo, hi = np.quantile(samples, [alpha/2, 1-alpha/2])
    return float(x.mean()), float(lo), float(hi)


def boot_diff_ci(x_a, x_b, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        a = x_a[rng.integers(0, len(x_a), len(x_a))].mean()
        b = x_b[rng.integers(0, len(x_b), len(x_b))].mean()
        samples.append(b - a)
    lo, hi = np.quantile(samples, [alpha/2, 1-alpha/2])
    return float(x_b.mean() - x_a.mean()), float(lo), float(hi)


def main():
    A_path = ROOT / "v3/results_vllm/attention_routing/Qwen2.5-3B-Instruct__normal.json"
    B_path = ROOT / "v3/results_vllm/attention_routing/Qwen2.5-3B-Instruct-baseline-block__block.json"
    C_path = ROOT / "v3/results_vllm/attention_routing/Qwen2.5-3B-Instruct-LoRA__normal.json"

    print(f"A (Base + Plain): {A_path}")
    print(f"B (Base + Block): {B_path}")
    print(f"C (LoRA + Plain): {C_path}")
    print()

    A = json.load(open(A_path))
    B = json.load(open(B_path))
    C = json.load(open(C_path))

    a_arr = per_trial_p_last_per_head(A)
    b_arr = per_trial_p_last_per_head(B)
    c_arr = per_trial_p_last_per_head(C)
    print(f"trials: A={len(a_arr)}, B={len(b_arr)}, C={len(c_arr)}")
    print()

    lines = []
    lines.append("=" * 110)
    lines.append("§6 Block-locus probe — does Block format activate the L30-L33 promoter heads?")
    lines.append("=" * 110)
    lines.append("")
    lines.append("Qwen2.5-3B-Instruct, K=2/N=30, expA_CVQ, 50 trials per condition.")
    lines.append("Conditions:")
    lines.append("  A = Base + Plain   (existing baseline)")
    lines.append("  B = Base + Block   (new — this probe)")
    lines.append("  C = LoRA + Plain   (existing post-LoRA)")
    lines.append("")
    lines.append("On each of the 15 LoRA-promoter heads identified in §5.1:")
    lines.append("")
    lines.append(f"{'Head':>6}  {'p_A [95% CI]':>22}  {'p_B [95% CI]':>22}  {'p_C [95% CI]':>22}  "
                 f"{'Δ(B-A) [CI]':>22}  {'Δ(B-C) [CI]':>22}")
    lines.append("-" * 130)

    results = []
    n_supports_locus = 0
    n_distinct = 0
    for L, H in LORA_PROMOTER_HEADS:
        a = a_arr[:, L, H]
        b = b_arr[:, L, H]
        c = c_arr[:, L, H]
        a_pt, a_lo, a_hi = boot_ci(a)
        b_pt, b_lo, b_hi = boot_ci(b)
        c_pt, c_lo, c_hi = boot_ci(c)
        ba_d, ba_lo, ba_hi = boot_diff_ci(a, b)
        bc_d, bc_lo, bc_hi = boot_diff_ci(c, b)

        lines.append(
            f"  L{L:>2}H{H:<2}  "
            f"{a_pt:.3f} [{a_lo:.3f}, {a_hi:.3f}]   "
            f"{b_pt:.3f} [{b_lo:.3f}, {b_hi:.3f}]   "
            f"{c_pt:.3f} [{c_lo:.3f}, {c_hi:.3f}]   "
            f"{ba_d:+.3f} [{ba_lo:+.3f}, {ba_hi:+.3f}]   "
            f"{bc_d:+.3f} [{bc_lo:+.3f}, {bc_hi:+.3f}]"
        )
        # Verdict per head:
        # "B matches C" if 95% CI of Δ_BC contains zero AND p_B > 0.4 (meaningfully active)
        b_matches_c = (bc_lo <= 0 <= bc_hi) and (b_pt > 0.4)
        b_matches_a = (ba_lo <= 0 <= ba_hi)
        results.append({
            "layer": L, "head": H,
            "p_A": a_pt, "p_B": b_pt, "p_C": c_pt,
            "delta_BA": ba_d, "delta_BA_lo": ba_lo, "delta_BA_hi": ba_hi,
            "delta_BC": bc_d, "delta_BC_lo": bc_lo, "delta_BC_hi": bc_hi,
            "b_matches_c": b_matches_c,
            "b_matches_a": b_matches_a,
        })
        if b_matches_c:
            n_supports_locus += 1
        if b_matches_a:
            n_distinct += 1

    lines.append("")
    lines.append("─" * 110)
    lines.append("VERDICT")
    lines.append("─" * 110)
    lines.append(f"Heads where Block reaches LoRA's level (B matches C, p_B > 0.4): {n_supports_locus} / 15")
    lines.append(f"Heads where Block leaves baseline unchanged (B matches A):       {n_distinct} / 15")
    lines.append("")
    if n_supports_locus >= 11:
        lines.append("→ COMMON LOCUS SUPPORTED. Block activates the same heads LoRA promotes.")
    elif n_distinct >= 11:
        lines.append("→ DISTINCT MECHANISMS. Block leaves the LoRA-target heads at baseline.")
    else:
        lines.append("→ PARTIAL OVERLAP. Block activates the LoRA-target heads partially.")
    lines.append("")
    # Headline stats (means across the 15 heads)
    lines.append(f"Pooled across the 15 heads (mean):")
    lines.append(f"  p_A (Base+Plain):  {np.mean([r['p_A'] for r in results]):.3f}")
    lines.append(f"  p_B (Base+Block):  {np.mean([r['p_B'] for r in results]):.3f}")
    lines.append(f"  p_C (LoRA+Plain):  {np.mean([r['p_C'] for r in results]):.3f}")
    lines.append(f"  Mean Δ(B-A):       {np.mean([r['delta_BA'] for r in results]):+.3f}")
    lines.append(f"  Mean Δ(B-C):       {np.mean([r['delta_BC'] for r in results]):+.3f}")

    text = "\n".join(lines)
    print(text)

    out_txt = ROOT / "lora_intervention" / "results" / "block_locus_comparison.txt"
    out_json = ROOT / "lora_intervention" / "results" / "block_locus_comparison.json"
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(text)
    with open(out_json, "w") as f:
        json.dump({
            "conditions": {
                "A": "Base + Plain (Qwen2.5-3B-Instruct__normal.json)",
                "B": "Base + Block (Qwen2.5-3B-Instruct-baseline-block__block.json)",
                "C": "LoRA + Plain (Qwen2.5-3B-Instruct-LoRA__normal.json)",
            },
            "lora_promoter_heads": [{"layer": L, "head": H} for L, H in LORA_PROMOTER_HEADS],
            "per_head": results,
            "verdict": {
                "n_supports_locus": n_supports_locus,
                "n_distinct": n_distinct,
            },
        }, f, indent=2)
    print(f"\n✓ Saved → {out_txt}")
    print(f"✓ Saved → {out_json}")


if __name__ == "__main__":
    main()
