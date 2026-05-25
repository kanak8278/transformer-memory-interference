"""
Bootstrap 95% confidence intervals for the mechanistic deltas reported in §7.

No GPU needed — pure resampling on existing JSON outputs.

Outputs: lora_intervention/results/ci_analysis.txt
"""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
N_BOOT = 2000
SEED = 17
rng = np.random.default_rng(SEED)


def boot_ci(values, statistic=np.mean, n_boot=N_BOOT, alpha=0.05):
    """Bootstrap CI for a statistic on a 1D array."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    point = statistic(values)
    idxs = rng.integers(0, n, size=(n_boot, n))
    samples = np.fromiter(
        (statistic(values[idxs[b]]) for b in range(n_boot)),
        dtype=float, count=n_boot,
    )
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


# ─── Probing ─────────────────────────────────────────────────────────────────
# probe_results stores per-layer cv_scores arrays from sklearn 5-fold CV.
# We bootstrap those folds. Returns CI on the mean fold accuracy.
def probing_cis():
    """Parametric CIs from cached mean ± std (sklearn 5-fold CV; raw fold scores
    not preserved). 95% CI ≈ mean ± 1.96 × std / sqrt(n_folds=5).
    Delta CI uses pooled-variance approximation."""
    print("─── Probing classifier (200 trials, K=2/N=5) ────────────────────────────")
    base = json.load(open(ROOT / "v3/results_vllm/probing/probing_Qwen2.5-3B-Instruct_2k_5u.json"))
    lora = json.load(open(ROOT / "v3/results_vllm/probing/probing_Qwen2.5-3B-Instruct-LoRA_2k_5u.json"))
    N_FOLDS = 5
    z = 1.96

    def acc_std(rec, probe_name, layer):
        probe = rec["probe_results"].get(probe_name, {})
        ld = probe.get(str(layer), probe.get(layer, {}))
        return ld.get("accuracy"), ld.get("std")

    layers = [27, 30, 31, 32, 33, 34, 35]
    probes = [
        ("condition_probe", "Cond discr (RI vs PI)"),
        ("RI_correct_probe", "RI correctness"),
        ("PI_correct_probe", "PI correctness"),
    ]
    lines = []
    lines.append(f"{'Probe':>22}  {'Layer':>5}  {'base [95% CI]':>22}  {'lora [95% CI]':>22}  {'Δ [95% CI]':>22}")
    for probe_name, label in probes:
        for L in layers:
            b_pt, b_std = acc_std(base, probe_name, L)
            l_pt, l_std = acc_std(lora, probe_name, L)
            if b_pt is None or l_pt is None:
                continue
            se_b = b_std / (N_FOLDS ** 0.5)
            se_l = l_std / (N_FOLDS ** 0.5)
            b_lo, b_hi = b_pt - z * se_b, b_pt + z * se_b
            l_lo, l_hi = l_pt - z * se_l, l_pt + z * se_l
            d_pt = l_pt - b_pt
            se_d = (se_b ** 2 + se_l ** 2) ** 0.5
            d_lo, d_hi = d_pt - z * se_d, d_pt + z * se_d
            lines.append(
                f"{label:>22}  {f'L{L}':>5}  "
                f"{b_pt:.3f} [{b_lo:.3f}, {b_hi:.3f}]   "
                f"{l_pt:.3f} [{l_lo:.3f}, {l_hi:.3f}]   "
                f"{d_pt:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}]"
            )
    return "\n".join(lines)


# ─── Logit lens ──────────────────────────────────────────────────────────────
# Per-trial value_probs_by_layer[vi][L]. Compute P(v_last) at L35 per trial,
# bootstrap across trials.
def logit_lens_cis():
    print("─── Logit lens (3 cells × 100 trials) ───────────────────────────────────")
    B = json.load(open(ROOT / "v3/results_vllm/logit_lens/Qwen2.5-3B-Instruct/stage2_logit_lens_20260409_054632.json"))
    L = json.load(open(ROOT / "v3/scripts/experiments/results/Qwen2.5-3B-Instruct-LoRA/stage2_logit_lens_20260524_194921.json"))

    def per_trial(data, condition, K, N, layer_idx, target_idx):
        rs = [a for a in data["analyses"]
              if a["condition"] == condition and a["num_keys"] == K and a["num_updates"] == N]
        return np.asarray([r["value_probs_by_layer"][target_idx][layer_idx] for r in rs])

    lines = []
    lines.append(f"{'Cell':>10}  {'cond':>4}  {'target':>7}  {'layer':>5}  {'base [95% CI]':>22}  {'lora [95% CI]':>22}  {'Δ [95% CI]':>22}")
    for K, N in [(2, 5), (2, 10), (2, 50)]:
        for condition, label, target_idx_fn in [("PI", "v_last", lambda N: N - 1), ("RI", "v_first", lambda N: 0)]:
            tgt = target_idx_fn(N)
            for layer_idx in [32, 33, 34, 35]:
                b = per_trial(B, condition, K, N, layer_idx, tgt)
                l = per_trial(L, condition, K, N, layer_idx, tgt)
                if len(b) == 0 or len(l) == 0:
                    continue
                b_pt, b_lo, b_hi = boot_ci(b)
                l_pt, l_lo, l_hi = boot_ci(l)
                # Unpaired delta bootstrap
                samples = []
                for _ in range(N_BOOT):
                    bs = b[rng.integers(0, len(b), len(b))].mean()
                    ls = l[rng.integers(0, len(l), len(l))].mean()
                    samples.append(ls - bs)
                d_pt = l_pt - b_pt
                d_lo, d_hi = np.quantile(samples, [0.025, 0.975])
                lines.append(
                    f"K={K},N={N:<3}  {condition:>4}  {label:>7}  L{layer_idx:<4}  "
                    f"{b_pt:.3f} [{b_lo:.3f}, {b_hi:.3f}]   "
                    f"{l_pt:.3f} [{l_lo:.3f}, {l_hi:.3f}]   "
                    f"{d_pt:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}]"
                )
    return "\n".join(lines)


# ─── Attention routing ───────────────────────────────────────────────────────
# Per-trial attn_from_gen shape (n_layers, n_heads, n_rounds, K). Compute
# P(attend to v_last round) per trial per head, bootstrap across trials.
def attention_routing_cis():
    print("─── Attention routing (K=2/N=30 CVQ, 50 trials) ─────────────────────────")
    B = json.load(open(ROOT / "v3/results_vllm/attention_routing/Qwen2.5-3B-Instruct__normal.json"))
    L = json.load(open(ROOT / "v3/results_vllm/attention_routing/Qwen2.5-3B-Instruct-LoRA__normal.json"))

    def per_trial_p_last(data, condition):
        out = []
        for t in data["trials"]:
            if condition not in t:
                continue
            attn = np.asarray(t[condition]["attn_from_gen"])  # (L, H, R, K)
            attn = attn.sum(axis=-1)  # collapse keys
            s = attn.sum(axis=-1, keepdims=True) + 1e-12
            attn_n = attn / s
            out.append(attn_n[..., -1])  # P(attend to last round) per (L,H)
        return np.stack(out)  # (n_trials, L, H)

    b_arr = per_trial_p_last(B, "expA_CVQ")
    l_arr = per_trial_p_last(L, "expA_CVQ")

    # Top heads by Δ (mean across trials)
    delta = l_arr.mean(axis=0) - b_arr.mean(axis=0)  # (L, H)
    n_layers, n_heads = delta.shape
    flat = [(L_idx, H_idx, delta[L_idx, H_idx]) for L_idx in range(n_layers) for H_idx in range(n_heads)]
    flat.sort(key=lambda x: -x[2])

    lines = []
    lines.append(f"{'Head':>8}  {'base [95% CI]':>22}  {'lora [95% CI]':>22}  {'Δ [95% CI]':>22}")
    for L_idx, H_idx, _ in flat[:15]:
        b = b_arr[:, L_idx, H_idx]
        l = l_arr[:, L_idx, H_idx]
        b_pt, b_lo, b_hi = boot_ci(b)
        l_pt, l_lo, l_hi = boot_ci(l)
        samples = []
        for _ in range(N_BOOT):
            bs = b[rng.integers(0, len(b), len(b))].mean()
            ls = l[rng.integers(0, len(l), len(l))].mean()
            samples.append(ls - bs)
        d_lo, d_hi = np.quantile(samples, [0.025, 0.975])
        d_pt = l_pt - b_pt
        lines.append(
            f"L{L_idx:>2}H{H_idx:<2}  "
            f"{b_pt:.3f} [{b_lo:.3f}, {b_hi:.3f}]   "
            f"{l_pt:.3f} [{l_lo:.3f}, {l_hi:.3f}]   "
            f"{d_pt:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}]"
        )
    return "\n".join(lines)


# ─── Stage 3 3C ──────────────────────────────────────────────────────────────
# Per-trial trajectories are NOT saved (only avg_normal_trajectory). So we
# bootstrap the per-trial deltas at the available cells. The stage3 file only
# saves p_last_normal_final and p_last_ablated_final scalars, so no resample
# is possible from these files. Report the point estimate only.
def stage3_summary():
    print("─── Stage 3 3C (50 trials, K=2/N=5) ──────────────────────────────────────")
    B = json.load(open(ROOT / "v3/results_vllm/causal/Qwen2.5-3B-Instruct/stage3_causal_20260409_055011.json"))
    L = json.load(open(ROOT / "v3/scripts/experiments/results/Qwen2.5-3B-Instruct-LoRA-clean5/stage3_causal_20260525_071300.json"))

    bn = np.asarray(B["experiments"]["3C"]["avg_normal_trajectory"])
    ba = np.asarray(B["experiments"]["3C"]["avg_ablated_trajectory"])
    ln = np.asarray(L["experiments"]["3C"]["avg_normal_trajectory"])
    la = np.asarray(L["experiments"]["3C"]["avg_ablated_trajectory"])

    lines = []
    lines.append("Stage 3C trajectories are PER-TRIAL-AVERAGED in the saved JSON; per-trial")
    lines.append("arrays are not preserved, so bootstrap CIs are not derivable from these")
    lines.append("files. Point estimates of (ablated − normal) at v_last × layer:")
    lines.append("")
    lines.append(f"{'Layer':>6}  {'BASE Δ':>10}  {'LoRA Δ':>10}")
    for L_idx in [31, 32, 33, 34, 35]:
        bd = ba[-1, L_idx] - bn[-1, L_idx]
        ld = la[-1, L_idx] - ln[-1, L_idx]
        lines.append(f"L{L_idx:<4}    {bd:+8.4f}      {ld:+8.4f}")
    lines.append("")
    lines.append("(For paired CIs, re-run stage3 with per-trial export enabled; non-trivial")
    lines.append("change to v3/scripts/experiments/stage3_causal.py.)")
    return "\n".join(lines)


def main():
    out_path = ROOT / "lora_intervention" / "results" / "ci_analysis.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sections = [
        ("PROBING CLASSIFIER", probing_cis()),
        ("LOGIT LENS", logit_lens_cis()),
        ("ATTENTION ROUTING", attention_routing_cis()),
        ("STAGE 3 3C", stage3_summary()),
    ]

    out_lines = [
        "=" * 100,
        "Bootstrap 95% CIs for §7 mechanistic deltas",
        f"n_boot = {N_BOOT}, seed = {SEED}",
        "=" * 100,
        "",
    ]
    for title, body in sections:
        out_lines.extend(["=" * 100, title, "=" * 100, body, ""])

    out_path.write_text("\n".join(out_lines))
    print(f"\n✓ Saved: {out_path}")


if __name__ == "__main__":
    main()
