"""
Bootstrap 95% CIs for §7 Gemma mechanistic deltas.

Pure resampling on JSON outputs of run_probing_lora_hf.py and
run_logit_lens_lora_hf.py. No GPU needed.

Output: lora_intervention/results/ci_analysis_gemma.txt
"""
from pathlib import Path
import glob
import json
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
N_BOOT = 2000
SEED = 17
rng = np.random.default_rng(SEED)


def boot_ci(values, statistic=np.mean, n_boot=N_BOOT, alpha=0.05):
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
def probing_cis():
    """Parametric CIs from sklearn 5-fold CV mean ± std.
    CI = mean ± 1.96 × std / sqrt(5).
    Δ CI uses pooled-variance approximation."""
    print("─── Gemma probing (K=2/N=5, 200 trials, HF-direct) ──────────────────────")
    base = json.load(open(ROOT / "v3/results_vllm/probing/probing_gemma-3-4b-it-baseline-hf_2k_5u.json"))
    lora = json.load(open(ROOT / "v3/results_vllm/probing/probing_gemma-3-4b-it-LoRA_2k_5u.json"))
    N_FOLDS = 5
    z = 1.96

    def acc_std(rec, probe_name, layer):
        probe = rec["probe_results"].get(probe_name, {})
        ld = probe.get(str(layer), probe.get(layer, {}))
        return ld.get("accuracy"), ld.get("std")

    # Gemma has 34 layers (0-indexed 0..33); focus on late layers
    n_layers = base.get("n_layers", 34)
    layers = list(range(max(0, n_layers - 8), n_layers))
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
            if b_pt is None or l_pt is None or b_std is None or l_std is None:
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
def _latest(glob_pattern):
    matches = sorted(glob.glob(str(glob_pattern)))
    if not matches:
        return None
    return matches[-1]


def logit_lens_cis():
    """Per-trial P(v_first)/P(v_last) per layer from HF logit lens runs.
    Saved at per_trial_p_first_per_layer and per_trial_p_last_per_layer (lists
    of [n_trials × n_layers] arrays).
    """
    print("─── Gemma logit lens (3 cells × 100 trials, HF-direct, V2 = pre-norm-fold) ──")
    # Use V2 files (without intermediate-layer norm fold-in). Intermediate-layer
    # P(v_*) is zero under this lens because HF's hidden_states are pre-final-norm
    # for L < n_layers - 1. Only L33 (= n_layers - 1) carries useful signal under
    # this configuration, and it's the headline metric anyway.
    base_path = ROOT / "v3/results_vllm/logit_lens/gemma-3-4b-it-baseline-hf/stage2_logit_lens_20260525_195739.json"
    lora_path = ROOT / "v3/results_vllm/logit_lens/gemma-3-4b-it-LoRA/stage2_logit_lens_20260525_200056.json"
    print(f"  baseline: {base_path}")
    print(f"  lora:     {lora_path}")
    B = json.load(open(base_path))
    L = json.load(open(lora_path))

    def per_trial(rec, cell_key, condition, layer_idx, target):
        """target in {'first', 'last'} → returns 1D array of length n_trials."""
        cell = rec["cells"][cell_key][condition]
        key = "per_trial_p_first_per_layer" if target == "first" else "per_trial_p_last_per_layer"
        if key not in cell:
            return np.array([])  # older file without per-trial
        arr = np.asarray(cell[key])  # [n_trials, n_layers]
        if arr.size == 0:
            return np.array([])
        return arr[:, layer_idx]

    lines = []
    lines.append(f"{'Cell':>10}  {'cond':>4}  {'target':>7}  {'layer':>5}  {'base [95% CI]':>22}  {'lora [95% CI]':>22}  {'Δ [95% CI]':>22}")

    # Only L33 (final layer) carries useful signal under the pre-norm-fold lens
    # used in V2 (see note above). Reporting only L33.
    for cell_key in ["2_5", "2_10", "2_50"]:
        for condition, target_label in [("PI", "last"), ("RI", "first")]:
            for layer_idx in [33]:
                b = per_trial(B, cell_key, condition, layer_idx, target_label)
                l = per_trial(L, cell_key, condition, layer_idx, target_label)
                if len(b) == 0 or len(l) == 0:
                    continue
                b_pt, b_lo, b_hi = boot_ci(b)
                l_pt, l_lo, l_hi = boot_ci(l)
                samples = []
                for _ in range(N_BOOT):
                    bs = b[rng.integers(0, len(b), len(b))].mean()
                    ls = l[rng.integers(0, len(l), len(l))].mean()
                    samples.append(ls - bs)
                d_pt = l_pt - b_pt
                d_lo, d_hi = np.quantile(samples, [0.025, 0.975])
                k, n = cell_key.split("_")
                v_label = "v_last" if target_label == "last" else "v_first"
                lines.append(
                    f"K={k},N={n:<3}  {condition:>4}  {v_label:>7}  L{layer_idx:<4}  "
                    f"{b_pt:.3f} [{b_lo:.3f}, {b_hi:.3f}]   "
                    f"{l_pt:.3f} [{l_lo:.3f}, {l_hi:.3f}]   "
                    f"{d_pt:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}]"
                )
    return "\n".join(lines)


def main():
    out_path = ROOT / "lora_intervention" / "results" / "ci_analysis_gemma.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sections = [
        ("PROBING CLASSIFIER (Gemma-3-4b-it)", probing_cis()),
        ("LOGIT LENS (Gemma-3-4b-it)", logit_lens_cis()),
    ]

    out_lines = [
        "=" * 100,
        "Bootstrap 95% CIs — §7 Gemma mechanistic deltas",
        f"n_boot = {N_BOOT}, seed = {SEED}",
        "Probing CIs are parametric (sklearn 5-fold mean ± 1.96 × std / sqrt(5)).",
        "Logit lens CIs are nonparametric bootstrap over per-trial means.",
        "=" * 100,
        "",
    ]
    for title, body in sections:
        out_lines.extend(["=" * 100, title, "=" * 100, body, ""])

    out_path.write_text("\n".join(out_lines))
    print(f"\n✓ Saved: {out_path}")
    print("\n" + "\n".join(out_lines))


if __name__ == "__main__":
    main()
