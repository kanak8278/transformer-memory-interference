"""
Per-trial attention-behavior correlation.

For each trial:
  1. Compute the "FVQ-primacy boost magnitude" = mean over the 11 discriminating heads
     of (attn_FVQ[L,H,round=0] - attn_CVQ[L,H,round=0]).  Source: attention_routing.json.
  2. Get the baseline FVQ correctness for the same trial (matched by seed).
     Source: causal_ablation baseline run.
  3. Likewise for CVQ-recency boost = mean over heads of
     (attn_CVQ[L,H,round=-1] - attn_FVQ[L,H,round=-1]).

Tests: Mann-Whitney U on (boost | correct) vs (boost | incorrect).
       Reports group means/medians, effect size, and a scatter plot.

Trials are matched by seed (verified identical content in both scripts).

Usage:
    .venv/bin/python v3/per_trial_correlation.py \\
      --attn v3/results_vllm/attention_routing/Qwen2.5-3B-Instruct__normal.json \\
      --abl  v3/results_vllm/causal_ablation/ablation__Qwen2.5-3B-Instruct__K2N30.json \\
      --heads-config Qwen2.5-3B-Instruct
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Same set used in causal_ablation.py
DISCRIMINATING_HEADS = {
    "Qwen2.5-3B-Instruct": [
        (31, 15), (31, 7), (31, 8), (31, 12),
        (30, 11),
        (29, 1), (29, 4), (29, 5),
        (27, 1),
        (32, 3), (32, 7),
    ],
    "gemma-3-4b-it": [
        (23, 3), (23, 1), (23, 0), (23, 6), (23, 7),
        (17, 5), (17, 0), (17, 2),
        (29, 4), (29, 3),
    ],
}


def per_trial_boost(attn_data, heads, source: str = "gen", round_idx: int = 0):
    """For each trial, compute mean attention to round_idx across the heads,
    for FVQ and CVQ separately. Returns dict {seed -> (fvq, cvq, diff)}.

    Uses the saved `attn_from_<source>` array shape [layers, heads, rounds, 2]
    (last dim = (key_span_sum, value_pos)).
    """
    attn_key = f"attn_from_{'gen' if source == 'gen' else 'qcat'}"
    out = {}
    for t in attn_data["trials"]:
        seed = t["seed"]
        fvq_arr = np.asarray(t["expA_FVQ"][attn_key], dtype=np.float32)  # [L,H,R,2]
        cvq_arr = np.asarray(t["expA_CVQ"][attn_key], dtype=np.float32)
        # Sum key-span attn + value-pos attn for the target round, per head
        fvq_per_head = fvq_arr[:, :, round_idx, :].sum(axis=-1)  # [L,H]
        cvq_per_head = cvq_arr[:, :, round_idx, :].sum(axis=-1)  # [L,H]
        fvq_vals = np.array([fvq_per_head[L, H] for L, H in heads])
        cvq_vals = np.array([cvq_per_head[L, H] for L, H in heads])
        out[seed] = {
            "fvq_mean": float(fvq_vals.mean()),
            "cvq_mean": float(cvq_vals.mean()),
            "diff_mean": float((fvq_vals - cvq_vals).mean()),
        }
    return out


def trial_correctness(abl_data, condition: str = "baseline"):
    """Return dict {seed -> {FVQ: bool, CVQ: bool}}."""
    out = {}
    for t in abl_data["trials"]:
        out[t["seed"]] = {
            "FVQ": t[condition]["FVQ"]["correct"],
            "CVQ": t[condition]["CVQ"]["correct"],
        }
    return out


def mann_whitney_u(group_a, group_b):
    """Two-sided Mann-Whitney U (no ties handling, no scipy dependency for portability)."""
    from itertools import chain
    combined = list(chain(((v, "a") for v in group_a), ((v, "b") for v in group_b)))
    combined.sort(key=lambda x: x[0])
    # Average rank for ties
    ranks = {}
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-indexed
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_a = sum(ranks[k] for k, (_, g) in enumerate(combined) if g == "a")
    n_a, n_b = len(group_a), len(group_b)
    U = rank_a - n_a * (n_a + 1) / 2
    # z-score approx (no ties correction)
    mu = n_a * n_b / 2
    sigma = (n_a * n_b * (n_a + n_b + 1) / 12) ** 0.5
    z = (U - mu) / sigma if sigma > 0 else 0.0
    # Two-sided p approx via normal CDF
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return {"U": U, "z": z, "p_two_sided": p}


def cohen_d(group_a, group_b):
    a, b = np.array(group_a), np.array(group_b)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_var = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    if pooled_var <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / np.sqrt(pooled_var))


def run_one(attn_path, abl_path, model_name, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(attn_path) as f: attn_data = json.load(f)
    with open(abl_path) as f:  abl_data  = json.load(f)

    heads = DISCRIMINATING_HEADS[model_name]
    print(f"\n=== {model_name}, {len(heads)} discriminating heads ===")
    print(f"  attn: {attn_path}")
    print(f"  abl:  {abl_path}")
    print(f"  K={attn_data['K']}, N={attn_data['N']}, trials={len(attn_data['trials'])}")

    # FVQ-primacy boost
    primacy_boost = per_trial_boost(attn_data, heads, source="gen", round_idx=0)
    # CVQ-recency boost (we negate so positive = CVQ attends more to last round)
    recency_diff = per_trial_boost(attn_data, heads, source="gen", round_idx=-1)
    # For recency, the "boost for CVQ" = CVQ - FVQ
    recency_boost = {s: {"diff_mean": -v["diff_mean"], "fvq_mean": v["fvq_mean"], "cvq_mean": v["cvq_mean"]}
                     for s, v in recency_diff.items()}

    correct = trial_correctness(abl_data, condition="baseline")
    seeds = sorted(set(primacy_boost) & set(correct))
    assert len(seeds) == len(primacy_boost), f"seed mismatch: {len(seeds)} shared vs {len(primacy_boost)} attn"

    # Group by FVQ correctness
    fvq_correct_seeds = [s for s in seeds if correct[s]["FVQ"]]
    fvq_wrong_seeds   = [s for s in seeds if not correct[s]["FVQ"]]
    cvq_correct_seeds = [s for s in seeds if correct[s]["CVQ"]]
    cvq_wrong_seeds   = [s for s in seeds if not correct[s]["CVQ"]]

    print(f"\n  FVQ baseline correctness: {len(fvq_correct_seeds)}/{len(seeds)}")
    print(f"  CVQ baseline correctness: {len(cvq_correct_seeds)}/{len(seeds)}")

    # ── FVQ-primacy boost vs FVQ correctness ──
    print(f"\n--- HYPOTHESIS 1: FVQ-primacy boost predicts FVQ correctness ---")
    print(f"  Boost metric: mean over 11 heads of (attn_FVQ - attn_CVQ) on round 0")
    g_correct = [primacy_boost[s]["diff_mean"] for s in fvq_correct_seeds]
    g_wrong   = [primacy_boost[s]["diff_mean"] for s in fvq_wrong_seeds]
    if g_correct and g_wrong:
        print(f"  FVQ correct (n={len(g_correct)}):  mean boost = {np.mean(g_correct):.4f}  median = {np.median(g_correct):.4f}")
        print(f"  FVQ wrong   (n={len(g_wrong)}):  mean boost = {np.mean(g_wrong):.4f}  median = {np.median(g_wrong):.4f}")
        mw = mann_whitney_u(g_correct, g_wrong)
        d = cohen_d(g_correct, g_wrong)
        print(f"  Mann-Whitney U={mw['U']:.1f}  z={mw['z']:+.2f}  p={mw['p_two_sided']:.4f}")
        print(f"  Cohen's d = {d:+.3f}  (positive = correct trials have larger boost)")

    # ── CVQ-recency boost vs CVQ correctness ──
    print(f"\n--- HYPOTHESIS 2: CVQ-recency boost predicts CVQ correctness ---")
    print(f"  Boost metric: mean over 11 heads of (attn_CVQ - attn_FVQ) on last round")
    g_correct = [recency_boost[s]["diff_mean"] for s in cvq_correct_seeds]
    g_wrong   = [recency_boost[s]["diff_mean"] for s in cvq_wrong_seeds]
    if g_correct and g_wrong:
        print(f"  CVQ correct (n={len(g_correct)}):  mean boost = {np.mean(g_correct):.4f}  median = {np.median(g_correct):.4f}")
        print(f"  CVQ wrong   (n={len(g_wrong)}):  mean boost = {np.mean(g_wrong):.4f}  median = {np.median(g_wrong):.4f}")
        mw = mann_whitney_u(g_correct, g_wrong)
        d = cohen_d(g_correct, g_wrong)
        print(f"  Mann-Whitney U={mw['U']:.1f}  z={mw['z']:+.2f}  p={mw['p_two_sided']:.4f}")
        print(f"  Cohen's d = {d:+.3f}  (positive = correct trials have larger boost)")

    # ── Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # FVQ-primacy boost histograms
    ax = axes[0, 0]
    ax.hist([primacy_boost[s]["diff_mean"] for s in fvq_correct_seeds],
            bins=12, alpha=0.6, label=f"FVQ correct (n={len(fvq_correct_seeds)})")
    ax.hist([primacy_boost[s]["diff_mean"] for s in fvq_wrong_seeds],
            bins=12, alpha=0.6, label=f"FVQ wrong (n={len(fvq_wrong_seeds)})")
    ax.set_xlabel("FVQ-primacy boost (mean over 11 heads)")
    ax.set_ylabel("# trials")
    ax.set_title("H1: trial-level FVQ-primacy boost by correctness")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # CVQ-recency boost histograms
    ax = axes[0, 1]
    ax.hist([recency_boost[s]["diff_mean"] for s in cvq_correct_seeds],
            bins=12, alpha=0.6, label=f"CVQ correct (n={len(cvq_correct_seeds)})")
    ax.hist([recency_boost[s]["diff_mean"] for s in cvq_wrong_seeds],
            bins=12, alpha=0.6, label=f"CVQ wrong (n={len(cvq_wrong_seeds)})")
    ax.set_xlabel("CVQ-recency boost (mean over 11 heads)")
    ax.set_ylabel("# trials")
    ax.set_title("H2: trial-level CVQ-recency boost by correctness")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Scatter: primacy boost vs recency boost, colored by FVQ correctness
    ax = axes[1, 0]
    x_c = [primacy_boost[s]["diff_mean"] for s in fvq_correct_seeds]
    y_c = [recency_boost[s]["diff_mean"] for s in fvq_correct_seeds]
    x_w = [primacy_boost[s]["diff_mean"] for s in fvq_wrong_seeds]
    y_w = [recency_boost[s]["diff_mean"] for s in fvq_wrong_seeds]
    ax.scatter(x_c, y_c, s=20, c="C0", alpha=0.6, label=f"FVQ correct (n={len(fvq_correct_seeds)})")
    ax.scatter(x_w, y_w, s=20, c="C1", alpha=0.6, label=f"FVQ wrong (n={len(fvq_wrong_seeds)})")
    ax.set_xlabel("FVQ-primacy boost")
    ax.set_ylabel("CVQ-recency boost")
    ax.set_title("Joint distribution of boosts vs FVQ correctness")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Same scatter colored by CVQ correctness
    ax = axes[1, 1]
    x_c = [primacy_boost[s]["diff_mean"] for s in cvq_correct_seeds]
    y_c = [recency_boost[s]["diff_mean"] for s in cvq_correct_seeds]
    x_w = [primacy_boost[s]["diff_mean"] for s in cvq_wrong_seeds]
    y_w = [recency_boost[s]["diff_mean"] for s in cvq_wrong_seeds]
    ax.scatter(x_c, y_c, s=20, c="C2", alpha=0.6, label=f"CVQ correct (n={len(cvq_correct_seeds)})")
    ax.scatter(x_w, y_w, s=20, c="C3", alpha=0.6, label=f"CVQ wrong (n={len(cvq_wrong_seeds)})")
    ax.set_xlabel("FVQ-primacy boost")
    ax.set_ylabel("CVQ-recency boost")
    ax.set_title("Joint distribution of boosts vs CVQ correctness")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Per-trial attention-behavior correlation — {model_name}\n"
                 f"K={attn_data['K']}, N={attn_data['N']}, n_trials={len(seeds)}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = out_dir / f"per_trial_correlation__{model_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n  saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", required=True, help="attention_routing JSON path")
    ap.add_argument("--abl", required=True, help="causal_ablation JSON path (for baseline correctness)")
    ap.add_argument("--heads-config", required=True, choices=list(DISCRIMINATING_HEADS.keys()))
    ap.add_argument("--out-dir", default="v3/plots/attention_routing/correlation")
    args = ap.parse_args()

    run_one(args.attn, args.abl, args.heads_config, args.out_dir)


if __name__ == "__main__":
    main()
