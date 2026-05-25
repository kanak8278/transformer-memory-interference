"""Generate Figure 8 — the §7 Mechanism summary (3-panel).

Panels:
  (a) Probing & Logit lens per-layer curves (baseline vs LoRA)
  (b) Attention routing — per-head Δ P(attend to v_last round) heatmap
  (c) Causal ablation — P(v_last) trajectory under LoRA promoter ablation

Run:
    .venv/bin/python paper/scripts/generate_mechanism_figure.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

# Source files (see lora_intervention/results/*_comparison.txt for cross-refs).
PROBING_BASE = _ROOT / "v3" / "results_vllm" / "probing" / "probing_Qwen2.5-3B-Instruct_2k_5u.json"
PROBING_LORA = _ROOT / "v3" / "results_vllm" / "probing" / "probing_Qwen2.5-3B-Instruct-LoRA_2k_5u.json"
LL_BASE      = _ROOT / "v3" / "results_vllm" / "logit_lens" / "Qwen2.5-3B-Instruct" / "stage2_logit_lens_20260409_054632.json"
LL_LORA      = _ROOT / "v3" / "scripts" / "experiments" / "results" / "Qwen2.5-3B-Instruct-LoRA" / "stage2_logit_lens_20260524_194921.json"
AR_BASE      = _ROOT / "v3" / "results_vllm" / "attention_routing" / "Qwen2.5-3B-Instruct__normal.json"
AR_LORA      = _ROOT / "v3" / "results_vllm" / "attention_routing" / "Qwen2.5-3B-Instruct-LoRA__normal.json"
CAUSAL_LORA  = _ROOT / "v3" / "scripts" / "experiments" / "results" / "Qwen2.5-3B-Instruct-LoRA-promoters" / "stage3_causal_20260525_112134.json"
CAUSAL_BASE  = _ROOT / "v3" / "scripts" / "experiments" / "results" / "Qwen2.5-3B-Instruct-baseline-promoters" / "stage3_causal_20260525_122807.json"


# ── Data extraction ──────────────────────────────────────────────────────────

def load_probe_curve(path, probe_name):
    """Return dict {layer_int: accuracy} for one probe (e.g. 'PI_correct_probe')."""
    d = json.load(open(path))
    res = d["probe_results"][probe_name]
    return {int(L): res[L]["accuracy"] for L in res}


def load_logit_lens_pvlast(path, point=(2, 5)):
    """Average P(v_last) per layer over all PI trials at the given (K, N).

    value_probs_by_layer shape is [n_values, n_layers]; the last value index
    is v_last so we slice ``vp[-1, :]`` for each trial.
    """
    K, N = point
    d = json.load(open(path))
    trials = [t for t in d["analyses"]
              if t["condition"] == "PI"
              and t["num_keys"] == K
              and t["num_updates"] == N]
    if not trials:
        return None
    n_layers = d["n_layers"]
    out = {L: [] for L in range(n_layers)}
    for t in trials:
        vp = np.asarray(t["value_probs_by_layer"])  # (n_values, n_layers)
        v_last_curve = vp[-1, :]  # P(v_last) at each layer
        for L in range(n_layers):
            out[L].append(float(v_last_curve[L]))
    return {L: float(np.mean(v)) for L, v in out.items() if v}


def load_attention_routing_per_head(path):
    """Average P(attend to v_last round) per (layer, head), summed across both
    test-category columns, over all trials. Shape: [n_layers, n_heads].
    """
    d = json.load(open(path))
    trials = d["trials"]
    # attn_from_gen shape: [n_layers, n_heads, n_rounds, n_cols=2]
    # P(attend to v_last round) per head = sum across both columns at last round
    accumulated = None
    n = 0
    for t in trials:
        afg = np.asarray(t["expA_CVQ"]["attn_from_gen"])
        # afg shape: [n_layers, n_heads, n_rounds, 2]
        # Sum across the 2 columns at the LAST round → [n_layers, n_heads]
        per_head = afg[:, :, -1, :].sum(axis=-1)
        if accumulated is None:
            accumulated = per_head
        else:
            accumulated = accumulated + per_head
        n += 1
    return accumulated / max(n, 1)  # mean


def load_causal_traj(path):
    """Return per-layer LoRA-normal and LoRA-ablated P(v_last) trajectories.

    avg_normal_trajectory shape is (n_values, n_layers); the last value index
    is v_last so we slice ``[-1, :]`` to get the per-layer curve.
    """
    d = json.load(open(path))
    e3c = d["experiments"]["3C"]
    norm = np.asarray(e3c["avg_normal_trajectory"])  # (n_values, n_layers)
    abl  = np.asarray(e3c["avg_ablated_trajectory"])
    return norm[-1, :], abl[-1, :]


# ── Plot helpers ─────────────────────────────────────────────────────────────

def _to_arr(d):
    """Dict {layer: val} -> sorted (layers, values) arrays."""
    if d is None:
        return np.array([]), np.array([])
    layers = sorted(d.keys())
    return np.array(layers), np.array([d[L] for L in layers])


def panel_probe_logitlens(ax):
    pi_base = load_probe_curve(PROBING_BASE, "PI_correct_probe")
    pi_lora = load_probe_curve(PROBING_LORA, "PI_correct_probe")
    ll_base = load_logit_lens_pvlast(LL_BASE, point=(2, 5))
    ll_lora = load_logit_lens_pvlast(LL_LORA, point=(2, 5))

    xb, yb = _to_arr(pi_base); xl, yl = _to_arr(pi_lora)
    ax.plot(xb, yb, color="#9ecae1", lw=1.2, marker="o", ms=3, ls="--",
            label="Probe PI-correctness (base)")
    ax.plot(xl, yl, color="#08519c", lw=1.5, marker="o", ms=3.5,
            label="Probe PI-correctness (+LoRA)")
    xb2, yb2 = _to_arr(ll_base); xl2, yl2 = _to_arr(ll_lora)
    ax.plot(xb2, yb2, color="#fc9272", lw=1.2, marker="s", ms=3, ls="--",
            label="Logit lens $P(v_\\mathrm{last})$ (base)")
    ax.plot(xl2, yl2, color="#a50f15", lw=1.5, marker="s", ms=3.5,
            label="Logit lens $P(v_\\mathrm{last})$ (+LoRA)")
    ax.axvspan(30, 33, color="#cccccc", alpha=0.25, zorder=0)
    ax.set_xlabel("Layer $L$")
    ax.set_ylabel("Probability / accuracy")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(-1, 36)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=2, frameon=False)
    ax.set_title("(a) Probing \\& logit lens", fontsize=10)


def panel_attention_routing(ax):
    base = load_attention_routing_per_head(AR_BASE)  # [n_layers, n_heads]
    lora = load_attention_routing_per_head(AR_LORA)
    delta = lora - base
    # Diverging colormap; vmin/vmax symmetric
    vmax = float(np.max(np.abs(delta)))
    im = ax.imshow(delta, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, origin="lower")
    ax.set_xlabel("Head index")
    ax.set_ylabel("Layer $L$")
    ax.set_title("(b) Attention routing — $\\Delta P(\\text{attend } v_\\mathrm{last})$", fontsize=10)
    # Mark the L30-L33 band
    ax.axhspan(29.5, 33.5, color="none", ec="black", lw=0.8, ls="--")
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("$+$LoRA $-$ base", fontsize=8)


def panel_causal(ax):
    lora_norm, lora_abl = load_causal_traj(CAUSAL_LORA)
    base_norm, base_abl = load_causal_traj(CAUSAL_BASE)
    layers = np.arange(len(lora_norm))
    # Baseline pair (grey tones)
    ax.plot(layers, base_norm, color="#9e9e9e", lw=1.2, marker="s", ms=3,
            label="Base (normal)")
    ax.plot(layers, base_abl, color="#5f5f5f", lw=1.3, marker="s", ms=3,
            ls="--", label="Base + 8-head ablation")
    # LoRA pair (blue tones)
    ax.plot(layers, lora_norm, color="#6baed6", lw=1.3, marker="o", ms=3,
            label="$+$LoRA (normal)")
    ax.plot(layers, lora_abl, color="#08519c", lw=1.5, marker="o", ms=3.5,
            ls="--", label="$+$LoRA + 8-head ablation")
    ax.axvspan(30, 33, color="#cccccc", alpha=0.25, zorder=0)
    ax.set_xlabel("Layer $L$")
    ax.set_ylabel("$P(v_\\mathrm{last})$")
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlim(-1, len(lora_norm))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=2, frameon=False)
    ax.set_title("(c) Causal ablation", fontsize=10)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    panel_probe_logitlens(axes[0])
    panel_attention_routing(axes[1])
    panel_causal(axes[2])
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out_pdf = OUT_DIR / "fig_mechanism.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(str(out_pdf).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"  saved {out_pdf}")


if __name__ == "__main__":
    main()
