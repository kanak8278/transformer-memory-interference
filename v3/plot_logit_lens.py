"""
Logit lens: P(v_last) and P(v_first) across layers.
Shows "found then suppressed" pattern for PI failures vs
clean monotonic rise for RI correct trials.

Two panels:
  Left:  PI failures — avg P(v_last) per layer (4 models overlaid)
  Right: RI correct  — avg P(v_first) per layer (same 4 models)

Data: v3/results_vllm/logit_lens/{model}/*.json
Output: v3/results_vllm/plots/logit_lens.png
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent
OUT  = BASE / "results_vllm" / "plots" / "logit_lens.png"

MODELS = [
    ("Qwen2.5-0.5B-Instruct",  "Qwen 0.5B  (24L)",  "#E53935"),
    ("Qwen2.5-1.5B-Instruct",  "Qwen 1.5B  (28L)",  "#1E88E5"),
    ("Qwen2.5-3B-Instruct",    "Qwen 3B    (36L)",  "#43A047"),
    ("gemma-3-1b-it",          "Gemma 1B   (26L)",  "#8E24AA"),
    ("pythia-410m",            "Pythia 410M (24L)", "#F57F17"),
]

def load_analyses(model_key):
    model_dir = BASE / "results_vllm" / "logit_lens" / model_key
    files = sorted(model_dir.glob("*.json"))
    if not files:
        return None, None
    with open(files[0]) as f:
        d = json.load(f)
    return d.get("analyses", []), d.get("n_layers")

# ── Compute per-model avg traces ──────────────────────────────────────────────

results = []   # (label, color, n_layers, pi_trace, ri_trace)

for model_key, label, color in MODELS:
    analyses, n_layers = load_analyses(model_key)
    if not analyses or not n_layers:
        continue

    # PI failures: condition=PI, correct=False, not garbage (total_value_prob > 0.05)
    pi_fail = [
        a for a in analyses
        if a["condition"] == "PI"
        and not a["correct"]
        and a.get("total_value_prob", 0) > 0.05
        and a.get("value_probs_by_layer")
    ]

    # RI correct: condition=RI, correct=True
    ri_ok = [
        a for a in analyses
        if a["condition"] == "RI"
        and a["correct"]
        and a.get("value_probs_by_layer")
    ]

    if not pi_fail or not ri_ok:
        print(f"  {model_key}: pi_fail={len(pi_fail)}, ri_ok={len(ri_ok)} — skipping")
        continue

    # P(v_last) per layer for PI failures — last value index = -1
    pi_traces = np.array([
        a["value_probs_by_layer"][-1]   # shape: [n_layers]
        for a in pi_fail
    ])
    pi_mean = pi_traces.mean(axis=0)
    pi_se   = pi_traces.std(axis=0) / np.sqrt(len(pi_fail))

    # P(v_first) per layer for RI correct — first value index = 0
    ri_traces = np.array([
        a["value_probs_by_layer"][0]
        for a in ri_ok
    ])
    ri_mean = ri_traces.mean(axis=0)
    ri_se   = ri_traces.std(axis=0) / np.sqrt(len(ri_ok))

    results.append((label, color, n_layers, pi_mean, pi_se, ri_mean, ri_se,
                    len(pi_fail), len(ri_ok)))
    print(f"  {model_key}: pi_fail={len(pi_fail)}, ri_ok={len(ri_ok)}")
    peak_layer = int(np.argmax(pi_mean))
    print(f"    PI peak at layer {peak_layer}/{n_layers} ({peak_layer/n_layers:.0%} depth), "
          f"P(v_last)={pi_mean[peak_layer]:.3f} → final={pi_mean[-1]:.3f}")
    print(f"    RI final P(v_first)={ri_mean[-1]:.3f}")

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, (ax_pi, ax_ri) = plt.subplots(1, 2, figsize=(14, 5.5))
fig.patch.set_facecolor("white")

for label, color, n_layers, pi_mean, pi_se, ri_mean, ri_se, n_pi, n_ri in results:
    # Normalize x to relative depth 0→1
    x = np.linspace(0, 1, n_layers)

    # ── PI panel ──
    ax_pi.plot(x, pi_mean, color=color, linewidth=2.2, label=label, zorder=3)
    ax_pi.fill_between(x, pi_mean - pi_se, pi_mean + pi_se,
                       color=color, alpha=0.12, zorder=2)

    # Mark peak
    peak_idx = int(np.argmax(pi_mean))
    ax_pi.scatter([x[peak_idx]], [pi_mean[peak_idx]],
                  color=color, s=60, zorder=5, edgecolors="white", linewidths=1.2)

    # ── RI panel ──
    ax_ri.plot(x, ri_mean, color=color, linewidth=2.2, label=label, zorder=3)
    ax_ri.fill_between(x, ri_mean - ri_se, ri_mean + ri_se,
                       color=color, alpha=0.12, zorder=2)

# ── Annotations ──────────────────────────────────────────────────────────────

# PI panel: mark 85% depth
ax_pi.axvline(0.85, color="#999999", linewidth=1.0, linestyle="--", alpha=0.6)
ax_pi.text(0.851, 0.185, "~85% depth\n(peak zone)", fontsize=8,
           color="#666666", va="top")
ax_pi.annotate("suppressed\nat output",
               xy=(0.995, 0.005), xytext=(0.73, 0.14),
               fontsize=7.5, color="#888888", style="italic",
               arrowprops=dict(arrowstyle="->", color="#aaa", lw=1.0))

# ── Styling ───────────────────────────────────────────────────────────────────

for ax, title, ylabel, xlim, ylim in [
    (ax_pi,
     "PI Failures: P(v_last) across layers\n"
     "Correct value is found — then lost in final layers",
     "P(v_last)  [zoomed]", (0.6, 1.0), (-0.005, 0.20)),
    (ax_ri,
     "RI Correct: P(v_first) across layers\n"
     "First value rises monotonically — no competition",
     "P(v_first)", (0.5, 1.0), (-0.02, 1.02)),
]:
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
    ax.set_xlabel("Relative network depth  (0 = input,  1 = output)", fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#EEEEEE", linewidth=0.7)
    ax.legend(fontsize=8.5, framealpha=0.92, edgecolor="#CCC", loc="upper left")

fig.suptitle(
    "Logit Lens: The Model Finds the Last Value — Then the Final Layers Throw It Away",
    fontsize=13.5, fontweight="bold", y=1.02
)

plt.tight_layout(pad=1.8)
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
print(f"\nSaved → {OUT}")
