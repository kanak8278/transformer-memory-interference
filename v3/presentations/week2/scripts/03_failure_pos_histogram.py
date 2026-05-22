"""
Slide 9b — Where in the value stream do PI failures land?

Two-panel horizontal layout:
  LEFT  : low N  — cell 2_5  (K=2 keys, N=5  updates)
  RIGHT : high N — cell 2_20 (K=2 keys, N=20 updates)

Each panel overlays the failure-position distributions for the three
instruct Qwen variants (0.5B, 1.5B, 3B). At low N failures cluster on the
penultimate position (off-by-one); at high N the distribution diffuses.

Source : v3/results_vllm/analysis/error_positions_arbitrary_single.json
Outputs:
  - presentations/week2/plots/slide9b_failure_pos.png
  - presentations/week2/plots/slide9b_failure_pos.json
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
SRC  = ROOT / "results_vllm" / "analysis" / "error_positions_arbitrary_single.json"
OUT_PNG  = HERE.parents[1] / "plots" / "slide9b_failure_pos.png"
OUT_JSON = HERE.parents[1] / "plots" / "slide9b_failure_pos.json"

MODELS = [
    ("Qwen2.5-0.5B-Instruct", "Qwen 0.5B Inst", "#E53935"),
    ("Qwen2.5-1.5B-Instruct", "Qwen 1.5B Inst", "#1E88E5"),
    ("Qwen2.5-3B-Instruct",   "Qwen 3B Inst",   "#43A047"),
]

# Panel definitions. The penultimate position is computed from the actual
# data spacing: positions are i/(N-1) for i in 0..N-1, so v_{N-1} sits at
# (N-2)/(N-1). (The "last" value is at position 1.0 and would mean a correct
# answer, so it never appears in failure_positions.)
PANELS = [
    {"key": "2_5",  "title": "Low N (K=2, N=5)",   "N": 5,
     "subtitle": "Failures cluster at v_{N-1} — off-by-one on the most-recent update."},
    {"key": "2_20", "title": "High N (K=2, N=20)", "N": 20,
     "subtitle": "Failures spread across the stream — recency signal washes out."},
]

NBINS = 10
MIN_FAILURES = 10  # below this we skip a (model, panel) cell

# ── Load ─────────────────────────────────────────────────────────────────────
with open(SRC) as f:
    raw = json.load(f)


def collect(panel_key):
    """Return list of (model_label, color, positions[]) for a panel."""
    out = []
    skipped = []
    for mkey, mlabel, color in MODELS:
        rec = raw.get(mkey, {}).get("per_cell", {}).get(panel_key)
        if rec is None:
            skipped.append((mlabel, "no cell"))
            continue
        positions = rec.get("failure_positions", []) or []
        if len(positions) < MIN_FAILURES:
            skipped.append((mlabel, f"only {len(positions)} failures"))
            continue
        out.append((mlabel, color, positions))
    return out, skipped


# ── Plot ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 5.5))
fig.patch.set_facecolor("white")

gs = fig.add_gridspec(1, 2, left=0.07, right=0.985, top=0.78, bottom=0.18,
                      wspace=0.18)
axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

bin_edges = np.linspace(0.0, 1.0, NBINS + 1)
bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Track per-panel data for JSON dump
json_data = {}

panel_records = []  # for axis-sync (max y)
y_max_global = 0.0

for ax, panel in zip(axes, PANELS):
    pkey = panel["key"]
    N = panel["N"]
    penult = (N - 2) / (N - 1) if N > 1 else 0.0

    series, skipped = collect(pkey)

    panel_data = {}
    for mlabel, color, positions in series:
        counts, _ = np.histogram(positions, bins=bin_edges)
        # density: integrates to 1 (counts / (n * bin_width))
        bin_width = bin_edges[1] - bin_edges[0]
        density = counts / (len(positions) * bin_width) if len(positions) else counts
        # step line
        # extend so step closes nicely
        ax.step(np.r_[bin_edges[0], bin_centres, bin_edges[-1]],
                np.r_[density[0], density, density[-1]],
                where="mid", color=color, linewidth=2.2, label=mlabel,
                solid_joinstyle="round", solid_capstyle="round")
        ax.fill_between(bin_centres, density, step="mid",
                        color=color, alpha=0.10, linewidth=0)

        panel_data[mlabel] = {
            "n_failures": int(len(positions)),
            "positions":  [float(p) for p in positions],
            "histogram": {
                "bin_edges": [float(x) for x in bin_edges],
                "counts":    [int(c) for c in counts],
                "density":   [float(c) for c in density],
            },
            "color": color,
        }
        y_max_global = max(y_max_global, float(density.max()) if density.size else 0.0)

    for mlabel, reason in skipped:
        panel_data[mlabel] = {"skipped": True, "reason": reason}

    # vertical reference line at penultimate position
    ax.axvline(penult, color="#444", linestyle=(0, (4, 3)), linewidth=1.4, alpha=0.85)
    ax.text(penult - 0.012, 0.97,
            r"$v_{N-1}$ (off-by-one target)",
            transform=ax.get_xaxis_transform(),
            ha="right", va="top",
            fontsize=9.5, color="#444", style="italic", rotation=90)

    # cosmetics
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0\n(first)", "0.25", "0.5", "0.75", "1\n(last)"], fontsize=10)
    ax.set_xlabel("Position of wrong answer in stream  (0 = first value, 1 = last)",
                  fontsize=10.5, color="#333", labelpad=6)

    ax.set_title(panel["title"], fontsize=12.5, fontweight="bold", color="#222", pad=22)
    ax.text(0.0, 1.04, panel["subtitle"], transform=ax.transAxes,
            ha="left", va="bottom", fontsize=10, color="#888", style="italic")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#CCC")
    ax.spines["bottom"].set_color("#CCC")
    ax.tick_params(left=False, bottom=False, color="#CCC")
    ax.grid(axis="x", color="#EEE", linewidth=0.7)
    ax.set_axisbelow(True)

    panel_records.append((ax, panel_data, penult))
    json_data[pkey] = {
        "title": panel["title"],
        "subtitle": panel["subtitle"],
        "N": N,
        "penultimate": penult,
        "models": panel_data,
    }

# Sync y-axes & set ylabel
y_top = y_max_global * 1.18 if y_max_global > 0 else 1.0
for i, (ax, _, _) in enumerate(panel_records):
    ax.set_ylim(0, y_top)
    if i == 0:
        ax.set_ylabel("Density of failures (per-model normalized)",
                      fontsize=10.5, color="#333", labelpad=6)

# Top band: title + subtitle
fig.text(0.5, 0.94,
         "Where do PI failures land? — Off-by-one at low N, diffuse at high N",
         ha="center", va="bottom", fontsize=15, fontweight="bold", color="#111")
fig.text(0.5, 0.895,
         "Failure-position distribution for the 3 instruct Qwen variants",
         ha="center", va="top", fontsize=10.5, color="#555", style="italic")

# Single horizontal legend at bottom
legend_handles = [mlines.Line2D([], [], color=c, linewidth=2.4, label=lbl)
                  for _, lbl, c in MODELS]
fig.legend(handles=legend_handles, loc="lower center",
           bbox_to_anchor=(0.5, 0.02), ncol=len(MODELS), frameon=False, fontsize=10.5)

# Footer
fig.text(0.5, -0.005,
         "Source: ARBITRARY-SINGLE · 100 trials per cell",
         ha="center", va="top", fontsize=9.5, color="#888", style="italic")

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUT_PNG}")

# ── Dump JSON ────────────────────────────────────────────────────────────────
json_out = {
    "title": "Where do PI failures land? — Off-by-one at low N, diffuse at high N",
    "subtitle": "Failure-position distribution for the 3 instruct Qwen variants",
    "footer":   "Source: ARBITRARY-SINGLE · 100 trials per cell",
    "panels": [
        {"key": p["key"], "title": p["title"], "subtitle": p["subtitle"],
         "N": p["N"], "penultimate": (p["N"] - 2) / (p["N"] - 1)}
        for p in PANELS
    ],
    "models": [{"key": k, "label": l, "color": c} for k, l, c in MODELS],
    "bins": {
        "n_bins":    NBINS,
        "bin_edges": [float(x) for x in bin_edges],
    },
    "min_failures_threshold": MIN_FAILURES,
    "data": json_data,
}
with open(OUT_JSON, "w") as f:
    json.dump(json_out, f, indent=2)
print(f"Saved → {OUT_JSON}")

# ── Verification & summary ───────────────────────────────────────────────────
assert OUT_PNG.exists() and OUT_PNG.stat().st_size > 0, "PNG missing or empty"
with open(OUT_JSON) as f:
    _ = json.load(f)  # validates JSON
print(f"PNG bytes : {OUT_PNG.stat().st_size:,}")
print(f"JSON OK   : {OUT_JSON.stat().st_size:,} bytes")

print("\nFailures per (panel, model):")
for pkey, prec in json_data.items():
    print(f"  {pkey}  (penultimate={prec['penultimate']:.4f})")
    for mlabel, mrec in prec["models"].items():
        if mrec.get("skipped"):
            print(f"    - {mlabel:18s} SKIPPED ({mrec['reason']})")
        else:
            print(f"    - {mlabel:18s} n_failures={mrec['n_failures']}")
