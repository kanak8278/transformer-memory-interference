"""
Plot training dynamics: RI−PI gap across SmolLM3 and SmolLM2 checkpoints.
Two-panel figure:
  Top:    SmolLM3-3B  — stage1 / stage2 / stage3 / SFT / alignment / final
  Bottom: SmolLM2-1.7B — pretraining only
X-axis: sequential index so every checkpoint gets equal width.
Output: v3/results_vllm/plots/training_dynamics_gap.png
"""

import json
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

BASE = Path(__file__).parent
OUT  = BASE / "results_vllm" / "plots" / "training_dynamics_gap.png"

STAGE_COLORS = {
    "stage1":       "#DDEEFF",
    "stage2":       "#D4EADC",
    "stage3":       "#FFF3CD",
    "sft":          "#FFE5CC",
    "mid_training": "#FFD5D5",
    "apo":          "#F0E6FF",
    "lc_expert":    "#E6F0FF",
    "final":        "#EEEEEE",
}
STAGE_DISPLAY = {
    "stage1":       "Pretraining\nStage 1",
    "stage2":       "Pretraining\nStage 2",
    "stage3":       "Pretraining\nStage 3",
    "sft":          "SFT",
    "mid_training": "Mid-\nTrain",
    "apo":          "APO",
    "lc_expert":    "LC-\nExpert",
    "final":        "Final",
}
STAGE_ORDER = ["stage1", "stage2", "stage3", "sft", "mid_training", "apo", "lc_expert", "final"]

def load_gap(filepath):
    with open(filepath) as f:
        d = json.load(f)
    cells = d.get("cells", {})
    if not cells:
        return None
    ri = sum(cells[k]["stats"]["RI"]["accuracy"] for k in cells) / len(cells)
    pi = sum(cells[k]["stats"]["PI"]["accuracy"] for k in cells) / len(cells)
    return ri - pi

# ── SmolLM3: collect & order ──────────────────────────────────────────────────

s3_dir = BASE / "results_vllm" / "training_dynamics_smollm3"
IT_DEFS = [
    ("it-SFT_completion",           "sft",          "SFT"),
    ("it-mid-training_completion",  "mid_training", "Mid-train"),
    ("it-soup-APO_completion",       "apo",          "APO"),
    ("it-LC-expert_completion",      "lc_expert",    "LC-Expert"),
    ("final_completion",             "final",        "Final"),
]

s3_raw = []  # (sort_key, gap, stage_key, tick_label)

for fpath in sorted(glob.glob(str(s3_dir / "stage*.json"))):
    fname = Path(fpath).stem
    if "_trials" in fname:
        continue
    gap = load_gap(fpath)
    if gap is None:
        continue
    parts  = fname.split("-")
    stage  = parts[0]
    step   = int(parts[-1])
    # sort_key: stage prefix 0/1/2 + step
    prefix = {"stage1": 0, "stage2": 1, "stage3": 2}[stage]
    s3_raw.append((prefix * 10_000_000 + step, gap, stage, f"{step//1000}K"))

for it_fname, stage_key, label in IT_DEFS:
    fpath = s3_dir / f"{it_fname}.json"
    if not fpath.exists():
        continue
    gap = load_gap(str(fpath))
    if gap is None:
        continue
    prefix_map = {"sft": 3, "mid_training": 4, "apo": 5, "lc_expert": 6, "final": 7}
    s3_raw.append((prefix_map[stage_key] * 10_000_000, gap, stage_key, label))

s3_raw.sort(key=lambda x: x[0])

# Sequential x (0, 1, 2, ...)
s3_x     = list(range(len(s3_raw)))
s3_gaps  = [r[1] for r in s3_raw]
s3_stage = [r[2] for r in s3_raw]
s3_ticks = [r[3] for r in s3_raw]

# Stage x-span boundaries (in sequential units)
stage_spans = {}
prev = None
for i, sk in enumerate(s3_stage):
    if sk != prev:
        if prev is not None:
            stage_spans[prev] = (stage_spans[prev][0], i - 0.5)
        stage_spans[sk] = (i - 0.5, None)
        prev = sk
stage_spans[prev] = (stage_spans[prev][0], len(s3_x) - 0.5)

# ── Draw ──────────────────────────────────────────────────────────────────────

YMIN, YMAX = -0.15, 0.75

fig, ax_top = plt.subplots(figsize=(16, 5.5))
ax_bot = None
fig.patch.set_facecolor("white")

for ax in (ax_top,):
    ax.axhline(0, color="#555", linewidth=0.8, linestyle=":", alpha=0.5, zorder=1)
    ax.set_ylim(YMIN, YMAX)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:+.0%}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6, zorder=0)
    ax.set_ylabel("RI − PI Gap", fontsize=11, labelpad=6)

# ─── Top panel: SmolLM3 ──────────────────────────────────────────────────────

for sk in STAGE_ORDER:
    if sk not in stage_spans:
        continue
    x0, x1 = stage_spans[sk]
    ax_top.axvspan(x0, x1, color=STAGE_COLORS[sk], alpha=1.0, zorder=0)
    mid = (x0 + x1) / 2
    ax_top.text(mid, YMAX - 0.02, STAGE_DISPLAY[sk],
                ha="center", va="top", fontsize=8.5,
                color="#333", fontweight="bold", linespacing=1.4, zorder=5)

# Boundaries
prev_sk = None
for sk in STAGE_ORDER:
    if sk not in stage_spans:
        continue
    if prev_sk is not None:
        ax_top.axvline(stage_spans[sk][0], color="#999",
                       linewidth=0.9, linestyle="--", alpha=0.5, zorder=2)
    prev_sk = sk

ax_top.plot(s3_x, s3_gaps,
            color="#1565C0", linewidth=2.2,
            marker="o", markersize=5.5, markerfacecolor="white",
            markeredgecolor="#1565C0", markeredgewidth=1.8, zorder=3)

# Annotate first checkpoint
ax_top.annotate(
    f"Step 40K  →  gap = {s3_gaps[0]:+.0%}\n(bias present from first checkpoint)",
    xy=(s3_x[0], s3_gaps[0]),
    xytext=(s3_x[0] + 4, s3_gaps[0] + 0.14),
    fontsize=8, color="#1565C0",
    arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.2),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1565C0", alpha=0.9),
    zorder=6
)

# x-ticks: sparse within stage1 (every 4th), all others
tick_x, tick_lbl = [], []
stage_counts = {}
for i, sk in enumerate(s3_stage):
    stage_counts.setdefault(sk, []).append(i)

for sk in STAGE_ORDER:
    idxs = stage_counts.get(sk, [])
    if not idxs:
        continue
    if len(idxs) > 6:
        # show first, every 4th, last
        selected = [idxs[0]] + idxs[4::4] + ([idxs[-1]] if idxs[-1] not in idxs[4::4] else [])
    else:
        selected = idxs
    for i in selected:
        tick_x.append(i)
        tick_lbl.append(s3_ticks[i])

ax_top.set_xticks(tick_x)
ax_top.set_xticklabels(tick_lbl, rotation=40, ha="right", fontsize=7.5)
ax_top.set_xlim(-0.7, len(s3_x) - 0.3)
ax_top.set_title(
    "Primacy Bias (RI − PI gap) Across All Training Stages — SmolLM3-3B  (37 checkpoints)",
    fontsize=13, fontweight="bold", pad=10
)

# Stage legend
stage_patches = [
    mpatches.Patch(color=STAGE_COLORS[sk],
                   label=STAGE_DISPLAY[sk].replace("\n", " "))
    for sk in STAGE_ORDER if sk in stage_spans
]
ax_top.legend(handles=stage_patches, loc="lower right",
              fontsize=8, framealpha=0.92, edgecolor="#CCC",
              ncol=4, borderpad=0.7, handlelength=1.5)

plt.tight_layout(pad=1.5)
plt.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUT}")
print(f"SmolLM3: {len(s3_x)} checkpoints")
