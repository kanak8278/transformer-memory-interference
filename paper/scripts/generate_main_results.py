"""
Single source of truth for §4 (Main Results) figures and tables.

Outputs (all under paper/figures/):
  - fig2_main_results.pdf       — Figure 2, 4-panel deep dive
  - tab1_proprietary.tex        — Table 1 (LaTeX tabular body)
  - tab2_openweight.tex         — Table 2 (LaTeX tabular body)

Re-run after any data change.

Usage:
    .venv/bin/python paper/figures/generate_main_results.py
"""

import csv
import glob
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════

_ROOT = Path(__file__).resolve().parent.parent.parent
SEM_DIR = _ROOT / "v3" / "results_vllm" / "semantic_multi"
PROP_CSV = _ROOT / "experiments_cloud" / "results" / "proprietary_semantic_multi.csv"
# Script lives in paper/scripts/; figures live in paper/figures/
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"

# Deep-dive models (Figure 2)
DEEP_DIVE = [
    ("Qwen2.5-3B-Instruct", "Qwen2.5-3B-Instruct"),
    ("gemma-3-4b-it", "Gemma-3-4b-it"),
]

# Open-weight models for combined table (in display order)
OPENWEIGHT_MODELS = [
    ("Qwen2.5-3B-Instruct", "Qwen2.5-3B-Instruct"),
    ("Qwen3.5-2B",          "Qwen3.5-2B"),
    ("Qwen3.5-4B",          "Qwen3.5-4B"),
    ("Qwen3.5-9B",          "Qwen3.5-9B"),
    ("gemma-3-1b-it",       "Gemma-3-1b-it"),
    ("gemma-3-4b-it",       "Gemma-3-4b-it"),
]

PROPRIETARY_MODELS = [
    ("gpt-4.1",          "GPT-4.1"),
    ("gpt-4.1-mini",     "GPT-4.1-mini"),
    ("claude-4.5-haiku", "Claude-4.5-Haiku"),
    ("claude-sonnet",    "Claude-4.5-Sonnet"),
    ("gemini-2.5-flash", "Gemini-2.5-Flash"),
    ("gemini-2.5-pro",   "Gemini-2.5-Pro"),
]

OPEN_CELL = (5, 30)    # K=5, N=30 — all open-weight models > +0.05 here
PROP_CELL = (20, 50)   # K=20, N=50 — all proprietary models > +0.05 here

# Appendix B/C: all open-weight models at four representative cells
ALL_MODELS = [
    ("Qwen2.5-0.5B-Instruct",    "Qwen2.5-0.5B-Instruct"),
    ("Qwen2.5-1.5B-Instruct",    "Qwen2.5-1.5B-Instruct"),
    ("Qwen2.5-3B-Instruct",      "Qwen2.5-3B-Instruct"),
    ("Qwen2.5-3B",               "Qwen2.5-3B (base)"),
    ("Qwen3.5-0.8B",             "Qwen3.5-0.8B"),
    ("Qwen3.5-2B",               "Qwen3.5-2B"),
    ("Qwen3.5-4B",               "Qwen3.5-4B"),
    ("Qwen3.5-9B",               "Qwen3.5-9B"),
    ("gemma-3-270m-it",          "Gemma-3-270m-it"),
    ("gemma-3-1b-it",            "Gemma-3-1b-it"),
    ("gemma-3-4b-it",            "Gemma-3-4b-it"),
    ("TinyLlama-1.1B-Chat-v1.0", "TinyLlama-1.1B"),
    ("stablelm-2-1_6b-chat",     "StableLM-2-1.6B"),
    ("pythia-410m",              "Pythia-410M"),
]
APPENDIX_CELLS = [(2, 10), (5, 30), (10, 50), (20, 50)]

# Format intervention data
UCURVE_VLLM_DIR = _ROOT / "experiments_cloud" / "results" / "ucurve_vllm"
UCURVE_PROP_CSV = _ROOT / "experiments_cloud" / "results" / "ucurve_proprietary_results.csv"

FORMAT_MODELS_OPEN = [
    ("Qwen2.5-3B-Instruct", "Qwen2.5-3B-Instruct"),
    ("Qwen3.5-2B",          "Qwen3.5-2B"),
    ("Qwen3.5-4B",          "Qwen3.5-4B"),
    ("gemma-3-4b-it",       "Gemma-3-4b-it"),
]
FORMAT_MODELS_PROP = [
    ("gpt-4.1",          "GPT-4.1"),
    ("gpt-4.1-mini",     "GPT-4.1-mini"),
    ("claude-4.5-haiku", "Claude-4.5-Haiku"),
    ("claude-sonnet",    "Claude-4.5-Sonnet"),
    ("gemini-2.5-flash", "Gemini-2.5-Flash"),
    ("gemini-2.5-pro",   "Gemini-2.5-Pro"),
]
FORMAT_KEYS = ["flat_nolabel", "flat_verbose", "block", "landmark"]
FORMAT_LABELS = {
    "flat_nolabel": "Plain",
    "flat_verbose": "Labeled",
    "block":        "Block",
    "landmark":     "Landmark",
}
FORMAT_CELL = (10, 50)  # K=10, N=50 — hardest cell available in ucurve sweep

# Training-dynamics dirs
SMOLLM2_DIR = _ROOT / "v3" / "results_vllm" / "training_dynamics"
SMOLLM3_DIR = _ROOT / "v3" / "results_vllm" / "training_dynamics_smollm3"

# SmolLM3 ordered post-training stages (completion format throughout).
# Order from the SmolLM3 blog: midtraining -> SFT -> APO -> LC-expert -> final.
SMOLLM3_POST_STAGES = [
    "it-mid-training",
    "it-SFT",
    "it-soup-APO",
    "it-LC-expert",
    "final",
]
SMOLLM3_POST_LABELS = {
    "it-mid-training": "Mid-train",
    "it-SFT":          "SFT",
    "it-soup-APO":     "APO",
    "it-LC-expert":    "LC-Expert",
    "final":           "Final",
}

# Cell partitioning for regime-dependent plotting.
# Training-dynamics grid: K in {2,3,5,7}, N in {2,3,5,7,10,15,20,30}.
# Three exhaustive bins (cover all 32 cells with no overlap).
LOW_LOAD_CELLS  = [(k, n) for k in [2, 3] for n in [2, 3, 5, 7, 10, 15, 20, 30]]
MED_LOAD_CELLS  = [(k, n) for k in [5, 7] for n in [2, 3, 5, 7, 10, 15]]
HIGH_LOAD_CELLS = [(k, n) for k in [5, 7] for n in [20, 30]]

# SmolLM2-1.7B pretraining stage boundaries (token-based; converted to steps
# using approx 2.15M tokens/step from the published 11T-token / ~5.125M-step
# correspondence). Source: Allal et al., arXiv 2502.02737, Sec. 3.
SMOLLM2_STAGES = [
    ("Pretraining\nStage 1", 0,         2_790_000),
    ("Pretraining\nStage 2", 2_790_000, 3_720_000),
    ("Pretraining\nStage 3", 3_720_000, 4_650_000),
    ("Pretraining\nStage 4", 4_650_000, 5_125_000),
]

# SmolLM3-3B pretraining stage boundaries (in steps; source: SmolLM3 blog,
# https://huggingface.co/blog/smollm3).
SMOLLM3_STAGES = [
    ("Pretraining\nStage 1", 0,         3_440_000),
    ("Pretraining\nStage 2", 3_440_000, 4_200_000),
    ("Pretraining\nStage 3", 4_200_000, 4_720_000),
]

# Figure 2 axis scans
VARY_K_AT_N      = 10   # vary K at fixed N=10  (low-load primacy panel)
VARY_N_AT_K      = 5    # vary N at fixed K=5   (low-load primacy panel)
VARY_K_AT_N_HIGH = 50   # vary K at fixed N=50  (high-load reversal panel)
K_SCAN_VALUES = [2, 3, 5, 7, 10, 15, 20]
N_SCAN_VALUES = [5, 7, 10, 15, 20, 30, 50]


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def wilson_half_width(p, n, z=1.96):
    """Half-width of Wilson 95% CI."""
    if n == 0: return 0.0
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return spread


def load_cell(dataset, model_dir, K, N):
    """dataset is 'semantic_multi' or 'arbitrary_single'."""
    base = _ROOT / "v3" / "results_vllm" / dataset
    files = sorted(glob.glob(str(base / model_dir / "stage1_sweep_*.json")))
    if not files: return None
    with open(files[-1]) as f: d = json.load(f)
    cell = d["cells"].get(f"{K}_{N}")
    if not cell or "stats" not in cell or cell.get("n_trials", 0) < 30:
        return None
    fvq = cell["stats"]["RI"]["accuracy"]
    cvq = cell["stats"]["PI"]["accuracy"]
    n = cell["n_trials"]
    return {
        "fvq": fvq, "cvq": cvq, "gap": fvq - cvq, "n": n,
        "fvq_hw": wilson_half_width(fvq, n),
        "cvq_hw": wilson_half_width(cvq, n),
    }


def load_open_weight_cell(model_dir, K, N):
    files = sorted(glob.glob(str(SEM_DIR / model_dir / "stage1_sweep_*.json")))
    if not files: return None
    with open(files[-1]) as f: d = json.load(f)
    cell = d["cells"].get(f"{K}_{N}")
    if not cell or "stats" not in cell or cell.get("n_trials", 0) < 30:
        return None
    fvq = cell["stats"]["RI"]["accuracy"]
    cvq = cell["stats"]["PI"]["accuracy"]
    n = cell["n_trials"]
    return {
        "fvq": fvq, "cvq": cvq, "gap": fvq - cvq, "n": n,
        "fvq_hw": wilson_half_width(fvq, n),
        "cvq_hw": wilson_half_width(cvq, n),
    }


def load_proprietary_cells():
    """Returns {model: {(K,N): {fvq, cvq, gap, n, fvq_hw, cvq_hw}}}"""
    out = {}
    with open(PROP_CSV) as f:
        for r in csv.DictReader(f):
            if int(r["n_trials"]) < 30: continue
            m = r["model"]
            K, N = int(r["num_keys"]), int(r["num_updates"])
            fvq = float(r["fvq_acc"]); cvq = float(r["cvq_acc"])
            n = int(r["n_trials"])
            out.setdefault(m, {})[(K, N)] = {
                "fvq": fvq, "cvq": cvq, "gap": fvq - cvq, "n": n,
                "fvq_hw": wilson_half_width(fvq, n),
                "cvq_hw": wilson_half_width(cvq, n),
            }
    return out


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — DEEP DIVE
# ═════════════════════════════════════════════════════════════════════════════

def _plot_vary_K(ax, model_dir, fixed_N, x_label_on, y_label_on, title_label, xticks_subsample=False):
    cells = [(K, load_open_weight_cell(model_dir, K, fixed_N)) for K in K_SCAN_VALUES]
    cells = [(K, c) for K, c in cells if c is not None]
    if not cells: return
    Ks  = [K for K, _ in cells]
    fvq = np.array([c["fvq"] for _, c in cells])
    cvq = np.array([c["cvq"] for _, c in cells])
    fvq_hw = np.array([c["fvq_hw"] for _, c in cells])
    cvq_hw = np.array([c["cvq_hw"] for _, c in cells])
    ax.errorbar(Ks, fvq, yerr=fvq_hw, marker="o", color="C0", label="FVQ",
                capsize=3, capthick=1.0, lw=1.2)
    ax.errorbar(Ks, cvq, yerr=cvq_hw, marker="s", color="C1", label="CVQ",
                capsize=3, capthick=1.0, lw=1.2)
    if x_label_on: ax.set_xlabel("Number of keys $K$")
    if y_label_on: ax.set_ylabel("Accuracy")
    ax.set_title(title_label, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(Ks)
    ax.grid(alpha=0.3)


def _plot_vary_N(ax, model_dir, fixed_K, x_label_on, y_label_on, title_label):
    cells = [(N, load_open_weight_cell(model_dir, fixed_K, N)) for N in N_SCAN_VALUES]
    cells = [(N, c) for N, c in cells if c is not None]
    if not cells: return
    Ns  = [N for N, _ in cells]
    fvq = np.array([c["fvq"] for _, c in cells])
    cvq = np.array([c["cvq"] for _, c in cells])
    fvq_hw = np.array([c["fvq_hw"] for _, c in cells])
    cvq_hw = np.array([c["cvq_hw"] for _, c in cells])
    ax.errorbar(Ns, fvq, yerr=fvq_hw, marker="o", color="C0", label="FVQ",
                capsize=3, capthick=1.0, lw=1.2)
    ax.errorbar(Ns, cvq, yerr=cvq_hw, marker="s", color="C1", label="CVQ",
                capsize=3, capthick=1.0, lw=1.2)
    if x_label_on: ax.set_xlabel("Updates per key $N$")
    if y_label_on: ax.set_ylabel("Accuracy")
    ax.set_title(title_label, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(Ns)
    ax.grid(alpha=0.3)


def make_figure_2(out_path):
    """6-panel: 2 models × 3 axis scans.
    Column 1: vary K at low N=10 (low-load primacy regime).
    Column 2: vary N at low K=5 (low-load primacy regime).
    Column 3: vary K at high N=50 (reaches reversal regime for Qwen-family).
    """
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 5.2), sharey=True)

    for row, (model_dir, model_label) in enumerate(DEEP_DIVE):
        is_bottom = (row == 1)
        # Column 0: vary K at N=10
        _plot_vary_K(
            axes[row, 0], model_dir, fixed_N=VARY_K_AT_N,
            x_label_on=is_bottom, y_label_on=True,
            title_label=f"{model_label} --- vary $K$ at $N{{=}}{VARY_K_AT_N}$",
        )
        # Column 1: vary N at K=5
        _plot_vary_N(
            axes[row, 1], model_dir, fixed_K=VARY_N_AT_K,
            x_label_on=is_bottom, y_label_on=False,
            title_label=f"{model_label} --- vary $N$ at $K{{=}}{VARY_N_AT_K}$",
        )
        # Column 2: vary K at high N=50
        _plot_vary_K(
            axes[row, 2], model_dir, fixed_N=VARY_K_AT_N_HIGH,
            x_label_on=is_bottom, y_label_on=False,
            title_label=f"{model_label} --- vary $K$ at $N{{=}}{VARY_K_AT_N_HIGH}$",
        )

    # Single shared legend at bottom (paper figure standard)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.01), ncol=2,
               frameon=False, fontsize=10)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# TABLES — LATEX
# ═════════════════════════════════════════════════════════════════════════════

def fmt_pm(acc, hw):
    """Format as ``0.86 \pm 0.07``."""
    return f"${acc:.2f} \\pm {hw:.2f}$"


def fmt_gap(gap):
    sign = "+" if gap >= 0 else "$-$"
    return f"{sign}{abs(gap):.2f}"


# ── Heatmap-shaded cell formatters ────────────────────────────────────────
# Cell intensity is computed here so the .tex output just contains literal
# \cellcolor{blue!36} etc.; the only preamble dependency is
# \usepackage[table]{xcolor}.
#
# Accuracy in [0, 1]:  shade by value, single-hue blue.  Cap at 50 so the
# darkest cell stays readable in B&W print.
# Gap in [-1, 1]:      diverging.  Negative gaps -> red; positive -> blue;
# 0 -> white.  Intensity capped at 60.

def _acc_shade(x):
    """accuracy [0,1] -> blue intensity (0-40), lowered cap for restraint."""
    return int(round(40 * max(0.0, min(1.0, float(x)))))


def _gap_shade(x):
    """|gap| -> red intensity (0-50).  Sign is carried by the +/- in the value,
    not by the colour.  Reader scan rule: dark red = large bias either way."""
    return int(round(50 * min(1.0, abs(float(x)))))


def fmt_acc_cell(x):
    """Shaded accuracy cell, e.g. \cellcolor{blue!36}0.71.
    Darker blue = higher accuracy = better."""
    return rf"\cellcolor{{blue!{_acc_shade(x)}}}{x:.2f}"


def fmt_gap_cell(gap):
    """Shaded gap cell, e.g. \cellcolor{red!28}+0.47.
    Darker red = larger bias (in either direction)."""
    sign = "+" if gap >= 0 else "$-$"
    return rf"\cellcolor{{red!{_gap_shade(gap)}}}{sign}{abs(gap):.2f}"


def load_context_usage():
    """Map model_label -> (tokens, ctx_window, pct). One row per Table 1 model."""
    path = OUT_DIR / "context_usage.csv"
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["model_label"]] = (
                int(r["tokens"]), int(r["ctx_window"]), float(r["pct"])
            )
    return out


def fmt_ctx(tokens, ctx, pct):
    """Format as '0.6%'.

    Compact percent-only display so the column fits the page; per-token
    counts and window sizes are documented in Appendix~D.
    """
    return f"{pct:.1f}\\%"


def make_combined_table(out_path, prop_cell=PROP_CELL, open_cell=OPEN_CELL):
    """One combined table: proprietary block (top) + open-weight block (bottom).

    Columns: Model, FVQ, CVQ, Gap, Ctx@cell.
    The Ctx column reports tokens / context window (percent) for one
    canonical Semantic-Multi prompt at the row's representative cell.
    All counts use tiktoken o200k_base for uniform cross-provider
    comparability (App~D documents this choice); full Wilson 95\% CIs on
    FVQ/CVQ live in App~D.
    """
    pK, pN = prop_cell
    oK, oN = open_cell
    prop = load_proprietary_cells()
    ctx = load_context_usage()

    lines = []
    lines.append("% Auto-generated by paper/figures/generate_main_results.py")
    lines.append(f"% Proprietary at K={pK},N={pN}; open-weight at K={oK},N={oN} on Semantic-Multi")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{FVQ} & \textbf{CVQ} & \textbf{Gap} & \textbf{\% Ctx} \\")
    lines.append(r"\midrule")

    lines.append(
        rf"\multicolumn{{5}}{{l}}{{\emph{{Proprietary models, $K{{=}}{pK}$, $N{{=}}{pN}$}}}} \\"
    )
    for model_key, label in PROPRIETARY_MODELS:
        c = prop.get(model_key, {}).get(prop_cell)
        ctx_cell = ctx.get(label)
        ctx_str = fmt_ctx(*ctx_cell) if ctx_cell else "---"
        if c is None:
            lines.append(f"  {label} & --- & --- & --- & {ctx_str} \\\\")
        else:
            lines.append(
                f"  {label} & "
                f"{fmt_acc_cell(c['fvq'])} & {fmt_acc_cell(c['cvq'])} & "
                f"{fmt_gap_cell(c['gap'])} & {ctx_str} \\\\"
            )

    lines.append(r"\midrule")
    lines.append(
        rf"\multicolumn{{5}}{{l}}{{\emph{{Open-weight models, $K{{=}}{oK}$, $N{{=}}{oN}$}}}} \\"
    )
    for model_dir, label in OPENWEIGHT_MODELS:
        c = load_open_weight_cell(model_dir, oK, oN)
        ctx_cell = ctx.get(label)
        ctx_str = fmt_ctx(*ctx_cell) if ctx_cell else "---"
        if c is None:
            lines.append(f"  {label} & --- & --- & --- & {ctx_str} \\\\")
        else:
            lines.append(
                f"  {label} & "
                f"{fmt_acc_cell(c['fvq'])} & {fmt_acc_cell(c['cvq'])} & "
                f"{fmt_gap_cell(c['gap'])} & {ctx_str} \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"  saved {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def load_format_open(model_dir, K, N):
    """Returns {format_key: {fvq, cvq, gap, n}} or None.

    Open-weight format keys in data are suffixed with `_last`.
    Falls back to checkpoint.json when no finalised ucurve_*.json
    exists (run was interrupted mid-sweep but per-cell data was
    persisted via the checkpointing mechanism).
    """
    candidates = sorted(glob.glob(str(UCURVE_VLLM_DIR / model_dir / "ucurve_*.json")))
    if not candidates:
        ckpt = UCURVE_VLLM_DIR / model_dir / "checkpoint.json"
        if ckpt.exists():
            candidates = [str(ckpt)]
    if not candidates: return None
    with open(candidates[-1]) as f: d = json.load(f)
    cell = d.get("cells", {}).get(f"{K}_{N}")
    if not cell: return None
    out = {}
    for fmt in FORMAT_KEYS:
        key = f"{fmt}_last"
        f_cell = cell.get(key)
        if not f_cell: continue
        out[fmt] = {
            "fvq": f_cell["first_acc"],
            "cvq": f_cell["last_acc"],
            "gap": f_cell["gap"],
            "n": f_cell["n_trials"],
        }
    return out if out else None


def load_format_prop():
    """Returns {model: {(K,N): {format: {fvq, cvq, gap, n}}}}.

    Computes fvq/cvq from positions '1' and 'last' in the per-position CSV.
    """
    rows_by = {}
    with open(UCURVE_PROP_CSV) as f:
        for r in csv.DictReader(f):
            key = (r["model"], r["format"], int(r["num_keys"]), int(r["num_updates"]))
            rows_by.setdefault(key, {})[r["position"]] = (
                float(r["accuracy"]), int(r["n_trials"])
            )
    out = {}
    for (model, fmt, K, N), posns in rows_by.items():
        first = posns.get("1"); last = posns.get("last")
        if first is None or last is None: continue
        fvq, n1 = first; cvq, n2 = last
        out.setdefault(model, {}).setdefault((K, N), {})[fmt] = {
            "fvq": fvq, "cvq": cvq, "gap": fvq - cvq, "n": min(n1, n2),
        }
    return out


def make_format_intervention_table(out_path, cell=FORMAT_CELL):
    """Combined format-intervention table: open-weight + proprietary.
    Columns: Model, gap per format. Smaller gap = format helps.
    """
    K, N = cell
    prop = load_format_prop()
    lines = []
    lines.append("% Auto-generated by paper/figures/generate_main_results.py")
    lines.append(f"% Format intervention: gap = FVQ - CVQ at K={K}, N={N}")
    col_spec = "l" + "c" * len(FORMAT_KEYS)
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    header = r"\textbf{Model}"
    for f in FORMAT_KEYS:
        header += rf" & \textbf{{{FORMAT_LABELS[f]}}}"
    lines.append(header + r" \\")
    lines.append(r"\midrule")

    lines.append(
        rf"\multicolumn{{{len(FORMAT_KEYS)+1}}}{{l}}{{\emph{{Open-weight models}}}} \\"
    )
    for model_dir, label in FORMAT_MODELS_OPEN:
        fdata = load_format_open(model_dir, K, N) or {}
        row = label
        for fmt in FORMAT_KEYS:
            d = fdata.get(fmt)
            row += f" & {fmt_gap(d['gap'])}" if d else r" & --- "
        lines.append(f"  {row} \\\\")

    lines.append(r"\midrule")
    lines.append(
        rf"\multicolumn{{{len(FORMAT_KEYS)+1}}}{{l}}{{\emph{{Proprietary models}}}} \\"
    )
    for model_key, label in FORMAT_MODELS_PROP:
        cells_for_model = prop.get(model_key, {})
        fdata = cells_for_model.get((K, N), {})
        row = label
        for fmt in FORMAT_KEYS:
            d = fdata.get(fmt)
            row += f" & {fmt_gap(d['gap'])}" if d else r" & --- "
        lines.append(f"  {row} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"  saved {out_path}")


def make_format_intervention_combined_table(out_path, cell=FORMAT_CELL):
    """Merged FVQ / CVQ per format. Replaces tab_format_gap + tab_format_cvq.

    Each cell shows `FVQ / CVQ` so the reader can see both queries in one
    glance.  Pattern at a glance:
      - "0.99 / 0.23"  large gap, Plain failure
      - "0.92 / 0.92"  Block recovery, both queries near ceiling
      - "0.39 / 0.39"  Labeled equalises but both degraded (failure shifted)
    """
    K, N = cell
    prop = load_format_prop()
    lines = []
    lines.append("% Auto-generated by paper/scripts/generate_main_results.py")
    lines.append(f"% Combined FVQ / CVQ by format at K={K}, N={N}")
    col_spec = "l" + "c" * len(FORMAT_KEYS)
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    header = r"\textbf{Model}"
    for f in FORMAT_KEYS:
        header += rf" & \textbf{{{FORMAT_LABELS[f]}}}"
    lines.append(header + r" \\")
    lines.append(r"\midrule")

    def _row(label, fdata):
        row = label
        for fmt in FORMAT_KEYS:
            d = fdata.get(fmt)
            if d:
                row += f" & {fmt_acc_cell(d['fvq'])} / {fmt_acc_cell(d['cvq'])}"
            else:
                row += " & --- "
        return f"  {row} \\\\"

    lines.append(
        rf"\multicolumn{{{len(FORMAT_KEYS)+1}}}{{l}}{{\emph{{Open-weight models}}}} \\"
    )
    for model_dir, label in FORMAT_MODELS_OPEN:
        fdata = load_format_open(model_dir, K, N) or {}
        lines.append(_row(label, fdata))

    lines.append(r"\midrule")
    lines.append(
        rf"\multicolumn{{{len(FORMAT_KEYS)+1}}}{{l}}{{\emph{{Proprietary models}}}} \\"
    )
    for model_key, label in FORMAT_MODELS_PROP:
        fdata = prop.get(model_key, {}).get((K, N), {})
        lines.append(_row(label, fdata))

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"  saved {out_path}")


def make_format_intervention_ci_table(out_path, cell=FORMAT_CELL):
    """Appendix companion to tab_format.tex with Wilson 95% half-widths.

    Each cell shows ``FVQ / CVQ`` (mean) on the first line and the matching
    Wilson half-widths ``±hw_FVQ / ±hw_CVQ`` on a second \\scriptsize line.
    Same (K, N, models, formats) as tab_format.tex.
    """
    K, N = cell
    prop = load_format_prop()
    lines = []
    lines.append("% Auto-generated by paper/scripts/generate_main_results.py")
    lines.append(f"% Format intervention with Wilson 95% half-widths at K={K}, N={N}")
    col_spec = "l" + "c" * len(FORMAT_KEYS)
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    header = r"\textbf{Model}"
    for f in FORMAT_KEYS:
        header += rf" & \textbf{{{FORMAT_LABELS[f]}}}"
    lines.append(header + r" \\")
    lines.append(r"\midrule")

    def _cell(d):
        if not d: return "---"
        hw_f = wilson_half_width(d["fvq"], d["n"])
        hw_c = wilson_half_width(d["cvq"], d["n"])
        return (
            rf"\begin{{tabular}}[t]{{@{{}}c@{{}}}}"
            rf"{d['fvq']:.2f}\,/\,{d['cvq']:.2f}\\"
            rf"{{\scriptsize$\pm${hw_f:.02f}\,/\,$\pm${hw_c:.02f}}}"
            rf"\end{{tabular}}"
        )

    def _row(label, fdata):
        row = label
        for fmt in FORMAT_KEYS:
            row += f" & {_cell(fdata.get(fmt))}"
        return f"  {row} \\\\"

    lines.append(
        rf"\multicolumn{{{len(FORMAT_KEYS)+1}}}{{l}}{{\emph{{Open-weight models}}}} \\"
    )
    for model_dir, label in FORMAT_MODELS_OPEN:
        fdata = load_format_open(model_dir, K, N) or {}
        lines.append(_row(label, fdata))

    lines.append(r"\midrule")
    lines.append(
        rf"\multicolumn{{{len(FORMAT_KEYS)+1}}}{{l}}{{\emph{{Proprietary models}}}} \\"
    )
    for model_key, label in FORMAT_MODELS_PROP:
        fdata = prop.get(model_key, {}).get((K, N), {})
        lines.append(_row(label, fdata))

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"  saved {out_path}")


def make_format_intervention_acc_table(out_path, cell=FORMAT_CELL):
    """Companion table: absolute CVQ accuracy per format (so reader sees
    the gap shrinks because CVQ rises, not because FVQ falls)."""
    K, N = cell
    prop = load_format_prop()
    lines = []
    lines.append("% Auto-generated by paper/figures/generate_main_results.py")
    lines.append(f"% CVQ (last-position) accuracy per format at K={K}, N={N}")
    col_spec = "l" + "c" * len(FORMAT_KEYS)
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    header = r"\textbf{Model}"
    for f in FORMAT_KEYS:
        header += rf" & \textbf{{{FORMAT_LABELS[f]}}}"
    lines.append(header + r" \\")
    lines.append(r"\midrule")

    lines.append(
        rf"\multicolumn{{{len(FORMAT_KEYS)+1}}}{{l}}{{\emph{{Open-weight models}}}} \\"
    )
    for model_dir, label in FORMAT_MODELS_OPEN:
        fdata = load_format_open(model_dir, K, N) or {}
        row = label
        for fmt in FORMAT_KEYS:
            d = fdata.get(fmt)
            row += f" & {d['cvq']:.2f}" if d else r" & --- "
        lines.append(f"  {row} \\\\")

    lines.append(r"\midrule")
    lines.append(
        rf"\multicolumn{{{len(FORMAT_KEYS)+1}}}{{l}}{{\emph{{Proprietary models}}}} \\"
    )
    for model_key, label in FORMAT_MODELS_PROP:
        cells_for_model = prop.get(model_key, {})
        fdata = cells_for_model.get((K, N), {})
        row = label
        for fmt in FORMAT_KEYS:
            d = fdata.get(fmt)
            row += f" & {d['cvq']:.2f}" if d else r" & --- "
        lines.append(f"  {row} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"  saved {out_path}")


def _cell_gap_or_none(d, K, N):
    cell = d.get("cells", {}).get(f"{K}_{N}")
    if not cell or "stats" not in cell: return None
    return cell["stats"]["RI"]["accuracy"] - cell["stats"]["PI"]["accuracy"]


def _mean_gap_over(d, cells):
    vals = [g for K, N in cells if (g := _cell_gap_or_none(d, K, N)) is not None]
    return sum(vals) / len(vals) if vals else None


def load_smollm2_trajectory():
    """Returns list of dicts with keys:
    step, ri, pi, gap_uniform, gap_low, gap_high.
    'final' is appended with a synthetic step value just past the last."""
    out = []
    files = sorted(glob.glob(str(SMOLLM2_DIR / "step-*.json")))
    files = [f for f in files if "_trials" not in f]
    for f in files:
        m = re.search(r"step-(\d+)\.json", f)
        if not m: continue
        step = int(m.group(1))
        d = json.load(open(f))
        s = d.get("summary") or {}
        if not s: continue
        out.append({
            "step": step,
            "ri": s["mean_ri"], "pi": s["mean_pi"], "gap_uniform": s["mean_gap"],
            "gap_low":  _mean_gap_over(d, LOW_LOAD_CELLS),
            "gap_med":  _mean_gap_over(d, MED_LOAD_CELLS),
            "gap_high": _mean_gap_over(d, HIGH_LOAD_CELLS),
            "is_final": False,
        })
    out.sort(key=lambda r: r["step"])
    fin_path = SMOLLM2_DIR / "final.json"
    if fin_path.exists():
        d = json.load(open(fin_path))
        s = d.get("summary") or {}
        if s:
            last_step = out[-1]["step"] if out else 0
            out.append({
                "step": last_step + 125_000,
                "ri": s["mean_ri"], "pi": s["mean_pi"], "gap_uniform": s["mean_gap"],
                "gap_low":  _mean_gap_over(d, LOW_LOAD_CELLS),
                "gap_med":  _mean_gap_over(d, MED_LOAD_CELLS),
                "gap_high": _mean_gap_over(d, HIGH_LOAD_CELLS),
                "is_final": True,
            })
    return out


def load_smollm3_pretraining():
    """SmolLM3 stage1/2/3 pretraining checkpoints in completion_few_shot format.
    Returns list of dicts (step, ri, pi, gap_uniform, gap_low, gap_high)."""
    out = []
    files = sorted(glob.glob(str(SMOLLM3_DIR / "stage*-step-*.json")))
    files = [f for f in files if "_trials" not in f]
    for f in files:
        d = json.load(open(f))
        if d.get("prompt_format") != "completion_few_shot": continue
        m = re.search(r"step-(\d+)\.json", f)
        if not m: continue
        step = int(m.group(1))
        s = d.get("summary") or {}
        if not s: continue
        out.append({
            "step": step,
            "ri": s["mean_ri"], "pi": s["mean_pi"], "gap_uniform": s["mean_gap"],
            "gap_low":  _mean_gap_over(d, LOW_LOAD_CELLS),
            "gap_med":  _mean_gap_over(d, MED_LOAD_CELLS),
            "gap_high": _mean_gap_over(d, HIGH_LOAD_CELLS),
        })
    out.sort(key=lambda r: r["step"])
    return out


def load_smollm3_posttraining(prompt_format="completion"):
    """SmolLM3 named post-training stages.
    Returns dict {stage: {ri, pi, gap_uniform, gap_low, gap_high, n_cells}}.
    """
    out = {}
    for stage in SMOLLM3_POST_STAGES:
        f = SMOLLM3_DIR / f"{stage}_{prompt_format}.json"
        if not f.exists():
            continue
        d = json.load(open(f))
        s = d.get("summary") or {}
        if not s: continue
        out[stage] = {
            "ri": s["mean_ri"], "pi": s["mean_pi"], "gap_uniform": s["mean_gap"],
            "gap_low":  _mean_gap_over(d, LOW_LOAD_CELLS),
            "gap_med":  _mean_gap_over(d, MED_LOAD_CELLS),
            "gap_high": _mean_gap_over(d, HIGH_LOAD_CELLS),
            "n_cells": s["n_cells"],
        }
    return out


_STAGE_COLORS = [
    "#c7e3f7",  # light blue
    "#cfeacb",  # light green
    "#fbeec2",  # light yellow
    "#f9d6c1",  # light salmon
    "#e1cef0",  # light purple
    "#d6e8d6",  # mint
    "#dcdcdc",  # grey
    "#f4c2c2",  # pink (final)
]


def _shade_stages(ax, stages, y_low, y_high, fontsize=8):
    """Shade stage bands and add labels at the top of the axes."""
    for i, (label, x0, x1) in enumerate(stages):
        ax.axvspan(x0, x1, color=_STAGE_COLORS[i % len(_STAGE_COLORS)],
                   alpha=0.55, zorder=0)
        cx = (x0 + x1) / 2
        ax.text(cx, y_high * 0.93, label, ha="center", va="top",
                fontsize=fontsize, color="black", alpha=0.85)


_REGIME_STYLE = [
    ("gap_low",  "#1f77b4", "o", f"Low ($K{{\\leq}}3$, any $N$; {len(LOW_LOAD_CELLS)} cells)"),
    ("gap_med",  "#7f7f7f", "^", f"Medium ($K{{\\geq}}5, N{{<}}20$; {len(MED_LOAD_CELLS)} cells)"),
    ("gap_high", "#d62728", "s", f"High ($K{{\\geq}}5, N{{\\geq}}20$; {len(HIGH_LOAD_CELLS)} cells)"),
]


def _x_fmt(ax):
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f"{int(x/1e6)}M" if x >= 1e6 else (f"{int(x/1e3)}k" if x > 0 else "0")))


def _plot_smollm2_panel(ax, sm2, y_low, y_high):
    steps = [r["step"] for r in sm2]
    stages = list(SMOLLM2_STAGES)
    if steps:
        stages.append(("Final", steps[-2] + 60_000, steps[-1] + 60_000))
    _shade_stages(ax, stages, y_low, y_high)
    for key, color, marker, label in _REGIME_STYLE:
        ys = [r[key] for r in sm2]
        ax.plot(steps, ys, marker=marker, markersize=3, lw=1.2,
                color=color, label=label)
    ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.7)
    ax.set_ylabel("Mean gap (FVQ$-$CVQ)")
    ax.set_xlabel("Training step")
    ax.set_title("SmolLM2-1.7B (42 checkpoints)", fontsize=10)
    ax.set_ylim(y_low, y_high)
    if steps:
        ax.set_xlim(steps[0] - 50_000, steps[-1] + 110_000)
    ax.grid(alpha=0.3, zorder=1)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, framealpha=0.95, frameon=False)
    _x_fmt(ax)


def _plot_smollm3_panel(ax, sm3_pre, sm3_post, y_low, y_high):
    pre_steps = [r["step"] for r in sm3_pre]
    if pre_steps:
        spacing = max((pre_steps[-1] - pre_steps[0]) / max(len(pre_steps) - 1, 1), 80_000)
        post_x = [pre_steps[-1] + spacing * (i + 1)
                  for i in range(len(SMOLLM3_POST_STAGES))]
    else:
        spacing = 1
        post_x = list(range(len(SMOLLM3_POST_STAGES)))

    # Single merged post-training band (replaces five narrow bands).
    sm3_stages = list(SMOLLM3_STAGES)
    if post_x:
        post_band_x0 = post_x[0] - spacing / 2
        post_band_x1 = post_x[-1] + spacing / 2
        sm3_stages.append(("Post-training\n(5 stages)", post_band_x0, post_band_x1))
    _shade_stages(ax, sm3_stages, y_low, y_high, fontsize=8)

    def _zip_filter(xs, ys):
        return zip(*[(x, y) for x, y in zip(xs, ys) if y is not None]) \
            if any(y is not None for y in ys) else ([], [])

    for key, color, marker, label in _REGIME_STYLE:
        pre_y = [r[key] for r in sm3_pre]
        post_y = [(sm3_post.get(s) or {}).get(key) for s in SMOLLM3_POST_STAGES]
        xs_p_raw, ys_p_raw = _zip_filter(post_x, post_y)
        xs_p, ys_p = list(xs_p_raw), list(ys_p_raw)
        # One continuous solid line spanning pretraining + post-training.
        all_x = list(pre_steps) + xs_p
        all_y = list(pre_y) + ys_p
        ax.plot(all_x, all_y, marker=marker, markersize=3, lw=1.2,
                color=color, label=label)

    if pre_steps and post_x:
        boundary = (pre_steps[-1] + post_x[0]) / 2
        ax.axvline(boundary, color="gray", lw=0.6, ls=":", alpha=0.8)
    ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.7)
    ax.set_ylabel("Mean gap (FVQ$-$CVQ)")
    ax.set_xlabel("Training step (pretraining) and post-training stage")
    ax.set_title("SmolLM3-3B (32 pretraining + 5 post-training checkpoints)",
                 fontsize=10)
    ax.set_ylim(y_low, y_high)
    if pre_steps:
        ax.set_xlim(pre_steps[0] - 50_000,
                    (post_x[-1] + spacing / 2) if post_x else 1)
    ax.grid(alpha=0.3, zorder=1)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, framealpha=0.95, frameon=False)
    _x_fmt(ax)


def _save_fig(fig, out_path):
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def make_training_dynamics_smollm3(out_path):
    """Single-panel SmolLM3-only training dynamics figure for the main paper."""
    sm3_pre = load_smollm3_pretraining()
    sm3_post = load_smollm3_posttraining("completion")
    fig, ax = plt.subplots(figsize=(9.0, 3.4))
    _plot_smollm3_panel(ax, sm3_pre, sm3_post, y_low=-0.55, y_high=0.95)
    _save_fig(fig, out_path)


def make_training_dynamics_smollm2(out_path):
    """Single-panel SmolLM2-only training dynamics figure for the appendix."""
    sm2 = load_smollm2_trajectory()
    fig, ax = plt.subplots(figsize=(9.0, 3.4))
    _plot_smollm2_panel(ax, sm2, y_low=-0.55, y_high=0.95)
    _save_fig(fig, out_path)


def make_training_dynamics_figure(out_path):
    """Vertically stacked SmolLM2 (top) and SmolLM3 (bottom) panels.

    Kept for back-compat / two-up viewing. The paper uses the single-panel
    variants make_training_dynamics_smollm2 and make_training_dynamics_smollm3.
    """
    sm2 = load_smollm2_trajectory()
    sm3_pre = load_smollm3_pretraining()
    sm3_post = load_smollm3_posttraining("completion")
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.6))
    _plot_smollm2_panel(axes[0], sm2, y_low=-0.55, y_high=0.85)
    _plot_smollm3_panel(axes[1], sm3_pre, sm3_post, y_low=-0.55, y_high=0.85)
    _save_fig(fig, out_path)


def make_training_dynamics_tables(out_dir):
    """Three appendix tables:
      - tab_smollm2_traj.tex    (full 42-row trajectory)
      - tab_smollm3_traj.tex    (pretraining + post-training, completion format)
      - tab_smollm3_formats.tex (post-training × {completion, chat_1024})
    """
    # 1. SmolLM2 full trajectory
    sm2 = load_smollm2_trajectory()
    lines = [
        "% Auto-generated",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Step} & \textbf{FVQ} & \textbf{CVQ} & \textbf{Gap} & \textbf{Cells} \\",
        r"\midrule",
    ]
    for i, r in enumerate(sm2):
        label = "final" if r.get("is_final") else f"{r['step']:,}"
        lines.append(f"  {label} & {fmt_acc_cell(r['ri'])} & {fmt_acc_cell(r['pi'])} & {fmt_gap_cell(r['gap_uniform'])} & 32 \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(out_dir / "tab_smollm2_traj.tex").write_text("\n".join(lines) + "\n")
    print(f"  saved tab_smollm2_traj.tex")

    # 2. SmolLM3 trajectory (completion format)
    sm3_pre = load_smollm3_pretraining()
    sm3_post = load_smollm3_posttraining("completion")
    lines = [
        "% Auto-generated; SmolLM3-3B, completion format",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Checkpoint} & \textbf{FVQ} & \textbf{CVQ} & \textbf{Gap} & \textbf{Cells} \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\emph{Pretraining (completion\_few\_shot)}} \\",
    ]
    for r in sm3_pre:
        step = r["step"]
        if step <= 3_440_000: tag = "stage1"
        elif step <= 4_200_000: tag = "stage2"
        else: tag = "stage3"
        lines.append(f"  {tag}-step-{step:,} & {fmt_acc_cell(r['ri'])} & {fmt_acc_cell(r['pi'])} & {fmt_gap_cell(r['gap_uniform'])} & 32 \\\\")
    lines += [
        r"\midrule",
        r"\multicolumn{5}{l}{\emph{Post-training (completion)}} \\",
    ]
    for stage in SMOLLM3_POST_STAGES:
        v = sm3_post.get(stage)
        if v is None:
            lines.append(f"  {stage} & --- & --- & --- & --- \\\\")
        else:
            lines.append(f"  {stage} & {fmt_acc_cell(v['ri'])} & {fmt_acc_cell(v['pi'])} & {fmt_gap_cell(v['gap_uniform'])} & {v['n_cells']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(out_dir / "tab_smollm3_traj.tex").write_text("\n".join(lines) + "\n")
    print(f"  saved tab_smollm3_traj.tex")

    # 3. SmolLM3 post-training × formats
    sm3_completion = load_smollm3_posttraining("completion")
    sm3_chat = load_smollm3_posttraining("chat_1024")
    lines = [
        "% Auto-generated; SmolLM3-3B post-training, completion vs chat_1024",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{\textbf{Completion}} & \multicolumn{2}{c}{\textbf{Chat (1024)}} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"\textbf{Stage} & \textbf{Gap} & \textbf{Cells} & \textbf{Gap} & \textbf{Cells} \\",
        r"\midrule",
    ]
    for stage in SMOLLM3_POST_STAGES:
        c = sm3_completion.get(stage)
        ch = sm3_chat.get(stage)
        g_c = fmt_gap_cell(c["gap_uniform"]) if c else "---"
        n_c = str(c["n_cells"]) if c else "---"
        g_h = fmt_gap_cell(ch["gap_uniform"]) if ch else "---"
        n_h = str(ch["n_cells"]) if ch else "---"
        lines.append(f"  {stage} & {g_c} & {n_c} & {g_h} & {n_h} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(out_dir / "tab_smollm3_formats.tex").write_text("\n".join(lines) + "\n")
    print(f"  saved tab_smollm3_formats.tex")


def make_multimodel_ivq_format_compare_figure(out_path, K=10, N=50):
    """Two-panel figure: per-position accuracy under Plain (left) vs Block (right)
    format at (K, N), for the five open-weight models in the ucurve sweep.
    Documents Block-format IVQ recovery to back §5.2 claim about format
    intervention restoring intermediate-position retrieval."""
    open_models_dirs = [
        ("Qwen2.5-3B-Instruct", "Qwen2.5-3B-Inst"),
        ("Qwen3.5-2B",          "Qwen3.5-2B"),
        ("Qwen3.5-4B",          "Qwen3.5-4B"),
        ("Qwen3.5-9B",          "Qwen3.5-9B"),
        ("gemma-3-4b-it",       "Gemma-3-4b-it"),
    ]

    def load_curves(fmt_key):
        curves = []
        for mdir, label in open_models_dirs:
            candidates = sorted(glob.glob(str(UCURVE_VLLM_DIR / mdir / "ucurve_*.json")))
            if not candidates:
                ckpt = UCURVE_VLLM_DIR / mdir / "checkpoint.json"
                if ckpt.exists(): candidates = [str(ckpt)]
            if not candidates: continue
            with open(candidates[-1]) as f: d = json.load(f)
            cell = d.get("cells", {}).get(f"{K}_{N}", {}).get(fmt_key)
            if not cell: continue
            posns = cell["positions"]
            pts = sorted((int(p), pinfo["accuracy"])
                         for p, pinfo in posns.items() if p != "last")
            if not pts: continue
            xs = [p for p, _ in pts]
            ys = [a for _, a in pts]
            last_acc = posns.get("last", {}).get("accuracy")
            if last_acc is not None and xs[-1] < N:
                xs.append(N); ys.append(last_acc)
            curves.append((label, xs, ys))
        return curves

    plain_curves = load_curves("flat_nolabel_last")
    block_curves = load_curves("block_last")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), sharey=True)
    for ax, curves, title in [
        (axes[0], plain_curves, f"Plain ($K{{=}}{K}, N{{=}}{N}$)"),
        (axes[1], block_curves, f"Block ($K{{=}}{K}, N{{=}}{N}$)"),
    ]:
        for label, xs, ys in curves:
            ax.plot(xs, ys, marker="o", markersize=3, lw=1.3, label=label)
        ax.set_xlabel(f"Query position $k$ (stream of $N{{=}}{N}$ updates)")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.03, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Retrieval accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=5, frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def make_multimodel_ivq_figure(out_path, K=10, N=50):
    """Two-panel figure: per-position accuracy on the default Plain (flat_nolabel)
    format at (K, N), for open-weight models on the left and proprietary on the
    right. Demonstrates 'general loss of position-indexed retrieval' across
    models with sufficient evidence for the claim in §4 Para 3.
    """
    # ---- Open-weight: load from ucurve_vllm JSONs ----
    open_models_dirs = [
        ("Qwen2.5-3B-Instruct", "Qwen2.5-3B-Inst"),
        ("Qwen3.5-2B",          "Qwen3.5-2B"),
        ("Qwen3.5-4B",          "Qwen3.5-4B"),
        ("Qwen3.5-9B",          "Qwen3.5-9B"),
        ("gemma-3-4b-it",       "Gemma-3-4b-it"),
    ]
    open_curves = []
    for mdir, label in open_models_dirs:
        candidates = sorted(glob.glob(str(UCURVE_VLLM_DIR / mdir / "ucurve_*.json")))
        if not candidates:
            ckpt = UCURVE_VLLM_DIR / mdir / "checkpoint.json"
            if ckpt.exists():
                candidates = [str(ckpt)]
        if not candidates: continue
        with open(candidates[-1]) as f: d = json.load(f)
        cell = d.get("cells", {}).get(f"{K}_{N}", {}).get("flat_nolabel_last")
        if not cell: continue
        posns = cell["positions"]
        pts = [(int(p), pinfo["accuracy"])
               for p, pinfo in posns.items() if p != "last"]
        pts.sort()
        if not pts: continue
        xs = [p for p, _ in pts]
        ys = [a for _, a in pts]
        # Append 'last' if present and not already at the final position
        last_acc = posns.get("last", {}).get("accuracy")
        if last_acc is not None and xs[-1] < N:
            xs.append(N)
            ys.append(last_acc)
        open_curves.append((label, xs, ys))

    # ---- Proprietary: load from ucurve_proprietary_results.csv ----
    prop_labels = {
        "gpt-4.1":          "GPT-4.1",
        "gpt-4.1-mini":     "GPT-4.1-mini",
        "claude-4.5-haiku": "Claude-4.5-Haiku",
        "claude-sonnet":    "Claude-4.5-Sonnet",
        "gemini-2.5-flash": "Gemini-2.5-Flash",
        "gemini-2.5-pro":   "Gemini-2.5-Pro",
    }
    rows_by_model = {}
    with open(UCURVE_PROP_CSV) as f:
        for r in csv.DictReader(f):
            if r["format"] != "flat_nolabel": continue
            if int(r["num_keys"]) != K or int(r["num_updates"]) != N: continue
            rows_by_model.setdefault(r["model"], []).append(
                (r["position"], float(r["accuracy"]))
            )
    prop_curves = []
    for mkey, label in prop_labels.items():
        pts = rows_by_model.get(mkey)
        if not pts: continue
        # Separate numeric positions and the 'last' position
        numeric = sorted([(int(p), a) for p, a in pts if p != "last"])
        last_acc = next((a for p, a in pts if p == "last"), None)
        if not numeric: continue
        xs = [p for p, _ in numeric]
        ys = [a for _, a in numeric]
        if last_acc is not None and xs[-1] < N:
            xs.append(N); ys.append(last_acc)
        prop_curves.append((label, xs, ys))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

    # Open-weight panel
    ax = axes[0]
    for label, xs, ys in open_curves:
        ax.plot(xs, ys, marker="o", markersize=3, lw=1.3, label=label)
    ax.set_xlabel(f"Query position $k$ (stream of $N{{=}}{N}$ updates)")
    ax.set_ylabel("Retrieval accuracy")
    ax.set_title(f"Open-weight ($K{{=}}{K}, N{{=}}{N}$, Plain format)",
                 fontsize=10)
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.3)
    open_handles, open_labels = ax.get_legend_handles_labels()

    # Proprietary panel
    ax = axes[1]
    for label, xs, ys in prop_curves:
        ax.plot(xs, ys, marker="s", markersize=3, lw=1.3, label=label)
    ax.set_xlabel(f"Query position $k$ (stream of $N{{=}}{N}$ updates)")
    ax.set_title(f"Proprietary ($K{{=}}{K}, N{{=}}{N}$, Plain format)",
                 fontsize=10)
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.3)
    prop_handles, prop_labels_ = ax.get_legend_handles_labels()

    # Per-panel legends BELOW each panel (paper figure standard).
    axes[0].legend(open_handles, open_labels,
                   loc="upper center", bbox_to_anchor=(0.5, -0.18),
                   ncol=3, frameon=False, fontsize=8)
    axes[1].legend(prop_handles, prop_labels_,
                   loc="upper center", bbox_to_anchor=(0.5, -0.18),
                   ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def make_proprietary_full_table(out_path):
    """Proprietary models on Semantic-Multi at four representative cells.
    Each cell shows gap (FVQ - CVQ) with approximate Wilson half-width.
    """
    prop = load_proprietary_cells()
    lines = []
    lines.append("% Auto-generated by paper/figures/generate_main_results.py")
    lines.append("% All proprietary models on Semantic-Multi at four representative (K,N) cells")
    col_spec = "l" + "c" * len(APPENDIX_CELLS)
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    header = r"\textbf{Model}"
    for K, N in APPENDIX_CELLS:
        header += rf" & $K{{=}}{K}, N{{=}}{N}$"
    lines.append(header + r" \\")
    lines.append(r"\midrule")
    for model_key, label in PROPRIETARY_MODELS:
        row = label
        for K, N in APPENDIX_CELLS:
            c = prop.get(model_key, {}).get((K, N))
            if c is None:
                row += r" & --- "
            else:
                hw = np.sqrt(c["fvq_hw"] ** 2 + c["cvq_hw"] ** 2)
                row += f" & {fmt_gap_cell(c['gap'])}\\,{{\\scriptsize($\\pm${hw:.2f})}}"
        lines.append(f"  {row} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"  saved {out_path}")


def make_full_results_table(out_path, dataset):
    """All 15 open-weight models × 4 representative cells.
    Each cell shows gap (FVQ - CVQ) ± Wilson half-width on the gap.
    """
    lines = []
    lines.append("% Auto-generated by paper/figures/generate_main_results.py")
    lines.append(f"% All open-weight models on {dataset} at four representative (K,N) cells")
    ncols = 1 + len(APPENDIX_CELLS)
    col_spec = "l" + "c" * len(APPENDIX_CELLS)
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    header = r"\textbf{Model}"
    for K, N in APPENDIX_CELLS:
        header += rf" & $K{{=}}{K}, N{{=}}{N}$"
    lines.append(header + r" \\")
    lines.append(r"\midrule")
    for model_dir, label in ALL_MODELS:
        row = label
        for K, N in APPENDIX_CELLS:
            c = load_cell(dataset, model_dir, K, N)
            if c is None:
                row += r" & --- "
            else:
                # gap half-width is approx sqrt(fvq_hw^2 + cvq_hw^2)
                hw = np.sqrt(c["fvq_hw"] ** 2 + c["cvq_hw"] ** 2)
                row += f" & {fmt_gap_cell(c['gap'])}\\,{{\\scriptsize($\\pm${hw:.2f})}}"
        lines.append(f"  {row} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"  saved {out_path}")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print(f"Generating §4 outputs to {OUT_DIR} …")
    make_figure_2(OUT_DIR / "fig2_main_results.pdf")
    make_combined_table(OUT_DIR / "tab_main.tex")
    print(f"Generating appendix tables …")
    make_full_results_table(OUT_DIR / "tab_full_sem.tex", "semantic_multi")
    make_full_results_table(OUT_DIR / "tab_full_arb.tex", "arbitrary_single")
    make_proprietary_full_table(OUT_DIR / "tab_full_sem_prop.tex")
    make_format_intervention_table(OUT_DIR / "tab_format_gap.tex")
    make_format_intervention_acc_table(OUT_DIR / "tab_format_cvq.tex")
    make_format_intervention_combined_table(OUT_DIR / "tab_format.tex")
    make_format_intervention_ci_table(OUT_DIR / "tab_format_ci.tex")
    make_multimodel_ivq_figure(OUT_DIR / "fig_multimodel_ivq.pdf")
    make_multimodel_ivq_format_compare_figure(OUT_DIR / "fig_multimodel_ivq_block.pdf")
    make_training_dynamics_smollm3(OUT_DIR / "fig_training_dynamics_smollm3.pdf")
    make_training_dynamics_smollm2(OUT_DIR / "fig_training_dynamics_smollm2.pdf")
    make_training_dynamics_figure(OUT_DIR / "fig3_training_dynamics.pdf")
    make_training_dynamics_tables(OUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
