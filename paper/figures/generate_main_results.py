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
OUT_DIR = Path(__file__).resolve().parent

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
    ("claude-sonnet",    "Claude-Sonnet"),
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
    ("Qwen3.5-9B",          "Qwen3.5-9B"),
    ("gemma-3-4b-it",       "Gemma-3-4b-it"),
]
FORMAT_MODELS_PROP = [
    ("gpt-4.1",          "GPT-4.1"),
    ("gpt-4.1-mini",     "GPT-4.1-mini"),
    ("claude-4.5-haiku", "Claude-4.5-Haiku"),
    ("claude-sonnet",    "Claude-Sonnet"),
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
LOW_LOAD_CELLS  = [(2, n) for n in [2, 3, 5, 7, 10, 15, 20, 30]] \
                + [(3, n) for n in [2, 3, 5, 7, 10, 15, 20, 30]]
HIGH_LOAD_CELLS = [(k, n) for k in [5, 7] for n in [20, 30]]

# SmolLM2-1.7B pretraining stage boundaries (token-based; converted to steps
# using approx 2.15M tokens/step from the published 11T-token / ~5.125M-step
# correspondence). Source: Allal et al., arXiv 2502.02737, Sec. 3.
SMOLLM2_STAGES = [
    ("Stage 1\n(web, 0-6T)",            0,         2_790_000),
    ("Stage 2\n(+math, 6-8T)",          2_790_000, 3_720_000),
    ("Stage 3\n(+Stack-Edu, 8-10T)",    3_720_000, 4_650_000),
    ("Stage 4\n(decay, 10-11T)",        4_650_000, 5_125_000),
]

# SmolLM3-3B pretraining stage boundaries (in steps; source: SmolLM3 blog,
# https://huggingface.co/blog/smollm3).
SMOLLM3_STAGES = [
    ("Pretraining\nStage 1", 0,         3_440_000),
    ("Pretraining\nStage 2", 3_440_000, 4_200_000),
    ("Pretraining\nStage 3", 4_200_000, 4_720_000),
]

# Figure 2 axis scans
VARY_K_AT_N = 10   # vary K at fixed N=10
VARY_N_AT_K = 5    # vary N at fixed K=5
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

def make_figure_2(out_path):
    """4-panel: 2 models × (vary K at N=10, vary N at K=5)."""

    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.0), sharey=True)

    for row, (model_dir, model_label) in enumerate(DEEP_DIVE):
        # Vary K at fixed N
        ax = axes[row, 0]
        cells = [(K, load_open_weight_cell(model_dir, K, VARY_K_AT_N)) for K in K_SCAN_VALUES]
        cells = [(K, c) for K, c in cells if c is not None]
        Ks  = [K for K, _ in cells]
        fvq = np.array([c["fvq"] for _, c in cells])
        cvq = np.array([c["cvq"] for _, c in cells])
        fvq_hw = np.array([c["fvq_hw"] for _, c in cells])
        cvq_hw = np.array([c["cvq_hw"] for _, c in cells])
        ax.errorbar(Ks, fvq, yerr=fvq_hw, marker="o", color="C0", label="FVQ",
                    capsize=3, capthick=1.0, lw=1.2)
        ax.errorbar(Ks, cvq, yerr=cvq_hw, marker="s", color="C1", label="CVQ",
                    capsize=3, capthick=1.0, lw=1.2)
        ax.set_xlabel("Number of keys $K$")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{model_label} — vary $K$ at $N{{=}}{VARY_K_AT_N}$", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(Ks)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="lower left")

        # Vary N at fixed K
        ax = axes[row, 1]
        cells = [(N, load_open_weight_cell(model_dir, VARY_N_AT_K, N)) for N in N_SCAN_VALUES]
        cells = [(N, c) for N, c in cells if c is not None]
        Ns  = [N for N, _ in cells]
        fvq = np.array([c["fvq"] for _, c in cells])
        cvq = np.array([c["cvq"] for _, c in cells])
        fvq_hw = np.array([c["fvq_hw"] for _, c in cells])
        cvq_hw = np.array([c["cvq_hw"] for _, c in cells])
        ax.errorbar(Ns, fvq, yerr=fvq_hw, marker="o", color="C0", label="FVQ",
                    capsize=3, capthick=1.0, lw=1.2)
        ax.errorbar(Ns, cvq, yerr=cvq_hw, marker="s", color="C1", label="CVQ",
                    capsize=3, capthick=1.0, lw=1.2)
        ax.set_xlabel("Updates per key $N$")
        ax.set_title(f"{model_label} — vary $N$ at $K{{=}}{VARY_N_AT_K}$", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(Ns)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="lower left")

    fig.tight_layout()
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


def make_combined_table(out_path, prop_cell=PROP_CELL, open_cell=OPEN_CELL):
    """One combined table: proprietary block (top) + open-weight block (bottom).

    Columns: Model, FVQ, CVQ, Gap.
    """
    pK, pN = prop_cell
    oK, oN = open_cell
    prop = load_proprietary_cells()

    lines = []
    lines.append("% Auto-generated by paper/figures/generate_main_results.py")
    lines.append(f"% Proprietary at K={pK},N={pN}; open-weight at K={oK},N={oN} on Semantic-Multi")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{FVQ} & \textbf{CVQ} & \textbf{Gap} \\")
    lines.append(r"\midrule")

    lines.append(
        rf"\multicolumn{{4}}{{l}}{{\emph{{Proprietary models, $K{{=}}{pK}$, $N{{=}}{pN}$}}}} \\"
    )
    for model_key, label in PROPRIETARY_MODELS:
        c = prop.get(model_key, {}).get(prop_cell)
        if c is None:
            lines.append(f"  {label} & --- & --- & --- \\\\")
        else:
            lines.append(
                f"  {label} & "
                f"{fmt_pm(c['fvq'], c['fvq_hw'])} & {fmt_pm(c['cvq'], c['cvq_hw'])} & "
                f"{fmt_gap(c['gap'])} \\\\"
            )

    lines.append(r"\midrule")
    lines.append(
        rf"\multicolumn{{4}}{{l}}{{\emph{{Open-weight models, $K{{=}}{oK}$, $N{{=}}{oN}$}}}} \\"
    )
    for model_dir, label in OPENWEIGHT_MODELS:
        c = load_open_weight_cell(model_dir, oK, oN)
        if c is None:
            lines.append(f"  {label} & --- & --- & --- \\\\")
        else:
            lines.append(
                f"  {label} & "
                f"{fmt_pm(c['fvq'], c['fvq_hw'])} & {fmt_pm(c['cvq'], c['cvq_hw'])} & "
                f"{fmt_gap(c['gap'])} \\\\"
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
    """
    files = sorted(glob.glob(str(UCURVE_VLLM_DIR / model_dir / "ucurve_*.json")))
    if not files: return None
    with open(files[-1]) as f: d = json.load(f)
    cell = d["cells"].get(f"{K}_{N}")
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


def make_training_dynamics_figure(out_path):
    """Vertically stacked SmolLM2 (top) and SmolLM3 (bottom) panels with
    stage bands and two regime lines per panel."""
    sm2 = load_smollm2_trajectory()
    sm3_pre = load_smollm3_pretraining()
    sm3_post = load_smollm3_posttraining("completion")

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.6))

    y_low, y_high = -0.55, 0.75

    # ── Top panel: SmolLM2 ─────────────────────────────────────────────────
    ax = axes[0]
    steps    = [r["step"] for r in sm2]
    gap_low  = [r["gap_low"]  for r in sm2]
    gap_high = [r["gap_high"] for r in sm2]
    # Stage bands
    stages = list(SMOLLM2_STAGES)
    # Append "Final" band at the right
    if steps:
        stages.append(("Final", steps[-2] + 60_000, steps[-1] + 60_000))
    _shade_stages(ax, stages, y_low, y_high)

    ax.plot(steps, gap_low,  marker="o", markersize=3, lw=1.2, color="#1f77b4",
            label="Low-load cells ($K{\\leq}3$): primacy regime")
    ax.plot(steps, gap_high, marker="s", markersize=3, lw=1.2, color="#d62728",
            label="High-load cells ($K{\\geq}5, N{\\geq}20$): recency regime")
    ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.7)
    ax.set_ylabel("Mean gap (FVQ$-$CVQ)")
    ax.set_title("SmolLM2-1.7B (42 checkpoints)", fontsize=10)
    ax.set_ylim(y_low, y_high)
    ax.set_xlim(steps[0] - 50_000, steps[-1] + 110_000)
    ax.grid(alpha=0.3, zorder=1)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.92)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))

    # ── Bottom panel: SmolLM3 ──────────────────────────────────────────────
    ax = axes[1]
    pre_steps    = [r["step"] for r in sm3_pre]
    pre_gap_low  = [r["gap_low"]  for r in sm3_pre]
    pre_gap_high = [r["gap_high"] for r in sm3_pre]

    if pre_steps:
        spacing = max((pre_steps[-1] - pre_steps[0]) / max(len(pre_steps) - 1, 1), 80_000)
        post_x = [pre_steps[-1] + spacing * (i + 1)
                  for i in range(len(SMOLLM3_POST_STAGES))]
    else:
        post_x = list(range(len(SMOLLM3_POST_STAGES)))
    post_gap_low  = [(sm3_post.get(s) or {}).get("gap_low")  for s in SMOLLM3_POST_STAGES]
    post_gap_high = [(sm3_post.get(s) or {}).get("gap_high") for s in SMOLLM3_POST_STAGES]
    # Drop None entries while keeping x in sync
    def _zip_filter(xs, ys):
        return zip(*[(x, y) for x, y in zip(xs, ys) if y is not None]) \
            if any(y is not None for y in ys) else ([], [])
    post_x_lo, post_y_lo = _zip_filter(post_x, post_gap_low)
    post_x_hi, post_y_hi = _zip_filter(post_x, post_gap_high)

    # Stage bands: pretraining stages + each post-training stage
    sm3_stages = list(SMOLLM3_STAGES)
    if pre_steps and post_x:
        for i, stage_key in enumerate(SMOLLM3_POST_STAGES):
            x_mid = post_x[i]
            x0 = x_mid - spacing / 2
            x1 = x_mid + spacing / 2
            sm3_stages.append((SMOLLM3_POST_LABELS[stage_key], x0, x1))
    _shade_stages(ax, sm3_stages, y_low, y_high, fontsize=7)

    ax.plot(list(pre_steps), pre_gap_low,  marker="o", markersize=3,
            lw=1.2, color="#1f77b4",
            label="Low-load cells ($K{\\leq}3$)")
    ax.plot(list(pre_steps), pre_gap_high, marker="s", markersize=3,
            lw=1.2, color="#d62728",
            label="High-load cells ($K{\\geq}5, N{\\geq}20$)")
    # Post-training points
    if post_x_lo:
        ax.plot(list(post_x_lo), list(post_y_lo), marker="o", markersize=5,
                lw=1.2, color="#1f77b4", linestyle="--")
    if post_x_hi:
        ax.plot(list(post_x_hi), list(post_y_hi), marker="s", markersize=5,
                lw=1.2, color="#d62728", linestyle="--")
    # Boundary between pre and post
    if pre_steps and post_x:
        boundary = (pre_steps[-1] + post_x[0]) / 2
        ax.axvline(boundary, color="gray", lw=0.6, ls=":", alpha=0.8)

    ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.7)
    ax.set_ylabel("Mean gap (FVQ$-$CVQ)")
    ax.set_xlabel("Training step (pretraining) and post-training stage")
    ax.set_title("SmolLM3-3B (32 pretraining + 5 post-training checkpoints)",
                 fontsize=10)
    ax.set_ylim(y_low, y_high)
    ax.set_xlim(pre_steps[0] - 50_000 if pre_steps else 0,
                (post_x[-1] + spacing / 2) if post_x else 1)
    ax.grid(alpha=0.3, zorder=1)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.92)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


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
        r"\textbf{Step} & \textbf{RI (FVQ)} & \textbf{PI (CVQ)} & \textbf{Gap} & \textbf{Cells} \\",
        r"\midrule",
    ]
    for i, r in enumerate(sm2):
        label = "final" if r.get("is_final") else f"{r['step']:,}"
        lines.append(f"  {label} & {r['ri']:.3f} & {r['pi']:.3f} & {fmt_gap(r['gap_uniform'])} & 32 \\\\")
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
        r"\textbf{Checkpoint} & \textbf{RI (FVQ)} & \textbf{PI (CVQ)} & \textbf{Gap} & \textbf{Cells} \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\emph{Pretraining (completion\_few\_shot)}} \\",
    ]
    for r in sm3_pre:
        step = r["step"]
        if step <= 3_440_000: tag = "stage1"
        elif step <= 4_200_000: tag = "stage2"
        else: tag = "stage3"
        lines.append(f"  {tag}-step-{step:,} & {r['ri']:.3f} & {r['pi']:.3f} & {fmt_gap(r['gap_uniform'])} & 32 \\\\")
    lines += [
        r"\midrule",
        r"\multicolumn{5}{l}{\emph{Post-training (completion)}} \\",
    ]
    for stage in SMOLLM3_POST_STAGES:
        v = sm3_post.get(stage)
        if v is None:
            lines.append(f"  {stage} & --- & --- & --- & --- \\\\")
        else:
            lines.append(f"  {stage} & {v['ri']:.3f} & {v['pi']:.3f} & {fmt_gap(v['gap_uniform'])} & {v['n_cells']} \\\\")
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
        g_c = fmt_gap(c["gap_uniform"]) if c else "---"
        n_c = str(c["n_cells"]) if c else "---"
        g_h = fmt_gap(ch["gap_uniform"]) if ch else "---"
        n_h = str(ch["n_cells"]) if ch else "---"
        lines.append(f"  {stage} & {g_c} & {n_c} & {g_h} & {n_h} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(out_dir / "tab_smollm3_formats.tex").write_text("\n".join(lines) + "\n")
    print(f"  saved tab_smollm3_formats.tex")


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
                row += f" & {fmt_gap(c['gap'])}\\,{{\\scriptsize($\\pm${hw:.2f})}}"
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
                row += f" & {fmt_gap(c['gap'])}\\,{{\\scriptsize($\\pm${hw:.2f})}}"
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
    make_training_dynamics_figure(OUT_DIR / "fig3_training_dynamics.pdf")
    make_training_dynamics_tables(OUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
