"""
2×2 ablation of Block format = grouping × numbering.

For each cell where all 4 formats {flat_nolabel, flat_verbose, block,
landmark} are present, compute:

  Plain        = flat_nolabel    (no grouping, no numbering)
  Labeled      = flat_verbose    (no grouping, with numbering)
  Landmark     = landmark        (grouping, no numbering)
  Block        = block           (grouping + numbering)

Contrasts (on CVQ accuracy, where the gap lives):
  num_no_group  = Labeled  - Plain      # numbering alone
  num_w_group   = Block    - Landmark   # numbering on top of grouping
  group_no_num  = Landmark - Plain      # grouping alone
  group_w_num   = Block    - Labeled    # grouping on top of numbering
  interaction   = Block    - (Labeled + Landmark - Plain)   # super-additivity

Bootstrap 95% CI on each contrast across cells × models.

Output:
  lora_intervention/results/block_2x2_table.txt
  lora_intervention/results/block_2x2_data.json
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "experiments_cloud" / "results" / "ucurve_proprietary_results.csv"
OUT_TXT = ROOT / "lora_intervention" / "results" / "block_2x2_table.txt"
OUT_JSON = ROOT / "lora_intervention" / "results" / "block_2x2_data.json"

FORMAT_KEYS = ["flat_nolabel", "flat_verbose", "block", "landmark"]
FORMAT_LABELS = {
    "flat_nolabel": "Plain",
    "flat_verbose": "Labeled",
    "block":        "Block",
    "landmark":     "Landmark",
}

N_BOOT = 2000
SEED = 17


def load_data():
    rows = list(csv.DictReader(open(CSV)))
    # (model, K, N) -> {format: {"fvq": acc, "cvq": acc, "n_fvq": n, "n_cvq": n}}
    grid = defaultdict(dict)
    for r in rows:
        key = (r["model"], int(r["num_keys"]), int(r["num_updates"]))
        fmt = r["format"]
        if fmt not in FORMAT_KEYS:
            continue
        slot = grid[key].setdefault(fmt, {})
        if r["position"] == "1":
            slot["fvq"] = float(r["accuracy"])
            slot["n_fvq"] = int(r["n_trials"])
        elif r["position"] == "last":
            slot["cvq"] = float(r["accuracy"])
            slot["n_cvq"] = int(r["n_trials"])
    # Keep only cells with full coverage
    full = {}
    for k, v in grid.items():
        if len(v) != 4:
            continue
        if not all(("fvq" in s and "cvq" in s) for s in v.values()):
            continue
        full[k] = v
    return full


def contrasts(cell, axis):
    """axis in {'fvq', 'cvq'}."""
    p = cell["flat_nolabel"][axis]
    lab = cell["flat_verbose"][axis]
    lmk = cell["landmark"][axis]
    blk = cell["block"][axis]
    return {
        "Plain":          p,
        "Labeled":        lab,
        "Landmark":       lmk,
        "Block":          blk,
        "num_no_group":   lab - p,
        "num_w_group":    blk - lmk,
        "group_no_num":   lmk - p,
        "group_w_num":    blk - lab,
        "interaction":    blk - (lab + lmk - p),
    }


def boot_ci(values, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idxs = rng.integers(0, n, size=(n_boot, n))
    samples = values[idxs].mean(axis=1)
    point = values.mean()
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def main():
    full = load_data()
    print(f"Loaded {len(full)} (model, K, N) cells with full 2×2 coverage.")
    print(f"Unique models: {sorted(set(k[0] for k in full))}")
    print()

    out_lines = []
    out_lines.append("=" * 100)
    out_lines.append("Block format 2×2 ablation — grouping × numbering")
    out_lines.append(f"Data: {CSV}")
    out_lines.append(f"Cells: {len(full)} unique (model, K, N) triples; n=200 trials per cell per format")
    out_lines.append(f"Bootstrap n={N_BOOT}, seed={SEED}")
    out_lines.append("=" * 100)
    out_lines.append("")

    # Pool all cells across models for the headline contrasts
    contrast_names = ["num_no_group", "num_w_group", "group_no_num", "group_w_num", "interaction"]
    contrast_pretty = {
        "num_no_group":  "Numbering effect (no grouping)    : Labeled  − Plain",
        "num_w_group":   "Numbering effect (with grouping)  : Block    − Landmark",
        "group_no_num":  "Grouping effect  (no numbering)   : Landmark − Plain",
        "group_w_num":   "Grouping effect  (with numbering) : Block    − Labeled",
        "interaction":   "Super-additivity (interaction)    : Block − (Labeled + Landmark − Plain)",
    }

    for axis_pretty, axis in [("FVQ (first value)", "fvq"), ("CVQ (last value)", "cvq")]:
        out_lines.append("─" * 100)
        out_lines.append(f"AXIS: {axis_pretty}")
        out_lines.append("─" * 100)

        # Per-format pooled means
        per_fmt = {fmt: [] for fmt in FORMAT_KEYS}
        per_contrast = {n: [] for n in contrast_names}
        for key, cell in full.items():
            con = contrasts(cell, axis)
            for fmt in FORMAT_KEYS:
                per_fmt[fmt].append(con[FORMAT_LABELS[fmt]])
            for n in contrast_names:
                per_contrast[n].append(con[n])

        out_lines.append(f"  {'Format':>10}  {'mean acc [95% CI]':>25}")
        for fmt in FORMAT_KEYS:
            point, lo, hi = boot_ci(per_fmt[fmt])
            out_lines.append(f"  {FORMAT_LABELS[fmt]:>10}  {point:.3f} [{lo:.3f}, {hi:.3f}]")

        out_lines.append("")
        out_lines.append(f"  {'Contrast':>50}  {'Δ [95% CI]':>22}  {'sig?':>6}")
        for n in contrast_names:
            point, lo, hi = boot_ci(per_contrast[n])
            sig = "YES" if (lo > 0 or hi < 0) else " no"
            out_lines.append(f"  {contrast_pretty[n]:>50}  {point:+.3f} [{lo:+.3f}, {hi:+.3f}]  {sig:>6}")
        out_lines.append("")

    # Per-model breakdown (just for CVQ — the headline metric)
    out_lines.append("─" * 100)
    out_lines.append("PER-MODEL BREAKDOWN — CVQ axis")
    out_lines.append("─" * 100)
    by_model = defaultdict(lambda: {n: [] for n in contrast_names})
    by_model_fmt = defaultdict(lambda: {fmt: [] for fmt in FORMAT_KEYS})
    for (model, K, N), cell in full.items():
        con = contrasts(cell, "cvq")
        for n in contrast_names:
            by_model[model][n].append(con[n])
        for fmt in FORMAT_KEYS:
            by_model_fmt[model][fmt].append(con[FORMAT_LABELS[fmt]])

    for model in sorted(by_model):
        out_lines.append(f"\n  {model}  (n cells = {len(by_model[model]['interaction'])})")
        out_lines.append(f"    {'Format':>10}  {'mean CVQ':>9}")
        for fmt in FORMAT_KEYS:
            m = np.mean(by_model_fmt[model][fmt])
            out_lines.append(f"    {FORMAT_LABELS[fmt]:>10}  {m:.3f}")
        out_lines.append(f"    {'Contrast':>50}  {'Δ':>8}")
        for n in contrast_names:
            m = np.mean(by_model[model][n])
            out_lines.append(f"    {contrast_pretty[n]:>50}  {m:+.3f}")

    # Per-cell breakdown for the hardest cell (largest N or K) per model
    out_lines.append("")
    out_lines.append("─" * 100)
    out_lines.append("PER-CELL BREAKDOWN — CVQ axis")
    out_lines.append("─" * 100)
    out_lines.append(f"  {'model':<25}  {'K':>3}  {'N':>3}  {'Plain':>7} {'Lab':>7} {'Lmk':>7} {'Blk':>7}  {'Lab-P':>7} {'Lmk-P':>7} {'Blk-Lab':>7} {'Blk-Lmk':>8} {'Inter':>7}")
    for (model, K, N), cell in sorted(full.items()):
        con = contrasts(cell, "cvq")
        out_lines.append(
            f"  {model:<25}  {K:>3}  {N:>3}  "
            f"{con['Plain']:.3f}  {con['Labeled']:.3f}  {con['Landmark']:.3f}  {con['Block']:.3f}  "
            f"{con['num_no_group']:+.3f}  {con['group_no_num']:+.3f}  {con['group_w_num']:+.3f}  "
            f"{con['num_w_group']:+.4f}  {con['interaction']:+.3f}"
        )

    out_txt = "\n".join(out_lines)
    print(out_txt)
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text(out_txt)
    print(f"\n✓ Saved table → {OUT_TXT}")

    # Save data
    serializable = {
        "cells": {f"{m}__{K}_{N}": cell for (m, K, N), cell in full.items()},
        "n_cells": len(full),
        "models": sorted(set(k[0] for k in full)),
        "n_boot": N_BOOT,
        "seed": SEED,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"✓ Saved data → {OUT_JSON}")


if __name__ == "__main__":
    main()
