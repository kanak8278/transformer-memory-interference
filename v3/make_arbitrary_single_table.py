"""
Generate per-cell tables for v3/results_vllm/arbitrary_single.
Outputs:
  - arbitrary_single_table.csv  (machine-readable, all models)
  - arbitrary_single_table.md   (markdown, grouped by model)
"""

import json
import os
import glob
import csv
from pathlib import Path

BASE = Path(__file__).parent / "results_vllm" / "arbitrary_single"
OUT_CSV = Path(__file__).parent / "arbitrary_single_table.csv"
OUT_MD  = Path(__file__).parent / "arbitrary_single_table.md"

# ── collect all rows ──────────────────────────────────────────────────────────

rows = []

for model_dir in sorted(BASE.iterdir()):
    if not model_dir.is_dir():
        continue
    sweeps = sorted(model_dir.glob("stage1_sweep_*.json"))
    if not sweeps:
        continue
    sweep_file = sweeps[-1]          # take the latest if somehow multiple

    with open(sweep_file) as f:
        d = json.load(f)

    model_full = d["model"]
    model_short = model_dir.name
    trials_per_cell = d["config"]["trials_per_cell"]
    cells = d["cells"]

    for cell_key, cell in sorted(
        cells.items(),
        key=lambda kv: (int(kv[0].split("_")[0]), int(kv[0].split("_")[1]))
    ):
        num_keys    = cell["num_keys"]
        num_updates = cell["num_updates"]
        n_trials    = cell.get("n_trials", trials_per_cell)
        regime      = cell.get("regime", "")

        ri = cell["stats"]["RI"]
        pi = cell["stats"]["PI"]

        rows.append({
            "model_short":    model_short,
            "model_full":     model_full,
            "num_keys":       num_keys,
            "num_updates":    num_updates,
            "n_trials":       n_trials,
            "regime":         regime,
            "RI_acc":         ri["accuracy"],
            "RI_ci_lower":    ri["ci_lower"],
            "RI_ci_upper":    ri["ci_upper"],
            "RI_n_failures":  ri["n_failures"],
            "RI_n_garbage":   ri.get("n_garbage", ""),
            "PI_acc":         pi["accuracy"],
            "PI_ci_lower":    pi["ci_lower"],
            "PI_ci_upper":    pi["ci_upper"],
            "PI_n_failures":  pi["n_failures"],
            "PI_n_garbage":   pi.get("n_garbage", ""),
            "gap_PI_minus_RI": round(pi["accuracy"] - ri["accuracy"], 4),
        })

# ── write CSV ─────────────────────────────────────────────────────────────────

fieldnames = list(rows[0].keys())

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows → {OUT_CSV}")

# ── write Markdown ────────────────────────────────────────────────────────────

def fmt_acc(v, lo, hi):
    return f"{v:.2f} [{lo:.2f}–{hi:.2f}]"

header = (
    "| keys | updates | trials | regime "
    "| RI acc [95% CI] | PI acc [95% CI] "
    "| RI fail | PI fail "
    "| gap (PI−RI) |\n"
    "|-----:|--------:|-------:|:------:"
    "|:----------------|:---------------|"
    "|--------:|--------:"
    "|------------:|"
)

# group by model
from itertools import groupby
from operator import itemgetter

lines = ["# Arbitrary Single — Per-Cell Results\n",
         f"**{len(rows)} total cells across {len(set(r['model_short'] for r in rows))} models**\n"]

for model_short, group in groupby(rows, key=itemgetter("model_short")):
    group = list(group)
    model_full = group[0]["model_full"]
    lines.append(f"\n## {model_short}\n")
    lines.append(f"*Full model id: `{model_full}`*\n")
    lines.append(header)
    for r in group:
        gap_str = f"{r['gap_PI_minus_RI']:+.2f}"
        ri_str = fmt_acc(r["RI_acc"], r["RI_ci_lower"], r["RI_ci_upper"])
        pi_str = fmt_acc(r["PI_acc"], r["PI_ci_lower"], r["PI_ci_upper"])
        lines.append(
            f"| {r['num_keys']:4d} | {r['num_updates']:7d} | {r['n_trials']:6d} | {r['regime']:6} "
            f"| {ri_str:<22} | {pi_str:<22} "
            f"| {r['RI_n_failures']:7} | {r['PI_n_failures']:7} "
            f"| {gap_str:>11} |"
        )

with open(OUT_MD, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"Wrote markdown → {OUT_MD}")

# ── quick terminal preview ────────────────────────────────────────────────────

print("\n── PREVIEW (first 5 rows per model) ──────────────────────────────────────")
for model_short, group in groupby(rows, key=itemgetter("model_short")):
    group = list(group)
    print(f"\n{'='*70}")
    print(f"  {model_short}  ({len(group)} cells)")
    print(f"{'='*70}")
    print(f"  {'keys':>4}  {'upd':>5}  {'n':>5}  {'rgm':3}  "
          f"{'RI':>6}  {'PI':>6}  {'gap':>7}")
    print(f"  {'-'*4}  {'-'*5}  {'-'*5}  {'-'*3}  "
          f"{'-'*6}  {'-'*6}  {'-'*7}")
    for r in group:
        print(f"  {r['num_keys']:4d}  {r['num_updates']:5d}  {r['n_trials']:5d}  "
              f"{r['regime']:3}  "
              f"{r['RI_acc']:6.2f}  {r['PI_acc']:6.2f}  "
              f"{r['gap_PI_minus_RI']:+7.2f}")
