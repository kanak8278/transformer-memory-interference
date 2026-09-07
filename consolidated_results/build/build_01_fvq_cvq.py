"""Theme 1 — First-value (FVQ) vs current/last-value (CVQ) retrieval.

Every (model, dataset, K, N) cell we have, as two rows (FVQ and CVQ), plus
gap-only rows where the raw data is gone and only the paper's .tex table
survives.

Sources
  RAW      v3/results_vllm/arbitrary_single/*/stage1_sweep_*.json
             15 open-weight models x up to 79 (K,N) cells, ARBITRARY_SINGLE
  RAW      experiments_cloud/results/proprietary_semantic_multi.csv
             6 proprietary models x 69 (K,N) cells, SEMANTIC_MULTI
  DERIVED  paper/figures/tab2_openweight.tex
             open-weight SEMANTIC_MULTI at K=5,N=30 (raw sweep is GPU-box only)
  DERIVED  paper/figures/tab_full_sem.tex
             open-weight SEMANTIC_MULTI gaps at 4 representative cells

paper/figures/tab_full_arb.tex is NOT emitted: it restates the arbitrary_single
sweeps we already read raw. It is used as a cross-check instead (see QA below).
"""

from __future__ import annotations

import csv
import glob
import json
import re
from pathlib import Path

from common import (DERIVED, RAW, ROOT, SEMANTIC, cond_from_ri_pi, canon_model,
                    emit, nums, offstream, rel, tex_body_rows, with_model)

THEME = "01_fvq_cvq"
OUT = Path(__file__).resolve().parent.parent / THEME

# Column headers of the two 4-cell .tex tables, in order.
FULL_CELLS = [(2, 10), (5, 30), (10, 50), (20, 50)]

_SIZE_SUFFIX = re.compile(r"\s*\(\d+(?:\.\d+)?\s*[BbMm]\)\s*$")


def _label_to_model(label: str) -> str:
    """'Qwen3.5-4B (4B)' -> 'Qwen3.5-4B'; '(base)' suffixes are preserved."""
    return canon_model(_SIZE_SUFFIX.sub("", label).strip())


# ── RAW: open-weight arbitrary_single stage-1 sweeps ─────────────────────────

def from_stage1_sweeps() -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(ROOT / "v3" / "results_vllm" /
                                     "arbitrary_single" / "*" /
                                     "stage1_sweep_*.json"))):
        d = json.load(open(path))
        src = rel(path)
        fmt = d.get("prompt_format", "")
        ds = d.get("dataset_type", "ARBITRARY_SINGLE").lower()
        for cell in d["cells"].values():
            for ri_pi, st in cell["stats"].items():
                cond = cond_from_ri_pi(ri_pi)
                n_off, acc_on, et_json = offstream(st)
                rows.append(with_model(
                    d["model"],
                    theme=THEME, variant="base", dataset=ds, prompt_format=fmt,
                    num_keys=cell["num_keys"], num_updates=cell["num_updates"],
                    condition=cond, query_type=SEMANTIC,
                    position="first" if cond == "FVQ" else "last",
                    accuracy=st["accuracy"], ci_lower=st.get("ci_lower"),
                    ci_upper=st.get("ci_upper"), n_trials=st.get("n"),
                    n_offstream=n_off, accuracy_onstream=acc_on,
                    error_types_json=et_json,
                    regime=cell.get("regime", ""),
                    provenance_tier=RAW, source_file=src,
                ))
    return rows


# ── RAW: proprietary semantic_multi sweep ────────────────────────────────────

def from_proprietary_csv() -> list[dict]:
    path = ROOT / "experiments_cloud" / "results" / "proprietary_semantic_multi.csv"
    src = rel(path)
    rows = []
    for r in csv.DictReader(open(path)):
        common = dict(
            theme=THEME, variant="base", dataset="semantic_multi",
            num_keys=int(r["num_keys"]), num_updates=int(r["num_updates"]),
            n_trials=int(r["n_trials"]), provenance_tier=RAW,
            source_file=src,
            notes=f"upstream source_file: {r['source_file']}",
        )
        rows.append(with_model(
            r["model"], condition="FVQ", query_type=SEMANTIC, position="first",
            accuracy=float(r["fvq_acc"]), ci_lower=float(r["fvq_ci_lower"]),
            ci_upper=float(r["fvq_ci_upper"]), **common))
        rows.append(with_model(
            r["model"], condition="CVQ", query_type=SEMANTIC, position="last",
            accuracy=float(r["cvq_acc"]), ci_lower=float(r["cvq_ci_lower"]),
            ci_upper=float(r["cvq_ci_upper"]), **common))
    return rows


# ── DERIVED: open-weight semantic_multi, back-extracted from the paper ───────

def from_tab2_openweight() -> list[dict]:
    """Model & Family & FVQ(+-hw) & CVQ(+-hw) & Gap, all at K=5, N=30."""
    path = ROOT / "paper" / "figures" / "tab2_openweight.tex"
    src = rel(path)
    rows = []
    for cells in tex_body_rows(path):
        model = _label_to_model(cells[0])
        fvq, fvq_hw = nums(cells[2])[:2]
        cvq, cvq_hw = nums(cells[3])[:2]
        common = dict(
            theme=THEME, variant="base", dataset="semantic_multi",
            num_keys=5, num_updates=30, provenance_tier=DERIVED,
            source_file=src,
            notes="back-extracted from committed paper table; "
                  "raw sweep v3/results_vllm/semantic_multi/ not local",
        )
        rows.append(with_model(model, condition="FVQ", query_type=SEMANTIC,
                               position="first", accuracy=fvq,
                               wilson_hw=fvq_hw, **common))
        rows.append(with_model(model, condition="CVQ", query_type=SEMANTIC,
                               position="last", accuracy=cvq,
                               wilson_hw=cvq_hw, **common))
    return rows


def from_tab_full_sem() -> list[dict]:
    """Model & gap(+-hw) x 4 cells. Only the gap survives in this table."""
    path = ROOT / "paper" / "figures" / "tab_full_sem.tex"
    src = rel(path)
    rows = []
    for cells in tex_body_rows(path):
        model = _label_to_model(cells[0])
        for (k, n), cell in zip(FULL_CELLS, cells[1:5]):
            vals = nums(cell)
            if not vals:          # '---' = cell not run for this model
                continue
            gap, hw = vals[0], (vals[1] if len(vals) > 1 else None)
            rows.append(with_model(
                model, theme=THEME, variant="base", dataset="semantic_multi",
                num_keys=k, num_updates=n, condition="FVQ-CVQ",
                query_type=SEMANTIC, gap=gap, wilson_hw=hw, provenance_tier=DERIVED, source_file=src,
                notes="gap only — per-condition accuracies not preserved in "
                      "the .tex; raw sweep not local",
            ))
    return rows


# ── QA: tab_full_arb.tex restates data we hold raw — check they agree ────────

def qa_against_tab_full_arb(raw_rows: list[dict]) -> None:
    path = ROOT / "paper" / "figures" / "tab_full_arb.tex"
    if not path.exists():
        return
    acc = {}
    for r in raw_rows:
        if r["dataset"] != "arbitrary_single":
            continue
        acc[(r["model"], r["num_keys"], r["num_updates"], r["condition"])] = \
            float(r["accuracy"])

    checked = mismatched = 0
    for cells in tex_body_rows(path):
        try:
            model = _label_to_model(cells[0])
        except KeyError:
            continue
        for (k, n), cell in zip(FULL_CELLS, cells[1:5]):
            vals = nums(cell)
            if not vals:
                continue
            f = acc.get((model, k, n, "FVQ"))
            c = acc.get((model, k, n, "CVQ"))
            if f is None or c is None:
                continue
            checked += 1
            if abs((f - c) - vals[0]) > 0.015:
                mismatched += 1
                print(f"     ! QA gap mismatch {model} K={k} N={n}: "
                      f"raw {f - c:+.2f} vs tab_full_arb {vals[0]:+.2f}")
    print(f"     QA vs tab_full_arb.tex: {checked} cells checked, "
          f"{mismatched} mismatched")


def main() -> None:
    print(f"[{THEME}] first-value vs last-value retrieval")
    raw_arb = from_stage1_sweeps()
    rows = raw_arb + from_proprietary_csv() + from_tab2_openweight() \
        + from_tab_full_sem()
    rows.sort(key=lambda r: (str(r["is_proprietary"]), r["model"],
                             r["dataset"], int(r["num_keys"] or 0),
                             int(r["num_updates"] or 0), r["condition"]))
    emit(rows, OUT, "fvq_cvq.csv")
    qa_against_tab_full_arb(raw_arb)


if __name__ == "__main__":
    main()
