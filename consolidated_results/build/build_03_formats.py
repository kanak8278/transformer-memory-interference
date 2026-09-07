"""Theme 3 — prompt formatting of the key/value updates.

Formats compared (names as they appear in `prompt_format`):
  flat_nolabel  "plain" running text, no per-update label
  flat_verbose  same text, each update explicitly labelled
  block         updates grouped into per-key blocks
  landmark      explicit positional landmarks interleaved with updates
  plain         the LoRA experiments' unformatted prompt (NOT byte-identical to
                flat_nolabel — different experiment, different template)
  chat_template / completion / chat_1024   chat-vs-completion wrapping

Deliberate overlap: the U-curve sweep rows and the block-vs-LoRA per-position
rows also appear in theme 02. They are the same measurements viewed along a
different axis (format here, position there); each theme is kept self-sufficient
so neither needs a join.

Sources
  RAW      experiments_cloud/results/ucurve_proprietary_results.csv
  RAW      lora_intervention/experiments/*/behavioral*/results.json  (plain vs block)
  RAW      lora_intervention/experiments/*/block*/results_K*.json    (behavioural acc)
  DERIVED  paper/figures/tab_format_ci.tex     open-weight rows only
  DERIVED  paper/figures/tab_smollm3_formats.tex
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from common import (DERIVED, ORDINAL, RAW, ROOT, SEMANTIC, cond_from_ri_pi,
                    canon_model, emit, model_meta, nums, pos_key, pct, rel,
                    tex_body_rows, with_model)

THEME = "03_prompt_format"
OUT = Path(__file__).resolve().parent.parent / THEME

QWEN = "Qwen2.5-3B-Instruct"
GEMMA = "gemma-3-4b-it"

# tab_format*.tex column header -> prompt_format id
TEX_FORMATS = ["flat_nolabel", "flat_verbose", "block", "landmark"]

BLOCK_CONDS = {
    "base_plain": ("base", "plain"),
    "base_block": ("base", "block"),
    "lora_plain": ("lora", "plain"),
}


def classify(position, num_updates: int) -> str:
    if position == "last" or int(position) == num_updates:
        return "CVQ"
    return "FVQ" if int(position) == 1 else "IVQ"


# ── RAW: proprietary format U-curve (all formats, all positions) ────────────

def from_ucurve() -> list[dict]:
    path = ROOT / "experiments_cloud" / "results" / "ucurve_proprietary_results.csv"
    src = rel(path)
    rows = []
    for r in csv.DictReader(open(path)):
        n = int(r["num_updates"])
        pos = r["position"]
        rows.append(with_model(
            r["model"], theme=THEME, variant="base",
            # SEMANTIC_MULTI, not arbitrary: run_ucurve_proprietary.sh passes
            # `--dataset SEMANTIC_MULTI` for all six models.
            dataset="semantic_multi", prompt_format=r["format"],
            num_keys=int(r["num_keys"]), num_updates=n,
            condition=classify(pos, n),
            query_type=SEMANTIC if pos == "last" else ORDINAL,
            position="last" if pos == "last" else int(pos),
            accuracy=float(r["accuracy"]), ci_lower=float(r["ci_lower"]),
            ci_upper=float(r["ci_upper"]), wilson_hw=float(r["wilson_hw"]),
            n_trials=int(r["n_trials"]), provenance_tier=RAW, source_file=src,
            notes=f"upstream source_file: {r['source_file']}",
        ))
    return rows


# ── RAW: plain vs block behavioural sweeps ──────────────────────────────────

def from_behavioral(path: Path, model: str) -> list[dict]:
    d = json.load(open(path))
    src = rel(path)
    rows = []
    for cell_key, conds in d.items():
        for cond_key, cv in conds.items():
            variant, fmt = BLOCK_CONDS[cond_key]
            n = cv["updates"]
            for p in cv["positions"]:
                rows.append(with_model(
                    model, theme=THEME, variant=variant,
                    dataset="arbitrary_single", prompt_format=fmt,
                    num_keys=cv["keys"], num_updates=n,
                    condition=classify(p["k"], n),
                    query_type=(ORDINAL if fmt == "block"
                                else SEMANTIC if p["k"] == n else ORDINAL),
                    position=p["k"],
                    accuracy=p["acc"], ci_lower=p.get("ci_lo"),
                    ci_upper=p.get("ci_hi"), wilson_hw=p.get("wilson_hw"),
                    n_trials=p["n"], n_correct=p.get("correct"),
                    provenance_tier=RAW, source_file=src,
                    notes=f"cell {cell_key}; condition key {cond_key}",
                ))
    return rows


# ── RAW: block-readout behavioural accuracies ───────────────────────────────

def from_block_readout(paths: list[Path], model: str) -> list[dict]:
    """Behavioural accuracy per condition; the 36-layer logit-lens curves in
    these files are mechanistic and stay in the source."""
    rows = []
    for path in paths:
        if not path.exists():
            continue
        d = json.load(open(path))
        src = rel(path)
        k, n = d["cell"]["keys"], d["cell"]["updates"]
        for cond_key, conds in d["conditions"].items():
            variant, fmt = BLOCK_CONDS[cond_key]
            for ri_pi, cv in conds.items():
                cond = cond_from_ri_pi(ri_pi)
                rows.append(with_model(
                    model, theme=THEME, variant=variant,
                    dataset="arbitrary_single", prompt_format=fmt,
                    num_keys=k, num_updates=n, condition=cond,
                    query_type=ORDINAL if fmt == "block" else SEMANTIC,
                    position="first" if cond == "FVQ" else "last",
                    accuracy=cv["behavioral_acc"], n_trials=cv["n"],
                    provenance_tier=RAW, source_file=src,
                    notes="behavioural accuracy only; per-layer P(v_last) / "
                          "P(v_first) curves remain in the source file",
                ))
    return rows


# ── DERIVED: open-weight format table (raw ucurve_vllm/ is GPU-box only) ────

def from_tab_format_ci() -> tuple[list[dict], list[tuple]]:
    """Returns (open-weight rows to emit, proprietary rows for QA)."""
    path = ROOT / "paper" / "figures" / "tab_format_ci.tex"
    src = rel(path)
    rows, qa = [], []
    for cells in tex_body_rows(path):
        model = canon_model(cells[0])
        _, _, is_prop = model_meta(model)
        for fmt, cell in zip(TEX_FORMATS, cells[1:5]):
            v = nums(cell)
            if len(v) < 4:
                continue
            fvq, cvq, fvq_hw, cvq_hw = v[0], v[1], v[2], v[3]
            if is_prop:
                # Same operating point exists raw in the U-curve CSV.
                qa.append((model, fmt, fvq, cvq))
                continue
            common = dict(
                theme=THEME, variant="base",
                # Semantic-Multi: paper/main.tex caption for tab:format reads
                # "Format-intervention results on Semantic-Multi (K=10, N=50)".
                dataset="semantic_multi",
                prompt_format=fmt, num_keys=10, num_updates=50,
                provenance_tier=DERIVED, source_file=src,
                notes="back-extracted from committed paper table; raw "
                      "experiments_cloud/results/ucurve_vllm/ not local",
            )
            rows.append(with_model(model, condition="FVQ", query_type=ORDINAL,
                                   position="first", accuracy=fvq,
                                   wilson_hw=fvq_hw, **common))
            rows.append(with_model(model, condition="CVQ", query_type=SEMANTIC,
                                   position="last", accuracy=cvq,
                                   wilson_hw=cvq_hw, **common))
    return rows, qa


def qa_tex_vs_ucurve(qa: list[tuple], ucurve_rows: list[dict]) -> None:
    """The proprietary rows of tab_format_ci restate raw U-curve numbers."""
    raw = {}
    for r in ucurve_rows:
        if int(r["num_keys"]) == 10 and int(r["num_updates"]) == 50:
            raw[(r["model"], r["prompt_format"], r["condition"])] = \
                float(r["accuracy"])
    checked = bad = 0
    for model, fmt, fvq, cvq in qa:
        for cond, want in (("FVQ", fvq), ("CVQ", cvq)):
            got = raw.get((model, fmt, cond))
            if got is None:
                continue
            checked += 1
            if abs(got - want) > 0.015:
                bad += 1
                print(f"     ! QA mismatch {model}/{fmt}/{cond}: "
                      f"raw {got:.2f} vs tab_format_ci {want:.2f}")
    print(f"     QA tab_format_ci.tex vs raw U-curve: {checked} values "
          f"checked, {bad} mismatched")


# ── DERIVED: SmolLM3 completion-vs-chat across post-training stages ─────────

def from_tab_smollm3_formats() -> list[dict]:
    path = ROOT / "paper" / "figures" / "tab_smollm3_formats.tex"
    src = rel(path)
    rows = []
    for cells in tex_body_rows(path):
        stage = cells[0].strip()
        if not stage:
            continue
        for fmt, gap_cell, cells_cell in (("completion", cells[1], cells[2]),
                                          ("chat_1024", cells[3], cells[4])):
            g = nums(gap_cell)
            n_cells = nums(cells_cell)
            if not g:
                continue
            rows.append(with_model(
                "SmolLM3-3B", theme=THEME, variant=stage,
                dataset="arbitrary_single", prompt_format=fmt,
                condition="FVQ-CVQ", query_type=SEMANTIC, gap=g[0],
                provenance_tier=DERIVED, source_file=src,
                notes=f"post-training stage {stage}; mean gap over "
                      f"{int(n_cells[0]) if n_cells else '?'} (K,N) cells "
                      "(K in 2,3,5,7 x N in 2,3,5,7,10,15,20,30; 100 trials/cell, "
                      "per paper App. training); raw "
                      "v3/results_vllm/training_dynamics_smollm3/ not local",
            ))
    return rows


def main() -> None:
    print(f"[{THEME}] prompt formatting of key/value updates")
    exp = ROOT / "lora_intervention" / "experiments"
    ucurve = from_ucurve()
    tex_rows, qa = from_tab_format_ci()
    rows = (
        ucurve
        + tex_rows
        + from_behavioral(exp / "behavioral_block_lora_results" / "results.json", QWEN)
        + from_behavioral(exp / "gemma_results" / "behavioral" / "results.json", GEMMA)
        + from_block_readout([exp / "block_readout_results" / "results_K2N30.json",
                              exp / "block_readout_results" / "results_K10N50.json"], QWEN)
        + from_block_readout([exp / "gemma_results" / "block" / "results_K2N30.json",
                              exp / "gemma_results" / "block" / "results_K10N50.json"], GEMMA)
        + from_tab_smollm3_formats()
    )
    rows.sort(key=lambda r: (
        str(r["is_proprietary"]), r["model"], r["prompt_format"], r["variant"],
        int(r["num_keys"] or 0), int(r["num_updates"] or 0),
        pos_key(r["position"])))
    emit(rows, OUT, "format.csv")
    qa_tex_vs_ucurve(qa, ucurve)


if __name__ == "__main__":
    main()
