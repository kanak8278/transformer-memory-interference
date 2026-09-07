"""Theme 4 — LoRA fine-tuning: which models, trained on what, evaluated where.

Two outputs:
  adapters.csv    one row per trained adapter (its own config schema, not the
                  measurement schema — these are hyperparameters, not results)
  lora_evals.csv  tidy-long measurements: base vs adapted, per (K,N) cell

Baseline (variant=base) rows on ARBITRARY_SINGLE are taken from the RAW stage-1
sweeps that `evaluate.py` itself compares against (DEFAULT_BASELINE), so they
carry full precision and CIs rather than the integer percents printed in the
*_comparison.txt files. Those .txt files are used as a cross-check instead.

SEMANTIC_MULTI: the Qwen sweep (base + LoRA, 4 cells) IS local — it survives as
lora_sem_validation_results.json, so those rows are RAW and carry per-condition
trial counts. The counts matter: the LoRA arm is n=20 on the first two cells and
n=10 on the last two (that script was killed mid-run and lora_sem_finish.py
re-ran the remainder), which the 2dp paper table cannot show. The paper table is
kept as a cross-check on those 16 values instead of as their source. Only the
Gemma SEM baseline is still back-extracted; its raw sweep is not local.

Sources
  RAW      lora_intervention/checkpoints/*/run_config.json + adapter_config.json
  RAW      lora_intervention/results/{main,gemma_main,gemma_smoke,dense_ivq,
             qwen_arith_control,qwen_arith_control_full}_eval_*.json
  RAW      lora_intervention/results/gsm8k_task_eval_*.json
  RAW      v3/results_vllm/arbitrary_single/<model>/stage1_sweep_*.json (baselines)
  RAW      lora_sem_validation_results.json (Qwen SEM base + LoRA)
  DERIVED  paper/figures/tab_lora_gemma_sem.tex (Gemma SEM baseline)
  QA       lora_intervention/results/*_comparison.txt
  QA       paper/figures/tab_lora_sem.tex (vs the raw Qwen SEM rows)
"""

from __future__ import annotations

import csv
import glob
import json
import re
from pathlib import Path

from common import (COLUMNS, DERIVED, ORDINAL, RAW, ROOT, SEMANTIC,
                    cond_from_ri_pi, emit, model_meta, nums, offstream, pct,
                    pos_key, rel, tex_body_rows, with_model, write_csv)

THEME = "04_lora"
OUT = Path(__file__).resolve().parent.parent / THEME
RESULTS = ROOT / "lora_intervention" / "results"

QWEN = "Qwen2.5-3B-Instruct"
GEMMA = "gemma-3-4b-it"

# The K,N grid the adapters were trained on (lora_intervention/data_gen.py).
TRAIN_KEYS = [2, 3, 5, 10]
TRAIN_UPDATES = [5, 10, 15, 20]

# eval run_name -> (base model, variant label, adapter dir, role)
#
# `variant` must be unique per (model, run): several runs share a base model and
# an adapter but are different evaluations with different numbers, and rows are
# identified downstream by (model, variant, K, N, condition).
RUNS = {
    "main":                    (QWEN,  "lora",                     "adapter",
                                "§5.2 main LoRA (Qwen)"),
    "gemma_main":              (GEMMA, "lora",                     "gemma_adapter",
                                "§5.2 cross-family control (Gemma)"),
    "gemma_smoke":             (GEMMA, "lora_smoke",               "gemma_adapter",
                                "Gemma smoke run — superseded by gemma_main"),
    "dense_ivq":               (QWEN,  "lora_dense_ivq",           None,
                                "dense-IVQ adapter (all positions in training)"),
    "qwen_arith_control":      (QWEN,  "lora_arith_control_5cell", "qwen_arith_adapter",
                                "App G negative control, 5-cell preview"),
    "qwen_arith_control_full": (QWEN,  "lora_arith_control",       "qwen_arith_adapter",
                                "App G negative control, full 28 cells"),
}

ADAPTERS = {
    "adapter":            (QWEN,  "§5.2 main LoRA", True),
    "gemma_adapter":      (GEMMA, "§5.2 cross-family control", True),
    "qwen_arith_adapter": (QWEN,  "App G negative control (GSM8K)", False),
}

ADAPTER_COLUMNS = [
    "adapter_dir", "run_name", "base_model", "model_family", "role_in_paper",
    "trained_on_kv_task", "train_data", "train_grid_keys", "train_grid_updates",
    "rank", "lora_alpha", "lr", "target_modules", "epochs_planned",
    "epochs_completed", "steps_completed", "steps_planned", "batch_size",
    "grad_accum", "max_seq_len", "dtype", "trainable_params", "trainable_pct",
    "stopped_early", "stop_reason", "adapter_bytes", "source_file", "notes",
]


# Per-position IVQ experiments (base-vs-LoRA comparisons). condition key -> (variant, format)
BLOCK_CONDS = {
    "base_plain": ("base", "plain"),
    "base_block": ("base", "block"),
    "lora_plain": ("lora", "plain"),
}


def _ivq_condition(position, num_updates: int) -> str:
    if position == "last" or int(position) == num_updates:
        return "CVQ"
    return "FVQ" if int(position) == 1 else "IVQ"


def _cell(key: str) -> tuple[int, int]:
    k, n = key.split("_")
    return int(k), int(n)


def _latest(pattern: str) -> Path | None:
    hits = sorted(glob.glob(str(RESULTS / pattern)))
    return Path(hits[-1]) if hits else None


# ── adapters.csv ────────────────────────────────────────────────────────────

def build_adapters() -> list[dict]:
    rows = []
    for adir, (base, role, on_task) in ADAPTERS.items():
        d = ROOT / "lora_intervention" / "checkpoints" / adir
        rc = json.load(open(d / "run_config.json"))
        ac_path = d / "adapter_config.json"
        ac = json.load(open(ac_path)) if ac_path.exists() else {}
        weights = d / "adapter_model.safetensors"
        _, fam, _ = model_meta(base)
        tm = rc.get("target_modules") or ac.get("target_modules") or []
        rows.append({
            "adapter_dir": rel(d),
            "run_name": rc.get("run_name", ""),
            "base_model": base,
            "model_family": fam,
            "role_in_paper": role,
            "trained_on_kv_task": on_task,
            "train_data": rc.get("train_file", ""),
            "train_grid_keys": ",".join(map(str, TRAIN_KEYS)) if on_task else "",
            "train_grid_updates": ",".join(map(str, TRAIN_UPDATES)) if on_task else "",
            "rank": rc.get("rank", ac.get("r", "")),
            "lora_alpha": rc.get("lora_alpha", ac.get("lora_alpha", "")),
            "lr": rc.get("lr", ""),
            "target_modules": ",".join(sorted(tm)),
            "epochs_planned": rc.get("epochs_planned", rc.get("epochs", "")),
            "epochs_completed": rc.get("epochs_completed", rc.get("epochs", "")),
            "steps_completed": rc.get("steps_completed", ""),
            "steps_planned": rc.get("steps_planned", ""),
            "batch_size": rc.get("batch_size", ""),
            "grad_accum": rc.get("grad_accum", ""),
            "max_seq_len": rc.get("max_seq_len", ""),
            "dtype": rc.get("dtype", ""),
            "trainable_params": rc.get("trainable_params", ""),
            "trainable_pct": rc.get("trainable_pct", ""),
            "stopped_early": rc.get("stopped_early", False),
            "stop_reason": rc.get("stop_reason", ""),
            "adapter_bytes": weights.stat().st_size if weights.exists() else "",
            "source_file": rel(d / "run_config.json"),
            "notes": ("K,N grid is not recorded in run_config.json — it comes "
                      "from lora_intervention/data_gen.py; the referenced "
                      "train/val .jsonl are not local but regenerate "
                      "deterministically" if on_task else
                      rc.get("deviation_from_PLAN_MD", "")),
        })
    return rows


# ── Baselines from the raw stage-1 sweeps ───────────────────────────────────

def load_baseline(model: str) -> tuple[dict, str, str]:
    """(cells, source_path, prompt_format).

    prompt_format is read from the sweep, not assumed: the stage-1 runs used
    chat_template for instruct models and completion_few_shot for base models,
    and mislabelling it would hide a real confound.
    """
    hits = sorted(glob.glob(str(ROOT / "v3" / "results_vllm" /
                                "arbitrary_single" / model / "stage1_sweep_*.json")))
    d = json.load(open(hits[-1]))
    return d["cells"], rel(hits[-1]), d.get("prompt_format", "")


def baseline_rows(model: str, cells_needed: set[str], run: str) -> list[dict]:
    cells, src, fmt = load_baseline(model)
    rows = []
    for key in sorted(cells_needed, key=_cell):
        cell = cells.get(key)
        if cell is None:
            continue                     # sweep never covered this cell
        k, n = _cell(key)
        for ri_pi, st in cell["stats"].items():
            cond = cond_from_ri_pi(ri_pi)
            n_off, acc_on, et_json = offstream(st)
            rows.append(with_model(
                model, theme=THEME, variant="base", dataset="arbitrary_single",
                prompt_format=fmt, num_keys=k, num_updates=n,
                condition=cond, query_type=SEMANTIC,
                position="first" if cond == "FVQ" else "last",
                accuracy=st["accuracy"], ci_lower=st.get("ci_lower"),
                ci_upper=st.get("ci_upper"), n_trials=st.get("n"),
                n_offstream=n_off, accuracy_onstream=acc_on,
                error_types_json=et_json,
                regime=cell.get("regime", ""), split="held_out",
                provenance_tier=RAW, source_file=src,
                notes=f"baseline paired with eval run '{run}'",
            ))
    return rows


# ── LoRA eval JSONs ─────────────────────────────────────────────────────────

def from_eval_jsons() -> tuple[list[dict], dict[str, set[str]]]:
    rows = []
    arb_cells: dict[str, set[str]] = {}
    for run, (model, variant, _adir, role) in RUNS.items():
        path = _latest(f"{run}_eval_*.json")
        if path is None:
            print(f"     (no eval json for run '{run}')")
            continue
        d = json.load(open(path))
        src = rel(path)
        for ds, cells in d["results"].items():
            for key, cell in cells.items():
                k, n = _cell(key)
                if ds == "ARBITRARY_SINGLE":
                    arb_cells.setdefault(model, set()).add(key)
                for ri_pi, st in cell["stats"].items():
                    cond = cond_from_ri_pi(ri_pi)
                    rows.append(with_model(
                        model, theme=THEME, variant=variant, dataset=ds.lower(),
                        prompt_format="chat_template", num_keys=k, num_updates=n,
                        condition=cond, query_type=SEMANTIC,
                        position="first" if cond == "FVQ" else "last",
                        accuracy=st["accuracy"], ci_lower=st.get("ci_lower"),
                        ci_upper=st.get("ci_upper"), n_trials=st.get("n"),
                        # eval JSONs record accuracy only — no output taxonomy
                        regime=cell.get("regime", ""), split="held_out",
                        provenance_tier=RAW, source_file=src,
                        notes=f"run '{run}' — {role}; adapter "
                              f"{d.get('adapter', '?')}"
                              + ("; early-stopped" if cell.get("stopped_early")
                                 else ""),
                    ))
    return rows, arb_cells


# ── RAW: Qwen SEM base + LoRA, from the run that survived on disk ───────────

SEM_RAW = ROOT / "lora_sem_validation_results.json"
# Preserved copy. The canonical file is a loose JSON at the repo root, which is
# exactly the kind of thing a cleanup deletes; keeping a byte-identical copy
# inside the folder means these 16 rows stay rebuildable and auditable here.
# The canonical path is still what source_file reports.
SEM_RAW_COPY = OUT / "sources" / "lora_sem_validation_results.json"


def from_lora_sem_raw(model: str) -> list[dict]:
    """Qwen SEMANTIC_MULTI, 4 cells x {base, lora} x {FVQ, CVQ}.

    lora_sem_validation.py was killed after writing its log but before writing
    JSON; the numbers were recovered from that log and the two unfinished LoRA
    cells re-run by lora_sem_finish.py. This file is the merged result and is
    the only place the per-condition trial counts exist -- the LoRA arm is n=20
    on (7,20) and (15,20) but n=10 on (15,30) and (20,30).
    """
    path = SEM_RAW if SEM_RAW.exists() else SEM_RAW_COPY
    if not path.exists():
        return []
    if path is SEM_RAW_COPY:
        print(f"     NOTE {rel(SEM_RAW)} is gone; rebuilt from the copy in "
              f"{rel(SEM_RAW_COPY.parent)}")
    d = json.load(open(path))
    # Report the canonical location either way, so provenance names the origin
    # of the measurement rather than wherever it was read from today.
    src = rel(SEM_RAW)
    rows = []
    for variant, key in (("base", "baseline"), ("lora", "lora")):
        for cell, st in d[key].items():
            k, n = st["num_keys"], st["num_updates"]
            origin = st.get("source", "lora_sem_validation.py run")
            common = dict(
                theme=THEME, variant=variant, dataset="semantic_multi",
                prompt_format="chat_template", num_keys=k, num_updates=n,
                query_type=SEMANTIC, split="ood", provenance_tier=RAW,
                source_file=src,
                notes=f"cell {cell}; {origin}; LoRA trained on "
                      "ARBITRARY_SINGLE only, so SEM is out-of-distribution "
                      "transfer",
            )
            rows.append(with_model(model, condition="FVQ", position="first",
                                   accuracy=st["fvq_acc"], n_trials=st["fvq_n"],
                                   n_correct=st["fvq_correct"], **common))
            rows.append(with_model(model, condition="CVQ", position="last",
                                   accuracy=st["cvq_acc"], n_trials=st["cvq_n"],
                                   n_correct=st["cvq_correct"], **common))
    return rows


def qa_lora_sem_tex(rows: list[dict], model: str) -> None:
    """Round-trip check: the paper table used to be the *source* of these 16
    rows, so confirm reading them raw reproduces what the table printed.

    NOT an independent measurement. paper/scripts/generate_lora_tables.py
    builds tab_lora_sem.tex from the same lora_sem_validation_results.json we
    now read directly, so this closes the loop on the tex parse and rounding
    only -- it cannot catch an error in the underlying run.
    """
    path = ROOT / "paper" / "figures" / "tab_lora_sem.tex"
    idx = {(r["variant"], int(r["num_keys"]), int(r["num_updates"]),
            r["condition"]): float(r["accuracy"])
           for r in rows
           if r["model"] == model and r["dataset"] == "semantic_multi"
           and r["accuracy"] != ""}

    checked = bad = 0
    for cells in tex_body_rows(path):
        kn = nums(cells[0])
        if len(kn) < 2:
            continue
        k, n = int(kn[0]), int(kn[1])
        vals = [nums(c) for c in cells[1:7]]
        if any(not v for v in vals):
            continue
        for variant, (fi, ci) in (("base", (0, 1)), ("lora", (3, 4))):
            for cond, i in (("FVQ", fi), ("CVQ", ci)):
                ours = idx.get((variant, k, n, cond))
                if ours is None:
                    continue
                checked += 1
                if abs(ours - pct(vals[i][0])) > 0.005:
                    bad += 1
                    print(f"     MISMATCH {model} {variant} K{k}N{n} {cond}: "
                          f"raw {ours} vs tex {pct(vals[i][0])}")
    print(f"     QA Qwen SEM raw vs tab_lora_sem.tex: {checked} values "
          f"checked, {bad} mismatched")


# ── DERIVED: SEM baselines from the paper's LoRA tables ─────────────────────

def from_lora_sem_tex(fname: str, model: str, want_variants=("base", "lora")) -> list[dict]:
    """(K,N) & base FVQ,CVQ,gap & lora FVQ,CVQ,gap — all as percents."""
    path = ROOT / "paper" / "figures" / fname
    if not path.exists():
        return []
    src = rel(path)
    rows = []
    for cells in tex_body_rows(path):
        kn = nums(cells[0])
        if len(kn) < 2:
            continue
        k, n = int(kn[0]), int(kn[1])
        vals = [nums(c) for c in cells[1:7]]
        if any(not v for v in vals):
            continue
        block = {"base": (vals[0][0], vals[1][0]), "lora": (vals[3][0], vals[4][0])}
        for variant in want_variants:
            fvq, cvq = block[variant]
            common = dict(
                theme=THEME, variant=variant, dataset="semantic_multi",
                prompt_format="chat_template", num_keys=k, num_updates=n,
                split="ood", provenance_tier=DERIVED, source_file=src,
                notes="back-extracted from committed paper table; raw "
                      "open-weight semantic sweep not local",
            )
            rows.append(with_model(model, condition="FVQ", query_type=SEMANTIC,
                                   position="first", accuracy=pct(fvq), **common))
            rows.append(with_model(model, condition="CVQ", query_type=SEMANTIC,
                                   position="last", accuracy=pct(cvq), **common))
    return rows


# ── GSM8K task control ──────────────────────────────────────────────────────

def from_gsm8k() -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(RESULTS / "gsm8k_task_eval_*.json"))):
        d = json.load(open(path))
        variant = "lora_arith_control" if d.get("adapter") else "base"
        rows.append(with_model(
            d["model"], theme=THEME, variant=variant, dataset="gsm8k",
            condition="TASK_ACC", accuracy=d["accuracy"], n_trials=d["n"],
            n_correct=d.get("n_correct"), provenance_tier=RAW,
            source_file=rel(path),
            notes=f"GSM8K task accuracy for run '{d.get('run_name', '')}' — "
                  "shows the arithmetic control adapter DID learn its task "
                  "while leaving the FVQ/CVQ gap open",
        ))
    return rows


# ── QA against the *_comparison.txt files ───────────────────────────────────

_CMP = re.compile(r"^\s*(\d+)k_\s*(\d+)u\s+(.+)$")


def qa_comparisons(rows: list[dict]) -> None:
    """Each *_comparison.txt prints baseline and fine-tuned percents per cell.
    Check our rows reproduce them (tolerance 1pp for the rounding in the txt)."""
    idx = {}
    for r in rows:
        if r["dataset"] != "arbitrary_single" or r["accuracy"] == "":
            continue
        idx[(r["model"], r["variant"], int(r["num_keys"]),
             int(r["num_updates"]), r["condition"])] = float(r["accuracy"])

    checked = bad = skipped = 0
    for path in sorted(RESULTS.glob("*_comparison.txt")):
        run = path.name[:-len("_comparison.txt")]
        if run not in RUNS:
            continue
        model, variant, _, _ = RUNS[run]
        for line in path.read_text().splitlines():
            m = _CMP.match(line)
            if not m:
                continue
            k, n = int(m.group(1)), int(m.group(2))
            v = nums(m.group(3))
            if len(v) < 6:
                skipped += 1        # '---' cells: no baseline in the sweep
                continue
            want = {("base", "FVQ"): v[0], ("base", "CVQ"): v[1],
                    (variant, "FVQ"): v[3], (variant, "CVQ"): v[4]}
            for (var, cond), pct_val in want.items():
                got = idx.get((model, var, k, n, cond))
                if got is None:
                    skipped += 1
                    continue
                checked += 1
                if abs(got - pct_val / 100.0) > 0.011:
                    bad += 1
                    print(f"     ! QA {run} {model} K={k} N={n} {var}/{cond}: "
                          f"ours {got:.2f} vs {path.name} {pct_val / 100:.2f}")
    print(f"     QA vs *_comparison.txt: {checked} values checked, {bad} "
          f"mismatched, {skipped} unavailable")


# ── Per-position IVQ: base vs LoRA (moved here from theme 02, 2026-07-22) ────
# These are interior-position comparisons whose whole purpose is base-vs-LoRA,
# so they belong with the LoRA work, not with the standalone proprietary IVQ.

def ivq_from_dense(path: Path) -> list[dict]:
    """dense-IVQ: Qwen2.5-3B-Instruct, base vs the (non-local) dense adapter."""
    if not path.exists():
        return []
    d = json.load(open(path))
    src = rel(path)
    rows = []
    for variant, splits in d["results"].items():
        v = "base" if variant == "base" else "lora_dense_ivq"
        for split, cells in splits.items():
            for cell_key, positions in cells.items():
                k, n = _cell(cell_key)
                for pos, st in positions.items():
                    rows.append(with_model(
                        QWEN, theme=THEME, variant=v,
                        dataset="arbitrary_single", prompt_format="plain",
                        num_keys=k, num_updates=n,
                        condition=_ivq_condition(pos, n),
                        query_type=(SEMANTIC if int(pos) == n else ORDINAL),
                        position=int(pos), accuracy=st["accuracy"],
                        n_trials=st["total"], n_correct=st["correct"], split=split,
                        provenance_tier=RAW, source_file=src,
                        notes="dense-IVQ; adapter checkpoints/dense_ivq/ is NOT "
                              "local; model per evaluate_ivq.py MODEL_ID",
                    ))
    return rows


def ivq_from_behavioral(path: Path, model: str) -> list[dict]:
    """3-way block-vs-LoRA per-position sweep (base_plain/base_block/lora_plain)."""
    if not path.exists():
        return []
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
                    condition=_ivq_condition(p["k"], n),
                    query_type=(ORDINAL if fmt == "block"
                                else SEMANTIC if p["k"] == n else ORDINAL),
                    position=p["k"], accuracy=p["acc"], ci_lower=p.get("ci_lo"),
                    ci_upper=p.get("ci_hi"), wilson_hw=p.get("wilson_hw"),
                    n_trials=p["n"], n_correct=p.get("correct"),
                    provenance_tier=RAW, source_file=src,
                    notes=f"block-vs-LoRA behavioural sweep; cell {cell_key}; "
                          f"condition key {cond_key}",
                ))
    return rows


def ivq_from_e1(path: Path, model: str) -> list[dict]:
    """E1 extrapolation frontier: main adapter pushed past its training grid."""
    if not path.exists():
        return []
    src = rel(path)
    rows = []
    for line in open(path):
        r = json.loads(line)
        cond = r["condition"]
        rows.append(with_model(
            model, theme=THEME, variant=r["model"],  # 'base' | 'lora'
            dataset=r["dataset"].lower(), prompt_format="plain",
            num_keys=r["num_keys"], num_updates=r["num_updates"],
            condition=cond.split("@")[0],
            query_type=(SEMANTIC if r["position_k"] in (1, r["num_updates"])
                        else ORDINAL),
            position=r["position_k"], accuracy=r["accuracy"],
            wilson_hw=r.get("wilson_hw"), n_trials=r["n_trials"],
            n_correct=r.get("n_correct"), split="extrapolation",
            provenance_tier=RAW, source_file=src,
            notes=f"E1 extrapolation; scan {r['scan']}; target={cond}"
                  + ("; early-stopped" if r.get("stopped_early") else ""),
        ))
    return rows


def build_lora_ivq() -> list[dict]:
    exp = ROOT / "lora_intervention" / "experiments"
    rows = (
        ivq_from_dense(RESULTS / "dense_ivq_ivq_20260523_101418.json")
        + ivq_from_behavioral(exp / "behavioral_block_lora_results" / "results.json", QWEN)
        + ivq_from_behavioral(exp / "gemma_results" / "behavioral" / "results.json", GEMMA)
        + ivq_from_e1(exp / "e1_results" / "results_multiples.jsonl", QWEN)
        + ivq_from_e1(exp / "gemma_results" / "e1" / "results.jsonl", GEMMA)
        # Scan C: big-cell interior sweep (K15-30 x N10-50), run on an H100 at the
        # canonical 200-trial policy. base vs main-adapter, ARBITRARY_SINGLE.
        + ivq_from_e1(exp / "bigcell_ivq" / "out" / "qwen" / "results.jsonl", QWEN)
        + ivq_from_e1(exp / "bigcell_ivq" / "out" / "gemma" / "results.jsonl", GEMMA)
    )
    rows.sort(key=lambda r: (
        r["model"], r["variant"], r["prompt_format"], r["split"],
        int(r["num_keys"] or 0), int(r["num_updates"] or 0),
        pos_key(r["position"])))
    return rows


def main() -> None:
    print(f"[{THEME}] LoRA fine-tuning: adapters, training configs, evals")

    adapters = build_adapters()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "adapters.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ADAPTER_COLUMNS)
        w.writeheader()
        w.writerows(adapters)
    print(f"  -> {THEME}/adapters.csv: {len(adapters)} adapters")

    eval_rows, arb_cells = from_eval_jsons()
    base_rows = []
    for model, keys in arb_cells.items():
        base_rows += baseline_rows(model, keys, "all ARB eval runs")
    # de-duplicate baselines shared by several runs on the same model
    seen, deduped = set(), []
    for r in base_rows:
        sig = (r["model"], r["num_keys"], r["num_updates"], r["condition"])
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(r)

    rows = (
        eval_rows + deduped
        # Qwen SEM: base and LoRA both survive raw, with trial counts.
        + from_lora_sem_raw(QWEN)
        # Gemma SEM: LoRA side already came in raw from gemma_main_eval;
        # the baseline sweep is the one that is genuinely not local.
        + from_lora_sem_tex("tab_lora_gemma_sem.tex", GEMMA, ("base",))
        + from_gsm8k()
    )
    rows.sort(key=lambda r: (r["model"], r["dataset"], r["variant"],
                             int(r["num_keys"] or 0), int(r["num_updates"] or 0),
                             r["condition"]))
    emit(rows, OUT, "lora_evals.csv")
    qa_comparisons(rows)
    qa_lora_sem_tex(rows, QWEN)

    # Per-position base-vs-LoRA IVQ, moved from theme 02. Own by_model subdir so
    # it doesn't clobber lora_evals' split.
    ivq_rows = build_lora_ivq()
    emit(ivq_rows, OUT, "lora_ivq.csv", bymodel="by_model_ivq")


if __name__ == "__main__":
    main()
