#!/usr/bin/env python3
"""
Comprehensive analysis of Qwen3.5 results vs all other models.
Covers: RI/PI accuracy, consistency, issues, dataset differences,
and context-length fill analysis per (num_keys, num_updates) cell.
"""

import json
import os
import math
from pathlib import Path

BASE = Path(__file__).parent
DATASETS = ["arbitrary_single", "semantic_multi", "narrative_dota2"]

# ─── token-count estimate ────────────────────────────────────────────────────
# prompt = system_prompt + chat_template + stream + query
# stream = num_keys * num_updates entries, each ~5 tok ("word: word\n")
# overhead (system, template, query) ≈ 200 tokens for instruct models
#                                     ≈ 450 tokens for completion (few-shot demos)
OVERHEAD_INSTRUCT  = 200
OVERHEAD_COMPLETION = 450
TOKENS_PER_ENTRY    = 5      # avg: category (~3 tok) + ": " + value (~1.5 tok) + newline

def estimate_tokens(num_keys, num_updates, is_instruct):
    overhead = OVERHEAD_INSTRUCT if is_instruct else OVERHEAD_COMPLETION
    return overhead + num_keys * num_updates * TOKENS_PER_ENTRY

def context_fill_pct(num_keys, num_updates, is_instruct, max_len=16384):
    return 100 * estimate_tokens(num_keys, num_updates, is_instruct) / max_len

# ─── model metadata ──────────────────────────────────────────────────────────
MODEL_META = {
    "Qwen3.5-0.8B":         {"params": 0.8,  "family": "Qwen3.5",   "instruct": False},
    "Qwen3.5-2B":           {"params": 2.0,  "family": "Qwen3.5",   "instruct": False},
    "Qwen3.5-4B":           {"params": 4.0,  "family": "Qwen3.5",   "instruct": False},
    "Qwen3.5-9B":           {"params": 9.0,  "family": "Qwen3.5",   "instruct": False},
    "Qwen2.5-0.5B-Instruct":{"params": 0.5,  "family": "Qwen2.5",   "instruct": True},
    "Qwen2.5-1.5B-Instruct":{"params": 1.5,  "family": "Qwen2.5",   "instruct": True},
    "Qwen2.5-3B":           {"params": 3.0,  "family": "Qwen2.5",   "instruct": False},
    "Qwen2.5-3B-Instruct":  {"params": 3.0,  "family": "Qwen2.5",   "instruct": True},
    "gemma-3-270m-it":      {"params": 0.27, "family": "Gemma3",    "instruct": True},
    "gemma-3-1b-it":        {"params": 1.0,  "family": "Gemma3",    "instruct": True},
    "gemma-3-4b-it":        {"params": 4.0,  "family": "Gemma3",    "instruct": True},
    "mamba-1.4b-hf":        {"params": 1.4,  "family": "Mamba",     "instruct": False},
    "pythia-410m":          {"params": 0.41, "family": "Pythia",    "instruct": False},
    "stablelm-2-1_6b-chat": {"params": 1.6,  "family": "StableLM",  "instruct": True},
    "TinyLlama-1.1B-Chat-v1.0": {"params": 1.1, "family": "TinyLlama", "instruct": True},
}

# ─── data loading ─────────────────────────────────────────────────────────────
def load_sweep(dataset, model):
    folder = BASE / dataset / model
    if not folder.exists():
        return None
    for f in sorted(folder.iterdir()):
        if "sweep" in f.name and f.suffix == ".json":
            with open(f) as fh:
                return json.load(fh)
    # fallback: checkpoint
    ckpt = folder / "stage1_checkpoint.json"
    if ckpt.exists():
        with open(ckpt) as fh:
            return json.load(fh)
    return None

def load_narrative(model):
    folder = BASE / "narrative_dota2" / model
    if not folder.exists():
        return None
    for f in sorted(folder.iterdir()):
        if f.suffix == ".json" and "trials" not in f.name:
            with open(f) as fh:
                return json.load(fh)
    return None

# ─── cell extraction ──────────────────────────────────────────────────────────
def extract_cells(sweep):
    """Return list of dicts: {num_keys, num_updates, RI, PI, gap, n, regime, ...}"""
    cells = []
    for cell_key, cell in sweep["cells"].items():
        ri = cell["stats"]["RI"]
        pi = cell["stats"]["PI"]
        nk = cell["num_keys"]
        nu = cell["num_updates"]
        cells.append({
            "num_keys":   nk,
            "num_updates": nu,
            "RI":   ri["accuracy"],
            "PI":   pi["accuracy"],
            "gap":  ri["accuracy"] - pi["accuracy"],   # positive = PI>RI expected behaviour
            "n":    ri["n"],
            "regime": cell.get("regime", "?"),
            "RI_garbage": ri.get("n_garbage", 0),
            "PI_garbage": pi.get("n_garbage", 0),
            "PI_primacy_intrusion": pi.get("error_types", {}).get("primacy_intrusion", 0),
            "PI_inter_intrusion":   pi.get("error_types", {}).get("intermediate_intrusion", 0),
        })
    return cells

def extract_narrative_cells(nd):
    cells = []
    for cell_key, cell in nd["results"].items():
        stats = cell["stats"]
        ri = stats["RI"]
        pi = stats["PI"]
        # parse cell key like "2k_5u"
        parts = cell_key.split("_")
        nk = int(parts[0].replace("k",""))
        nu = int(parts[1].replace("u",""))
        cells.append({
            "num_keys":   nk,
            "num_updates": nu,
            "RI":   ri["accuracy"],
            "PI":   pi["accuracy"],
            "gap":  ri["accuracy"] - pi["accuracy"],
            "n":    ri["n"],
        })
    return cells

# ─── helpers ─────────────────────────────────────────────────────────────────
def mean(xs): return sum(xs)/len(xs) if xs else float("nan")
def pct_positive(xs): return 100*sum(1 for x in xs if x > 0)/len(xs) if xs else float("nan")

def bar(val, width=20, max_val=1.0):
    filled = int(round(val / max_val * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)

def fmt(v): return f"{v:.3f}" if not math.isnan(v) else "  nan"
def fmtp(v): return f"{v:5.1f}%" if not math.isnan(v) else "  nan "

# ─── SECTION 1: Dataset overview ─────────────────────────────────────────────
def section_dataset_overview():
    print("=" * 80)
    print("SECTION 1: DATASET OVERVIEW")
    print("=" * 80)

    info = {
        "arbitrary_single": {
            "desc": "Arbitrary key=value pairs; keys=category names (sports/food/...), values=random single English words.",
            "difficulty": "Pure memory — no semantic signal in value. Forces verbatim recall.",
            "pi_bias_expected": "High: no semantic anchor for 'last' value, recency signal weak.",
        },
        "semantic_multi": {
            "desc": "Same structure but values have semantic coherence with keys (e.g. sport→football).",
            "difficulty": "Easier: semantic priors can help recall. Tests if semantics rescue PI.",
            "pi_bias_expected": "Lower gap than arbitrary — semantics provide a shortcut.",
        },
        "narrative_dota2": {
            "desc": "Values embedded in Dota 2 narrative text. Key=player/item, value=stat embedded in a sentence.",
            "difficulty": "Hardest: context is long prose, no clean key: value format.",
            "pi_bias_expected": "Largest gap expected — retrieval must parse narrative structure.",
        },
    }
    for ds, d in info.items():
        print(f"\n  [{ds}]")
        print(f"    Description : {d['desc']}")
        print(f"    Difficulty  : {d['difficulty']}")
        print(f"    PI bias     : {d['pi_bias_expected']}")

    # How many models have data per dataset
    for ds in DATASETS:
        models = sorted((BASE / ds).iterdir()) if (BASE / ds).exists() else []
        models = [m.name for m in models if m.is_dir()]
        qwen35 = [m for m in models if "Qwen3.5" in m]
        others = [m for m in models if "Qwen3.5" not in m]
        print(f"\n  {ds}: {len(models)} models total — Qwen3.5: {sorted(qwen35)} | others: {sorted(others)}")


# ─── SECTION 2: Cell grid & context length ───────────────────────────────────
def section_context_length(model="Qwen3.5-0.8B", dataset="arbitrary_single"):
    print("\n" + "=" * 80)
    print("SECTION 2: CONTEXT LENGTH FILL PER CELL  (Qwen3.5-0.8B × arbitrary_single)")
    print("=" * 80)

    sweep = load_sweep(dataset, model)
    if sweep is None:
        print("  [not found]"); return

    meta = MODEL_META.get(model, {})
    is_instruct = meta.get("instruct", False)

    key_levels    = sorted(set(c["num_keys"]    for c in extract_cells(sweep)))
    update_levels = sorted(set(c["num_updates"] for c in extract_cells(sweep)))

    # Build lookup: (nk, nu) -> cell
    cell_lookup = {(c["num_keys"], c["num_updates"]): c for c in extract_cells(sweep)}

    print(f"\n  Prompt format : {'chat/instruct' if is_instruct else 'completion few-shot'}")
    print(f"  Token overhead: {OVERHEAD_INSTRUCT if is_instruct else OVERHEAD_COMPLETION} tok")
    print(f"  Tokens/entry  : {TOKENS_PER_ENTRY} tok  (num_keys × num_updates entries total)")
    print(f"  Max model len : 16384 tokens  (85% limit = {int(16384*0.85)} tok)")
    print(f"  Cutoff at     : ~{(int(16384*0.85) - (OVERHEAD_INSTRUCT if is_instruct else OVERHEAD_COMPLETION)) // TOKENS_PER_ENTRY} total entries")

    print("\n  Grid: rows=num_keys, cols=num_updates  |  [fill%  RI   PI  gap] or [SKIP]")
    nk_nu_label = "nk\\nu"
    header = f"  {nk_nu_label:>5} " + "".join(f"  {nu:>4}" for nu in update_levels)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for nk in key_levels:
        row = f"  {nk:>5}k  "
        for nu in update_levels:
            est = estimate_tokens(nk, nu, is_instruct)
            fill = 100 * est / 16384
            if (nk, nu) in cell_lookup:
                c = cell_lookup[(nk, nu)]
                row += f"[{fill:4.0f}% {c['RI']:.2f} {c['PI']:.2f} {c['gap']:+.2f}] "
            else:
                row += f"[{fill:4.0f}% ---- SKIP ----] "
        print(row)

    # Flag high-context cells (>60% fill) and their performance
    print("\n  High-context cells (>60% fill):")
    high_ctx = [(c, estimate_tokens(c['num_keys'], c['num_updates'], is_instruct))
                for c in extract_cells(sweep)
                if estimate_tokens(c['num_keys'], c['num_updates'], is_instruct) > 0.60 * 16384]
    high_ctx.sort(key=lambda x: x[1], reverse=True)
    for c, tok in high_ctx[:10]:
        fill = 100 * tok / 16384
        print(f"    nk={c['num_keys']:2d} nu={c['num_updates']:3d}  "
              f"~{tok:5d} tok ({fill:4.1f}%)  RI={c['RI']:.3f}  PI={c['PI']:.3f}  gap={c['gap']:+.3f}")


# ─── SECTION 3: Per-model summary across datasets ────────────────────────────
def section_model_summary():
    print("\n" + "=" * 80)
    print("SECTION 3: PER-MODEL SUMMARY  (mean RI / PI / gap / %cells PI>RI)")
    print("=" * 80)

    all_models = sorted(MODEL_META.keys())

    for ds in ["arbitrary_single", "semantic_multi"]:
        print(f"\n  Dataset: {ds}")
        print(f"  {'Model':<28} {'Params':>6}  {'mean_RI':>7}  {'mean_PI':>7}  {'mean_gap':>8}  {'%PI>RI':>7}  {'cells':>5}  {'issues'}")
        print("  " + "-" * 90)

        rows = []
        for model in all_models:
            sweep = load_sweep(ds, model)
            if sweep is None: continue
            cells = extract_cells(sweep)
            if not cells: continue
            ri_vals  = [c["RI"]  for c in cells]
            pi_vals  = [c["PI"]  for c in cells]
            gap_vals = [c["gap"] for c in cells]
            params = MODEL_META[model]["params"]
            issues = []
            # Check for reversed cells (PI > RI unexpected)
            reversed_cells = [c for c in cells if c["gap"] < 0]
            if reversed_cells:
                issues.append(f"{len(reversed_cells)} reversed")
            # Check for low RI (model failing basic task)
            low_ri = [c for c in cells if c["RI"] < 0.5]
            if low_ri:
                issues.append(f"{len(low_ri)} low-RI(<0.5)")
            # High garbage
            high_garbage = [c for c in cells if (c["RI_garbage"] + c["PI_garbage"]) / max(c["n"]*2, 1) > 0.15]
            if high_garbage:
                issues.append(f"{len(high_garbage)} hi-garbage")
            rows.append((params, model, mean(ri_vals), mean(pi_vals), mean(gap_vals),
                         pct_positive(gap_vals), len(cells), ", ".join(issues) or "—"))

        rows.sort(key=lambda r: r[0])
        for params, model, mri, mpi, mgap, pct, ncells, issues in rows:
            is_q35 = "Qwen3.5" in model
            tag = " ◄" if is_q35 else "  "
            print(f"  {model:<28} {params:>5.1f}B  {fmt(mri)}  {fmt(mpi)}  {fmt(mgap):>8}  "
                  f"{fmtp(pct)}  {ncells:>5}  {issues}{tag}")

    # Narrative separately
    print(f"\n  Dataset: narrative_dota2")
    print(f"  {'Model':<28} {'Params':>6}  {'mean_RI':>7}  {'mean_PI':>7}  {'mean_gap':>8}  {'%PI>RI':>7}  {'cells':>5}")
    print("  " + "-" * 80)
    rows = []
    for model in all_models:
        nd = load_narrative(model)
        if nd is None: continue
        cells = extract_narrative_cells(nd)
        if not cells: continue
        ri_vals  = [c["RI"]  for c in cells]
        pi_vals  = [c["PI"]  for c in cells]
        gap_vals = [c["gap"] for c in cells]
        params = MODEL_META[model]["params"]
        rows.append((params, model, mean(ri_vals), mean(pi_vals), mean(gap_vals),
                     pct_positive(gap_vals), len(cells)))

    rows.sort(key=lambda r: r[0])
    for params, model, mri, mpi, mgap, pct, ncells in rows:
        is_q35 = "Qwen3.5" in model
        tag = " ◄" if is_q35 else "  "
        print(f"  {model:<28} {params:>5.1f}B  {fmt(mri)}  {fmt(mpi)}  {fmt(mgap):>8}  "
              f"{fmtp(pct)}  {ncells:>5}{tag}")


# ─── SECTION 4: Qwen3.5 vs Qwen2.5 deep dive ─────────────────────────────────
def section_qwen_comparison():
    print("\n" + "=" * 80)
    print("SECTION 4: Qwen3.5 vs Qwen2.5 — FAMILY COMPARISON")
    print("=" * 80)

    qwen35 = ["Qwen3.5-0.8B", "Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B"]
    qwen25 = ["Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct", "Qwen2.5-3B", "Qwen2.5-3B-Instruct"]

    for ds in ["arbitrary_single", "semantic_multi"]:
        print(f"\n  [{ds}]")
        print(f"  {'Model':<28}  RI↑  PI↑  gap↑  %PI>RI   Notes")
        print("  " + "-"*70)

        all_models = qwen25 + qwen35
        for m in all_models:
            sweep = load_sweep(ds, m)
            if sweep is None:
                print(f"  {m:<28}  [no data]"); continue
            cells = extract_cells(sweep)
            ri   = mean([c["RI"]  for c in cells])
            pi   = mean([c["PI"]  for c in cells])
            gap  = mean([c["gap"] for c in cells])
            pct  = pct_positive([c["gap"] for c in cells])
            tag = "◄ Qwen3.5" if "Qwen3.5" in m else "  Qwen2.5"
            # notable: PI > RI cells (reversed)
            reversed_n = sum(1 for c in cells if c["gap"] < 0)
            notes = f"  {reversed_n} rev-cells" if reversed_n else ""
            print(f"  {m:<28}  {ri:.3f}  {pi:.3f}  {gap:+.3f}  {pct:5.1f}%  {tag}{notes}")

    # Gap vs size within Qwen3.5
    print(f"\n  Qwen3.5 gap vs model size (arbitrary_single):")
    for m in qwen35:
        sweep = load_sweep("arbitrary_single", m)
        if sweep is None: continue
        cells = extract_cells(sweep)
        gap = mean([c["gap"] for c in cells])
        params = MODEL_META[m]["params"]
        print(f"    {m:<22} {params:4.1f}B  gap={gap:+.3f}  {bar(gap, 20, 0.5)}")


# ─── SECTION 5: Error pattern analysis ──────────────────────────────────────
def section_error_patterns():
    print("\n" + "=" * 80)
    print("SECTION 5: ERROR PATTERN ANALYSIS")
    print("=" * 80)

    print("\n  For PI failures: primacy_intrusion = recalled FIRST value (worse than random)")
    print("  intermediate_intrusion = recalled a middle value (serial-position effect)")
    print("  garbage = output outside the value set (model confused/hallucinating)")

    models_to_show = ["Qwen3.5-0.8B","Qwen3.5-2B","Qwen3.5-4B","Qwen3.5-9B",
                      "Qwen2.5-0.5B-Instruct","Qwen2.5-1.5B-Instruct",
                      "Qwen2.5-3B-Instruct","gemma-3-1b-it","mamba-1.4b-hf"]

    for ds in ["arbitrary_single", "semantic_multi"]:
        print(f"\n  [{ds}] — PI error breakdown (aggregated over all cells)")
        print(f"  {'Model':<28}  n_PI_fail  primacy%  intermed%  garbage%  RI_garb%")
        print("  " + "-"*78)

        for m in models_to_show:
            sweep = load_sweep(ds, m)
            if sweep is None: continue
            cells = extract_cells(sweep)

            total_pi_trials = sum(c["n"] for c in cells)
            total_pi_correct = sum(int(c["PI"] * c["n"]) for c in cells)
            total_pi_fail = total_pi_trials - total_pi_correct

            total_primacy = sum(c["PI_primacy_intrusion"] for c in cells)
            total_inter   = sum(c["PI_inter_intrusion"]   for c in cells)
            total_pi_garb = sum(c["PI_garbage"]           for c in cells)
            total_ri_garb = sum(c["RI_garbage"]           for c in cells)
            total_ri_trials= sum(c["n"] for c in cells)

            def pct_of_fail(n): return 100*n/total_pi_fail if total_pi_fail > 0 else 0
            def pct_of_ri(n):   return 100*n/total_ri_trials if total_ri_trials > 0 else 0

            tag = " ◄" if "Qwen3.5" in m else "  "
            print(f"  {m:<28}  {total_pi_fail:>8}   "
                  f"{pct_of_fail(total_primacy):5.1f}%  "
                  f"{pct_of_fail(total_inter):8.1f}%  "
                  f"{pct_of_fail(total_pi_garb):7.1f}%  "
                  f"{pct_of_ri(total_ri_garb):6.1f}%{tag}")


# ─── SECTION 6: Qwen3.5 cell-level deep dive ─────────────────────────────────
def section_qwen35_cells():
    print("\n" + "=" * 80)
    print("SECTION 6: Qwen3.5 — CELL-LEVEL DEEP DIVE (arbitrary_single)")
    print("=" * 80)

    models = ["Qwen3.5-0.8B","Qwen3.5-2B","Qwen3.5-4B","Qwen3.5-9B"]

    for model in models:
        sweep = load_sweep("arbitrary_single", model)
        if sweep is None: continue
        cells = extract_cells(sweep)
        meta = MODEL_META[model]

        # sort by num_updates (primary), num_keys (secondary)
        cells.sort(key=lambda c: (c["num_updates"], c["num_keys"]))

        print(f"\n  {model}  ({meta['params']}B params)")
        print(f"  {'nk':>3} {'nu':>3}  {'fill%':>6}  {'RI':>5}  {'PI':>5}  {'gap':>6}  regime  issues")
        print("  " + "-"*60)

        for c in cells:
            is_inst = meta["instruct"]
            tok = estimate_tokens(c["num_keys"], c["num_updates"], is_inst)
            fill = 100 * tok / 16384
            flags = []
            if c["gap"] < 0:
                flags.append("REVERSED!")
            if c["RI"] < 0.5:
                flags.append("low-RI")
            if c["PI"] < 0.2:
                flags.append("PI-collapse")
            garb_rate = (c["RI_garbage"] + c["PI_garbage"]) / (2 * c["n"]) if c["n"] > 0 else 0
            if garb_rate > 0.15:
                flags.append(f"garb={garb_rate:.0%}")
            flag_str = ", ".join(flags)
            print(f"  {c['num_keys']:>3} {c['num_updates']:>3}  {fill:5.1f}%  "
                  f"{c['RI']:.3f}  {c['PI']:.3f}  {c['gap']:+.3f}  "
                  f"{'['+c['regime']+']':>7}  {flag_str}")


# ─── SECTION 7: Performance vs context fill ──────────────────────────────────
def section_context_fill_vs_perf():
    print("\n" + "=" * 80)
    print("SECTION 7: PERFORMANCE vs CONTEXT FILL  (all Qwen3.5, arbitrary_single)")
    print("=" * 80)

    print("\n  Do RI/PI scores degrade as context fills?")
    print("  Bucketed by estimated context fill %.\n")

    buckets = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    models = ["Qwen3.5-0.8B","Qwen3.5-2B","Qwen3.5-4B","Qwen3.5-9B"]

    for model in models:
        sweep = load_sweep("arbitrary_single", model)
        if sweep is None: continue
        meta  = MODEL_META[model]
        cells = extract_cells(sweep)

        print(f"  {model}:")
        print(f"  {'fill bucket':>14}  {'mean RI':>7}  {'mean PI':>7}  {'mean gap':>8}  {'n_cells':>7}")
        for lo, hi in buckets:
            bucket_cells = [c for c in cells
                            if lo <= context_fill_pct(c["num_keys"], c["num_updates"], meta["instruct"]) < hi]
            if not bucket_cells: continue
            ri  = mean([c["RI"]  for c in bucket_cells])
            pi  = mean([c["PI"]  for c in bucket_cells])
            gap = mean([c["gap"] for c in bucket_cells])
            print(f"  {lo:3d}–{hi:3d}%          "
                  f"{fmt(ri)}  {fmt(pi)}  {fmt(gap):>8}  {len(bucket_cells):>7}")
        print()


# ─── SECTION 8: Consistency — narrative vs structured ────────────────────────
def section_narrative_consistency():
    print("\n" + "=" * 80)
    print("SECTION 8: NARRATIVE vs STRUCTURED — CONSISTENCY CHECK")
    print("=" * 80)

    models = ["Qwen3.5-0.8B","Qwen3.5-2B","Qwen3.5-4B","Qwen3.5-9B",
              "Qwen2.5-0.5B-Instruct","Qwen2.5-1.5B-Instruct","Qwen2.5-3B-Instruct",
              "gemma-3-1b-it","gemma-3-4b-it"]

    print(f"\n  {'Model':<28}  arb_gap   sem_gap   narr_gap  consistent?")
    print("  " + "-"*68)

    for m in models:
        arb  = load_sweep("arbitrary_single", m)
        sem  = load_sweep("semantic_multi", m)
        nd   = load_narrative(m)

        arb_gap  = mean([c["gap"] for c in extract_cells(arb)])  if arb else float("nan")
        sem_gap  = mean([c["gap"] for c in extract_cells(sem)])  if sem else float("nan")
        narr_gap = mean([c["gap"] for c in extract_narrative_cells(nd)]) if nd else float("nan")

        # Consistency: all gaps same sign?
        signs = [g > 0 for g in [arb_gap, sem_gap, narr_gap] if not math.isnan(g)]
        consistent = "YES" if len(set(signs)) == 1 else "NO ← INCONSISTENT"

        tag = " ◄" if "Qwen3.5" in m else "  "
        print(f"  {m:<28}  {fmt(arb_gap)}   {fmt(sem_gap)}   {fmt(narr_gap)}  {consistent}{tag}")


# ─── SECTION 9: Issues flagged for Qwen3.5 ──────────────────────────────────
def section_issues():
    print("\n" + "=" * 80)
    print("SECTION 9: ISSUES FOUND IN Qwen3.5")
    print("=" * 80)

    for model in ["Qwen3.5-0.8B","Qwen3.5-2B","Qwen3.5-4B","Qwen3.5-9B"]:
        issues = []
        meta = MODEL_META[model]

        for ds in ["arbitrary_single", "semantic_multi"]:
            sweep = load_sweep(ds, model)
            if sweep is None:
                issues.append(f"[{ds}] NO DATA")
                continue
            cells = extract_cells(sweep)

            # 1. reversed cells (RI < PI — unusual)
            rev = [c for c in cells if c["gap"] < 0]
            if rev:
                for c in rev:
                    issues.append(f"[{ds}] REVERSED cell nk={c['num_keys']} nu={c['num_updates']}: "
                                  f"RI={c['RI']:.3f} PI={c['PI']:.3f} gap={c['gap']:+.3f}")

            # 2. Low RI (model failing even to retrieve first value)
            low_ri = [c for c in cells if c["RI"] < 0.5]
            for c in low_ri:
                issues.append(f"[{ds}] LOW RI nk={c['num_keys']} nu={c['num_updates']}: "
                               f"RI={c['RI']:.3f} (below chance)")

            # 3. High garbage — model outputting non-values
            hi_garb = [c for c in cells
                       if (c["RI_garbage"] + c["PI_garbage"]) / (2 * c["n"]) > 0.20]
            for c in hi_garb:
                gr = (c["RI_garbage"] + c["PI_garbage"]) / (2 * c["n"])
                issues.append(f"[{ds}] HIGH GARBAGE nk={c['num_keys']} nu={c['num_updates']}: "
                               f"avg garbage rate={gr:.1%}")

            # 4. PI collapse (PI very near zero — total confusion)
            pi_collapse = [c for c in cells if c["PI"] < 0.15]
            for c in pi_collapse:
                issues.append(f"[{ds}] PI COLLAPSE nk={c['num_keys']} nu={c['num_updates']}: "
                               f"PI={c['PI']:.3f}")

            # 5. Suspicious perfect RI (1.0 exactly on many cells — might be leakage/format issue)
            perfect_ri = [c for c in cells if c["RI"] >= 0.99]
            if len(perfect_ri) >= 5:
                issues.append(f"[{ds}] {len(perfect_ri)} cells with RI≥0.99 — possible format/leakage?")

            # 6. Note prompt format (Qwen3.5 is NOT instruct here)
            if not meta["instruct"]:
                fmt_used = sweep.get("prompt_format","?")
                issues.append(f"[{ds}] Using '{fmt_used}' format (NOT chat/instruct)")

        print(f"\n  {model} ({meta['params']}B):")
        if not issues:
            print("    No significant issues found.")
        for issue in issues:
            print(f"    ⚠ {issue}")


# ─── SECTION 10: Cross-architecture comparison ───────────────────────────────
def section_cross_arch():
    print("\n" + "=" * 80)
    print("SECTION 10: CROSS-ARCHITECTURE COMPARISON")
    print("=" * 80)

    arch_groups = {
        "Qwen3.5 (dense, completion)": ["Qwen3.5-0.8B","Qwen3.5-2B","Qwen3.5-4B","Qwen3.5-9B"],
        "Qwen2.5 (dense, instruct)":   ["Qwen2.5-0.5B-Instruct","Qwen2.5-1.5B-Instruct","Qwen2.5-3B-Instruct"],
        "Gemma3 (dense, instruct)":    ["gemma-3-270m-it","gemma-3-1b-it","gemma-3-4b-it"],
        "Mamba (SSM, completion)":     ["mamba-1.4b-hf"],
        "Other (completion/instruct)": ["pythia-410m","stablelm-2-1_6b-chat","TinyLlama-1.1B-Chat-v1.0"],
    }

    print(f"\n  arbitrary_single — mean gap by architecture group")
    print(f"  {'Architecture':<36}  {'mean_RI':>7}  {'mean_PI':>7}  {'mean_gap':>8}  {'%PI>RI':>7}")
    print("  " + "-"*75)

    for group, models in arch_groups.items():
        all_ri, all_pi, all_gap = [], [], []
        for m in models:
            sweep = load_sweep("arbitrary_single", m)
            if sweep is None: continue
            cells = extract_cells(sweep)
            all_ri  += [c["RI"]  for c in cells]
            all_pi  += [c["PI"]  for c in cells]
            all_gap += [c["gap"] for c in cells]
        if not all_gap: continue
        print(f"  {group:<36}  {fmt(mean(all_ri))}  {fmt(mean(all_pi))}  "
              f"{fmt(mean(all_gap)):>8}  {fmtp(pct_positive(all_gap))}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.chdir(BASE)

    section_dataset_overview()
    section_context_length()
    section_model_summary()
    section_qwen_comparison()
    section_error_patterns()
    section_qwen35_cells()
    section_context_fill_vs_perf()
    section_narrative_consistency()
    section_issues()
    section_cross_arch()

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
