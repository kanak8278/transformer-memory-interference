#!/usr/bin/env python3
"""
Phase 0 — shape existing v3 / paper result JSON into clean static files for the
interactive showcase page.  No model runs; pure reshaping of vetted data.

Outputs (web/data/):
  examples.json        §1  real single-key streams + base-model predictions
  models_scatter.json  §2  per-model FVQ (RI) vs CVQ (PI), Wilson CIs
  logit_lens.json      §3  per-layer P(target) baseline vs +LoRA, per cell
  probing.json         §4  per-layer probe accuracy baseline vs +LoRA
  intervention.json    §5  headline before/after of the LoRA fix

FVQ = First-Value Query  = RI  (models GOOD)
CVQ = Current-Value Query = PI  (models BAD)
Gap = FVQ - CVQ

Run:  .venv/bin/python web/build_data.py
"""
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

SEM_DIR = ROOT / "v3" / "results_vllm" / "semantic_multi"
PROP_CSV = ROOT / "experiments_cloud" / "results" / "proprietary_semantic_multi.csv"

# Logit-lens (Qwen2.5-3B-Instruct) — baseline vs +LoRA  (per-trial analyses schema)
LL_BASE = ROOT / "v3/results_vllm/logit_lens/Qwen2.5-3B-Instruct/stage2_logit_lens_20260409_054632.json"
LL_LORA = ROOT / "v3/scripts/experiments/results/Qwen2.5-3B-Instruct-LoRA/stage2_logit_lens_20260524_194921.json"

# Logit-lens (Gemma-3-4b-it) — aggregated per-position schema; files chosen to
# match paper/figures/tab_gemma_logit_lens.tex (P(v_last) base 0.80/0.56/0.59).
GEMMA_LL_BASE = ROOT / "v3/results_vllm/logit_lens/gemma-3-4b-it-baseline-hf/stage2_logit_lens_20260525_195739.json"
GEMMA_LL_LORA = ROOT / "v3/results_vllm/logit_lens/gemma-3-4b-it-LoRA/stage2_logit_lens_20260525_200056.json"

# Probing (Qwen2.5-3B-Instruct, K=2 N=5, 200 trials) — baseline vs +LoRA
PROBE_BASE = ROOT / "v3/results_vllm/probing/probing_Qwen2.5-3B-Instruct_2k_5u.json"
PROBE_LORA = ROOT / "v3/results_vllm/probing/probing_Qwen2.5-3B-Instruct-LoRA_2k_5u.json"

LORA_BEHAVIOR = ROOT / "lora_sem_finish_results.json"

# Dense-IVQ per-position curve (Qwen2.5-3B, arbitrary single-token, dense updates)
DENSE_IVQ = ROOT / "lora_intervention/results/dense_ivq_ivq_20260523_101418.json"


def wilson_hw(p, n, z=1.96):
    if n == 0:
        return 0.0
    denom = 1 + z * z / n
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return spread


# Display metadata for the scatter (family + size in B). Unknown -> omitted.
META = {
    "Qwen2.5-0.5B-Instruct": ("Qwen2.5", 0.5),
    "Qwen2.5-1.5B-Instruct": ("Qwen2.5", 1.5),
    "Qwen2.5-3B-Instruct": ("Qwen2.5", 3.0),
    "Qwen2.5-3B": ("Qwen2.5", 3.0),
    "Qwen3.5-0.8B": ("Qwen3.5", 0.8),
    "Qwen3.5-2B": ("Qwen3.5", 2.0),
    "Qwen3.5-4B": ("Qwen3.5", 4.0),
    "Qwen3.5-9B": ("Qwen3.5", 9.0),
    "gemma-3-270m-it": ("Gemma-3", 0.27),
    "gemma-3-1b-it": ("Gemma-3", 1.0),
    "gemma-3-4b-it": ("Gemma-3", 4.0),
    "TinyLlama-1.1B-Chat-v1.0": ("TinyLlama", 1.1),
    "stablelm-2-1_6b-chat": ("StableLM", 1.6),
    "pythia-410m": ("Pythia", 0.41),
    "mamba-1.4b-hf": ("Mamba", 1.4),
    "gpt-4.1": ("GPT", 1760.0),
    "gpt-4.1-mini": ("GPT", None),
    "claude-4.5-haiku": ("Claude", None),
    "claude-sonnet": ("Claude", None),
    "gemini-2.5-flash": ("Gemini", None),
    "gemini-2.5-pro": ("Gemini", None),
}
DISPLAY = {
    "gpt-4.1": "GPT-4.1", "gpt-4.1-mini": "GPT-4.1-mini",
    "claude-4.5-haiku": "Claude-4.5-Haiku", "claude-sonnet": "Claude-4.5-Sonnet",
    "gemini-2.5-flash": "Gemini-2.5-Flash", "gemini-2.5-pro": "Gemini-2.5-Pro",
    "gemma-3-270m-it": "Gemma-3-270m-it", "gemma-3-1b-it": "Gemma-3-1b-it",
    "gemma-3-4b-it": "Gemma-3-4b-it", "stablelm-2-1_6b-chat": "StableLM-2-1.6B",
    "pythia-410m": "Pythia-410M", "TinyLlama-1.1B-Chat-v1.0": "TinyLlama-1.1B",
    "mamba-1.4b-hf": "Mamba-1.4B", "Qwen2.5-3B": "Qwen2.5-3B (base)",
}


def disp(m):
    return DISPLAY.get(m, m)


# ── §2  scatter ──────────────────────────────────────────────────────────────
def build_scatter():
    OPEN_CELL = (5, 30)   # paper's open-weight cell
    PROP_CELL = (20, 50)  # paper's proprietary cell
    pts = []

    # Mamba is an SSM *control* in the paper, not one of the main 20 — exclude it.
    EXCLUDE = {"mamba-1.4b-hf"}
    for d in sorted(SEM_DIR.glob("*")):
        if not d.is_dir() or d.name in EXCLUDE:
            continue
        files = sorted(glob.glob(str(d / "stage1_sweep_*.json")))
        if not files:
            continue
        data = json.load(open(files[-1]))
        cell = data["cells"].get(f"{OPEN_CELL[0]}_{OPEN_CELL[1]}")
        if not cell or "stats" not in cell or cell.get("n_trials", 0) < 30:
            continue
        fvq = cell["stats"]["RI"]["accuracy"]
        cvq = cell["stats"]["PI"]["accuracy"]
        n = cell["n_trials"]
        fam, size = META.get(d.name, (None, None))
        pts.append({
            "model": disp(d.name), "family": fam, "size_b": size,
            "fvq": round(fvq, 3), "cvq": round(cvq, 3), "gap": round(fvq - cvq, 3),
            "fvq_hw": round(wilson_hw(fvq, n), 3), "cvq_hw": round(wilson_hw(cvq, n), 3),
            "n": n, "cell": {"K": OPEN_CELL[0], "N": OPEN_CELL[1]}, "proprietary": False,
        })

    # proprietary at the harder cell
    prop = defaultdict(dict)
    with open(PROP_CSV) as f:
        for r in csv.DictReader(f):
            if int(r["n_trials"]) < 30:
                continue
            prop[r["model"]][(int(r["num_keys"]), int(r["num_updates"]))] = r
    for m, cells in prop.items():
        r = cells.get(PROP_CELL)
        if not r:
            continue
        fvq, cvq, n = float(r["fvq_acc"]), float(r["cvq_acc"]), int(r["n_trials"])
        fam, size = META.get(m, (None, None))
        pts.append({
            "model": disp(m), "family": fam, "size_b": size,
            "fvq": round(fvq, 3), "cvq": round(cvq, 3), "gap": round(fvq - cvq, 3),
            "fvq_hw": round(wilson_hw(fvq, n), 3), "cvq_hw": round(wilson_hw(cvq, n), 3),
            "n": n, "cell": {"K": PROP_CELL[0], "N": PROP_CELL[1]}, "proprietary": True,
        })

    pts.sort(key=lambda p: -p["gap"])
    n_pos = sum(1 for p in pts if p["gap"] > 0)
    out = {
        "headline": {
            # Matches paper/main.tex: 20 models (14 open-weight + 6 proprietary),
            # FVQ-CVQ gaps range +6 to +53 pp, majority > +20 pp.
            "n_models_paper": 20, "gap_range_pp": [6, 53],
            "n_shown": len(pts), "n_gap_positive": n_pos,
            "mean_gap": round(sum(p["gap"] for p in pts) / len(pts), 3),
        },
        "points": pts,
        "note": ("FVQ = First-Value Query (“what was the FIRST value?”; models good). "
                 "CVQ = Current-Value Query (“what is the CURRENT/last value?”; models bad). "
                 "Open-weight models scored at K=5, N=30; proprietary at the harder K=20, N=50, "
                 "matching the paper. Mamba (an SSM control) is excluded from the main set."),
    }
    json.dump(out, open(OUT / "models_scatter.json", "w"), indent=1)
    print(f"  models_scatter.json — {len(pts)} models ({n_pos} with gap>0)")


DEMO_RUNS = OUT / "demo_runs.json"


# ── §1  examples (real multi-key interleaved runs) ───────────────────────────
def build_demo():
    if not DEMO_RUNS.exists():
        print("  examples.json — SKIPPED (run build_demo_data.py first)")
        return
    raw = json.load(open(DEMO_RUNS))
    n_vals = raw["n_values"]

    def annotate(scn):
        target = scn["target_key"]
        keys = scn["keys"]
        out_by_n = {}
        for N, b in scn["by_n"].items():
            stream = b["stream"]
            tvals = b["target_values"]
            nt = len(tvals)
            # mark each stream item with target flag + position among target values
            tpos_counter = 0
            items = []
            for it in stream:
                is_t = it["category"] == target
                tpos = tpos_counter if is_t else None
                if is_t:
                    tpos_counter += 1
                items.append({"key": it["category"], "value": it["value"],
                              "target": is_t, "tpos": tpos})

            def locate(pred):
                """Return (tpos, other_key) for the predicted value."""
                pl = pred.strip().lower()
                for i, v in enumerate(tvals):
                    if v.lower() == pl:
                        return i, None
                for it in stream:
                    if it["value"].lower() == pl and it["category"] != target:
                        return None, it["category"]
                return None, None

            def note(qd, expected_pos):
                tpos, other = locate(qd["predicted"])
                if qd["correct"]:
                    return "exact match"
                if tpos is not None:
                    rel = "earlier" if tpos < expected_pos else "later"
                    return f"returned an {rel} value of “{target}” (update {tpos+1} of {nt})"
                if other:
                    return f"returned a value from another key (“{other}”)"
                return "returned a value not in the stream"

            fvq, cvq = dict(b["fvq"]), dict(b["cvq"])
            fvq["predicted_tpos"], fvq["predicted_key"] = locate(fvq["predicted"])
            cvq["predicted_tpos"], cvq["predicted_key"] = locate(cvq["predicted"])
            fvq["note"] = note(fvq, 0)
            cvq["note"] = note(cvq, nt - 1)

            out_by_n[N] = {"stream": items, "n_target": nt,
                           "memory_now": b["memory_now"], "fvq": fvq, "cvq": cvq}
        return {"target_key": target, "keys": keys, "by_n": out_by_n}

    # curate: scenarios where FVQ holds and CVQ degrades with N (the clean story
    # first), but keep all real runs.
    def score(scn):
        bn = scn["by_n"]
        fv = sum(bn[str(n)]["fvq"]["correct"] for n in n_vals)
        # reward CVQ correct at low N and wrong at high N (a visible transition)
        lo = bn[str(n_vals[1])]["cvq"]["correct"]  # N=10
        hi = not bn[str(n_vals[-1])]["cvq"]["correct"]  # N=30 wrong
        return (fv, lo + hi)

    scen = sorted([annotate(s) for s in raw["scenarios"]],
                  key=score, reverse=True)
    json.dump({
        "model": raw["model"], "K": raw.get("K", 3), "n_values": n_vals,
        "default_n": 20,
        "note": ("Real Qwen2.5-3B-Instruct runs on the paper's interleaved key-value "
                 "task. K keys, each updated N times, shuffled so no two consecutive "
                 "updates share a key. “First/last value” = first/last occurrence of "
                 "the target key in the stream. Slide N up: the first value stays "
                 "recallable, the last value gets lost among the updates."),
        "scenarios": scen,
    }, open(OUT / "examples.json", "w"), indent=1)
    tn = scen[0]["target_key"]
    print(f"  examples.json — {len(scen)} scenarios, lead='{tn}', N={n_vals}")


# ── (legacy single-stream examples, kept for reference) ──────────────────────
def build_examples():
    data = json.load(open(LL_BASE))
    analyses = data["analyses"]
    # curate: clear RI-correct and PI-incorrect single-key streams at low load
    buckets = defaultdict(list)
    for a in analyses:
        if a["num_keys"] != 2 or a["num_updates"] not in (5, 10):
            continue
        vals = a["all_values"]
        if not (4 <= len(vals) <= 10):
            continue
        # For wrong answers, keep only clean *intrusions* — the model returned a
        # real other value from the stream — not single-token-argmax garbage
        # ("ug" for "ugly"). Intrusions are what makes the interference legible.
        if not a["correct"]:
            if a.get("predicted") not in vals:
                continue
            if a.get("error_type") not in ("intermediate_intrusion", "recency_intrusion",
                                           "primacy_intrusion"):
                continue
        buckets[(a["condition"], a["correct"])].append(a)

    examples = []
    seen = set()
    # interleave so the page shows the asymmetry: RI win, PI fail, ...
    order = [("RI", True), ("PI", False)] * 6 + [("PI", True), ("RI", False)] * 2
    idx = defaultdict(int)
    for key in order:
        pool = buckets.get(key, [])
        while idx[key] < len(pool):
            a = pool[idx[key]]
            idx[key] += 1
            sig = tuple(a["all_values"])
            if sig in seen:
                continue
            seen.add(sig)
            examples.append({
                "values": a["all_values"],
                "query": a["query_word"],          # "first" | "last"
                "condition": a["condition"],        # RI | PI
                "expected": a["expected"],
                "predicted": a["predicted"],
                "correct": a["correct"],
                "error_type": a.get("error_type"),
                "predicted_idx": a.get("predicted_idx"),
                "expected_idx": a.get("expected_idx"),
            })
            break
        if len(examples) >= 16:
            break

    json.dump({
        "model": "Qwen2.5-3B-Instruct",
        "note": ("Real base-model predictions on single-key streams. The key is "
                 "updated several times; we ask for its FIRST value (RI) or LAST "
                 "value (PI). Models reliably recall the first, but overwrite and "
                 "lose the last."),
        "examples": examples,
    }, open(OUT / "examples.json", "w"), indent=1)
    n_ri = sum(1 for e in examples if e["condition"] == "RI")
    print(f"  examples.json — {len(examples)} streams ({n_ri} RI / {len(examples)-n_ri} PI)")


# ── §3  logit lens ───────────────────────────────────────────────────────────
def _curves_by_layer(analyses, condition, K, N, n_layers):
    """Per-layer averaged probabilities for matching trials:
       correct   = P(the correct value)
       predicted = P(the value the model actually outputs)   [0 if a non-value token]
       top_other = P(the strongest *wrong* tracked value)    (the competitor)
       total     = P(any tracked value)                      (total readable mass)
    """
    sel = [a for a in analyses
           if a["condition"] == condition and a["num_keys"] == K and a["num_updates"] == N]
    if not sel:
        return None
    corr = [0.0] * n_layers
    pred = [0.0] * n_layers
    other = [0.0] * n_layers
    total = [0.0] * n_layers
    for a in sel:
        vp = a["value_probs_by_layer"]          # [n_values][n_layers]
        eidx = a["expected_idx"]
        pidx = a.get("predicted_idx", -1)
        nv = len(vp)
        for L in range(n_layers):
            corr[L] += vp[eidx][L]
            if pidx is not None and 0 <= pidx < nv:
                pred[L] += vp[pidx][L]
            total[L] += sum(vp[v][L] for v in range(nv))
            # strongest wrong tracked value at this layer
            other[L] += max((vp[v][L] for v in range(nv) if v != eidx), default=0.0)
    n = len(sel)
    acc = sum(1 for a in sel if a.get("correct")) / n
    return {
        "correct":   [round(corr[L] / n, 4) for L in range(n_layers)],
        "predicted": [round(pred[L] / n, 4) for L in range(n_layers)],
        "top_other": [round(other[L] / n, 4) for L in range(n_layers)],
        "total":     [round(total[L] / n, 4) for L in range(n_layers)],
        "accuracy":  round(acc, 3),
        "n":         n,
    }


def _gemma_curves(cc, condition):
    """Reconstruct correct/predicted/top_other/total from Gemma's aggregated
    per-position probabilities (p_pos_per_layer)."""
    pp = cc["p_pos_per_layer"]               # [n_pos][n_layers]
    n_layers = len(pp[0]); npos = len(pp)
    expected = 0 if condition == "RI" else npos - 1
    final = [pp[k][n_layers - 1] for k in range(npos)]
    pred = max(range(npos), key=lambda k: final[k])   # most-favored position at output
    return {
        "correct":   [round(pp[expected][L], 4) for L in range(n_layers)],
        "predicted": [round(pp[pred][L], 4) for L in range(n_layers)],
        "top_other": [round(max((pp[k][L] for k in range(npos) if k != expected), default=0.0), 4)
                      for L in range(n_layers)],
        "total":     [round(sum(pp[k][L] for k in range(npos)), 4) for L in range(n_layers)],
        "accuracy":  round(cc.get("behavioral_accuracy", 0.0), 3),
        "n":         cc.get("n_trials", 0),
    }


def build_logit_lens():
    cells = [(2, 5), (2, 10), (2, 50)]
    cell_labels = [{"K": k, "N": n, "label": f"K={k}, N={n}"} for k, n in cells]
    models = {}

    # ── Qwen (per-trial analyses schema) ──
    qb, ql = json.load(open(LL_BASE)), json.load(open(LL_LORA))
    nlq = qb["n_layers"]
    qser = {"base": {"PI": {}, "RI": {}}, "lora": {"PI": {}, "RI": {}}}
    for tag, data in (("base", qb), ("lora", ql)):
        for cond in ("PI", "RI"):
            for (K, N) in cells:
                v = _curves_by_layer(data["analyses"], cond, K, N, nlq)
                if v is not None:
                    qser[tag][cond][f"{K}_{N}"] = v
    models["Qwen2.5-3B-Instruct"] = {
        "n_layers": nlq, "cells": cell_labels, "series": qser,
        "dataset": "arbitrary single-token (paper §7 probe)"}

    out = {
        "models": models,
        "default_model": "Qwen2.5-3B-Instruct",
        "default_cond": "PI",
        "default_cell": "2_5",   # clearest surfaced-then-suppressed peak
        "note": ("Logit lens: probability mass on tracked values when decoding the "
                 "residual stream at each layer. correct = the queried value; competitor = "
                 "the strongest wrong value; total = mass on any tracked value. Values are "
                 "unreadable (≈0) until the last few layers. For the current-value query, "
                 "the correct value surfaces in the late layers then is suppressed before "
                 "the output; +LoRA removes the suppression."),
    }
    json.dump(out, open(OUT / "logit_lens.json", "w"), indent=1)
    print(f"  logit_lens.json — Qwen ({nlq}L), 4 curves/cell")


# ── §4  probing ──────────────────────────────────────────────────────────────
def _probe_series(d, key):
    pr = d["probe_results"][key]
    layers = sorted((int(k) for k in pr), key=int)
    return [round(pr[str(L)]["accuracy"], 4) for L in layers], layers


def build_probing():
    base = json.load(open(PROBE_BASE))
    lora = json.load(open(PROBE_LORA))
    _, layers = _probe_series(base, "condition_probe")
    out = {
        "model": "Qwen2.5-3B-Instruct",
        "point": base["point"],
        "trials": base["trials"],
        "layers": layers,
        "behavioral": {
            "base": {"RI": base["behavioral"]["RI"]["accuracy"],
                     "PI": base["behavioral"]["PI"]["accuracy"]},
            "lora": {"RI": lora["behavioral"]["RI"]["accuracy"],
                     "PI": lora["behavioral"]["PI"]["accuracy"]},
        },
        "probes": {},
        "note": ("Linear probe accuracy decoding the residual stream. "
                 "condition: can we tell RI vs PI is being asked (always ~100%). "
                 "RI_correct / PI_correct: does the model encode whether it will get "
                 "the answer right? Baseline PI-correctness sits at chance in late "
                 "layers; +LoRA installs the signal (L27-L33 jump to 77-94%)."),
    }
    for key in ("condition_probe", "RI_correct_probe", "PI_correct_probe"):
        b, _ = _probe_series(base, key)
        l, _ = _probe_series(lora, key)
        out["probes"][key] = {"base": b, "lora": l}
    json.dump(out, open(OUT / "probing.json", "w"), indent=1)
    print(f"  probing.json — {len(layers)} layers, "
          f"PI base->lora L33: {out['probes']['PI_correct_probe']['base'][-1]:.2f}"
          f"->{out['probes']['PI_correct_probe']['lora'][-1]:.2f}")


# ── §5  intervention headline ────────────────────────────────────────────────
def build_intervention():
    # Exact published values from paper/figures/tab_lora_sem.tex
    # (LoRA trained on ARB, evaluated on SEM — cross-distribution transfer).
    # FVQ / CVQ as fractions.
    cells = [
        {"K": 7,  "N": 20, "base": {"fvq": 0.60, "cvq": 0.55}, "lora": {"fvq": 1.0, "cvq": 1.0}},
        {"K": 15, "N": 20, "base": {"fvq": 0.25, "cvq": 0.55}, "lora": {"fvq": 1.0, "cvq": 1.0}},
        {"K": 15, "N": 30, "base": {"fvq": 0.35, "cvq": 0.80}, "lora": {"fvq": 1.0, "cvq": 1.0}},
        {"K": 20, "N": 30, "base": {"fvq": 0.25, "cvq": 0.65}, "lora": {"fvq": 1.0, "cvq": 1.0}},
    ]
    for c in cells:
        for v in ("base", "lora"):
            c[v]["gap"] = round(c[v]["fvq"] - c[v]["cvq"], 3)

    # Mechanism deltas from paper/figures/tab_mechanism.tex (bootstrap 95% CIs).
    mechanism = {
        "probe": [
            {"q": "CVQ-correctness probe, L33", "base": 0.58, "lora": 0.94, "ci": [0.21, 0.50]},
            {"q": "CVQ-correctness probe, L32", "base": 0.65, "lora": 0.94, "ci": [0.22, 0.36]},
            {"q": "FVQ-correctness probe, L33", "base": 0.81, "lora": 0.93, "ci": [0.05, 0.19]},
        ],
        "logit_lens": [
            {"q": "P(v_last) at L32, K=2,N=5",  "base": 0.34, "lora": 0.99, "ci": [0.55, 0.73]},
            {"q": "P(v_last) at L32, K=2,N=50", "base": 0.02, "lora": 0.62, "ci": [0.50, 0.69]},
            {"q": "P(v_last) at L35, K=2,N=50", "base": 0.11, "lora": 0.90, "ci": [0.71, 0.87]},
        ],
        "attention": [
            {"q": "P(attend v_last), L32H3",  "base": 0.11, "lora": 0.78, "ci": [0.65, 0.71]},
            {"q": "P(attend v_last), L31H12", "base": 0.16, "lora": 0.81, "ci": [0.60, 0.71]},
        ],
        "story": ("The skill was latent, not missing. Condition-discrimination, the "
                  "late-layer readout, and the v_last representation all pre-existed in "
                  "the baseline. LoRA fixes a calibration / readout failure — re-routing "
                  "attention to the last value and lifting it through the final layers — "
                  "not a representational one."),
    }
    out = {
        "model": "Qwen2.5-3B-Instruct",
        "adapter": {"type": "LoRA", "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
                    "r": 16, "alpha": 32, "size_mb": 28, "trained_on": "arbitrary-token task (ARB)",
                    "evaluated_on": "semantic task (SEM) — cross-distribution transfer"},
        "cells": cells,
        "mechanism": mechanism,
        "note": ("Behavioural numbers are the published values from the paper's LoRA "
                 "transfer table (Semantic-Multi eval); mechanism deltas carry the paper's "
                 "bootstrap 95% CIs."),
        "source": "paper/figures/tab_lora_sem.tex, tab_mechanism.tex",
    }
    json.dump(out, open(OUT / "intervention.json", "w"), indent=1)
    print(f"  intervention.json — {len(cells)} cells, all +LoRA -> 100%/100%")


# ── §3 (new)  position curve — where in the stream does memory fail? ─────────
def build_position_curve():
    if not DENSE_IVQ.exists():
        print("  position_curve.json — SKIPPED (dense_ivq file missing)")
        return
    d = json.load(open(DENSE_IVQ))
    r = d["results"]
    cells, seen = [], set()
    for split in ("train", "held_moderate", "held_hard"):
        for key in r["base"].get(split, {}):
            if key in seen:
                continue
            seen.add(key)
            K, N = map(int, key.split("_"))
            # positions actually sampled (may be a subset of 1..N for large cells)
            positions = sorted((int(p) for p in r["base"][split][key]), key=int)
            def curve(ver):
                cc = r[ver][split][key]
                return [round(cc[str(p)]["accuracy"], 3) for p in positions]
            cells.append({"key": key, "K": K, "N": N, "positions": positions,
                          "base": curve("base"), "lora": curve("lora")})
    cells.sort(key=lambda c: (c["K"], c["N"]))
    out = {
        "model": "Qwen2.5-3B-Instruct",
        "dataset": "arbitrary single-token, dense updates",
        "trials_per_position": d["trials_per_position"],
        "cells": cells,
        "default_cell": "5_10",
        "note": ("Ask for the value at every position k (not just first/last). Baseline: "
                 "a strong primacy spike at position 1, a mild recency bump at the last "
                 "position, and a dead zone in between — the model keeps the first write "
                 "and loses the rest. +LoRA flattens the whole curve: every position "
                 f"becomes retrievable. {d['trials_per_position']} trials/position, so "
                 "per-point CIs are wide; the shape is the result."),
    }
    json.dump(out, open(OUT / "position_curve.json", "w"), indent=1)
    print(f"  position_curve.json — {len(cells)} cells (positions 1..N), base vs LoRA")


# ── §4 (new)  CVQ vs load (N) ────────────────────────────────────────────────
LOAD_MODELS = ["Qwen2.5-3B-Instruct", "gemma-3-4b-it", "Qwen3.5-9B", "Qwen3.5-2B", "gemma-3-1b-it"]
def build_load():
    K = 5
    Ns = [5, 7, 10, 15, 20, 30, 50]
    models = []
    for mdir in LOAD_MODELS:
        files = sorted(glob.glob(str(SEM_DIR / mdir / "stage1_sweep_*.json")))
        if not files:
            continue
        cells = json.load(open(files[-1]))["cells"]
        fvq, cvq = [], []
        for N in Ns:
            c = cells.get(f"{K}_{N}")
            if c and "stats" in c and c.get("n_trials", 0) >= 30:
                fvq.append(round(c["stats"]["RI"]["accuracy"], 3))
                cvq.append(round(c["stats"]["PI"]["accuracy"], 3))
            else:
                fvq.append(None); cvq.append(None)
        fam = META.get(mdir, (None, None))[0]
        models.append({"model": disp(mdir), "family": fam, "fvq": fvq, "cvq": cvq})
    out = {
        "K": K, "N_values": Ns, "models": models,
        "default_model": "Gemma-3-4b-it",   # cleanest FVQ-flat / CVQ-decays profile
        "note": (f"First- vs current-value accuracy at K={K} keys as the number of updates "
                 "N grows. First-value recall stays high; current-value recall decays with "
                 "load — the gap is driven by how much gets written after the target, not by "
                 "context length (the prompts use <4% of the context window)."),
    }
    json.dump(out, open(OUT / "load.json", "w"), indent=1)
    print(f"  load.json — {len(models)} models, K={K}, N={Ns}")


if __name__ == "__main__":
    print("Building web/data/ ...")
    build_demo()
    build_scatter()
    build_position_curve()
    build_load()
    build_logit_lens()
    build_probing()
    build_intervention()
    print("Done.")
