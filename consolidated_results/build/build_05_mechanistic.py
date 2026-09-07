"""Theme 5 — mechanistic interpretability: HOW LoRA closes the CVQ gap.

Four methods, four natural shapes, one CSV each (a single tidy schema does not
fit — layers, heads, and per-cell trajectories are different units):

  logit_lens.csv        per (model, variant, cell, condition, layer) -> P(target
                        token) through the residual stream. "Is v_last decodable
                        late, and does LoRA raise it?"
  probing.csv           per (model, variant, probe, layer) -> linear-probe
                        accuracy. "Is the info linearly present regardless of
                        whether the model outputs it?"
  attention_routing.csv per (model, regime, layer|head) -> P(gen-token attends to
                        the v_last round), base vs LoRA. "Does LoRA re-point
                        attention at the last update?"
  causal_ablation.csv   per (model, variant, cell, layer) -> P(v_last) with vs
                        without the promoter heads ablated. "Are those heads
                        causally responsible?"

Provenance tiers as elsewhere: raw = from a results JSON on disk; derived =
back-extracted from a committed .tex / precomputed .txt because the raw run is
GPU-box only. Naming: RI->FVQ, PI->CVQ (paper terminology); v_first/v_last kept
as-is since that is the mechanistic quantity.

This theme is behavioural-scope's complement: themes 1-4 are what the model
outputs; this is what happens inside. Qwen2.5-3B-Instruct is the primary target,
gemma-3-4b-it the cross-family replication; base vs LoRA throughout.
"""

from __future__ import annotations

import csv
import glob
import json
import re
from pathlib import Path

from common import DERIVED, RAW, ROOT, canon_model, model_meta, rel

OUT = Path(__file__).resolve().parent.parent / "05_mechanistic"
QWEN = "Qwen2.5-3B-Instruct"
GEMMA = "gemma-3-4b-it"
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _model_variant(tag: str) -> tuple[str, str]:
    """'gemma-3-4b-it-LoRA' -> (gemma-3-4b-it, lora); '...-baseline-hf' -> base."""
    t = tag.replace("-baseline-hf", "").replace("-baseline", "")
    if t.endswith("-LoRA"):
        return canon_model(t[:-5]), "lora"
    return canon_model(t), "base"


def write(rows: list[dict], cols: list[str], name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    tiers = {}
    for r in rows:
        tiers[r.get("provenance_tier", "")] = tiers.get(r.get("provenance_tier", ""), 0) + 1
    print(f"  -> 05_mechanistic/{name}: {len(rows)} rows  "
          f"({', '.join(f'{k}={v}' for k, v in sorted(tiers.items()))})")


def _cvq(cond: str) -> str:
    return {"RI": "FVQ", "PI": "CVQ"}.get(cond, cond)


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOGIT LENS
# ═══════════════════════════════════════════════════════════════════════════
LL_COLS = ["model", "model_family", "variant", "dataset", "num_keys",
           "num_updates", "condition", "target", "layer", "p_target",
           "behavioral_acc", "ci_lower", "ci_upper", "provenance_tier",
           "source_file", "notes"]


def logit_lens_raw() -> list[dict]:
    """Gemma base + LoRA: per-layer P(v_first) and P(v_last), both conditions."""
    rows = []
    for path in sorted(glob.glob(str(ROOT / "v3" / "results_vllm" / "logit_lens" /
                                     "*" / "stage2_logit_lens_*.json"))):
        d = json.load(open(path))
        model, variant = _model_variant(d["model"])
        src = rel(path)
        for cell_key, conds in d["cells"].items():
            k, n = (int(x) for x in cell_key.split("_"))
            for cond, cv in conds.items():
                fvq_cvq = _cvq(cond)
                # the mechanistic target is v_first for RI/FVQ, v_last for PI/CVQ
                target = "v_first" if cond == "RI" else "v_last"
                series = cv["p_first_per_layer"] if cond == "RI" else cv["p_last_per_layer"]
                for layer, p in enumerate(series):
                    rows.append(dict(
                        model=model, model_family=model_meta(model)[1],
                        variant=variant, dataset="arbitrary_single",
                        num_keys=k, num_updates=n, condition=fvq_cvq,
                        target=target, layer=layer, p_target=round(p, 4),
                        behavioral_acc=cv.get("behavioral_accuracy", ""),
                        provenance_tier=RAW, source_file=src,
                        notes=f"latest of 2 timestamps kept" if "20260525_200056" in src
                              or "20260525_195739" in src else "",
                    ))
    # keep only the latest timestamp per (model,variant) to avoid dupes
    latest = {}
    for path in glob.glob(str(ROOT / "v3" / "results_vllm" / "logit_lens" / "*" /
                              "stage2_logit_lens_*.json")):
        tag = path.split("/")[-2]
        latest[tag] = max(latest.get(tag, ""), path.split("/")[-1])
    rows = [r for r in rows
            if r["source_file"].endswith(latest[r["source_file"].split("/")[-2]])]
    return rows


_LL_TXT = re.compile(
    r"^K=(\d+),N=(\d+)\s+(RI|PI)\s+(v_first|v_last)\s+L(\d+)\s+"
    r"([\d.]+)\s+\[([\d.]+),\s*([\d.]+)\]\s+([\d.]+)\s+\[([\d.]+),\s*([\d.]+)\]")


def logit_lens_from_ci(path: Path, model: str) -> list[dict]:
    """Qwen: base+lora per-layer P(target) with CIs from ci_analysis.txt."""
    if not path.exists():
        return []
    src = rel(path)
    text = path.read_text()
    # isolate the LOGIT LENS section
    if "LOGIT LENS" in text:
        text = text.split("LOGIT LENS", 1)[1]
    rows = []
    for line in text.splitlines():
        m = _LL_TXT.match(line.strip())
        if not m:
            continue
        k, n, cond, target, layer, bmean, blo, bhi, lmean, llo, lhi = m.groups()
        for variant, mean, lo, hi in (("base", bmean, blo, bhi),
                                      ("lora", lmean, llo, lhi)):
            rows.append(dict(
                model=model, model_family=model_meta(model)[1], variant=variant,
                dataset="arbitrary_single", num_keys=int(k), num_updates=int(n),
                condition=_cvq(cond), target=target, layer=int(layer),
                p_target=float(mean), ci_lower=float(lo), ci_upper=float(hi),
                provenance_tier=DERIVED, source_file=src,
                notes="from precomputed bootstrap CIs; raw Qwen logit-lens JSON not local",
            ))
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# 2. PROBING
# ═══════════════════════════════════════════════════════════════════════════
PR_COLS = ["model", "model_family", "variant", "num_keys", "num_updates",
           "probe", "layer", "accuracy", "std", "ci_lower", "ci_upper",
           "behavioral_acc", "provenance_tier", "source_file", "notes"]

_PROBE_NAME = {"condition_probe": "cond_discrimination",
               "RI_correct_probe": "FVQ_correct",
               "PI_correct_probe": "CVQ_correct"}


def probing_raw() -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(ROOT / "v3" / "results_vllm" / "probing" /
                                     "probing_*_2k_5u.json"))):
        d = json.load(open(path))
        model, variant = _model_variant(d["model"])
        src = rel(path)
        pt = d.get("point", {})
        for probe, layers in d["probe_results"].items():
            for layer, cv in layers.items():
                rows.append(dict(
                    model=model, model_family=model_meta(model)[1], variant=variant,
                    num_keys=pt.get("keys", 2), num_updates=pt.get("updates", 5),
                    probe=_PROBE_NAME.get(probe, probe), layer=int(layer),
                    accuracy=round(cv["accuracy"], 4), std=round(cv.get("std", 0), 4),
                    provenance_tier=RAW, source_file=src,
                    notes="8 late layers indexed 0-7 (absolute layer in ci_analysis)",
                ))
    return rows


_PR_TXT = re.compile(
    r"^(Cond discr \(RI vs PI\)|RI correctness|PI correctness)\s+L(\d+)\s+"
    r"([\d.]+)\s+\[([\d.]+),\s*([\d.]+)\]\s+([\d.]+)\s+\[([\d.]+),\s*([\d.]+)\]")

_PR_TXT_NAME = {"Cond discr (RI vs PI)": "cond_discrimination",
                "RI correctness": "FVQ_correct", "PI correctness": "CVQ_correct"}


def probing_from_ci(path: Path, model: str) -> list[dict]:
    """base+lora probe accuracy with CIs at absolute layers, from ci_analysis.txt."""
    if not path.exists():
        return []
    src = rel(path)
    text = path.read_text()
    text = text.split("LOGIT LENS", 1)[0]  # probing is before the logit-lens block
    rows = []
    for line in text.splitlines():
        m = _PR_TXT.match(line.strip())
        if not m:
            continue
        name, layer, bmean, blo, bhi, lmean, llo, lhi = m.groups()
        for variant, mean, lo, hi in (("base", bmean, blo, bhi),
                                      ("lora", lmean, llo, lhi)):
            rows.append(dict(
                model=model, model_family=model_meta(model)[1], variant=variant,
                num_keys=2, num_updates=5, probe=_PR_TXT_NAME[name],
                layer=int(layer), accuracy=float(mean), ci_lower=float(lo),
                ci_upper=float(hi), provenance_tier=DERIVED, source_file=src,
                notes="precomputed bootstrap CIs (absolute layer index)",
            ))
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# 3. ATTENTION ROUTING
# ═══════════════════════════════════════════════════════════════════════════
AR_COLS = ["model", "model_family", "regime", "num_keys", "num_updates", "unit",
           "layer", "head", "base", "lora", "delta", "provenance_tier",
           "source_file", "notes"]

_AR_LAYER = re.compile(r"^L\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([+-][\d.]+)")
_AR_HEAD = re.compile(r"L(\d+)H(\d+)\s+\(([\d.]+)\s*→\s*([\d.]+),\s*([+-][\d.]+)\)")


def attention_routing_txt(path: Path, model: str, regime: str, k: int, n: int) -> list[dict]:
    if not path.exists():
        return []
    src = rel(path)
    rows = []
    for line in path.read_text().splitlines():
        s = line.strip()
        ml = _AR_LAYER.match(s)
        if ml:
            L, b, l, dl = ml.groups()
            rows.append(dict(
                model=model, model_family=model_meta(model)[1], regime=regime,
                num_keys=k, num_updates=n, unit="layer_mean", layer=int(L),
                base=float(b), lora=float(l), delta=float(dl),
                provenance_tier=DERIVED, source_file=src,
                notes="head-averaged P(attend to v_last round), CVQ condition; "
                      "raw per-head JSON not local"))
        for mh in _AR_HEAD.finditer(line):
            L, H, b, l, dl = mh.groups()
            rows.append(dict(
                model=model, model_family=model_meta(model)[1], regime=regime,
                num_keys=k, num_updates=n, unit="head", layer=int(L), head=int(H),
                base=float(b), lora=float(l), delta=float(dl),
                provenance_tier=DERIVED, source_file=src,
                notes="top mover head (promoter); P(attend to v_last round)"))
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# 4. CAUSAL ABLATION (stage 3)
# ═══════════════════════════════════════════════════════════════════════════
CA_COLS = ["model", "model_family", "variant", "condition_set", "num_keys",
           "num_updates", "layer", "p_last_normal", "p_last_ablated", "delta",
           "n_heads_ablated", "provenance_tier", "source_file", "notes"]


def causal_ablation_stage3() -> list[dict]:
    """Per-layer P(v_last) trajectory, normal vs promoter-head-ablated."""
    rows = []
    for path in sorted(glob.glob(str(ROOT / "v3" / "scripts" / "experiments" /
                                     "results" / "*" / "stage3_causal_*.json"))):
        d = json.load(open(path))
        tag = d["model"]
        # tag e.g. 'Qwen2.5-3B-Instruct-LoRA-promoters' / '-baseline-promoters'
        variant = "lora" if "-LoRA" in tag else "base"
        cset = tag.split("-")[-1]  # promoters | clean5
        model = QWEN
        src = rel(path)
        op = d.get("operating_point", {})
        exp = d.get("experiments", {}).get("3C", {})
        normal = exp.get("avg_normal_trajectory")
        ablated = exp.get("avg_ablated_trajectory")
        if not normal or not ablated:
            continue
        # trajectories are per-layer lists of [p_first, p_last] or scalars
        def plast(x):
            return x[1] if isinstance(x, (list, tuple)) else x
        for layer, (nrm, abl) in enumerate(zip(normal, ablated)):
            pn, pa = plast(nrm), plast(abl)
            rows.append(dict(
                model=model, model_family=model_meta(model)[1], variant=variant,
                condition_set=cset, num_keys=op.get("keys", 2),
                num_updates=op.get("updates", 5), layer=layer,
                p_last_normal=round(pn, 4), p_last_ablated=round(pa, 4),
                delta=round(pa - pn, 4),
                n_heads_ablated=len(exp.get("heads_ablated", [])),
                provenance_tier=RAW, source_file=src,
                notes=f"stage3 3C; heads={','.join(exp.get('heads_ablated', [])[:3])}..."))
    return rows


def causal_ablation_rerun() -> list[dict]:
    """n=200 paired ablation (Qwen + Gemma): per-layer mean p_last, base & lora."""
    rows = []
    specs = [(ROOT / "lora_intervention/experiments/ablation_rerun_results/results.json", QWEN),
             (ROOT / "lora_intervention/experiments/gemma_results/ablation/results.json", GEMMA)]
    for path, model in specs:
        if not path.exists():
            continue
        d = json.load(open(path))
        src = rel(path)
        heads = d.get("promoter_heads", [])
        for cell_key, variants in d["cells"].items():
            k, n = re.match(r"K(\d+)N(\d+)", cell_key).groups()
            for variant, vd in variants.items():
                per_trial = vd.get("per_trial", [])
                if not per_trial:
                    continue
                # per_trial: list of dicts {layer_str: [p_normal, p_ablated]}
                layers = sorted({int(L) for t in per_trial for L in t}, )
                for L in layers:
                    ns = [t[str(L)][0] for t in per_trial if str(L) in t]
                    as_ = [t[str(L)][1] for t in per_trial if str(L) in t]
                    if not ns:
                        continue
                    mn, ma = sum(ns) / len(ns), sum(as_) / len(as_)
                    rows.append(dict(
                        model=model, model_family=model_meta(model)[1],
                        variant=variant, condition_set="promoters_n200",
                        num_keys=int(k), num_updates=int(n), layer=L,
                        p_last_normal=round(mn, 4), p_last_ablated=round(ma, 4),
                        delta=round(ma - mn, 4), n_heads_ablated=len(heads),
                        provenance_tier=RAW, source_file=src,
                        notes=f"n=200 paired; {cell_key}; heads cluster {heads[0] if heads else ''}"))
    return rows


def main() -> None:
    print("[05_mechanistic] interpretability: how LoRA closes the gap")

    # 1. logit lens: gemma raw + qwen derived
    ll = logit_lens_raw() + logit_lens_from_ci(
        ROOT / "lora_intervention/results/ci_analysis.txt", QWEN)
    ll.sort(key=lambda r: (r["model"], r["variant"], r["num_keys"],
                           r["num_updates"], r["condition"], int(r["layer"])))
    write(ll, LL_COLS, "logit_lens.csv")

    # 2. probing: gemma+qwen-lora raw + qwen base/lora derived (with CIs)
    pr = probing_raw() + probing_from_ci(
        ROOT / "lora_intervention/results/ci_analysis.txt", QWEN)
    pr.sort(key=lambda r: (r["model"], r["variant"], r["probe"], int(r["layer"])))
    write(pr, PR_COLS, "probing.csv")

    # 3. attention routing: qwen primacy + reversal (txt; raw per-head JSON gone)
    ar = attention_routing_txt(
        ROOT / "lora_intervention/results/attention_routing_comparison.txt",
        QWEN, "primacy", 2, 30)
    ar += attention_routing_txt(
        ROOT / "lora_intervention/results/attention_routing_reversal_comparison.txt",
        QWEN, "reversal", 15, 20)
    ar.sort(key=lambda r: (r["regime"], r["unit"], int(r["layer"]),
                           int(r.get("head") or -1)))
    write(ar, AR_COLS, "attention_routing.csv")

    # 4. causal ablation: stage3 trajectories + n=200 rerun (qwen+gemma)
    ca = causal_ablation_stage3() + causal_ablation_rerun()
    ca.sort(key=lambda r: (r["model"], r["condition_set"], r["variant"],
                           int(r["num_keys"]), int(r["num_updates"]), int(r["layer"])))
    write(ca, CA_COLS, "causal_ablation.csv")


if __name__ == "__main__":
    main()
