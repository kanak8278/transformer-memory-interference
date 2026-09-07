"""Theme 5, Entropy-Lens — writes into 05_mechanistic/entropy_lens/.

Entropy-Lens (arXiv:2502.16570) on the KV-interference task: one forward pass
per trial, and at the answer position a logit lens at every layer --
`softmax(unembed(final_norm(resid_post_L)))` over the full vocab -> Shannon
entropy. Stored as raw nats and normalised `H/log(V)`. Correctness comes free
from the same forward (single-token argmax), so profiles split into
all / correct / wrong.

Layout, one subfolder per ARM (see the naming note below):

  entropy_lens/
    summary.csv                 the cross-arm comparison, 3 arms x 5 conditions
    qwen_base/    profiles.csv  per-layer entropy, cell x condition x subset x layer
                  confusion.csv output-layer entropy, correct vs wrong, per cell
                  behavioral.csv the free correctness labels, in the SHARED
                                 tidy-long schema so they can be concatenated
                                 with themes 01-04/06
    qwen_lora/    (same three)
    gpt2_scratch/ (same three)

NAMING: subfolders are by *arm*, not by model, because `base` and `lora` are the
SAME model (Qwen2.5-3B-Instruct) with and without the adapter -- a by-model
split would collide. The `model` and `variant` columns carry the distinction on
every row.

MODELS: Qwen + the from-scratch GPT-2 only. **There is no Gemma arm.** Within
theme 05 that puts entropy alongside `attention_routing.csv`, which is also
Qwen-only. A Gemma run is a flag change, not new code -- `run_entropy_lens.py`
takes `--base/--adapter` and `lora_intervention/checkpoints/gemma_adapter`
exists -- and would cost ~30 min of GPU time.

GRID: K in {2,4,6,8,10,12} x N in {4,6,8,10,12} = 30 cells, 150 trials per
(cell, condition), 5 conditions = 22,500 scored trials per arm. This grid was
chosen as **the from-scratch model's native training range**, the only range all
three arms can run. It therefore does NOT overlap `04_lora/lora_ivq.csv` in a
single cell (that grid starts at N=10 and pairs it only with K in {5,15,...}),
so the Qwen rows here are the small-K/small-N regime that is otherwise absent
from this folder. Do not read LoRA's 0.81-0.93 interior here as contradicting
the 0.20-0.29 quoted in theme 04 -- different regime, not a disagreement.

QUERY TYPE IS NOT UNIFORM, and it is not uniform across arms either:
  qwen base/lora   FVQ,CVQ -> semantic ("What was the first/last value of X?")
                   IVQ_d*  -> ordinal  ("What was the 6th value of X?")
  scratch          all     -> ordinal  (a step token <S1>..<S12>)
So the IVQ comparisons are like-for-like across arms; the FVQ/CVQ ones are not
(Qwen gets a semantic anchor, the scratch model gets a numeric index). This
folder's own headline finding is that the two phrasings can differ by 138x on
the same target, so `query_type` is set per condition per arm, never uniformly.
`entropy_lens/RUN.md` claims the arms are matched on "query-type"; they are not,
for the endpoints.

Sources
  RAW  lora_intervention/experiments/entropy_lens/results/entropy_base.json
  RAW  lora_intervention/experiments/entropy_lens/results/entropy_lora.json
  RAW  lora_intervention/experiments/entropy_lens/results/entropy_scratch.json
  QA   the scratch behavioural labels vs 06_from_scratch/from_scratch_ivq.csv
       (independent re-measurement of the same model on the same 30 cells)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from common import (COLUMNS, ORDINAL, RAW, ROOT, SEMANTIC, model_meta, rel,
                    with_model, write_csv)

OUT = Path(__file__).resolve().parent.parent / "05_mechanistic" / "entropy_lens"
SRC = (ROOT / "lora_intervention" / "experiments" / "entropy_lens" / "results")

# arm -> (subfolder, model, variant, dataset, prompt_format)
ARMS = {
    "base":    ("qwen_base", "Qwen2.5-3B-Instruct", "base",
                "arbitrary_single", "chat_template"),
    "lora":    ("qwen_lora", "Qwen2.5-3B-Instruct", "lora",
                "arbitrary_single", "chat_template"),
    "scratch": ("gpt2_scratch", "gpt2-small", "scratch_cosine",
                "synthetic_kv", "token_stream"),
}
CONDS = ["FVQ", "CVQ", "IVQ_d25", "IVQ_d50", "IVQ_d75"]
IVQ_DEPTHS = {"IVQ_d25": 0.25, "IVQ_d50": 0.50, "IVQ_d75": 0.75}
SUBSETS = ("all", "correct", "wrong")


def ivq_step(depth: float, n: int) -> int:
    """clip(round(d*N), 2, N-1) -- same rule as linear_probing and the run."""
    return min(max(round(depth * n), 2), n - 1)


def cell_kn(name: str) -> tuple[int, int]:
    """'K12_N10' -> (12, 10)."""
    k, n = name.split("_")
    return int(k[1:]), int(n[1:])


def query_type_for(arm: str, cond: str) -> str:
    if arm == "scratch":
        return ORDINAL            # a step token, even for first/last
    return SEMANTIC if cond in ("FVQ", "CVQ") else ORDINAL


def position_for(cond: str, n: int):
    if cond == "FVQ":
        return "first"
    if cond == "CVQ":
        return "last"
    return ivq_step(IVQ_DEPTHS[cond], n)


# ── per-layer profiles ──────────────────────────────────────────────────────

PROFILE_COLS = ["arm", "model", "variant", "num_keys", "num_updates",
                "condition", "query_type", "subset", "layer", "rel_depth",
                "entropy_nats", "entropy_norm", "n_trials", "provenance_tier",
                "source_file"]


def build_profiles(arm: str, d: dict, src: str) -> list[dict]:
    """Arrays are indexed layer 1..n_layers; hidden_states[0] (the embedding
    output) is deliberately not lensed. rel_depth = layer/n_layers in (0,1],
    which is what makes 36-layer Qwen and 12-layer GPT-2 comparable."""
    _, model, variant, _, _ = ARMS[arm]
    nl = d["n_layers"]
    rows = []
    for cname, cell in d["cells"].items():
        k, n = cell_kn(cname)
        for cond in CONDS:
            v = cell.get(cond)
            if not v:
                continue
            for sub in SUBSETS:
                s = v.get(sub)
                if not s or not s.get("n"):
                    continue      # e.g. `wrong` is None when the arm was perfect
                for i, (nats, nrm) in enumerate(zip(s["raw"], s["norm"]), start=1):
                    rows.append({
                        "arm": arm, "model": model, "variant": variant,
                        "num_keys": k, "num_updates": n, "condition": cond,
                        "query_type": query_type_for(arm, cond), "subset": sub,
                        "layer": i, "rel_depth": round(i / nl, 6),
                        "entropy_nats": nats, "entropy_norm": nrm,
                        "n_trials": s["n"], "provenance_tier": RAW,
                        "source_file": src,
                    })
    return rows


# ── the confusion view: output-layer entropy, correct vs wrong ──────────────

CONFUSION_COLS = ["arm", "model", "variant", "num_keys", "num_updates",
                  "condition", "query_type", "behavioral_accuracy", "n_trials",
                  "n_correct", "n_wrong", "H_out_all", "H_out_correct",
                  "H_out_wrong", "confusion_gap", "provenance_tier",
                  "source_file", "notes"]


def build_confusion(arm: str, d: dict, src: str) -> list[dict]:
    """`confusion_gap` = H_out(wrong) - H_out(correct), normalised entropy.

    ~0 means the arm is equally confident whether right or wrong ("confidently
    wrong"); large positive means its uncertainty tracks its correctness. Blank
    when either subset is empty -- read `n_wrong` before trusting a gap, some
    LoRA cells have n_wrong=1.
    """
    _, model, variant, _, _ = ARMS[arm]
    rows = []
    for cname, cell in d["cells"].items():
        k, n = cell_kn(cname)
        for cond in CONDS:
            v = cell.get(cond)
            if not v:
                continue
            out = {}
            for sub in SUBSETS:
                s = v.get(sub)
                out[sub] = (s["norm"][-1], s["n"]) if s and s.get("n") else (None, 0)
            hc, nc = out["correct"]
            hw, nw = out["wrong"]
            gap = (hw - hc) if (hc is not None and hw is not None) else ""
            note = f"chance {1/10:.2f}" if arm == "scratch" else ""
            if nw and nw < 20:
                note = (note + "; " if note else "") + \
                    f"n_wrong={nw} -- gap is noisy, do not quote this cell alone"
            rows.append({
                "arm": arm, "model": model, "variant": variant, "num_keys": k,
                "num_updates": n, "condition": cond,
                "query_type": query_type_for(arm, cond),
                "behavioral_accuracy": v["behavioral_accuracy"],
                "n_trials": v["n"], "n_correct": nc, "n_wrong": nw,
                "H_out_all": out["all"][0], "H_out_correct": hc,
                "H_out_wrong": hw, "confusion_gap": gap,
                "provenance_tier": RAW, "source_file": src, "notes": note,
            })
    return rows


# ── the free behavioural labels, in the SHARED schema ───────────────────────

def build_behavioral(arm: str, d: dict, src: str) -> list[dict]:
    subdir, model, variant, dataset, fmt = ARMS[arm]
    rows = []
    for cname, cell in d["cells"].items():
        k, n = cell_kn(cname)
        for cond in CONDS:
            v = cell.get(cond)
            if not v:
                continue
            base_cond = "IVQ" if cond.startswith("IVQ") else cond
            acc = v["behavioral_accuracy"]
            nt = v["n"]
            note = (f"Entropy-Lens run; correctness scored free from the "
                    f"logit-lens forward pass (single-token argmax)")
            if cond.startswith("IVQ"):
                note += (f"; relative depth {IVQ_DEPTHS[cond]:.2f} -> "
                         f"step {ivq_step(IVQ_DEPTHS[cond], n)} of {n}")
            if arm == "scratch":
                note += "; chance 0.10; query is a step token, so FVQ/CVQ are ordinal too"
            rows.append(with_model(
                model, theme="05_mechanistic", variant=variant, dataset=dataset,
                prompt_format=fmt, num_keys=k, num_updates=n,
                condition=base_cond, query_type=query_type_for(arm, cond),
                position=position_for(cond, n), accuracy=acc, n_trials=nt,
                n_correct=int(round(acc * nt)), provenance_tier=RAW,
                source_file=src, notes=note,
            ))
    return rows


# ── the cross-arm summary ───────────────────────────────────────────────────

SUMMARY_COLS = ["arm", "model", "model_family", "variant", "n_layers", "vocab",
                "condition", "query_type", "behavioral_accuracy", "n_trials",
                "output_entropy", "peak_entropy", "peak_depth",
                "expansion_amplitude", "pruning_drop", "H_out_correct",
                "H_out_wrong", "confusion_gap", "n_wrong", "provenance_tier",
                "source_file", "notes"]


def build_summary(loaded: dict, confusion: dict, behav: dict) -> list[dict]:
    """One row per (arm, condition): the geometry scalars from
    entropy_summary.json plus the pooled behavioural accuracy and pooled
    confusion gap recomputed from the per-cell records."""
    summ = json.load(open(SRC / "entropy_summary.json"))
    rows = []
    for arm, (subdir, model, variant, _, _) in ARMS.items():
        m, fam, _ = model_meta(model)
        d = loaded[arm]
        for cond in CONDS:
            g = summ[arm]["conditions"][cond]
            crows = [r for r in confusion[arm] if r["condition"] == cond]
            nt = sum(r["n_trials"] for r in crows)
            ncorr = sum(r["n_correct"] for r in crows)
            nw = sum(r["n_wrong"] for r in crows)
            wsum = sum(r["H_out_wrong"] * r["n_wrong"] for r in crows
                       if r["H_out_wrong"] is not None)
            csum = sum(r["H_out_correct"] * r["n_correct"] for r in crows
                       if r["H_out_correct"] is not None)
            hw = wsum / nw if nw else None
            hc = csum / ncorr if ncorr else None
            rows.append({
                "arm": arm, "model": m, "model_family": fam, "variant": variant,
                "n_layers": d["n_layers"], "vocab": d["vocab_size"],
                "condition": cond, "query_type": query_type_for(arm, cond),
                "behavioral_accuracy": round(ncorr / nt, 6) if nt else "",
                "n_trials": nt,
                "output_entropy": g["output_entropy"],
                "peak_entropy": g["peak_entropy"],
                "peak_depth": g["peak_depth"],
                "expansion_amplitude": g["expansion_amplitude"],
                "pruning_drop": g["pruning_drop"],
                "H_out_correct": hc if hc is not None else "",
                "H_out_wrong": hw if hw is not None else "",
                "confusion_gap": (hw - hc) if (hw is not None and hc is not None) else "",
                "n_wrong": nw, "provenance_tier": RAW,
                "source_file": rel(SRC / "entropy_summary.json"),
                "notes": ("geometry scalars from entropy_summary.json; "
                          "behavioural + confusion pooled over the 30 cells"),
            })
    return rows


# ── QA: the scratch arm independently re-measures theme 06 ─────────────────

def qa_scratch_vs_theme06(behav_scratch: list[dict]) -> None:
    """The scratch arm re-runs the theme-06 model on the same 30 cells with a
    different seed formula and n=150 (vs 200/500), loading the checkpoint from
    the HF Hub rather than the local .pt. So it is an independent replication;
    the two should agree within binomial noise."""
    p = OUT.parent.parent / "06_from_scratch" / "from_scratch_ivq.csv"
    if not p.exists():
        print("     QA vs theme 06: skipped (06_from_scratch not built yet)")
        return
    ref = {}
    for r in csv.DictReader(open(p)):
        k, n = int(r["num_keys"]), int(r["num_updates"])
        pos = r["position"]
        step = 1 if pos == "first" else n if pos == "last" else int(pos)
        ref[(k, n, step)] = float(r["accuracy"])

    diffs = []
    for r in behav_scratch:
        k, n = int(r["num_keys"]), int(r["num_updates"])
        pos = r["position"]
        step = 1 if pos == "first" else n if pos == "last" else int(pos)
        got = ref.get((k, n, step))
        if got is not None:
            diffs.append(abs(float(r["accuracy"]) - got))
    if not diffs:
        print("     QA vs theme 06: no matched cells")
        return
    diffs.sort()
    mean = sum(diffs) / len(diffs)
    print(f"     QA scratch arm vs 06_from_scratch (independent re-measure): "
          f"{len(diffs)} cells, mean |diff| {mean:.4f}, median "
          f"{diffs[len(diffs)//2]:.4f}, max {diffs[-1]:.4f}, "
          f"{sum(1 for x in diffs if x <= 0.10)}/{len(diffs)} within 0.10")


def write(rows: list[dict], cols: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def main() -> None:
    print("[05_mechanistic/entropy_lens] Entropy-Lens: base vs LoRA vs from-scratch")
    loaded, confusion, behav = {}, {}, {}
    for arm in ARMS:
        path = SRC / f"entropy_{arm}.json"
        loaded[arm] = json.load(open(path))
        src = rel(path)
        subdir = ARMS[arm][0]
        prof = build_profiles(arm, loaded[arm], src)
        confusion[arm] = build_confusion(arm, loaded[arm], src)
        behav[arm] = build_behavioral(arm, loaded[arm], src)
        write(prof, PROFILE_COLS, OUT / subdir / "profiles.csv")
        write(confusion[arm], CONFUSION_COLS, OUT / subdir / "confusion.csv")
        write_csv(behav[arm], OUT / subdir / "behavioral.csv")
        print(f"  -> entropy_lens/{subdir}/: profiles {len(prof)}, "
              f"confusion {len(confusion[arm])}, behavioral {len(behav[arm])} rows "
              f"({loaded[arm]['n_layers']}L, V={loaded[arm]['vocab_size']})")

    summ = build_summary(loaded, confusion, behav)
    write(summ, SUMMARY_COLS, OUT / "summary.csv")
    print(f"  -> entropy_lens/summary.csv: {len(summ)} rows "
          f"({len(ARMS)} arms x {len(CONDS)} conditions)")
    qa_scratch_vs_theme06(behav["scratch"])


if __name__ == "__main__":
    main()
