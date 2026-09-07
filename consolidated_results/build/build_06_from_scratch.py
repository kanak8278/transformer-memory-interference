"""Theme 6 — from-scratch training: the effect is not inherited from pretraining.

A GPT-2-small *architecture* with a 51-symbol closed vocabulary, trained from
random init on the synthetic key-value task (`synthetic_scratch_training/
05_gpt2_scratch_h100`). No pretrained weights, no natural-language tokens, no
prompt — the input is a raw token stream, so `prompt_format=token_stream` and
`query_type=ordinal` on every row (the query is a step *token* `<S1>..<S12>`,
a numeric position index; this experiment has no semantic "the last value"
anchor arm at all).

SCOPE (decided 2026-09-07): this theme consolidates the `h100_cosine` arm
ONLY -- the run that escaped the loss plateau and reached iid 0.800 /
held-out 0.633, and the only arm any figure in `present/ONE_PAGER.tex` (S7) is
built from.

The sibling `h100_plateau` arm is NOT here. It is a real, complete 30,000-step
run (same seed, same data, LR schedule the only difference) that never escaped
the plateau and finished at iid 0.283 / held-out 0.208. It is the control that
separates "the LR decay enabled this" from "we just trained longer", and if that
claim is ever made in print it needs to come back in. Left at
`synthetic_scratch_training/05_gpt2_scratch_h100/results/h100_plateau/`.

Nothing from experiments 01-04 or 06-07 is here either; see this theme's
README.md for what each of those is and why it was left.

Three outputs, because the natural units differ:

  from_scratch_ivq.csv       per-position accuracy on the standard tidy-long
                             schema — the joint (K, N, step) grid at the best
                             checkpoint. This is what backs ONE_PAGER.tex §7.
  from_scratch_training.csv  per-checkpoint validation accuracy split by role,
                             joined to LR and train loss. This is the phase
                             transition itself, and it needs a train_step
                             column the shared schema does not have.
  from_scratch_summary.csv   final_eval aggregates (overall, by_role, by_step,
                             by_cell, by_duplicate) for the same checkpoint.

Chance is 0.10 throughout (10 value symbols, i.i.d. with replacement).

Sources
  RAW  synthetic_scratch_training/05_gpt2_scratch_h100/results/h100_cosine/
         joint_knstep.json, final_eval.json, val_agg_step*.json, train_log.json
  QA   joint_knstep.json re-aggregated vs final_eval.json (same checkpoint,
       same frozen eval set, independent aggregation path)
"""

from __future__ import annotations

import csv
import glob
import json
import re
from pathlib import Path

from common import (ORDINAL, RAW, ROOT, emit, rel, with_model, write_csv)

THEME = "06_from_scratch"
OUT = Path(__file__).resolve().parent.parent / THEME
SRC = ROOT / "synthetic_scratch_training" / "05_gpt2_scratch_h100" / "results"

MODEL = "gpt2-small"
CHANCE = 0.10
# cosine only -- see SCOPE in the module docstring. Adding the plateau arm back
# is a one-line change here; every builder function iterates this mapping.
ARMS = {"h100_cosine": "scratch_cosine"}
ROLE_COND = {"first": "FVQ", "last": "CVQ", "intermediate": "IVQ"}


def _n_correct(acc: float, n: int) -> int | str:
    """acc was computed as correct/n, so recover the integer count exactly."""
    c = acc * n
    return int(round(c)) if abs(c - round(c)) < 1e-6 else ""


# ── 1. Per-position joint (K, N, step) grid — the U-curve ───────────────────

def build_ivq() -> list[dict]:
    """joint_knstep.json: 204 trained + 48 held-out (K, N, step) cells.

    `held_out_cells` in the source are (N, step) pairs, NOT (K, N) cells --
    (12, 8) means N=12/step=8, held out across all six K values. Every other
    theme in this folder uses "cell" to mean (K, N), so the distinction is
    recorded in `notes` on each row rather than left to the reader.
    """
    path = SRC / "h100_cosine" / "joint_knstep.json"
    d = json.load(open(path))
    src = rel(path)
    held_pairs = {(int(n), int(s)) for n, s in d["held_out_cells"]}
    rows = []
    for split_key, split in (("iid", "train"), ("heldout", "held_out")):
        for key, v in d[split_key].items():
            k, n, step = (int(x) for x in key.split(","))
            cond = "FVQ" if step == 1 else "CVQ" if step == n else "IVQ"
            note = (f"query step {step} of {n}; chance {CHANCE:.2f}; "
                    f"checkpoint step {d['ckpt_step']}")
            if (n, step) in held_pairs:
                note += (f"; held-out column = (N={n}, step={step}) across all K "
                         "-- an (N,step) pair, not a (K,N) cell")
            rows.append(with_model(
                MODEL, theme=THEME, variant=ARMS["h100_cosine"],
                dataset="synthetic_kv", prompt_format="token_stream",
                num_keys=k, num_updates=n, condition=cond, query_type=ORDINAL,
                position="first" if step == 1 else "last" if step == n else step,
                accuracy=v["acc"], n_trials=v["n"],
                n_correct=_n_correct(v["acc"], v["n"]),
                split=split, provenance_tier=RAW, source_file=src, notes=note,
            ))
    rows.sort(key=lambda r: (r["split"], int(r["num_keys"]),
                             int(r["num_updates"]), int(r["position"])
                             if str(r["position"]).isdigit()
                             else (0 if r["position"] == "first" else 9999)))
    return rows


# ── 2. The phase transition: per-checkpoint validation, split by role ───────

TRAIN_COLS = ["model", "model_family", "variant", "schedule", "train_step",
              "lr", "train_loss", "condition", "role", "accuracy", "n_trials",
              "n_correct", "provenance_tier", "source_file", "notes"]


def build_training() -> list[dict]:
    """val_agg_step*.json per arm, joined to lr/loss from train_log.json.

    condition=ALL is the pooled validation accuracy; FVQ/CVQ/IVQ are the
    by_role split. The three roles transition at *different* steps, which is
    the finding, so the role breakdown is kept rather than only the pooled
    number.
    """
    rows = []
    for tag, variant in ARMS.items():
        log = {r["step"]: r for r in json.load(open(SRC / tag / "train_log.json"))}
        vsrc = rel(SRC / tag)
        for f in sorted(glob.glob(str(SRC / tag / "val_agg_step*.json")),
                        key=lambda p: int(re.search(r"step(\d+)", p).group(1))):
            d = json.load(open(f))
            step = d.get("step") or int(re.search(r"step(\d+)", f).group(1))
            lg = log.get(step, {})
            buckets = [("ALL", "", d.get("overall"), d.get("n"))]
            for role, cond in ROLE_COND.items():
                b = (d.get("by_role") or {}).get(role) or {}
                buckets.append((cond, role, b.get("acc"), b.get("n")))
            for cond, role, acc, n in buckets:
                if acc is None:
                    continue
                rows.append({
                    "model": MODEL, "model_family": "GPT-2", "variant": variant,
                    "schedule": tag.replace("h100_", ""), "train_step": step,
                    "lr": lg.get("lr", ""), "train_loss": lg.get("loss", ""),
                    "condition": cond, "role": role, "accuracy": acc,
                    "n_trials": n, "n_correct": _n_correct(acc, n) if n else "",
                    "provenance_tier": RAW,
                    "source_file": f"{vsrc}/val_agg_step{step}.json",
                    "notes": f"chance {CHANCE:.2f}"
                             + ("; lr/loss from train_log.json" if lg else ""),
                })
    return rows


# ── 3. final_eval aggregates for both arms ──────────────────────────────────

SUM_COLS = ["model", "model_family", "variant", "schedule", "ckpt_step",
            "split", "breakdown", "bucket", "accuracy", "n_trials",
            "n_correct", "provenance_tier", "source_file", "notes"]


def build_summary() -> list[dict]:
    rows = []
    for tag, variant in ARMS.items():
        path = SRC / tag / "final_eval.json"
        d = json.load(open(path))
        src = rel(path)
        base = dict(model=MODEL, model_family="GPT-2", variant=variant,
                    schedule=d.get("schedule", tag.replace("h100_", "")),
                    ckpt_step=d.get("trained_to_step", ""),
                    provenance_tier=RAW, source_file=src)
        for split_key, split in (("iid_test", "train"),
                                 ("heldout_test", "held_out")):
            blk = d.get(split_key) or {}
            rows.append({**base, "split": split, "breakdown": "overall",
                         "bucket": "", "accuracy": blk.get("overall"),
                         "n_trials": blk.get("n"),
                         "n_correct": _n_correct(blk["overall"], blk["n"])
                         if blk.get("n") else "",
                         "notes": f"chance {CHANCE:.2f}"})
            for bd in ("by_role", "by_step", "by_cell", "by_duplicate"):
                for bucket, v in (blk.get(bd) or {}).items():
                    if v.get("acc") is None:
                        continue
                    note = f"chance {CHANCE:.2f}"
                    if bd == "by_duplicate":
                        note += ("; True = the target value also occurs "
                                 "elsewhere in the queried key's history, so a "
                                 "'most frequent value' shortcut can score; "
                                 "False is the shortcut-free measure")
                    rows.append({**base, "split": split, "breakdown": bd,
                                 "bucket": bucket, "accuracy": v["acc"],
                                 "n_trials": v.get("n"),
                                 "n_correct": _n_correct(v["acc"], v["n"])
                                 if v.get("n") else "", "notes": note})
    return rows


# ── QA: joint_knstep re-aggregated must reproduce final_eval ────────────────

# joint_knstep.json and final_eval.json are two separate eval passes over the
# SAME frozen set (both `build_frozen_eval_set(SEED+2 / SEED+3, ..., 200/500)`,
# seed 42, chunk 256) at the SAME checkpoint -- so pooling one must reproduce
# the other. In practice they differ by 1 prediction in 40,800 (iid) and 3 in
# 24,000 (held-out): 4 flips in 64,800, i.e. 0.006%. Same weights and same data
# leave only non-deterministic float reduction order under TF32/bf16 flipping
# near-tie argmaxes. So the check is expressed as a budget of flipped
# predictions rather than a float epsilon -- a real regression (a moved source,
# a wrong checkpoint, a mis-parsed grid) would blow past 5 examples immediately.
FLIP_BUDGET = 5


def qa_joint_vs_final(ivq_rows: list[dict]) -> None:
    """Pool the joint (K,N,step) grid and compare to final_eval's aggregates."""
    fin = json.load(open(SRC / "h100_cosine" / "final_eval.json"))
    checked = bad = 0
    flipped: dict[str, float] = {}
    for split, key in (("train", "iid_test"), ("held_out", "heldout_test")):
        sub = [r for r in ivq_rows if r["split"] == split]
        exp = fin[key]

        def pooled(rs):
            n = sum(int(r["n_trials"]) for r in rs)
            c = sum(int(r["n_correct"]) for r in rs)
            return (c / n if n else None), n

        buckets = [("overall", sub, {"acc": exp["overall"], "n": exp["n"]})]
        for role, cond in ROLE_COND.items():
            want = (exp.get("by_role") or {}).get(role) or {}
            if want.get("acc") is not None:
                buckets.append((role, [r for r in sub if r["condition"] == cond],
                                want))
        for label, rs, want in buckets:
            got, n = pooled(rs)
            checked += 1
            if n != want["n"]:
                bad += 1
                print(f"     MISMATCH {split} {label}: n {n} vs {want['n']}")
                continue
            flips = abs((got - want["acc"]) * n)
            flipped[f"{split}/{label}"] = flips
            if flips > FLIP_BUDGET:
                bad += 1
                print(f"     MISMATCH {split} {label}: {got:.6f} vs "
                      f"{want['acc']:.6f} = {flips:.1f} predictions")
    worst = max(flipped.values()) if flipped else 0
    print(f"     QA joint_knstep re-aggregated vs final_eval: {checked} values "
          f"checked, {bad} outside the {FLIP_BUDGET}-prediction budget "
          f"(worst {worst:.1f})")


def write(rows: list[dict], cols: list[str], name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"  -> {THEME}/{name}: {len(rows)} rows  (raw={len(rows)})")


def main() -> None:
    print(f"[{THEME}] from-scratch GPT-2-small on the synthetic KV task")
    ivq = build_ivq()
    emit(ivq, OUT, "from_scratch_ivq.csv")
    qa_joint_vs_final(ivq)
    write(build_training(), TRAIN_COLS, "from_scratch_training.csv")
    write(build_summary(), SUM_COLS, "from_scratch_summary.csv")


if __name__ == "__main__":
    main()
