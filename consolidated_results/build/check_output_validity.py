"""Output-validity audit for theme 1.

Answers: when a model "fails" FVQ, did it retrieve the wrong value, or did it
fail to emit a stream value at all (key name, filler word, refusal, truncation)?

The stage-1 sweeps already classify every generation into an outcome taxonomy
(`error_types`), so most of this is recoverable from data on disk. What is NOT
recoverable is the raw generation text — the sweeps record counts only.

Run:
    cd consolidated_results/build && python3 check_output_validity.py
"""

from __future__ import annotations

import collections
import glob
import itertools
import json
import random
import sys

from common import ROOT, OFFSTREAM_TYPES

SWEEPS = str(ROOT / "v3" / "results_vllm" / "arbitrary_single" / "*" /
             "stage1_sweep_*.json")


def load() -> dict:
    """model -> condition -> Counter(error_type)."""
    out = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for path in sorted(glob.glob(SWEEPS)):
        d = json.load(open(path))
        model = path.split("/")[-2]
        for cell in d["cells"].values():
            for ri_pi, st in cell["stats"].items():
                cond = {"RI": "FVQ", "PI": "CVQ"}[ri_pi]
                out[model][cond].update(st["error_types"])
    return out


def part1_taxonomy(data) -> None:
    print("=" * 74)
    print("1. What does the model actually emit?  (all open-weight sweeps)")
    print("=" * 74)
    tot = collections.defaultdict(collections.Counter)
    for conds in data.values():
        for cond, c in conds.items():
            tot[cond].update(c)
    for cond in ("FVQ", "CVQ"):
        c = tot[cond]
        n = sum(c.values())
        print(f"\n  {cond}  (N={n:,} trials)")
        for et, k in c.most_common():
            flag = "  <-- NOT a retrieval error" if et in OFFSTREAM_TYPES else ""
            print(f"    {et:24s} {k:7,}  {k / n:6.1%}{flag}")


def part2_per_model(data) -> None:
    print()
    print("=" * 74)
    print("2. Off-stream rate by model, and the gap with those trials removed")
    print("=" * 74)
    print("   acc  = correct / all trials          (as reported today)")
    print("   acc* = correct / on-stream trials    (retrieval accuracy proper)")
    print()
    print(f"  {'model':24s} {'off%FVQ':>8s} {'off%CVQ':>8s} | {'acc F':>6s} {'acc C':>6s} "
          f"{'gap':>6s} | {'acc*F':>6s} {'acc*C':>6s} {'gap*':>6s}")
    rows = []
    for model, conds in data.items():
        r = {"model": model}
        for cond in ("FVQ", "CVQ"):
            c = conds[cond]
            n = sum(c.values())
            off = sum(c.get(k, 0) for k in OFFSTREAM_TYPES)
            r[cond] = {
                "off": off / n if n else 0,
                "acc": c["correct"] / n if n else 0,
                "acc_on": c["correct"] / (n - off) if n - off else 0,
            }
        rows.append(r)
    rows.sort(key=lambda r: -r["FVQ"]["off"])

    flips = 0
    for r in rows:
        gap = r["FVQ"]["acc"] - r["CVQ"]["acc"]
        gap_on = r["FVQ"]["acc_on"] - r["CVQ"]["acc_on"]
        if (gap > 0) != (gap_on > 0):
            flips += 1
        mark = "  ** sign flip" if (gap > 0) != (gap_on > 0) else ""
        print(f"  {r['model']:24s} {r['FVQ']['off']:8.1%} {r['CVQ']['off']:8.1%} | "
              f"{r['FVQ']['acc']:6.2f} {r['CVQ']['acc']:6.2f} {gap:+6.2f} | "
              f"{r['FVQ']['acc_on']:6.2f} {r['CVQ']['acc_on']:6.2f} {gap_on:+6.2f}{mark}")
    print(f"\n  gap sign changes when off-stream trials are excluded: "
          f"{flips} / {len(rows)} models")


def part3_false_positives() -> None:
    print()
    print("=" * 74)
    print("3. Can the lenient matcher score a WRONG answer as correct?")
    print("=" * 74)
    print("   classify_error uses `expected in predicted` (substring, case-folded),")
    print("   and checks 'correct' FIRST. So if another value in the same trial")
    print("   CONTAINS the expected value, an intrusion can be scored correct.")
    print()
    try:
        sys.path.insert(0, str(ROOT))
        from mechanistic_probing_v2.core.dataset_configs import (
            get_eligible_categories, generate_values_for_trial)
    except ImportError as e:                                   # pragma: no cover
        print(f"   (skipped — cannot import dataset_configs: {e})")
        return

    cats_all = get_eligible_categories("ARBITRARY_SINGLE", min_values=5)
    print(f"  {'N updates':>10s} {'FVQ risk':>10s} {'CVQ risk':>10s}   (10,000 simulated trials each)")
    for n_upd in (5, 10, 20, 50, 100):
        hits = {"FVQ": 0, "CVQ": 0}
        trials = 10_000
        for t in range(trials):
            rng = random.Random(t)
            cats = rng.sample(cats_all, min(5, len(cats_all)))
            try:
                vpc = generate_values_for_trial("ARBITRARY_SINGLE", cats, n_upd, rng)
            except Exception:
                continue
            vals = [v.lower() for v in vpc[cats[t % len(cats)]]]
            for cond, exp in (("FVQ", vals[0]), ("CVQ", vals[-1])):
                if any(exp in other for other in vals if other != exp):
                    hits[cond] += 1
        print(f"  {n_upd:>10d} {hits['FVQ'] / trials:>10.2%} {hits['CVQ'] / trials:>10.2%}")


def part4_what_is_missing() -> None:
    print()
    print("=" * 74)
    print("4. What CANNOT be checked from data on disk")
    print("=" * 74)
    probe = ["output_raw", "predicted", "per_trial", '"gen"']
    for path in sorted(glob.glob(SWEEPS))[:3]:
        blob = open(path).read()
        found = [k for k in probe if k in blob]
        print(f"  {path.split('/')[-2]:24s} raw generations stored: "
              f"{found if found else 'NO'}")
    print()
    print("  The sweeps aggregate to counts and discard `output_raw`, so we cannot")
    print("  post-hoc inspect what the 21.6% off-stream generations actually said,")
    print("  nor audit whether a 30-token answer contained extra values alongside")
    print("  the correct one. Both need a re-run that persists generations.")


if __name__ == "__main__":
    data = load()
    part1_taxonomy(data)
    part2_per_model(data)
    part3_false_positives()
    part4_what_is_missing()
