#!/usr/bin/env python3
"""Museum M0 endpoint RI/PI sweep — REUSES sweep_arbitrary.py by import.

This does NOT modify sweep_arbitrary.py (the file that produced the plain KV
baseline). It imports its validated machinery verbatim so museum M0 is scored
identically to plain:

  from sweep_arbitrary:  classify_error, get_error_detail, wilson_half_width,
                         has_converged, bootstrap_ci, init/update/is_saturated,
                         get_thread_model, save_results, print_summary,
                         model_short_name, and every threshold constant
                         (CI_THRESHOLD=0.07, NEAR_ZERO, RECOVERY, SAT_COUNT, ...)

Only three things are museum-specific and live here:
  1. run_trial       — builds a museum M0 narrative instead of a KV stream
  2. run_cell        — same batch/convergence/saturation loop, calling the
                       imported wilson/has_converged/bootstrap_ci
  3. main            — the endpoint grid + orchestration

Determinism / comparability:
  - seed = abs(hash((nk,nu,trial_idx)))%(2**31)   — identical to sweep_arbitrary
  - the MuseumTrialGenerator seeds its own RNG from that seed, so any model
    rerun on the same (nk,nu,trial_idx) sees a byte-identical narrative
  - temp 0 (create_model default) → deterministic decoding
  - RI/PI prompt = narrative + question + the SAME instruction string plain uses

Usage (project .venv, ANTHROPIC_API_KEY in .env):
  .venv/bin/python experiments_cloud/museum_endpoint_sweep.py --smoke
  .venv/bin/python experiments_cloud/museum_endpoint_sweep.py --model claude-haiku
"""
import argparse
import os
import sys
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

# reuse the plain sweep's machinery verbatim (import, do not modify/copy)
import sweep_arbitrary as sa
from narrative_generator import MuseumTrialGenerator

# endpoint grid: K x N, minus degenerate (N=1) and pool-capped (40x50,45x50);
# the last two are auto-skipped by the pool-feasibility check below.
KEY_LEVELS = [1, 2, 5, 10, 15, 20, 25, 30, 40, 45]
UPDATE_LEVELS = [5, 10, 15, 20, 30, 50]
SAVE_DIR = "experiments_cloud/results/museum_endpoint"
ANSWER_INSTRUCTION = "\nAnswer with ONLY the exact value. No explanation."
RENDER_MODE = "M0"
POOL_MODE = "disjoint_pool"

_GEN = None


def gen():
    global _GEN
    if _GEN is None:
        _GEN = MuseumTrialGenerator()
    return _GEN


def feasible(nk, nu):
    """disjoint_pool needs nk*nu unique titles; also require nu>=2 (RI!=PI)."""
    return nu >= 2 and nk * nu <= len(gen().ARTWORKS)


# ─── museum trial: mirrors sa.run_trial, museum branch only ──────────────────
def run_trial(model_name, num_keys, num_updates, trial_idx, _eligible=None):
    model = sa.get_thread_model(model_name)
    seed = abs(hash((num_keys, num_updates, trial_idx))) % (2 ** 31)  # == sa scheme

    trial = gen().generate_trial(num_keys, num_updates, None, seed,
                                 render_mode=RENDER_MODE, pool_mode=POOL_MODE)
    qv = trial["config"]["queried_visitor"]
    all_values = trial["entity_tracking"][f"{qv} / artwork"]
    initial_value, final_value = all_values[0], all_values[-1]
    narrative = trial["narrative"]

    results = {}
    for condition in ["RI", "PI"]:
        expected = initial_value if condition == "RI" else final_value
        prompt = narrative + "\n\n" + trial["questions"][condition]["question"] + ANSWER_INSTRUCTION
        response = model.generate(prompt)
        token_usage = {
            "input_tokens": model.last_input_tokens,
            "output_tokens": model.last_output_tokens,
            "was_truncated": getattr(model, "last_was_truncated", None),
            "error": model.last_error,
        }
        predicted = response.strip()
        # identical scoring to plain
        error_type = sa.classify_error(predicted, expected, initial_value,
                                       final_value, all_values, condition)
        error_detail = sa.get_error_detail(predicted, all_values)
        results[condition] = {
            "trial_idx": trial_idx, "seed": seed, "condition": condition,
            "test_category": qv, "categories_in_sequence": trial["config"]["roster"],
            "num_keys": num_keys, "num_updates": num_updates,
            "expected": expected, "predicted": predicted,
            "raw_output": response,                       # keep full generation
            "correct": error_type == "correct",
            "error_type": error_type, "error_detail": error_detail,
            "token_usage": token_usage,
        }
    return results


# ─── cell loop: same structure as sa.run_cell, imported CI/stats ─────────────
def run_cell(model_name, num_keys, num_updates, trials_per_cell, workers):
    if not feasible(num_keys, num_updates):
        return None, None
    all_res = {"RI": [], "PI": []}
    cell_start = time.time()
    trial_idx = 0
    stopped_early = False

    with ThreadPoolExecutor(max_workers=workers) as ex:
        while trial_idx < trials_per_cell:
            bs = min(workers, trials_per_cell - trial_idx)
            futs = [ex.submit(run_trial, model_name, num_keys, num_updates, trial_idx + i)
                    for i in range(bs)]
            for fut in as_completed(futs):
                tr = fut.result()
                for cond in ["RI", "PI"]:
                    all_res[cond].append(tr[cond])
            trial_idx += bs
            n = trial_idx
            ri = np.mean([r["correct"] for r in all_res["RI"]])
            pi = np.mean([r["correct"] for r in all_res["PI"]])
            rihw = sa.wilson_half_width(n, sum(r["correct"] for r in all_res["RI"]))
            pihw = sa.wilson_half_width(n, sum(r["correct"] for r in all_res["PI"]))
            print(f"    {n} trials: RI={ri:.0%}±{rihw:.0%}  PI={pi:.0%}±{pihw:.0%}", end="")
            if sa.has_converged(all_res):
                print("  → converged"); stopped_early = True; break
            print()

    cell_stats = {}
    for cond in ["RI", "PI"]:
        corr = [r["correct"] for r in all_res[cond]]
        mean, lo, hi = sa.bootstrap_ci(corr)
        ec = {}
        for r in all_res[cond]:
            ec[r["error_type"]] = ec.get(r["error_type"], 0) + 1
        cell_stats[cond] = {"accuracy": round(mean, 4), "ci_lower": round(lo, 4),
                            "ci_upper": round(hi, 4), "n": len(corr), "error_types": ec}
    ri_acc, pi_acc = cell_stats["RI"]["accuracy"], cell_stats["PI"]["accuracy"]
    regime = ("A" if ri_acc >= .5 and pi_acc >= .5 else "B" if ri_acc >= .5 and pi_acc < .5
              else "D" if ri_acc < .5 and pi_acc >= .5 else "C")
    n_actual = len(all_res["RI"])
    summary = {
        "num_keys": num_keys, "num_updates": num_updates, "regime": regime,
        "elapsed_sec": round(time.time() - cell_start, 1), "n_trials": n_actual,
        "n_observations": n_actual, "stopped_early": stopped_early,
        "max_trials": trials_per_cell, "stats": cell_stats,
    }
    return summary, all_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku")
    ap.add_argument("--trials", type=int, default=sa.DEFAULT_TRIALS)
    ap.add_argument("--workers", type=int, default=sa.DEFAULT_WORKERS)
    ap.add_argument("--save-dir", default=SAVE_DIR)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--key-levels", type=int, nargs="+", default=None)
    ap.add_argument("--update-levels", type=int, nargs="+", default=None)
    args = ap.parse_args()

    # load .env (no python-dotenv dependency)
    envf = os.path.join(_ROOT, ".env")
    if os.path.exists(envf):
        for line in open(envf):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    assert os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY not in .env"

    key_levels = args.key_levels or KEY_LEVELS
    update_levels = args.update_levels or UPDATE_LEVELS
    if args.smoke:
        key_levels, update_levels, args.trials = [3], [5], 3

    print("=" * 70)
    print("MUSEUM M0 ENDPOINT SWEEP (reuses sweep_arbitrary machinery)")
    print(f"  model={args.model}  render={RENDER_MODE}  pool={POOL_MODE}")
    print(f"  grid: {key_levels} x {update_levels}   trials/cell={args.trials}")
    print(f"  CI_THRESHOLD={sa.CI_THRESHOLD}  saturation: near0={sa.NEAR_ZERO}, sat={sa.SAT_COUNT}")
    print("=" * 70)
    sa.create_model(args.model)  # fail fast

    results = {
        "model": args.model, "dataset_type": "museum_M0",
        "config": {"trials_per_cell": args.trials, "workers": args.workers,
                   "key_levels": key_levels, "update_levels": update_levels,
                   "render_mode": RENDER_MODE, "pool_mode": POOL_MODE,
                   "seed_formula": "abs(hash((nk,nu,t)))%(2**31)  [== sweep_arbitrary]",
                   "reuses": "sweep_arbitrary.classify_error/wilson/saturation/save",
                   "comparator": "plain sweep_arbitrary ARBITRARY_SINGLE",
                   "near_zero_thresh": sa.NEAR_ZERO, "recovery_thresh": sa.RECOVERY,
                   "sat_count": sa.SAT_COUNT},
        "cells": {}, "start_time": datetime.now(timezone.utc).isoformat(),
    }
    full_trials = {}
    sat = sa.init_saturation(key_levels)
    done = 0
    total = len(key_levels) * len(update_levels)

    for nk in key_levels:
        for nu in update_levels:
            ck = f"{nk}_{nu}"
            if sa.is_saturated(sat, nk):
                print(f"  [{done+1}/{total}] {nk}x{nu}: SKIPPED (saturated)"); done += 1; continue
            if not feasible(nk, nu):
                print(f"  [{done+1}/{total}] {nk}x{nu}: SKIPPED (infeasible: needs "
                      f"{nk*nu} titles / pool {len(gen().ARTWORKS)}, or N<2)"); done += 1; continue
            print(f"\n  [{done+1}/{total}] keys={nk:>2}, updates={nu:>3}")
            cs, ct = run_cell(args.model, nk, nu, args.trials, args.workers)
            if cs is None:
                done += 1; continue
            results["cells"][ck] = cs
            full_trials[ck] = ct
            for cond in ["RI", "PI"]:
                sa.update_saturation(sat, nk, cond, cs["stats"][cond]["accuracy"])
            print(f"    RI={cs['stats']['RI']['accuracy']:.0%}  "
                  f"PI={cs['stats']['PI']['accuracy']:.0%}  regime={cs['regime']}  "
                  f"n={cs['n_trials']}  {'(early)' if cs['stopped_early'] else ''}")
            done += 1
            sa.save_results(results, full_trials, args.model, args.save_dir, partial=True)

    results["end_time"] = datetime.now(timezone.utc).isoformat()
    sa.save_results(results, full_trials, args.model, args.save_dir, partial=False)
    sa.print_summary(results, key_levels, update_levels)


if __name__ == "__main__":
    main()
