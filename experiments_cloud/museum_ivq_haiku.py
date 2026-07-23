#!/usr/bin/env python3
"""IVQ-dense U-curve on the museum M0 (atemporal) domain.

DESIGN GOAL: match the earlier PLAIN Haiku U-curve (ucurve_sweep.py, run on
claude-4.5-haiku) EXACTLY, so museum M0 is directly comparable to plain's
`flat_nolabel` format (both: no order marker, interior positions require the
model to recover order from reading position). We therefore REUSE plain's
machinery verbatim:

  - model:     create_model(name, {"verbose": False})  -> max_tokens 2000, temp 0
  - scoring:   ucurve_sweep.classify()  (correct / in_sequence / garbage)
  - CI/stop:   ucurve_sweep.wilson_hw + converged(); MIN_TRIALS=50, MAX_TRIALS=200,
               CI_THRESHOLD=0.05, converge when ALL positions' half-width <= 5%
  - positions: query_positions(nu) (16-point scheme) + semantic 'first'/'last'
  - prompt:    narrative + question + "Answer with ONLY the exact value. No explanation."

NOTE on chain-of-thought: Haiku sometimes reasons out loud on interior ordinal
queries, and containment scoring can then credit the enumeration. This behaviour
was ALSO present in the plain Haiku U-curve (same model, same machinery), so it
is a constant across both arms and does not bias the plain-vs-museum comparison.
We deliberately do NOT suppress it, to keep the two experiments identical.

Reproducibility: seed = abs(hash((nk,nu,trial_idx)))%(2**31), the same int-only
scheme as sweep_arbitrary.py:272 (stable across processes). Every query's raw
model output is saved, so any other model rerun on the same seeds is comparable.

Usage (project .venv, ANTHROPIC_API_KEY in .env):
  .venv/bin/python experiments_cloud/museum_ivq_haiku.py --smoke
  .venv/bin/python experiments_cloud/museum_ivq_haiku.py --model claude-haiku
"""
import argparse
import json
import math
import os
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments_cloud"))

from narrative_generator import MuseumTrialGenerator          # noqa: E402
from models.model_factory import create_model                 # noqa: E402
# reuse plain U-curve machinery verbatim
from ucurve_sweep import classify, wilson_hw, converged        # noqa: E402

# ─── fixed config — matches plain Haiku U-curve ──────────────────────────────
GRID_KEYS = [5, 10]
GRID_UPDATES = [10, 20, 50]
MIN_TRIALS = 50
MAX_TRIALS = 200
CI_THRESHOLD = 0.05
RENDER_MODE = "M0"
POOL_MODE = "disjoint_pool"
ANSWER_INSTRUCTION = "\nAnswer with ONLY the exact value. No explanation."

_thread_local = threading.local()


def get_model(model_name):
    if not hasattr(_thread_local, "model"):
        _thread_local.model = create_model(model_name, {"verbose": False})
    return _thread_local.model


def seed_for(nk, nu, t):
    """Same scheme as sweep_arbitrary.py:272 — int-only tuple hash, stable
    across processes, so any model rerun on (nk,nu,t) sees the same narrative."""
    return abs(hash((nk, nu, t))) % (2 ** 31)


def pos_label(q):
    """Convergence/aggregation key for one probed position."""
    if q["style"] == "semantic":
        return q["anchor"]              # 'first' / 'last'
    return q["position"]                # ordinal int


def _matches(pred_low, value):
    v = value.lower()
    return v == pred_low or v in pred_low or pred_low.startswith(v)


def classify_fine(base_error_type, raw, context_index):
    """Sub-divide ucurve's `garbage` into `in_context` vs true `garbage`.

    Does NOT change correct / in_sequence — so the accuracy metric stays
    identical to plain. context_index maps a value -> the OTHER visitor whose
    sequence it belongs to (queried visitor's own values already excluded).

    Returns (fine_error_type, intrusion_visitor|None).
    """
    if base_error_type != "garbage":
        return base_error_type, None
    pred = raw.lower().strip()
    for value, owner in context_index.items():
        if _matches(pred, value):
            return "in_context", owner      # another visitor's artwork
    return "garbage", None                  # appears nowhere in the narrative


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-trials", type=int, default=MAX_TRIALS)
    args = ap.parse_args()

    # load .env (no python-dotenv dependency)
    envf = ROOT / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    assert os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY not in .env"

    gen = MuseumTrialGenerator()
    create_model(args.model)  # fail fast if key/model bad

    if args.smoke:
        cells = [(5, 10)]
        max_trials, min_trials = 3, 3
        tag = "smoke"
    else:
        cells = [(nk, nu) for nk in GRID_KEYS for nu in GRID_UPDATES]
        max_trials, min_trials = args.max_trials, MIN_TRIALS
        tag = "full"

    out_dir = ROOT / "results" / "museum_ivq" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    q_path = out_dir / f"ivq_{tag}_{ts}_queries.jsonl"
    s_path = out_dir / f"ivq_{tag}_{ts}_summary.json"

    print(f"model={args.model} render={RENDER_MODE} pool={POOL_MODE} "
          f"grid={cells} max_trials={max_trials} ci={CI_THRESHOLD}")
    print(f"queries -> {q_path}")

    summary = []
    t0 = time.time()
    qf = open(q_path, "w")

    def run_query(task):
        model = get_model(args.model)
        raw = model.generate(task["prompt"])
        cls = classify(raw, task["expected"], task["all_values"])   # unchanged vs plain
        fine, intruder = classify_fine(cls["error_type"], raw, task["context_index"])
        rec = dict(task)
        rec.pop("prompt")            # keep the file lean; prompt = narrative+question
        rec.pop("context_index")     # bulky; owner captured in intrusion_visitor
        rec.update({
            "raw_output": raw,
            "predicted": raw.strip(),
            "error_type": cls["error_type"],          # correct|in_sequence|garbage (== plain)
            "error_type_fine": fine,                  # ...garbage split into in_context|garbage
            "intrusion_visitor": intruder,            # other visitor, if in_context
            "correct": cls["error_type"] == "correct",
            "predicted_idx": cls.get("predicted_idx"),
            "predicted_relative_pos": cls.get("predicted_relative_pos"),
            "input_tokens": getattr(model, "last_input_tokens", None),
            "output_tokens": getattr(model, "last_output_tokens", None),
        })
        return rec

    for (nk, nu) in cells:
        correct_by_pos = defaultdict(list)                 # pos_label -> [bool,...]
        errors_by_pos = defaultdict(lambda: defaultdict(int))  # pos_label -> {fine: n}
        n_done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            t = 0
            while n_done < max_trials:
                # one batch = `workers` trials, each firing all its position queries
                batch_tasks = []
                for _ in range(args.workers):
                    if t >= max_trials:
                        break
                    seed = seed_for(nk, nu, t)
                    trial = gen.generate_trial(nk, nu, None, seed,
                                               render_mode=RENDER_MODE,
                                               pool_mode=POOL_MODE,
                                               all_positions=True)
                    gen.validate_trial(trial)
                    qv = trial["config"]["queried_visitor"]
                    all_values = trial["entity_tracking"][f"{qv} / artwork"]
                    # value -> owning visitor, for the OTHER visitors only
                    # (used to split garbage into cross-entity in_context vs true garbage)
                    own = set(all_values)
                    context_index = {}
                    for key, vals in trial["entity_tracking"].items():
                        owner = key.rsplit(" / ", 1)[0]
                        if owner == qv:
                            continue
                        for v in vals:
                            if v not in own:
                                context_index[v] = owner
                    for q in trial["position_questions"]:
                        batch_tasks.append({
                            "cell": f"{nk}k_{nu}u", "num_keys": nk, "num_updates": nu,
                            "trial_index": t, "seed": seed, "queried_visitor": qv,
                            "position": q["position"], "style": q["style"],
                            "anchor": q.get("anchor"),
                            "pos_label": str(pos_label(q)),
                            "question": q["question"],
                            "expected": q["expected_answer"],
                            "all_values": all_values,
                            "context_index": context_index,
                            "prompt": trial["narrative"] + "\n\n" + q["question"]
                                      + ANSWER_INSTRUCTION,
                        })
                    t += 1
                if not batch_tasks:
                    break

                for fut in as_completed({ex.submit(run_query, bt): bt
                                         for bt in batch_tasks}):
                    rec = fut.result()
                    correct_by_pos[rec["pos_label"]].append(rec["correct"])
                    errors_by_pos[rec["pos_label"]][rec["error_type_fine"]] += 1
                    qf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                qf.flush()
                n_done = t

                if n_done >= min_trials and converged(correct_by_pos, CI_THRESHOLD):
                    break

        # aggregate this cell
        for label, res in sorted(correct_by_pos.items(),
                                 key=lambda kv: (kv[0].isdigit(), kv[0])):
            n, k = len(res), sum(res)
            acc = k / n if n else 0.0
            hw = wilson_hw(n, k)
            fine = errors_by_pos[label]
            summary.append({
                "cell": f"{nk}k_{nu}u", "num_keys": nk, "num_updates": nu,
                "pos_label": label, "n": n, "n_correct": k,
                "acc": round(acc, 4), "wilson_hw": round(hw, 4),
                "ci_low": round(max(0.0, acc - hw), 4),
                "ci_high": round(min(1.0, acc + hw), 4),
                "n_in_sequence": fine.get("in_sequence", 0),
                "n_in_context": fine.get("in_context", 0),
                "n_garbage": fine.get("garbage", 0),
            })
        rate = (t) / max(time.time() - t0, 0.1)
        print(f"  {nk}k_{nu}u: {n_done} trials, converged={n_done < max_trials} "
              f"({rate:.1f} trials/s cumulative)")

    qf.close()
    meta = {
        "model": args.model,
        "model_id": getattr(create_model(args.model), "model_id", None),
        "render_mode": RENDER_MODE, "pool_mode": POOL_MODE,
        "grid_keys": GRID_KEYS, "grid_updates": GRID_UPDATES,
        "min_trials": min_trials, "max_trials": max_trials,
        "ci_threshold": CI_THRESHOLD,
        "seed_formula": "abs(hash((nk,nu,t)))%(2**31)  [== sweep_arbitrary.py:272]",
        "reuses": "ucurve_sweep.classify / wilson_hw / converged",
        "comparator": "plain flat_nolabel in ucurve_proprietary_results.csv",
        "timestamp": ts, "elapsed_sec": round(time.time() - t0, 1),
    }
    s_path.write_text(json.dumps({"metadata": meta, "summary": summary},
                                 indent=2, ensure_ascii=False))
    print(f"\nsummary -> {s_path}\ndone in {meta['elapsed_sec']}s")


if __name__ == "__main__":
    main()
