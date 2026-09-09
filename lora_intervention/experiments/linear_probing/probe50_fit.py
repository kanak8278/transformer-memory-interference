"""
Closed-pool value-identity probing: probe fitting.

Runs entirely on CPU off the arrays saved by probe50_collect.py, so every
re-fit (C grid x label sets x layers x subsets) costs no GPU time.

For each (condition, subset, label_set, layer, C) it fits

    Pipeline([StandardScaler(), LogisticRegression(C, max_iter, lbfgs)])

on the full d_model residual (no PCA) and reports 5-fold CV top-1, mean rank of
the true class, and MRR. See PROBE50_DESIGN.md for why C is one global value
rather than tuned per cell, and why squared-loss/closed-form and PCA variants
were rejected.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

C_GRID = [0.0003, 0.001, 0.003, 0.01, 0.1, 1.0]
LABEL_SETS = ["expected", "shuffled", "cv_first", "cv_last"]
SUBSETS = ["wrong", "correct", "all"]


# ─── label construction ──────────────────────────────────────────────────────

def build_labels(records, pool, label_set, rng_seed=0):
    """Integer class labels over the closed pool.

    `expected`  -- the trial's ground-truth answer, cv[k-1]. The result.
    `shuffled`  -- `expected` permuted across trials. Control task: must fall to
                   chance, otherwise the pipeline leaks or the probe overfits.
    `cv_first`  -- the test key's first value. Primacy control.
    `cv_last`   -- the test key's last value. Recency control; if this decodes
                   better than `expected` at interior k, the effect is recency
                   tracking rather than depth-indexed retrieval.
    """
    idx = {v: i for i, v in enumerate(pool)}
    key = {"expected": "expected", "shuffled": "expected",
           "cv_first": "cv_first", "cv_last": "cv_last"}[label_set]
    y = np.array([idx[r[key]] for r in records], dtype=np.int32)
    if label_set == "shuffled":
        y = y.copy()
        np.random.default_rng(rng_seed).shuffle(y)
    return y


def subset_mask(records, subset, pool=None):
    """Trial mask for a subset.

    `wrong` excludes trials whose single-token argmax is not a pool value at
    all. Every pool value appears in every stream (closed-pool permutation), so
    an off-pool argmax means the model emitted no candidate -- in practice it
    echoed the key, e.g. expected 'club', generated 'bird species: club'. That
    is a format failure, not a retrieval failure, and the content is often
    correct, so such trials must not sit in a subset whose whole purpose is
    "the answer was not emitted".

    Measured rate: 0.000-0.004 for every interior slot and CVQ, but 0.051 for
    FVQ (128 of 2500), i.e. 20% of FVQ's wrong subset. It is the one condition
    where this matters.
    """
    correct = np.array([r["correct_single_token"] for r in records])
    if subset == "correct":
        return correct
    if subset == "all":
        return np.ones(len(records), dtype=bool)
    if subset == "wrong":
        if pool is None:
            return ~correct
        poolset = {v.lower() for v in pool}
        in_pool = np.array([(r["pred_single_token"] or "").strip().lower() in poolset
                            for r in records])
        return (~correct) & in_pool
    if subset == "off_pool":
        poolset = {v.lower() for v in pool} if pool else set()
        return np.array([(r["pred_single_token"] or "").strip().lower() not in poolset
                         for r in records])
    raise ValueError(f"unknown subset: {subset}")


# ─── metrics ─────────────────────────────────────────────────────────────────

def rank_metrics(proba, y_true, classes):
    """Mean rank of the true class (1 = best) and MRR.

    Rank uses the full score ordering, so it degrades gracefully where top-1 is
    near the 1/n_classes floor and reads as noise.
    """
    pos = {c: j for j, c in enumerate(classes)}
    ranks = []
    order = np.argsort(-proba, axis=1)
    for i, yt in enumerate(y_true):
        j = pos.get(int(yt))
        if j is None:          # true class absent from this training fold
            continue
        ranks.append(int(np.where(order[i] == j)[0][0]) + 1)
    if not ranks:
        return None, None
    ranks = np.asarray(ranks, dtype=float)
    return float(ranks.mean()), float((1.0 / ranks).mean())


def fit_one(X, y, C, max_iter, n_splits, seed):
    """5-fold CV for one (layer, C). Returns top-1, mean rank, MRR, convergence."""
    counts = np.bincount(y, minlength=int(y.max()) + 1)
    present = counts[counts > 0]
    n_splits = int(min(n_splits, present.min()))
    if n_splits < 2:
        return {"top1": None, "mean_rank": None, "mrr": None,
                "note": f"min class support {int(present.min())} < 2 folds"}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    top1, mr, mrr, hit_cap = [], [], [], False
    for tr, te in skf.split(X, y):
        clf = Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")),
        ])
        clf.fit(X[tr], y[tr])
        lr = clf.named_steps["lr"]
        if int(lr.n_iter_[0]) >= max_iter:
            hit_cap = True
        p = clf.predict_proba(X[te])
        pred = lr.classes_[p.argmax(1)]
        top1.append(float((pred == y[te]).mean()))
        a, b = rank_metrics(p, y[te], lr.classes_)
        if a is not None:
            mr.append(a); mrr.append(b)
    return {
        "top1": float(np.mean(top1)),
        "top1_std": float(np.std(top1)),
        "mean_rank": float(np.mean(mr)) if mr else None,
        "mrr": float(np.mean(mrr)) if mrr else None,
        "n_splits": n_splits,
        "hit_max_iter": hit_cap,
    }


# ─── driver ──────────────────────────────────────────────────────────────────

def fit_condition(reps, records, pool, cond, args):
    """All (subset, label_set, layer, C) combinations for one condition."""
    n_layers = reps.shape[1]
    out = {}
    for subset in args.subsets:
        m = subset_mask(records, subset, pool)
        n = int(m.sum())
        recs = [r for r, keep in zip(records, m) if keep]
        entry = {"n": n, "label_sets": {}}
        if subset == "wrong":
            n_raw = int((~np.array([r["correct_single_token"]
                                    for r in records])).sum())
            entry["n_before_off_pool_filter"] = n_raw
            entry["n_excluded_off_pool"] = n_raw - n
            if n_raw != n:
                print(f"      [{cond}/wrong] excluded {n_raw-n} off-pool-argmax "
                      f"trials (format failures, not retrieval failures): "
                      f"{n_raw} -> {n}", flush=True)
        if n < args.min_n:
            entry["note"] = f"n={n} < min_n={args.min_n}, skipped"
            out[subset] = entry
            continue

        # Do NOT materialise reps[m].astype(float32): that is the whole
        # (n, n_layers, d) block, e.g. 1800 x 37 x 2048 x 4 B = 1.05 GB, and
        # with n_jobs workers on top it will OOM a machine already in swap.
        # Extract one layer at a time instead -- 1800 x 2048 x 4 B = 14.7 MB.
        rows = np.flatnonzero(m)
        for ls in args.label_sets:
            y = build_labels(recs, pool, ls)
            per_C = {}
            for C in args.C_grid:
                t0 = time.time()
                res = Parallel(n_jobs=args.n_jobs)(
                    delayed(fit_one)(
                        np.asarray(reps[rows, L, :], dtype=np.float32),
                        y, C, args.max_iter, args.n_splits, args.seed)
                    for L in range(n_layers))
                layers = {str(L): r for L, r in enumerate(res)}
                valid = [(L, r["top1"]) for L, r in enumerate(res)
                         if r.get("top1") is not None]
                best = max(valid, key=lambda t: t[1]) if valid else (None, None)
                per_C[str(C)] = {
                    "layers": layers,
                    "best_layer": best[0], "best_top1": best[1],
                    "mean_top1": float(np.mean([v for _, v in valid])) if valid else None,
                    "elapsed_sec": time.time() - t0,
                }
                print(f"      [{cond}/{subset}/{ls}] C={C:<7} best L{best[0]}="
                      f"{best[1]:.3f} mean={per_C[str(C)]['mean_top1']:.3f} "
                      f"({time.time()-t0:.0f}s)" if valid else
                      f"      [{cond}/{subset}/{ls}] C={C:<7} no valid layers",
                      flush=True)
            entry["label_sets"][ls] = per_C
        out[subset] = entry
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", help="directory written by probe50_collect.py")
    p.add_argument("--C_grid", type=float, nargs="+", default=C_GRID)
    p.add_argument("--label_sets", nargs="+", default=LABEL_SETS)
    p.add_argument("--subsets", nargs="+", default=SUBSETS)
    p.add_argument("--conditions", nargs="+", default=None,
                   help="default: every condition in the manifest")
    p.add_argument("--max_iter", type=int, default=2000)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--min_n", type=int, default=200,
                   help="skip a subset with fewer trials than this")
    p.add_argument("--n_jobs", type=int, default=4,
                   help="joblib workers. Kept at 4, not 10: this box has "
                        "24 GB and runs ~7.5 GB into swap under normal "
                        "desktop load, so more workers risk an OOM kill.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    manifest = json.load(open(run_dir / "manifest.json"))
    pool = manifest["pool"]
    conds = args.conditions or list(manifest["conditions"].keys())

    print(f"{manifest['base_model']}  cell={manifest['cell']}  "
          f"n_layers={manifest['n_layers']}  d_model={manifest['d_model']}  "
          f"classes={len(pool)}  chance={1/len(pool):.3f}")
    print(f"C grid: {args.C_grid}")
    print(f"label sets: {args.label_sets}   subsets: {args.subsets}")

    results = {"manifest": {k: v for k, v in manifest.items() if k != "conditions"},
               "fit_config": {"C_grid": args.C_grid, "label_sets": args.label_sets,
                              "subsets": args.subsets, "max_iter": args.max_iter,
                              "n_splits": args.n_splits, "min_n": args.min_n,
                              "seed": args.seed},
               "conditions": {}}

    out_path = Path(args.out) if args.out else run_dir / "probe_fits.json"
    for cond in conds:
        npy = run_dir / f"reps_{cond}.npy"
        if not npy.exists():
            print(f"  [{cond}] no reps_{cond}.npy (pilot run?), skipping")
            continue
        reps = np.load(npy, mmap_mode="r")
        records = manifest["conditions"][cond]["records"]
        assert len(records) == reps.shape[0], \
            f"{cond}: {len(records)} records vs {reps.shape[0]} activation rows"
        acc = manifest["conditions"][cond]["accuracy_single_token"]
        print(f"\n  [{cond}] k={manifest['conditions'][cond]['k']} "
              f"n={reps.shape[0]} acc={acc:.3f}", flush=True)
        results["conditions"][cond] = {
            "k": manifest["conditions"][cond]["k"],
            "accuracy_single_token": acc,
            "subsets": fit_condition(reps, records, pool, cond, args),
        }
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    # global C selection, per PROBE50_DESIGN.md: the value maximizing mean
    # top-1 averaged over all layers on CVQ / wrong / expected. Chosen once,
    # then applied everywhere -- tuning per (layer, condition) would confound
    # the layer curve with the regularization strength.
    ref = results["conditions"].get("CVQ", {}).get("subsets", {}) \
        .get("wrong", {}).get("label_sets", {}).get("expected")
    if ref:
        scored = {C: v["mean_top1"] for C, v in ref.items()
                  if v.get("mean_top1") is not None}
        if scored:
            best_C = max(scored, key=scored.get)
            results["global_C"] = {"value": float(best_C),
                                   "rule": "max mean top-1 over layers, "
                                           "CVQ/wrong/expected",
                                   "mean_top1_by_C": scored}
            print(f"\nGlobal C = {best_C}  (mean top-1 by C: "
                  f"{ {k: round(v,4) for k, v in scored.items()} })")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
