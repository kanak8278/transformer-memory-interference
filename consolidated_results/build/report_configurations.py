"""Regenerate the theme-1 configuration report (01_fvq_cvq/CONFIGURATIONS.md).

Prints, per model: run-level config from the raw sweeps, the exact (K, N) cells
that ran, trial counts, and the shared-cell intersection used for fair
cross-model comparison.

Run:
    cd consolidated_results/build && python3 report_configurations.py
"""

from __future__ import annotations

import collections
import csv
import glob
import json
from pathlib import Path

from common import EXCLUDED_MODELS, ROOT

CSV = Path(__file__).resolve().parent.parent / "01_fvq_cvq" / "fvq_cvq.csv"
ARB_K = [2, 3, 5, 7, 10, 15, 20, 25, 30]
ARB_N = [5, 7, 10, 15, 20, 30, 50, 75, 100]
SEM_K = [1, 2, 5, 10, 15, 20, 25, 30, 40, 45]
SEM_N = [1, 5, 10, 15, 20, 30, 50]


def run_configs() -> None:
    print("=" * 78)
    print("1. Run-level config — open-weight stage-1 sweeps (retained models)")
    print("=" * 78)
    fmt_groups = collections.defaultdict(list)
    for path in sorted(glob.glob(str(ROOT / "v3" / "results_vllm" /
                                     "arbitrary_single" / "*" /
                                     "stage1_sweep_*.json"))):
        model = path.split("/")[-2]
        if model in EXCLUDED_MODELS:
            continue
        d = json.load(open(path))
        c = d["config"]
        fmt_groups[d["prompt_format"]].append(model)
        eng = c.get("engine", {})
        warn = ""
        if not d.get("total_prompts"):
            warn = "   <-- total_prompts=0: interrupted/resumed run, counters unwritten"
        print(f"\n{model}  [{d['prompt_format']}]{warn}")
        print(f"   id={d['model']}  backend={d['backend']}  dataset={d['dataset_type']}")
        print(f"   grid K={c['key_levels']}")
        print(f"        N={c['update_levels']}")
        print(f"   trials/cell={c['trials_per_cell']}  max_new_tokens={c['max_new_tokens']}"
              f"  min_updates={d['min_updates']}")
        print(f"   sampling={c['sampling']}")
        print(f"   dtype={eng.get('dtype')}  max_model_len={eng.get('max_model_len')}"
              f"  gpu_mem_util={eng.get('gpu_memory_utilization')}")
        print(f"   cells={len(d['cells'])}/{len(c['key_levels']) * len(c['update_levels'])}"
              f"  prompts={d.get('total_prompts')}  runtime={d.get('inference_time_sec')}s")

    print("\n" + "!" * 78)
    print("! PROMPT-FORMAT CONFOUND — these groups are not directly comparable:")
    for fmt, models in sorted(fmt_groups.items()):
        print(f"!   {fmt:20s} {', '.join(sorted(models))}")
    print("!" * 78)


def grid(dataset: str, ks: list, ns: list, proprietary: bool) -> None:
    rows = [x for x in csv.DictReader(open(CSV))
            if x["accuracy"] and x["dataset"] == dataset
            and (x["is_proprietary"] == "True") == proprietary]
    cells = collections.defaultdict(set)
    fmts, trials = {}, collections.defaultdict(list)
    for x in rows:
        m = x["model"]
        cells[m].add((int(x["num_keys"]), int(x["num_updates"])))
        fmts[m] = x["prompt_format"]
        trials[m].append(int(x["n_trials"]))

    label = "proprietary" if proprietary else "open-weight"
    print("\n" + "=" * 78)
    print(f"2. Cell coverage — {label}, {dataset}   (X = ran, . = absent)")
    print("=" * 78)
    for m in sorted(cells, key=lambda m: (fmts.get(m, ""), m)):
        t = trials[m]
        tt = (f"n={t[0]}" if len(set(t)) == 1
              else f"n={min(t)}-{max(t)} ({len(set(t))} distinct)")
        print(f"\n{m}  [{fmts.get(m) or 'n/a'}]  {len(cells[m])}/{len(ks) * len(ns)} cells  {tt}")
        print("      N=  " + " ".join(f"{n:>4}" for n in ns))
        for k in ks:
            marks = " ".join("   X" if (k, n) in cells[m] else "   ." for n in ns)
            print(f"  K={k:<3} {marks}")

    if len(cells) > 1:
        shared = set.intersection(*cells.values())
        print(f"\n  Cells present for ALL {len(cells)} {label} models: {len(shared)}"
              f" of {len(ks) * len(ns)}")
        if shared:
            kk = sorted({k for k, _ in shared})
            nn = sorted({n for _, n in shared})
            print(f"    K up to {max(kk)}, N up to {max(nn)}")
        print("    -> restrict to these for a fair cross-model comparison.")


if __name__ == "__main__":
    run_configs()
    grid("arbitrary_single", ARB_K, ARB_N, proprietary=False)
    grid("semantic_multi", SEM_K, SEM_N, proprietary=True)
