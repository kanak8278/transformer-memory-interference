#!/usr/bin/env python3
"""
Deep-dive into REVERSAL cases (RI < PI) and garbage-filtered results.
Goal: distinguish instruction-following failure vs. key-count capacity
      limit vs. context-length attention decay vs. recency prior.

All analyses DROP garbage trials and recompute clean RI/PI.
"""

import json, os, math
from pathlib import Path

BASE = Path(__file__).parent
os.chdir(BASE)

# ─── model meta ──────────────────────────────────────────────────────────────
MODEL_META = {
    "Qwen3.5-0.8B":            {"params": 0.8,  "family": "Qwen3.5",  "instruct": False},
    "Qwen3.5-2B":              {"params": 2.0,  "family": "Qwen3.5",  "instruct": False},
    "Qwen3.5-4B":              {"params": 4.0,  "family": "Qwen3.5",  "instruct": False},
    "Qwen3.5-9B":              {"params": 9.0,  "family": "Qwen3.5",  "instruct": False},
    "Qwen2.5-0.5B-Instruct":   {"params": 0.5,  "family": "Qwen2.5",  "instruct": True},
    "Qwen2.5-1.5B-Instruct":   {"params": 1.5,  "family": "Qwen2.5",  "instruct": True},
    "Qwen2.5-3B":              {"params": 3.0,  "family": "Qwen2.5",  "instruct": False},
    "Qwen2.5-3B-Instruct":     {"params": 3.0,  "family": "Qwen2.5",  "instruct": True},
    "gemma-3-270m-it":         {"params": 0.27, "family": "Gemma3",   "instruct": True},
    "gemma-3-1b-it":           {"params": 1.0,  "family": "Gemma3",   "instruct": True},
    "gemma-3-4b-it":           {"params": 4.0,  "family": "Gemma3",   "instruct": True},
    "mamba-1.4b-hf":           {"params": 1.4,  "family": "Mamba",    "instruct": False},
    "pythia-410m":             {"params": 0.41, "family": "Pythia",   "instruct": False},
    "stablelm-2-1_6b-chat":    {"params": 1.6,  "family": "StableLM", "instruct": True},
    "TinyLlama-1.1B-Chat-v1.0":{"params": 1.1,  "family": "TinyLlama","instruct": True},
}

def mean(xs): return sum(xs)/len(xs) if xs else float("nan")
def fmt(v):   return f"{v:+.3f}" if not math.isnan(v) else "  nan "


# ─── load trial-level data ───────────────────────────────────────────────────
def load_trials(dataset, model):
    folder = BASE / dataset / model
    if not folder.exists():
        return None
    for f in sorted(folder.iterdir()):
        if "trials" in f.name and "checkpoint" not in f.name and f.suffix == ".json":
            # prefer the most recent (latest timestamp in name)
            with open(f) as fh:
                d = json.load(fh)
            if "trial_details" in d:
                return d
    return None

def load_sweep(dataset, model):
    folder = BASE / dataset / model
    if not folder.exists():
        return None
    for f in sorted(folder.iterdir()):
        if "sweep" in f.name and f.suffix == ".json":
            with open(f) as fh:
                return json.load(fh)
    ckpt = folder / "stage1_checkpoint.json"
    if ckpt.exists():
        with open(ckpt) as fh:
            return json.load(fh)
    return None


# ─── garbage-filtered cell stats from trial_details ──────────────────────────
def clean_cell_stats(trial_data, cell_key):
    """
    For a given cell, compute RI/PI accuracy *excluding garbage trials*.
    Also compute: for RI failures, what did the model predict?
                  for PI failures, what did the model predict?
    Returns dict or None.
    """
    td = trial_data.get("trial_details", {})
    if cell_key not in td:
        return None
    cell = td[cell_key]

    results = {}
    for cond in ["RI", "PI"]:
        trials = cell.get(cond, [])
        non_garb = [t for t in trials if t.get("error_type") != "garbage"]
        if not non_garb:
            results[cond] = {"n": 0, "acc": float("nan"), "failure_positions": []}
            continue
        correct = sum(1 for t in non_garb if t["correct"])
        acc = correct / len(non_garb)
        # For failures: collect relative position of predicted value
        failures = [t for t in non_garb if not t["correct"]]
        failure_positions = [t.get("predicted_relative_pos") for t in failures
                             if t.get("predicted_relative_pos") is not None]
        # For RI failures: does model predict LAST value (rel pos = 1.0)?
        # For PI failures: does model predict FIRST value (rel pos = 0.0)?
        results[cond] = {
            "n": len(non_garb),
            "acc": acc,
            "n_fail": len(failures),
            "failure_positions": failure_positions,
            "frac_last": sum(1 for p in failure_positions if p >= 0.99) / len(failure_positions) if failure_positions else 0,
            "frac_first": sum(1 for p in failure_positions if p <= 0.01) / len(failure_positions) if failure_positions else 0,
            "mean_fail_pos": mean(failure_positions) if failure_positions else float("nan"),
        }
    return results


# ─── SECTION A: Garbage-filtered global summary ──────────────────────────────
def section_clean_summary(dataset="arbitrary_single"):
    print("=" * 80)
    print(f"SECTION A: GARBAGE-FILTERED SUMMARY  [{dataset}]")
    print("=" * 80)
    print("  Recomputing RI/PI accuracy excluding garbage outputs.")
    print("  gap_raw = raw gap;  gap_clean = gap after dropping garbage\n")

    print(f"  {'Model':<28} {'params':>6}  {'raw_gap':>8}  {'clean_RI':>8}  {'clean_PI':>8}  {'clean_gap':>9}  {'change':>7}  {'%cells_rev_clean':>16}")
    print("  " + "-"*100)

    all_models = sorted(MODEL_META.keys(), key=lambda m: MODEL_META[m]["params"])
    for model in all_models:
        trials_data = load_trials(dataset, model)
        sweep_data  = load_sweep(dataset, model)
        if sweep_data is None:
            continue

        # raw stats from sweep
        raw_gaps = []
        for cell in sweep_data["cells"].values():
            raw_gaps.append(cell["stats"]["RI"]["accuracy"] - cell["stats"]["PI"]["accuracy"])
        raw_gap = mean(raw_gaps)

        # clean stats from trials
        if trials_data is None:
            print(f"  {model:<28} {MODEL_META[model]['params']:>5.1f}B  {fmt(raw_gap)}  [no trial data]")
            continue

        clean_ri_list, clean_pi_list, clean_gap_list = [], [], []
        for ck in sweep_data["cells"]:
            cstats = clean_cell_stats(trials_data, ck)
            if cstats is None:
                continue
            ri = cstats["RI"]["acc"]
            pi = cstats["PI"]["acc"]
            if not math.isnan(ri) and not math.isnan(pi):
                clean_ri_list.append(ri)
                clean_pi_list.append(pi)
                clean_gap_list.append(ri - pi)

        clean_ri  = mean(clean_ri_list)
        clean_pi  = mean(clean_pi_list)
        clean_gap = mean(clean_gap_list)
        change    = clean_gap - raw_gap
        pct_rev_clean = 100 * sum(1 for g in clean_gap_list if g < 0) / len(clean_gap_list) if clean_gap_list else float("nan")

        tag = " ◄" if "Qwen3.5" in model else "  "
        print(f"  {model:<28} {MODEL_META[model]['params']:>5.1f}B  "
              f"{fmt(raw_gap)}   {clean_ri:>7.3f}   {clean_pi:>7.3f}   "
              f"{fmt(clean_gap)}   {fmt(change)}  {pct_rev_clean:>6.1f}%{tag}")


# ─── SECTION B: RI failure mode — what does the model predict? ───────────────
def section_ri_failure_mode(dataset="arbitrary_single"):
    print("\n" + "=" * 80)
    print(f"SECTION B: RI FAILURE MODE  [{dataset}]  (garbage excluded)")
    print("=" * 80)
    print("  When RI fails (model can't recall FIRST value), what does it predict?")
    print("  frac_last = fraction of RI failures where model predicted the LAST value")
    print("  mean_pos  = mean relative position predicted (0=first, 1=last)\n")
    print("  If frac_last ≈ 1.0 → model is ignoring 'first', defaulting to recency.")
    print("  If mean_pos ≈ 0.5  → model is guessing a middle value (confusion).\n")

    print(f"  {'Model':<28}  {'n_RI_fail':>9}  {'frac_last':>9}  {'frac_first':>10}  {'mean_pos':>8}  interpretation")
    print("  " + "-"*90)

    all_models = sorted(MODEL_META.keys(), key=lambda m: MODEL_META[m]["params"])
    for model in all_models:
        trials_data = load_trials(dataset, model)
        sweep_data  = load_sweep(dataset, model)
        if sweep_data is None or trials_data is None:
            continue

        all_fail_pos = []
        for ck in sweep_data["cells"]:
            cs = clean_cell_stats(trials_data, ck)
            if cs is None: continue
            all_fail_pos += cs["RI"]["failure_positions"]

        if not all_fail_pos:
            print(f"  {model:<28}  [no RI failures or no trial data]")
            continue

        n = len(all_fail_pos)
        frac_last  = sum(1 for p in all_fail_pos if p >= 0.99) / n
        frac_first = sum(1 for p in all_fail_pos if p <= 0.01) / n
        mean_pos   = mean(all_fail_pos)

        if frac_last > 0.6:
            interp = "RECENCY DEFAULT — model ignores 'first', predicts last"
        elif frac_last > 0.35:
            interp = "recency-leaning — often predicts last"
        elif mean_pos > 0.55:
            interp = "recency-leaning (intermediate)"
        elif mean_pos < 0.45:
            interp = "primacy-leaning (surprising)"
        else:
            interp = "uniform confusion — genuine uncertainty"

        tag = " ◄" if "Qwen3.5" in model else "  "
        print(f"  {model:<28}  {n:>9}  {frac_last:>9.3f}  {frac_first:>10.3f}  "
              f"{mean_pos:>8.3f}  {interp}{tag}")


# ─── SECTION C: Reversed cells — num_keys vs num_updates driver ──────────────
def section_reversal_driver(dataset="arbitrary_single"):
    print("\n" + "=" * 80)
    print(f"SECTION C: WHAT DRIVES REVERSALS?  [{dataset}]  (garbage excluded)")
    print("=" * 80)
    print("  Hypothesis 1 (KEY CAPACITY): reversal driven by num_keys → high-nk cells reverse")
    print("  Hypothesis 2 (CONTEXT FILL): reversal driven by fill% → high-fill cells reverse")
    print("  Hypothesis 3 (INSTRUCTION):  reversal random/uniform → any cell can reverse\n")

    OVERHEAD = 450  # completion format

    focus_models = ["Qwen3.5-2B", "Qwen2.5-3B", "pythia-410m",
                    "Qwen2.5-3B-Instruct", "TinyLlama-1.1B-Chat-v1.0", "Qwen3.5-9B"]

    for model in focus_models:
        trials_data = load_trials(dataset, model)
        sweep_data  = load_sweep(dataset, model)
        if sweep_data is None or trials_data is None:
            continue

        cells = []
        for ck, raw_cell in sweep_data["cells"].items():
            nk = raw_cell["num_keys"]
            nu = raw_cell["num_updates"]
            fill = 100 * (OVERHEAD + nk * nu * 5) / 16384
            cs = clean_cell_stats(trials_data, ck)
            if cs is None: continue
            ri, pi = cs["RI"]["acc"], cs["PI"]["acc"]
            if math.isnan(ri) or math.isnan(pi): continue
            cells.append({"nk": nk, "nu": nu, "fill": fill,
                          "ri": ri, "pi": pi, "gap": ri - pi,
                          "reversed": ri < pi})

        if not cells: continue

        n_rev = sum(1 for c in cells if c["reversed"])
        n_tot = len(cells)

        # Mean num_keys for reversed vs normal
        rev_nk   = mean([c["nk"] for c in cells if c["reversed"]])
        norm_nk  = mean([c["nk"] for c in cells if not c["reversed"]])
        rev_fill = mean([c["fill"] for c in cells if c["reversed"]])
        norm_fill= mean([c["fill"] for c in cells if not c["reversed"]])
        rev_nu   = mean([c["nu"] for c in cells if c["reversed"]])
        norm_nu  = mean([c["nu"] for c in cells if not c["reversed"]])

        print(f"\n  {model} ({MODEL_META[model]['params']}B)  —  {n_rev}/{n_tot} reversed cells (clean)")
        print(f"  {'':>20}  {'mean_nk':>8}  {'mean_nu':>8}  {'mean_fill%':>10}  {'mean_gap':>9}")
        print(f"  {'Reversed cells':>20}  {rev_nk:>8.1f}  {rev_nu:>8.1f}  {rev_fill:>10.1f}  "
              f"{mean([c['gap'] for c in cells if c['reversed']]):>9.3f}")
        print(f"  {'Normal cells':>20}  {norm_nk:>8.1f}  {norm_nu:>8.1f}  {norm_fill:>10.1f}  "
              f"{mean([c['gap'] for c in cells if not c['reversed']]):>9.3f}")

        # Check: at fixed low fill (<15%), is nk still the driver?
        low_fill = [c for c in cells if c["fill"] < 15]
        if low_fill:
            rev_lf = [c for c in low_fill if c["reversed"]]
            print(f"  At <15% fill ({len(low_fill)} cells): {len(rev_lf)} reversed, "
                  f"reversed_mean_nk={mean([c['nk'] for c in rev_lf]):.1f}  "
                  f"normal_mean_nk={mean([c['nk'] for c in low_fill if not c['reversed']]):.1f}")

        # Show reversal rate by num_keys bucket
        print(f"  Reversal rate by num_keys:")
        for nk_thresh in [2, 3, 5, 7, 10, 15, 20, 25, 30]:
            bucket = [c for c in cells if c["nk"] == nk_thresh]
            if not bucket: continue
            rev_b = sum(1 for c in bucket if c["reversed"])
            mean_gap_b = mean([c["gap"] for c in bucket])
            print(f"    nk={nk_thresh:2d}: {rev_b:2d}/{len(bucket):2d} reversed  mean_gap={mean_gap_b:+.3f}")


# ─── SECTION D: RI failure by num_keys (does it recall LAST at high nk?) ──────
def section_ri_failure_by_nk(dataset="arbitrary_single"):
    print("\n" + "=" * 80)
    print(f"SECTION D: RI FAILURE MODE BY num_keys  [{dataset}]  (garbage excluded)")
    print("=" * 80)
    print("  For each num_keys level: frac of RI failures that predict the LAST value")
    print("  Key question: does high num_keys push model toward recency default?\n")

    focus_models = ["Qwen3.5-2B", "Qwen2.5-3B", "Qwen2.5-3B-Instruct",
                    "pythia-410m", "Qwen3.5-9B", "gemma-3-4b-it"]

    for model in focus_models:
        trials_data = load_trials(dataset, model)
        sweep_data  = load_sweep(dataset, model)
        if sweep_data is None or trials_data is None:
            continue

        # bucket by nk
        nk_data = {}
        for ck, raw_cell in sweep_data["cells"].items():
            nk = raw_cell["num_keys"]
            cs = clean_cell_stats(trials_data, ck)
            if cs is None: continue
            ri_stats = cs["RI"]
            if nk not in nk_data:
                nk_data[nk] = {"fail_pos": [], "n_trials": 0, "n_fail": 0, "acc_list": []}
            nk_data[nk]["fail_pos"] += ri_stats["failure_positions"]
            nk_data[nk]["n_trials"] += ri_stats["n"]
            nk_data[nk]["n_fail"]   += ri_stats.get("n_fail", 0)
            if not math.isnan(ri_stats["acc"]):
                nk_data[nk]["acc_list"].append(ri_stats["acc"])

        print(f"\n  {model}:")
        print(f"  {'nk':>4}  {'RI_acc':>7}  {'n_fail':>7}  {'pred_last%':>10}  {'mean_fail_pos':>14}")
        for nk in sorted(nk_data):
            d = nk_data[nk]
            fp = d["fail_pos"]
            fl = sum(1 for p in fp if p >= 0.99) / len(fp) if fp else 0
            mp = mean(fp) if fp else float("nan")
            ri_acc = mean(d["acc_list"])
            print(f"  {nk:>4}  {ri_acc:>7.3f}  {d['n_fail']:>7}  {fl:>10.3f}  {mp:>14.3f}")


# ─── SECTION E: Instruction-following hypothesis test ────────────────────────
def section_instruct_hypothesis(dataset="arbitrary_single"):
    print("\n" + "=" * 80)
    print(f"SECTION E: INSTRUCTION-FOLLOWING HYPOTHESIS  [{dataset}]  (garbage excluded)")
    print("=" * 80)
    print("  If reversal = instruction failure, BOTH RI and PI should look like 'predict last'.")
    print("  i.e., RI fails because model predicts last; PI succeeds because model predicts last.")
    print("  This means: RI_fail_pos → last; PI_acc → high; PI_fail_pos → NOT last.\n")
    print("  If reversal = capacity limit, RI fails at high nk EVEN when format is followed.\n")

    all_models = sorted(MODEL_META.keys(), key=lambda m: MODEL_META[m]["params"])
    print(f"  {'Model':<28}  {'instruct':>8}  {'RI_fail→last%':>14}  {'PI_fail→first%':>15}  {'gap_clean':>9}  diagnosis")
    print("  " + "-"*110)

    for model in all_models:
        trials_data = load_trials(dataset, model)
        sweep_data  = load_sweep(dataset, model)
        if sweep_data is None or trials_data is None:
            continue

        ri_fail_pos, pi_fail_pos = [], []
        clean_gaps = []
        for ck in sweep_data["cells"]:
            cs = clean_cell_stats(trials_data, ck)
            if cs is None: continue
            ri_fail_pos += cs["RI"]["failure_positions"]
            pi_fail_pos += cs["PI"]["failure_positions"]
            ri, pi = cs["RI"]["acc"], cs["PI"]["acc"]
            if not math.isnan(ri) and not math.isnan(pi):
                clean_gaps.append(ri - pi)

        ri_last = sum(1 for p in ri_fail_pos if p >= 0.99) / len(ri_fail_pos) if ri_fail_pos else float("nan")
        pi_first= sum(1 for p in pi_fail_pos if p <= 0.01) / len(pi_fail_pos) if pi_fail_pos else float("nan")
        gap     = mean(clean_gaps)

        is_inst = MODEL_META[model]["instruct"]

        if not math.isnan(ri_last):
            if ri_last > 0.6 and not math.isnan(pi_first) and pi_first < 0.1:
                diag = "RECENCY DEFAULT (instruction fail)"
            elif ri_last > 0.3 and not math.isnan(gap) and gap < 0:
                diag = "Partial recency default + reversed"
            elif ri_last < 0.2 and not math.isnan(gap) and gap < 0:
                diag = "NOT recency default — capacity/attention limit"
            elif ri_last < 0.2:
                diag = "Clean — RI failures are intermediate values"
            else:
                diag = "Mixed"
        else:
            diag = "[no data]"

        tag = " ◄" if "Qwen3.5" in model else "  "
        print(f"  {model:<28}  {'yes' if is_inst else 'no':>8}  "
              f"{ri_last:>14.3f}  {pi_first:>15.3f}  "
              f"{fmt(gap)}  {diag}{tag}")


# ─── SECTION F: Clean gap by (num_keys, num_updates) for key outliers ────────
def section_clean_heatmap(dataset="arbitrary_single"):
    print("\n" + "=" * 80)
    print(f"SECTION F: CLEAN GAP HEATMAP — KEY OUTLIERS  [{dataset}]")
    print("=" * 80)

    focus = [("Qwen3.5-2B", "Qwen3.5-9B"),   # anomalous vs clean in same family
             ("Qwen2.5-3B", "Qwen2.5-3B-Instruct"),   # base vs instruct same size
             ("pythia-410m", "gemma-3-1b-it")]          # two base-like small models

    for m1, m2 in focus:
        for model in [m1, m2]:
            trials_data = load_trials(dataset, model)
            sweep_data  = load_sweep(dataset, model)
            if sweep_data is None or trials_data is None:
                print(f"\n  {model}: [no data]"); continue

            cells_raw = sweep_data["cells"]
            nk_levels = sorted(set(c["num_keys"]    for c in cells_raw.values()))
            nu_levels = sorted(set(c["num_updates"] for c in cells_raw.values()))

            # build clean gap lookup
            clean_gap = {}
            for ck, raw_cell in cells_raw.items():
                cs = clean_cell_stats(trials_data, ck)
                if cs is None: continue
                ri, pi = cs["RI"]["acc"], cs["PI"]["acc"]
                if not math.isnan(ri) and not math.isnan(pi):
                    clean_gap[(raw_cell["num_keys"], raw_cell["num_updates"])] = ri - pi

            print(f"\n  {model} — clean gap  (+ = PI>RI expected, - = REVERSED)")
            nk_nu_lbl = "nk\\nu"
            header = f"  {nk_nu_lbl:>5} " + "".join(f"  {nu:>5}" for nu in nu_levels)
            print(header)
            print("  " + "-"*(len(header)-2))
            for nk in nk_levels:
                row = f"  {nk:>5} "
                for nu in nu_levels:
                    g = clean_gap.get((nk, nu), float("nan"))
                    if math.isnan(g):
                        row += "   SKIP"
                    elif g < -0.05:
                        row += f"  {g:+5.2f}"   # reversed
                    else:
                        row += f"   {g:+.2f}"
                print(row)
        print()


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for ds in ["arbitrary_single", "semantic_multi"]:
        section_clean_summary(ds)
        print()

    section_ri_failure_mode("arbitrary_single")
    section_reversal_driver("arbitrary_single")
    section_ri_failure_by_nk("arbitrary_single")
    section_instruct_hypothesis("arbitrary_single")
    section_clean_heatmap("arbitrary_single")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
