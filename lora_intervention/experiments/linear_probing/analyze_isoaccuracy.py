"""
Fix 4 -- iso-accuracy analysis of the linear-probing results.

The per-cell base-vs-LoRA comparison is confounded: the two states are
class-balanced at *different* loads (base is ~50% correct at high load, LoRA at
low load), so "base decodable, LoRA not" partly reflects which cells happened to
be testable, not a representational difference. This script removes that
confound two ways:

  1. CORRECTNESS probe, iso-accuracy binned. AUC's chance is 0.5 regardless of
     skew (Fix 1), so we can pool every testable (cell, condition) by its
     *behavioral accuracy* and ask: at a matched accuracy, is correctness more
     decodable in base than LoRA? If the two curves overlie, the asymmetry was
     an artifact; if base sits above LoRA at matched accuracy, it is real.

  2. RETRIEVAL probe (Fix 3), which is correctness-INDEPENDENT and therefore
     defined at *every* cell including the saturated ones the correctness probe
     can't touch. Two readouts:
       - retrieval[all]:  is the answer value linearly represented at all?
       - retrieval[wrong]: on trials the model got WRONG, is the correct value
         still decodable? -> the "tracked but suppressed" test.

Reads results/probe_*-v3_*.json (both states). Prints tables; writes a PNG and a
markdown summary next to the results.
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
RESULTS = _HERE / "results"


def best_corr_auc(cond):
    """Best per-layer correctness-probe AUC (+layer), or (None, None) if the
    condition is degenerate everywhere."""
    best_auc, best_L = None, None
    for L, ld in (cond.get("layers") or {}).items():
        if ld.get("degenerate"):
            continue
        a = ld.get("cv_auc")
        if a is not None and (best_auc is None or a > best_auc):
            best_auc, best_L = a, int(L)
    return best_auc, best_L


def best_retr_auc(cond, subset):
    r = cond.get("retrieval")
    if not r or subset not in r:
        return None, None, 0
    s = r[subset]
    n = s.get("n_trials", 0)
    summ = s.get("summary")
    if not summ or summ.get("best_auc") is None:
        return None, None, n
    return summ["best_auc"], summ["best_layer"], n


def load_rows(version="v3", model="Qwen2.5-3B-Instruct"):
    rows = []
    for state in ("base", "lora"):
        for f in sorted(RESULTS.glob(f"probe_{model}-{state}-{version}_*.json")):
            d = json.load(open(f))
            cell = f"{d['cell']['keys']}k{d['cell']['updates']}u"
            for cname, cond in d["conditions"].items():
                ca, cL = best_corr_auc(cond)
                ra_all, rL_all, n_all = best_retr_auc(cond, "all")
                ra_wr, rL_wr, n_wr = best_retr_auc(cond, "wrong")
                rows.append({
                    "state": state, "cell": cell, "cond": cname,
                    "beh": cond["behavioral_accuracy"], "n": cond["n"],
                    "corr_auc": ca, "corr_L": cL,
                    "retr_all": ra_all, "retr_all_L": rL_all, "n_all": n_all,
                    "retr_wrong": ra_wr, "retr_wrong_L": rL_wr, "n_wrong": n_wr,
                })
    return rows


def iso_accuracy_bins(rows, metric, bins=((0.0, 0.2), (0.2, 0.4), (0.4, 0.6),
                                          (0.6, 0.8), (0.8, 1.01))):
    """Mean of `metric` per behavioral-accuracy bin, base vs LoRA."""
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = r.get(metric)
        if v is None:
            continue
        for lo, hi in bins:
            if lo <= r["beh"] < hi:
                agg[(lo, hi)][r["state"]].append(v)
                break
    out = []
    for lo, hi in bins:
        b = agg[(lo, hi)]["base"]
        l = agg[(lo, hi)]["lora"]
        out.append({
            "band": f"{lo:.1f}-{hi:.1f}",
            "base_mean": sum(b) / len(b) if b else None, "base_n": len(b),
            "lora_mean": sum(l) / len(l) if l else None, "lora_n": len(l),
        })
    return out


def _fmt(x, pct=False):
    if x is None:
        return "  --"
    return f"{x*100:4.0f}" if pct else f"{x:4.2f}"


def _print_bin_table(bins):
    print(f"{'beh-acc band':13} {'base AUC (n)':>16} {'lora AUC (n)':>16}")
    for b in bins:
        base_cell = f"{_fmt(b['base_mean'])}  ({b['base_n']})"
        lora_cell = f"{_fmt(b['lora_mean'])}  ({b['lora_n']})"
        print(f"{b['band']:13} {base_cell:>16} {lora_cell:>16}")


def print_tables(rows):
    print("\n" + "=" * 78)
    print("FIX 4a -- CORRECTNESS probe (AUC), iso-accuracy binned  [chance = 0.50]")
    print("At a matched behavioral accuracy, is correctness more decodable in one state?")
    print("=" * 78)
    _print_bin_table(iso_accuracy_bins(rows, "corr_auc"))

    print("\n" + "=" * 78)
    print("FIX 4b -- RETRIEVAL probe [all] (AUC), iso-accuracy binned  [chance = 0.50]")
    print("Correctness-INDEPENDENT: is the answer value represented, at matched accuracy?")
    print("=" * 78)
    _print_bin_table(iso_accuracy_bins(rows, "retr_all"))

    print("\n" + "=" * 78)
    print("FIX 3 -- 'TRACKED BUT SUPPRESSED': retrieval AUC on WRONG-answer trials")
    print("Correct value decodable despite the model emitting a wrong answer (n_wrong>=30).")
    print("=" * 78)
    supp = [r for r in rows if r["retr_wrong"] is not None and r["n_wrong"] >= 30]
    supp.sort(key=lambda r: -r["retr_wrong"])
    print(f"{'state':5} {'cell':7} {'cond':9} {'beh':>4} {'n_wrong':>7} "
          f"{'retr[wrong]':>11} {'@L':>3} {'corr_auc':>8}")
    for r in supp:
        print(f"{r['state']:5} {r['cell']:7} {r['cond']:9} {_fmt(r['beh'],1):>4} "
              f"{r['n_wrong']:7} {_fmt(r['retr_wrong']):>11} {str(r['retr_wrong_L']):>3} "
              f"{_fmt(r['corr_auc']):>8}")
    return supp


def make_plot(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"\n(matplotlib unavailable, skipping PNG: {e})")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = {"base": "#1f77b4", "lora": "#d62728"}
    for i, (metric, title) in enumerate([
        ("corr_auc", "Correctness probe: best-layer AUC"),
        ("retr_all", "Retrieval probe [all]: best-layer AUC"),
    ]):
        ax = axes[i]
        for state in ("base", "lora"):
            xs = [r["beh"] for r in rows if r["state"] == state and r.get(metric) is not None]
            ys = [r[metric] for r in rows if r["state"] == state and r.get(metric) is not None]
            ax.scatter(xs, ys, c=colors[state], label=state, alpha=0.7, s=42,
                       edgecolors="white", linewidths=0.5)
        ax.axhline(0.5, color="gray", ls="--", lw=1, label="chance")
        ax.set_xlabel("behavioral accuracy (model)")
        ax.set_ylabel("best-layer AUC")
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0.3, 1.02)
        ax.set_xlim(-0.02, 1.02)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
    fig.suptitle("Iso-accuracy view: probe decodability at matched behavioral accuracy "
                 "(base vs LoRA)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    print(f"\nSaved plot: {path}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen2.5-3B-Instruct",
                   help="Model-name prefix of the probe_*.json files "
                        "(e.g. gemma-3-4b-it).")
    p.add_argument("--tag", default=None, help="Output filename tag (default: model).")
    args = p.parse_args()
    tag = args.tag or args.model
    rows = load_rows("v3", args.model)
    if not rows:
        print(f"No v3 results for model={args.model} in", RESULTS)
        return
    print(f"\n### Model: {args.model}  ({len(rows)} state×cell×condition rows) ###")
    supp = print_tables(rows)
    make_plot(rows, RESULTS / f"isoaccuracy_v3_{tag}.png")
    with open(RESULTS / f"isoaccuracy_v3_{tag}_summary.json", "w") as f:
        json.dump({"model": args.model, "rows": rows,
                   "corr_bins": iso_accuracy_bins(rows, "corr_auc"),
                   "retr_all_bins": iso_accuracy_bins(rows, "retr_all"),
                   "suppression": supp}, f, indent=2)
    print(f"Saved summary: {RESULTS / f'isoaccuracy_v3_{tag}_summary.json'}")


if __name__ == "__main__":
    main()
