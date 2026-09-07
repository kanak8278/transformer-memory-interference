"""Refusal watch for the museum CoT runs.

`stop_reason == 'refusal'` is the API's own label, not ours — the model returns
no text block and 0 output tokens, which the scorer records as `no_answer` and
excludes from the accuracy denominator. That exclusion is correct per call and
dangerous in aggregate: on the sonnet nocot arm the refusal rate ran 3.3% at
K5/N10 to 90.5% at K10/N50, i.e. the missing data is concentrated exactly where
interference is highest. An accuracy computed over the survivors is then
conditioned on the variable under study.

So this reports refusal rate per cell and per position, and flags any slice
above a threshold. Read it before quoting any accuracy from these runs.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results" / "museum_cot"
CELL_ORDER = ["5k10u", "10k10u", "5k20u", "10k20u", "5k50u", "10k50u"]


def cell_of(r):
    return f"{r['num_keys']}k{r['num_updates']}u"


def pos_sort(p):
    return (not str(p).isdigit(), int(p) if str(p).isdigit() else str(p))


def report(run_dir, warn):
    f = run_dir / "trials.jsonl"
    if not f.exists():
        return
    recs = [json.loads(l) for l in f.open() if l.strip()]
    if not recs:
        return
    n = len(recs)
    ref = [r for r in recs if r.get("stop_reason") == "refusal"]
    rate = len(ref) / n

    flag = "  <-- REFUSALS" if rate > warn else ""
    print(f"\n=== {run_dir.name}")
    print(f"    calls={n}  refusals={len(ref)} ({100 * rate:.1f}%){flag}")
    if not ref:
        print("    clean")
        return

    # Empty-response refusals vs ones that emitted a token or two first. Only
    # the first kind is a total loss; both are excluded as not well-formed.
    empty = sum(1 for r in ref if not r.get("text_block"))
    print(f"    of those: {empty} emitted nothing, {len(ref) - empty} cut off mid-answer")

    tot, rf = Counter(), Counter()
    for r in recs:
        tot[cell_of(r)] += 1
        if r.get("stop_reason") == "refusal":
            rf[cell_of(r)] += 1
    print("    by cell:")
    for c in CELL_ORDER:
        if tot[c]:
            bar = "#" * int(30 * rf[c] / tot[c])
            print(f"      {c:>7}: {rf[c]:5d}/{tot[c]:5d} = {100*rf[c]/tot[c]:5.1f}% {bar}")

    tp, rp = Counter(), Counter()
    for r in recs:
        tp[str(r["position"])] += 1
        if r.get("stop_reason") == "refusal":
            rp[str(r["position"])] += 1
    worst = sorted(tp, key=lambda p: -(rp[p] / tp[p]))[:5]
    print("    worst positions: " + "  ".join(
        f"{p}:{100*rp[p]/tp[p]:.0f}%" for p in sorted(worst, key=pos_sort)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(RESULTS))
    ap.add_argument("--glob", default="*")
    ap.add_argument("--warn", type=float, default=0.02,
                    help="flag a run above this refusal rate")
    a = ap.parse_args()
    runs = sorted(d for d in Path(a.results_dir).glob(a.glob) if d.is_dir()
                  and not d.name.startswith("_"))
    for d in runs:
        report(d, a.warn)
    print()


if __name__ == "__main__":
    main()
