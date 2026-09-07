"""Inspect saved CoT-IVQ runs: arm hygiene, extraction quality, accuracy.

Two modes. `--summary` (default) audits every run directory: the checks that
decide whether a sweep is trustworthy — did the `nocot` arm stay free of
reasoning, did every response parse, did anything truncate. `--show N` prints
N raw records so the thinking and the answer can be read by eye.

    python experiments_cloud/cot_ivq_inspect.py
    python experiments_cloud/cot_ivq_inspect.py --show 3 --position 50
"""
import argparse
import json
import statistics as st
from collections import Counter
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results" / "cot_ivq"


def load(run_dir):
    f = run_dir / "trials.jsonl"
    if not f.exists():
        return []
    return [json.loads(line) for line in f.open() if line.strip()]


def summarize(run_dir, recs):
    ok = [r for r in recs if not r.get("error")]
    errs = [r for r in recs if r.get("error")]
    print(f"\n=== {run_dir.name}   n={len(recs)}  errors={len(errs)}")
    if errs:
        print(f"    first error: {errs[0]['error'][:160]}")
    if not ok:
        return

    # Arm hygiene. thinking_enabled is the config, thinking_blocks is what came
    # back — a mismatch means the request and the arm disagree.
    n_think = sum(1 for r in ok if r.get("thinking_blocks"))
    enabled = Counter(r.get("thinking_enabled") for r in ok)
    print(f"    thinking_enabled={dict(enabled)}  responses_with_thinking={n_think}/{len(ok)}")

    # In the no-reasoning arm the visible text should be just the value. Long
    # text there means the prefill+stop_sequence guard failed and the control
    # arm is quietly doing CoT.
    if not enabled.get(True):
        leak = [r for r in ok if len(r.get("text_block") or "") > 40]
        print(f"    long_text_leaks={len(leak)}/{len(ok)}")
        if leak:
            print(f"    worst leak: {leak[0]['text_block'][:200]!r}")

    print(f"    extraction_tier={dict(Counter(r.get('extraction_tier') for r in ok))}")
    print(f"    stop_reason={dict(Counter(r.get('stop_reason') for r in ok))}")
    print(f"    outcome={dict(Counter(r.get('outcome') for r in ok))}")
    print(f"    malformed(not well_formed)={sum(1 for r in ok if not r.get('well_formed'))}"
          f"  truncated={sum(1 for r in ok if r.get('truncated'))}")

    out = sorted(r["output_tokens"] for r in ok if r.get("output_tokens"))
    if out:
        p95 = out[min(len(out) - 1, int(0.95 * len(out)))]
        print(f"    output_tokens: max={max(out)} p95={p95} mean={st.mean(out):.0f} "
              f"median={st.median(out)}")

    print(f"    accuracy={sum(1 for r in ok if r.get('correct')) / len(ok):.3f}")
    by_pos = {}
    for r in ok:
        by_pos.setdefault(r["position"], []).append(1 if r.get("correct") else 0)
    curve = "  ".join(f"{p}:{sum(v)/len(v):.2f}" for p, v in sorted(by_pos.items(), key=lambda x: str(x[0])))
    print(f"    per-position: {curve}")


def show(run_dir, recs, n, position):
    pool = [r for r in recs if position is None or str(r.get("position")) == str(position)]
    for r in pool[:n]:
        print(f"\n----- {run_dir.name}  trial={r.get('trial_idx')} pos={r.get('position')}")
        print(f"  expected={r.get('expected')!r}  extracted={r.get('answer_extracted')!r}"
              f"  outcome={r.get('outcome')}  tier={r.get('extraction_tier')}")
        for b in r.get("thinking_blocks") or []:
            print(f"  [thinking] {(b.get('text') or '')[:2000]}")
        print(f"  [text] {(r.get('text_block') or '')[:1000]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(RESULTS))
    ap.add_argument("--glob", default="*", help="filter run dirs, e.g. '*cot_thinking*'")
    ap.add_argument("--show", type=int, default=0, help="print N raw records per run")
    ap.add_argument("--position", default=None, help="restrict --show to one position")
    args = ap.parse_args()

    root = Path(args.results_dir)
    runs = sorted(d for d in root.glob(args.glob) if d.is_dir())
    if not runs:
        print(f"no run directories under {root}")
        return
    for d in runs:
        recs = load(d)
        if not recs:
            print(f"\n=== {d.name}   (no trials.jsonl)")
            continue
        summarize(d, recs)
        if args.show:
            show(d, recs, args.show, args.position)


if __name__ == "__main__":
    main()
