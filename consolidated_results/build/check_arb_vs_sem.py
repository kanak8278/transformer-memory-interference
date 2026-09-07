"""Bridge test: does Arbitrary-Single reproduce Semantic-Multi?

Semantic-Multi is the main experiment. Arbitrary-Single exists to (a) supply
single-token values for the mechanistic work (LoRA, logit lens, probing,
ablation) and (b) show the phenomenon is not an artefact of semantic values.
Claim (b) is what licenses transferring the mechanistic findings back to the
main result, so it is worth measuring rather than asserting.

Test: for every (model, K, N) where both datasets have a gap, compare them.

Run:
    cd consolidated_results/build && python3 check_arb_vs_sem.py
"""

from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

CSV = Path(__file__).resolve().parent.parent / "01_fvq_cvq" / "fvq_cvq.csv"


def load() -> tuple[dict, dict]:
    """(semantic gaps, arbitrary gaps) keyed by (model, K, N)."""
    sem, acc = {}, {}
    for x in csv.DictReader(open(CSV)):
        key = (x["model"], int(x["num_keys"]), int(x["num_updates"]))
        if x["dataset"] == "semantic_multi" and x["condition"] == "FVQ-CVQ" and x["gap"]:
            sem[key] = float(x["gap"])
        elif x["dataset"] == "arbitrary_single" and x["accuracy"]:
            acc[key + (x["condition"],)] = float(x["accuracy"])
    arb = {}
    for (m, k, n, cond), v in acc.items():
        if cond == "FVQ" and (m, k, n, "CVQ") in acc:
            arb[(m, k, n)] = v - acc[(m, k, n, "CVQ")]
    return sem, arb


def main() -> None:
    sem, arb = load()
    print("BRIDGE TEST — Arbitrary-Single vs Semantic-Multi")
    print("gap = FVQ - CVQ, matched on (model, K, N)\n")
    print(f"{'model':24s} {'cell':>10s} {'SEM':>8s} {'ARB':>8s} {'diff':>7s} {'sign':>6s}")
    print("-" * 68)

    pairs = []
    for (m, k, n), sg in sorted(sem.items()):
        ag = arb.get((m, k, n))
        if ag is None:
            continue
        same = (sg > 0) == (ag > 0)
        pairs.append((sg, ag, same))
        print(f"{m:24s} {f'K{k}/N{n}':>10s} {sg:+8.2f} {ag:+8.2f} "
              f"{ag - sg:+7.2f} {'ok' if same else 'DIFF':>6s}")

    if not pairs:
        print("no paired cells")
        return

    s = [p[0] for p in pairs]
    a = [p[1] for p in pairs]
    agree = sum(1 for p in pairs if p[2])
    mx, my = st.mean(s), st.mean(a)
    num = sum((x - mx) * (y - my) for x, y in zip(s, a))
    den = (sum((x - mx) ** 2 for x in s) * sum((y - my) ** 2 for y in a)) ** 0.5

    print("-" * 68)
    print(f"{'':24s} {'MEAN':>10s} {mx:+8.2f} {my:+8.2f} {my - mx:+7.2f}")
    print(f"\n  paired cells   : {len(pairs)}")
    print(f"  sign agreement : {agree}/{len(pairs)} = {agree / len(pairs):.0%}")
    print(f"  mean |ARB-SEM| : {st.mean([abs(x - y) for x, y in zip(a, s)]):.3f}")
    print(f"  correlation r  : {num / den:.3f}   (r^2 = {(num / den) ** 2:.3f})")
    print(f"  ARB bias       : {my - mx:+.3f}  (ARB reads hotter than SEM)")
    print("\n  NOTE: the SEM side is `derived` — back-extracted from")
    print("  paper/figures/tab_full_sem.tex at 2dp, with no trial counts.")
    print("  This test is therefore as precise as the paper's own table.")


if __name__ == "__main__":
    main()
