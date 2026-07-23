#!/usr/bin/env python3
"""Pilot M0 renderer — §13 gate.

NOT the full MuseumTrialGenerator. Just enough to render M0 (atemporal)
narratives so we can READ them and decide whether the observational-log genre
holds up under generation, or degenerates into a table.

Honours the invariants that affect how the prose reads:
  - observational past tense, full names, no pronouns, no connectives (§2.1/2.2)
  - one record per sentence (§2.4)
  - disjoint pool, within-visitor uniqueness via deal-without-replacement (§2.3.1)
  - rank-blind template selection (§2.8)
  - filler never names a tracked visitor (§2.9)
  - atemporal header + roster shuffled independently of narrative order (§5.3)

Run:  python3 narrative_generator/museum/pilot_render.py
"""
import json
import random
from pathlib import Path

DATA = json.loads((Path(__file__).parent / "data" / "museum_data.json").read_text())
ARTWORKS = DATA["ARTWORKS"]
VISITORS = DATA["VISITOR_NAMES"]
GALLERIES = DATA["GALLERIES"]
SEATING = DATA["SEATING"]
MUSEUMS = DATA["MUSEUMS"]

# ── M0 template bank (spec §7.1) ─────────────────────────────────────────────
M0_TEMPLATES = [
    "{visitor} was recorded at {value}.",
    "{visitor} was logged at {value}.",
    "{visitor} was observed at {value}.",
    "{visitor} stood before {value}.",
    "{visitor} spent the observation at {value}.",
    "The docent noted {visitor} at {value}.",
    "{visitor}'s attention rested on {value}.",
    "Floor staff logged {visitor} at {value}.",
    "{visitor} was noted at {value}.",
    "The record for {visitor} gives {value}.",
]

# ── filler bank (stative, atemporal, never names a tracked visitor) ──────────
FILLER_TEMPLATES = [
    "The skylight over the {gallery} was partly shuttered.",
    "Bench seating occupied the centre of the {gallery}.",
    "A guard station stood at the {gallery} doorway.",
    "Wall placards in the {gallery} gave title and medium only.",
    "The {seating} faced the long wall.",
    "A school group occupied the {gallery}.",
    "Lighting in the {gallery} was kept low for conservation.",
    "The floor plan marked the {gallery} in grey.",
]


def render(nk, nu, seed, filler=False):
    rng = random.Random(seed)

    # roster
    roster = rng.sample(VISITORS, nk)

    # deal artworks: disjoint across visitors, unique within (§2.3.1)
    pool = rng.sample(ARTWORKS, nk * nu)
    decks = {v: pool[i * nu:(i + 1) * nu] for i, v in enumerate(roster)}

    # ground truth keyed to reading order (filled as we emit)
    tracking = {v: [] for v in roster}

    # emission order: each visitor appears nu times, interleaved, avoid
    # long consecutive runs of the same visitor
    slots = [v for v in roster for _ in range(nu)]
    for _ in range(200):
        rng.shuffle(slots)
        if all(slots[i] != slots[i + 1] for i in range(len(slots) - 1)):
            break

    # header (roster order shuffled independently of narrative order, §5.3)
    museum, building = rng.choice(MUSEUMS)
    header_roster = roster[:]
    rng.shuffle(header_roster)
    header = (
        f"Floor observations, {museum}, {building}. Statements appear in the "
        f"order they were logged. A visitor may be logged at several works.\n"
        f"Visitors on record: {', '.join(header_roster)}."
    )

    # body
    cursor = {v: 0 for v in roster}
    lines = []
    for v in slots:
        val = decks[v][cursor[v]]
        cursor[v] += 1
        tracking[v].append(val)
        tmpl = rng.choice(M0_TEMPLATES)
        lines.append(tmpl.format(visitor=v, value=val))
        if filler and rng.random() < 0.5:
            f = rng.choice(FILLER_TEMPLATES)
            lines.append(f.format(gallery=rng.choice(GALLERIES),
                                  seating=rng.choice(SEATING)))

    narrative = header + "\n\n" + " ".join(lines)

    # query a visitor whose first != last (guaranteed by disjoint deal)
    q = roster[seed % nk]
    ri = tracking[q][0]
    pi = tracking[q][-1]
    questions = {
        "RI": (f"In this report, what was the first artwork recorded for {q}?", ri),
        "PI": (f"In this report, what was the last artwork recorded for {q}?", pi),
    }
    return narrative, questions, roster


def main():
    print("=" * 78)
    print("PILOT — M0 atemporal, NO FILLER (the default / exposed config, §13)")
    print("=" * 78)
    for (nk, nu, seed) in [(3, 5, 1), (5, 10, 2), (10, 5, 3)]:
        narr, qs, roster = render(nk, nu, seed, filler=False)
        print(f"\n### K={nk} N={nu} seed={seed}  ({nk*nu} records)\n")
        print(narr)
        print()
        for cond, (q, a) in qs.items():
            print(f"  [{cond}] {q}  ->  {a}")
        print("-" * 78)

    print("\n" + "=" * 78)
    print("PILOT — M0 atemporal, WITH FILLER (comparison)")
    print("=" * 78)
    narr, qs, roster = render(3, 5, 1, filler=True)
    print(f"\n### K=3 N=5 seed=1 (with filler)\n")
    print(narr)
    print()
    for cond, (q, a) in qs.items():
        print(f"  [{cond}] {q}  ->  {a}")


if __name__ == "__main__":
    main()
