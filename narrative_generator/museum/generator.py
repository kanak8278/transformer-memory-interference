"""Museum floor-observation narrative interference generator.

Domain 36 — an experimental CONTROL, not another naturalistic setting. Its
defining property is the *absence* of an order cue: order lives only in sequence
position, exactly as in synthetic plain key-value streams. See
data/narrative_interference/domain_specs/36_museum_ATEMPORAL_DESIGN.md.

Render modes (§7):
  M0 atemporal   — no order marker at all           (plain KV analogue) [default]
  M1 timestamped — a monotone clock time per record (wildlife/ATC analogue)
  M2 indexed     — a per-visitor ordinal per record (block KV analogue)

Values are the SAME pool plain KV draws from (arbitrary single words), so museum
and plain are value-identical and differ only in the sentence frame. Values are
neutral one-word labels, not curated art titles (§5.1, §13 gate decision).
"""
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ..base import NarrativeTrialGenerator

_DATA = json.loads((Path(__file__).parent / "data" / "museum_data.json").read_text())


# ── M0 template bank (§7.1) — every frame takes a noun-slot value ────────────
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

# Filler — stative, atemporal, never names a tracked visitor (§2.9, §6.1).
FILLER_TEMPLATES = [
    "The skylight over the {gallery} was partly shuttered.",
    "Bench seating occupied the centre of the {gallery}.",
    "A guard station stood at the {gallery} doorway.",
    "Wall placards in the {gallery} gave title and medium only.",
    "The {seating} faced the long wall.",
    "A school group occupied the {gallery}.",
    "Lighting in the {gallery} was kept low for conservation.",
    "The floor plan marked the {gallery} in grey.",
    "Cases in the {gallery} held works on paper.",
    "The {seating} was unoccupied.",
]

# filler probability per record, by budget (§2.7, Universal Principle 4)
FILLER_BUDGETS = {"minimal": 0.05, "light": 0.25, "medium": 0.45, "heavy": 0.65}

RENDER_MODES = ("M0", "M1", "M2")
POOL_MODES = ("disjoint_pool", "shared_pool")


# ── order-cue linter (§9) ─────────────────────────────────────────────────────
# Category -> compiled pattern. Categories a render mode INTENTIONALLY injects
# are exempted via the `allow` set. Everything else must never appear in any
# mode. Patterns are lowercase/word-bounded; values and names are Title-case so
# they do not trip the lowercase pronoun/connective checks.
_LINT = {
    # 1. explicit temporal (M1 injects clock times)
    "temporal": re.compile(
        r"\b\d{1,2}:\d{2}\b"
        r"|\b\d{1,3}\s*-?\s*minute\b"
        r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d",
        re.I,
    ),
    # 2. relative temporal deictics ("before"/"after" excluded — spatial in our
    #    frames, e.g. "stood before")
    "deictic": re.compile(
        r"\b(?:now|then|later|subsequently|afterwards?|previously|earlier|"
        r"meanwhile|thereafter|next|again|already|finally|soon|once more)\b",
        re.I,
    ),
    # 3. perfect / anteriority aspect
    "aspect": re.compile(r"\b(?:had|has been|have been|having)\b", re.I),
    # 4. back-reference
    "backref": re.compile(
        r"\b(?:previous|prior|last seen|up from|down from|as before)\b", re.I
    ),
    # 6. causality
    "causal": re.compile(
        r"\b(?:because|prompting|leading to|so that|which drew|therefore)\b", re.I
    ),
    # 7. discourse ordinals (M2 injects per-visitor ordinals)
    "ordinal": re.compile(
        r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|"
        r"tenth|\d{1,3}(?:st|nd|rd|th))\b",
        re.I,
    ),
    # 8. third-person pronouns (lowercase only; Title-case values are exempt)
    "pronoun": re.compile(
        r"\b(?:he|she|they|him|her|them|his|hers|their|theirs)\b"
    ),
}

# which categories each mode is allowed to contain
_LINT_ALLOW = {
    "M0": set(),
    "M1": {"temporal"},
    "M2": {"ordinal"},
}


def _lint(text: str, render_mode: str) -> List[str]:
    """Return list of 'category:match' violations, honouring per-mode exemptions."""
    allow = _LINT_ALLOW[render_mode]
    hits = []
    for cat, pat in _LINT.items():
        if cat in allow:
            continue
        m = pat.search(text)
        if m:
            hits.append(f"{cat}:{m.group(0)!r}")
    return hits


def _ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _ordinal_word(n: int) -> str:
    words = ["zeroth", "first", "second", "third", "fourth", "fifth", "sixth",
             "seventh", "eighth", "ninth", "tenth"]
    return words[n] if n < len(words) else f"{_ordinal(n)}"


def query_positions(nu: int, n_points: int = 16) -> List[int]:
    """Per-visitor positions to probe for the U-curve (§11). First 7 fixed +
    evenly spaced fill + final. Mirrors experiments_cloud/ucurve_prompts.py."""
    if nu <= n_points:
        return list(range(1, nu + 1))
    fixed = list(range(1, 8))
    n_fill = n_points - len(fixed) - 1
    fill = []
    if n_fill > 0 and nu > 8:
        for i in range(n_fill):
            pos = round(8 + i * (nu - 1 - 8) / max(n_fill - 1, 1))
            fill.append(max(8, min(nu - 1, pos)))
    return sorted(set(fixed + fill + [nu]))


class MuseumTrialGenerator(NarrativeTrialGenerator):
    """Atemporal museum floor-observation trials."""

    DOMAIN = "museum"

    def __init__(self):
        # Drop any pool word that would itself trip the M0 linter — a handful of
        # the arbitrary-word values are function words ("First", "Then", "Had",
        # "Because") that both leak an order cue and read badly as a label.
        self.ARTWORKS = [a for a in _DATA["ARTWORKS"] if not _lint(a, "M0")]
        self.VISITORS = _DATA["VISITOR_NAMES"]
        self.GALLERIES = _DATA["GALLERIES"]
        self.SEATING = _DATA["SEATING"]
        self.MUSEUMS = [tuple(m) for m in _DATA["MUSEUMS"]]

        # §5.1 invariant: no value is a substring of another (containment scorer)
        low = [a.lower() for a in self.ARTWORKS]
        sset = set(low)
        assert len(sset) == len(low), "duplicate artwork values"
        # substring check is O(n^2); done once at construction against a sorted
        # copy grouped by length to keep it cheap enough for 1861 entries
        by_len = sorted(low, key=len)
        for i, w in enumerate(by_len):
            for o in by_len[i + 1:]:
                if len(o) == len(w):
                    continue
                if w in o:
                    raise AssertionError(f"'{w}' is a substring of '{o}'")

    # ─── setup ────────────────────────────────────────────────────────────────
    def _auto_config(self, num_keys, num_updates, rng, **opts):
        cfg = {
            "render_mode": opts.get("render_mode", "M0"),
            "pool_mode": opts.get("pool_mode", "disjoint_pool"),
            "working_set_multiplier": float(opts.get("working_set_multiplier", 1.0)),
            "filler": bool(opts.get("filler", False)),
            "filler_budget": opts.get("filler_budget")
            or (rng.choice(list(FILLER_BUDGETS)) if opts.get("filler") else "none"),
            "include_header": bool(opts.get("include_header", False)),
        }
        assert cfg["render_mode"] in RENDER_MODES, cfg["render_mode"]
        assert cfg["pool_mode"] in POOL_MODES, cfg["pool_mode"]
        return cfg

    def _deal(self, roster, nu, rng, pool_mode, wsm):
        """Return {visitor: [v1..vnu]} — unique within visitor (§2.3.1)."""
        nk = len(roster)
        if pool_mode == "disjoint_pool":
            need = nk * nu
            if need > len(self.ARTWORKS):
                raise ValueError(
                    f"disjoint_pool needs {need} titles, pool has {len(self.ARTWORKS)}"
                )
            selected = rng.sample(self.ARTWORKS, need)
            return {v: selected[i * nu:(i + 1) * nu] for i, v in enumerate(roster)}

        # shared_pool — working set (§2.5.1)
        W = max(nu, min(int(-(-wsm * nu // 1)), nk * nu))  # ceil(wsm*nu) clamped
        W = min(W, len(self.ARTWORKS))
        working = rng.sample(self.ARTWORKS, W)
        return {v: rng.sample(working, nu) for v in roster}

    def _emission_order(self, roster, nu, rng):
        """Interleaved slot list; avoid consecutive same visitor (best effort)."""
        slots = [v for v in roster for _ in range(nu)]
        if len(roster) == 1:
            return slots
        for _ in range(200):
            rng.shuffle(slots)
            if all(slots[i] != slots[i + 1] for i in range(len(slots) - 1)):
                break
        return slots

    def _assign_times(self, n_slots, rng):
        """Monotone clock times 09:00 → ~17:00 for M1."""
        start, end = 9 * 60, 17 * 60
        span = max(end - start, n_slots)
        times = []
        cur = start
        step = span / max(n_slots, 1)
        for _ in range(n_slots):
            cur += step
            m = int(min(cur, end - 1))
            times.append(f"{m // 60}:{m % 60:02d}")
        return times

    # ─── main ───────────────────────────────────────────────────────────────
    def generate_trial(self, num_keys: int, num_updates: int,
                       condition: Optional[str], seed: int, **opts) -> dict:
        rng = random.Random(seed)
        cfg = self._auto_config(num_keys, num_updates, rng, **opts)
        rm = cfg["render_mode"]

        if num_keys > len(self.VISITORS):
            raise ValueError(f"need {num_keys} visitors, pool has {len(self.VISITORS)}")
        roster = rng.sample(self.VISITORS, num_keys)

        decks = self._deal(roster, num_updates, rng,
                           cfg["pool_mode"], cfg["working_set_multiplier"])
        slots = self._emission_order(roster, num_updates, rng)
        times = self._assign_times(len(slots), rng) if rm == "M1" else None

        # header (§5.3) — OFF by default. When on, roster order is shuffled
        # independently of narrative order so it cannot leak first-mention rank.
        museum, building = rng.choice(self.MUSEUMS)
        header = None
        if cfg["include_header"]:
            header_roster = roster[:]
            rng.shuffle(header_roster)
            header = (
                f"Floor observations, {museum}, {building}. Statements are listed "
                f"in logging order. A visitor may be logged at several works.\n"
                f"Visitors on record: {', '.join(header_roster)}."
            )

        # body
        entity_tracking: Dict[str, List[str]] = {f"{v} / artwork": [] for v in roster}
        mention_counts: Dict[str, int] = {v: 0 for v in roster}
        full_state_log = []
        cursor = {v: 0 for v in roster}
        lines = []

        for slot_idx, v in enumerate(slots):
            val = decks[v][cursor[v]]
            cursor[v] += 1
            rank = cursor[v]  # per-visitor 1-based mention rank
            entity_tracking[f"{v} / artwork"].append(val)
            mention_counts[v] += 1

            sentence = rng.choice(M0_TEMPLATES).format(visitor=v, value=val)
            if rm == "M1":
                # lowercase the lead-in only for common-word starts, never names
                if sentence.startswith(("The ", "Floor ")):
                    sentence = sentence[0].lower() + sentence[1:]
                sentence = f"At {times[slot_idx]}, {sentence}"
            elif rm == "M2":
                sentence = f"{v}'s {_ordinal(rank)} recorded stop: {val}."

            lines.append(sentence)
            full_state_log.append({
                "slot_index": slot_idx,
                "entity": v,
                "attribute": "artwork",
                "value": val,
                "visitor_rank": rank,
                "marker": times[slot_idx] if rm == "M1" else (
                    _ordinal(rank) if rm == "M2" else None),
                "mentioned_in_narrative": True,
            })

            if cfg["filler"]:
                p = FILLER_BUDGETS.get(cfg["filler_budget"], 0.0)
                if rng.random() < p:
                    lines.append(rng.choice(FILLER_TEMPLATES).format(
                        gallery=rng.choice(self.GALLERIES),
                        seating=rng.choice(self.SEATING)))

        body = " ".join(lines)
        narrative = (header + "\n\n" + body) if header else body

        # queried visitor — guaranteed first != last by within-visitor uniqueness
        queried = roster[seed % num_keys]
        key = f"{queried} / artwork"
        vals = entity_tracking[key]
        questions = self._semantic_questions(queried, vals, rng)

        trial = {
            "id": f"museum_{seed:06d}",
            "domain": "museum",
            "num_keys": num_keys,
            "num_updates": num_updates,
            "narrative": narrative,
            "questions": questions,
            "entity_tracking": entity_tracking,
            "full_state_log": full_state_log,
            "mention_counts": mention_counts,
            "config": {
                "seed": seed,
                "num_keys": num_keys,
                "num_updates": num_updates,
                "tracked_attribute": "artwork",
                "render_mode": rm,
                "pool_mode": cfg["pool_mode"],
                "working_set_multiplier": cfg["working_set_multiplier"],
                "filler": cfg["filler"],
                "filler_budget": cfg["filler_budget"],
                "include_header": cfg["include_header"],
                "museum": museum,
                "building": building,
                "roster": roster,
                "queried_visitor": queried,
            },
        }

        # optional U-curve position questions (§11 intermediate config)
        if opts.get("all_positions"):
            trial["position_questions"] = self._position_questions(queried, vals)

        return trial

    # ─── question builders (Universal Principle 8: cheap, post-hoc) ───────────
    def _semantic_questions(self, visitor, vals, rng):
        RI = [
            f"In this report, what was the first artwork recorded for {visitor}?",
            f"What was the first artwork logged for {visitor} in this report?",
            f"At the first record for {visitor} in this report, what was the artwork?",
        ]
        PI = [
            f"In this report, what was the last artwork recorded for {visitor}?",
            f"What was the final artwork logged for {visitor} in this report?",
            f"At the last record for {visitor} in this report, what was the artwork?",
        ]
        return {
            "RI": {"question": rng.choice(RI), "expected_answer": vals[0],
                   "target_entity": visitor, "target_attribute": "artwork"},
            "PI": {"question": rng.choice(PI), "expected_answer": vals[-1],
                   "target_entity": visitor, "target_attribute": "artwork"},
        }

    def _position_questions(self, visitor, vals):
        """Ordinal query at each probed position; plus semantic anchor at the
        two endpoints (§11 — 'kth (last)' AND 'last' both probed)."""
        nu = len(vals)
        out = []
        for k in query_positions(nu):
            out.append({
                "style": "ordinal",
                "position": k,
                "question": (f"In this report, what was the {_ordinal(k)} artwork "
                             f"recorded for {visitor}?"),
                "expected_answer": vals[k - 1],
                "target_entity": visitor,
                "target_attribute": "artwork",
            })
        # semantic endpoints, same targets as position 1 and nu
        out.append({
            "style": "semantic", "position": 1, "anchor": "first",
            "question": f"In this report, what was the first artwork recorded for {visitor}?",
            "expected_answer": vals[0],
            "target_entity": visitor, "target_attribute": "artwork",
        })
        out.append({
            "style": "semantic", "position": nu, "anchor": "last",
            "question": f"In this report, what was the last artwork recorded for {visitor}?",
            "expected_answer": vals[-1],
            "target_entity": visitor, "target_attribute": "artwork",
        })
        return out

    # ─── validation (§10) ─────────────────────────────────────────────────────
    def validate_trial(self, trial: dict) -> bool:
        narr = trial["narrative"]
        et = trial["entity_tracking"]
        q = trial["questions"]
        rm = trial["config"]["render_mode"]

        # 1. every tracked value appears verbatim
        for key, vals in et.items():
            for val in vals:
                assert val in narr, f"value '{val}' for '{key}' missing from narrative"

        # 2-4. RI/PI well-posed and keyed to reading order
        ri, pi = q["RI"]["expected_answer"], q["PI"]["expected_answer"]
        assert ri != pi, f"RI==PI ('{ri}'): degenerate trial (num_updates>=2 required)"
        key = f"{q['RI']['target_entity']} / artwork"
        vals = et[key]
        assert len(vals) >= 2, f"queried entity '{key}' has <2 mentions"
        assert vals[0] == ri, f"RI must be first value {vals[0]!r}, got {ri!r}"
        assert vals[-1] == pi, f"PI must be last value {vals[-1]!r}, got {pi!r}"

        # 5. within-visitor uniqueness (§2.3.1)
        for k, vs in et.items():
            assert len(vs) == len(set(vs)), f"repeated value within visitor '{k}'"

        # 6. disjoint => no value shared across visitors
        if trial["config"]["pool_mode"] == "disjoint_pool":
            allv = [v for vs in et.values() for v in vs]
            assert len(allv) == len(set(allv)), "disjoint_pool: value shared across visitors"

        # 7. order-cue linter (§9), mode-aware
        hits = _lint(narr, rm)
        assert not hits, f"order-cue leak in {rm}: {hits}"

        return True

    # ─── batch (siblings' interface) ──────────────────────────────────────────
    @staticmethod
    def _get_data_dir() -> Path:
        current = Path(__file__).resolve().parent
        for _ in range(10):
            if (current / ".git").exists() or (current / "CLAUDE.md").exists():
                break
            current = current.parent
        return current / "data" / "narrative_interference" / "museum"

    def generate_batch(self, key_levels: List[int] = None,
                       update_levels: List[int] = None,
                       trials_per_cell: int = 30,
                       seed_start: int = 360000, **kwargs) -> dict:
        if key_levels is None:
            key_levels = [1, 2, 5, 10, 15, 20, 25, 30, 40, 45]
        if update_levels is None:
            update_levels = [5, 10, 15, 20, 30, 50]  # N=1 excluded (degenerate)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        trials, failures, seed = [], [], seed_start
        total = len(key_levels) * len(update_levels)
        ci, t0 = 0, time.time()

        for nk in key_levels:
            for nu in update_levels:
                ci += 1
                ok = 0
                for _ in range(trials_per_cell):
                    try:
                        t = self.generate_trial(nk, nu, None, seed, **kwargs)
                        self.validate_trial(t)
                        trials.append(t)
                        ok += 1
                    except Exception as e:
                        failures.append({"seed": seed, "nk": nk, "nu": nu,
                                         "error": str(e)[:120]})
                    seed += 1
                rate = len(trials) / max(time.time() - t0, 0.1)
                print(f"  [{ci}/{total}] keys={nk:>2} updates={nu:>3}: "
                      f"{ok}/{trials_per_cell} ok ({rate:.0f}/s)")

        print(f"\nGenerated {len(trials)} trials ({len(failures)} failures)")
        return {
            "metadata": {
                "domain": self.DOMAIN, "generator": self.__class__.__name__,
                "total_trials": len(trials), "failures": len(failures),
                "failure_detail": failures[:50],
                "timestamp": ts,
                "grid": {"num_keys": key_levels, "num_updates": update_levels},
                "trials_per_cell": trials_per_cell,
                "seed_range": f"{seed_start}-{seed - 1}",
                "overrides": kwargs if kwargs else "all_random",
            },
            "trials": trials,
        }

    def save_batch(self, batch: dict, tag: str = None) -> Path:
        ts = batch["metadata"]["timestamp"]
        parts = [self.DOMAIN] + ([tag] if tag else []) + [ts]
        out_dir = self._get_data_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / ("_".join(parts) + ".json")
        with open(out_path, "w") as f:
            json.dump(batch, f, indent=2, ensure_ascii=False)
        print(f"Saved to: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
        return out_path
