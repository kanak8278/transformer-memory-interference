"""Prompts for the CoT vs non-CoT sweep on the museum M0 narrative domain.

Stimulus is a `MuseumTrialGenerator` M0 narrative — naturalistic prose, no order
marker of any kind, so recovering position k requires counting a visitor's
mentions while reading. The naturalistic counterpart of
`ucurve_prompts.prompt_flat_nolabel`, which is what `cot_ivq_prompts` uses.

The answer contract is imported verbatim from `cot_ivq_prompts` rather than
restated, so the two stimulus families cannot drift apart on the one thing that
has to be identical for their results to be comparable.

The prompt is BYTE-IDENTICAL in both arms. The only manipulated variable is
whether extended thinking is enabled on the API call.

Seeding
-------
blake2b via `make_seed`, not the `abs(hash((nk,nu,t)))%2**31` scheme the July
museum runs used. That scheme is fine for int-only tuples but the project has
string-containing variants of it, and PYTHONHASHSEED salting makes those
irreproducible across processes; blake2b removes the footgun entirely.

Consequence worth stating plainly: SEED_VERSION differs from the July runs, so
these trials are NOT paired with those. The two arms here are paired with each
other, which is what the contrast needs.

Question phrasing
-----------------
The fixed phrasings from `generator._position_questions`, not the 3-paraphrase
`rng.choice` bank in `_semantic_questions`. The July endpoint sweep used the
random bank and so averaged over three wordings; the July IVQ sweep used the
fixed one. Fixed everywhere here, so wording is held constant rather than being
a silent source of variance.
"""
import hashlib
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from experiments_cloud.cot_ivq_prompts import (  # noqa: E402
    ANSWER_INSTRUCTION, cached_blocks,
)
from experiments_cloud.ucurve_prompts import ordinal, query_positions  # noqa: E402
from narrative_generator import MuseumTrialGenerator  # noqa: E402

SEED_VERSION = "museum_cot_v1"

RENDER_MODE = "M0"          # atemporal: no clock times, no per-visitor ordinals
POOL_MODE = "disjoint_pool"  # every visitor's titles are unique -> a cross-visitor
#                              answer is unambiguously identifiable as such
ATTRIBUTE = "artwork"

_GEN = None


def gen() -> MuseumTrialGenerator:
    """Process-wide generator. Construction reads and lints the title pool, so
    it is worth doing once; `generate_trial` itself is pure given a seed."""
    global _GEN
    if _GEN is None:
        _GEN = MuseumTrialGenerator()
    return _GEN


def make_seed(nk: int, nu: int, trial_idx: int, version: str = SEED_VERSION) -> int:
    """Deterministic, PYTHONHASHSEED-independent seed.

    Same construction as `cot_ivq_prompts.make_seed`, different version string,
    so the museum and plain sweeps never collide on a stream.
    """
    key = "|".join(str(p) for p in (nk, nu, trial_idx, version)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big")


def _suffix_ordinal(visitor: str, k: int) -> str:
    return (f"In this report, what was the {ordinal(k)} {ATTRIBUTE} recorded "
            f"for {visitor}?\n{ANSWER_INSTRUCTION}")


def _suffix_anchor(visitor: str, anchor: str) -> str:
    """`anchor` is 'first' or 'last' — the semantic phrasing.

    Kept distinct from the ordinal query at the same target. In the July museum
    data 'first' and '1st' agree to within 1% while 'last' and 'Nth' diverge by
    6-47x, so collapsing them would hide the effect.
    """
    return (f"In this report, what was the {anchor} {ATTRIBUTE} recorded "
            f"for {visitor}?\n{ANSWER_INSTRUCTION}")


def build_trial(nk: int, nu: int, trial_idx: int, positions: list,
                cache: bool = False, render_mode: str = RENDER_MODE,
                pool_mode: str = POOL_MODE) -> dict:
    """One narrative, every position query built from it.

    All queries in a trial share one narrative, so the per-position curve within
    a trial is paired by construction. With cache=True each prompt is a two-block
    list carrying a cache breakpoint over that shared narrative; the text is
    identical either way.
    """
    seed = make_seed(nk, nu, trial_idx)
    trial = gen().generate_trial(nk, nu, None, seed,
                                 render_mode=render_mode, pool_mode=pool_mode)

    cfg = trial["config"]
    visitor = cfg["queried_visitor"]
    et = trial["entity_tracking"]

    # Verified across all six cells: entity_tracking order == narrative
    # appearance order == chronological update order. Unlike flat_nolabel, where
    # stream order and update order differ and only the former is countable,
    # the museum narrative has no such split.
    stream_vals = list(et[f"{visitor} / {ATTRIBUTE}"])

    # {visitor: [values]} for the scorer's out_of_context test.
    values_by_category = {k.split(" / ")[0]: list(v) for k, v in et.items()}

    # Every record in narrative order, for dist_from_narrative_end. Verified
    # sorted by slot_index and by narrative character offset, length K*N.
    narrative_records = [
        {"entity": e["entity"], "value": e["value"]}
        for e in trial["full_state_log"]
    ]

    prefix = trial["narrative"] + "\n\n"

    def _build(suffix):
        return cached_blocks(prefix, suffix) if cache else prefix + suffix

    queries = {}
    for k in positions:
        if k <= len(stream_vals):
            queries[str(k)] = (_build(_suffix_ordinal(visitor, k)),
                               stream_vals[k - 1])
    # Semantic anchors at both endpoints, same targets as position 1 and N.
    queries["first"] = (_build(_suffix_anchor(visitor, "first")), stream_vals[0])
    queries["last"] = (_build(_suffix_anchor(visitor, "last")), stream_vals[-1])

    return {
        "seed": seed,
        "test_category": visitor,
        "categories": list(cfg["roster"]),
        "stream_vals": stream_vals,
        "values_by_category": values_by_category,
        "narrative_records": narrative_records,
        "render_mode": render_mode,
        "pool_mode": pool_mode,
        "queries": queries,
    }


def position_keys(nu: int, n_points: int = 16) -> list:
    """The probed slices for one cell: ordinals + both semantic anchors.

    12 for N=10, 18 for N=20 and N=50 — the same set the July museum IVQ run
    used, so the two are slice-for-slice comparable.
    """
    return [str(k) for k in query_positions(nu, n_points)] + ["first", "last"]


__all__ = [
    "ANSWER_INSTRUCTION", "ATTRIBUTE", "POOL_MODE", "RENDER_MODE",
    "SEED_VERSION", "build_trial", "gen", "make_seed", "position_keys",
    "query_positions",
]
