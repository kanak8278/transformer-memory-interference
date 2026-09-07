"""Prompts for the CoT vs non-CoT interior-value-query (IVQ) sweep.

Stimulus is `ucurve_prompts.prompt_flat_nolabel`: a flat interleaved key-value
stream with no order markers, so recovering position k requires counting
occurrences of the queried key. One change from that builder — the answer is
requested inside <answer>...</answer> so it can be extracted as an exact span
rather than matched by substring containment.

The prompt is BYTE-IDENTICAL in both arms. The only manipulated variable is
whether extended thinking is enabled on the API call, which keeps the contrast
clean: nothing in the stimulus differs, only the reasoning channel.

Seeding uses blake2b rather than `ucurve_prompts.make_seed`, whose tuple contains
a string and is therefore PYTHONHASHSEED-salted (not reproducible across
processes). Identical seeds here mean the two arms see identical streams, which
makes the comparison paired trial-by-trial.
"""
import hashlib
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from experiments_cloud.ucurve_prompts import (  # noqa: E402
    generate_stream, ordinal, query_positions,
)

SEED_VERSION = "cot_ivq_v1"

# Requested answer format. Both arms carry this identically.
ANSWER_INSTRUCTION = (
    "Respond with only:\n"
    "<answer>the exact value</answer>"
)


def make_seed(nk: int, nu: int, trial_idx: int, version: str = SEED_VERSION) -> int:
    """Deterministic, PYTHONHASHSEED-independent seed.

    Matches the scheme in lora_intervention/data_gen.py:stable_seed.
    """
    key = "|".join(str(p) for p in (nk, nu, trial_idx, version)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big")


def _stream_text(flat_items: list) -> str:
    return "\n".join(f"{it['category']}: {it['value']}" for it in flat_items)


def _prefix(flat_items: list) -> str:
    """Preamble + stream: the part every query in a trial shares verbatim.

    Split out so callers can put a prompt-cache breakpoint here. At large N the
    stream dominates the prompt and is re-sent once per queried position, so
    caching it turns 17 full-price reads of the same 12k tokens into one write
    plus 16 cheap reads.
    """
    return f"{_PREAMBLE}\n\n{_stream_text(flat_items)}\n\n"


def _suffix_ordinal(test_category: str, k: int) -> str:
    return f"What was the {ordinal(k)} value of {test_category}?\n{ANSWER_INSTRUCTION}"


def _suffix_last(test_category: str) -> str:
    return f"What was the last value of {test_category}?\n{ANSWER_INSTRUCTION}"


def cached_blocks(prefix: str, suffix: str) -> list:
    """Content blocks with an ephemeral cache breakpoint after the prefix.

    Concatenating the two blocks reproduces the single-string prompt exactly, so
    the stimulus is unchanged. Note the block boundary can shift tokenization by
    a token or two at the seam versus one flat string; irrelevant when every
    cell in a sweep is built the same way, but do not mix cached and uncached
    cells inside one comparison.
    """
    return [
        {"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": suffix},
    ]


_PREAMBLE = (
    "Read the following key-value stream. Each key appears multiple times "
    "as it gets updated. Count each occurrence of a key as one update."
)


def prompt_ordinal(flat_items: list, test_category: str, k: int) -> str:
    """Ordinal position query: 'the k-th value of X'."""
    return _prefix(flat_items) + _suffix_ordinal(test_category, k)


def prompt_last(flat_items: list, test_category: str) -> str:
    """Semantic endpoint query: 'the last value of X'.

    Distinct from the ordinal query at k=N — same target, different phrasing —
    and the two are known to diverge, so both are collected.
    """
    return _prefix(flat_items) + _suffix_last(test_category)


def build_trial(nk: int, nu: int, trial_idx: int, positions: list,
                dataset: str = "SEMANTIC_MULTI", cache: bool = False) -> dict:
    """One stream, all position queries built from it.

    Returns the stream metadata plus {position_key: (prompt, expected)}. All
    positions in a trial share one stream, so the per-position curve within a
    trial is paired by construction.

    With cache=True each prompt is a two-block list instead of a string, the
    first block carrying a cache breakpoint over the shared stream. The text is
    identical either way.
    """
    seed = make_seed(nk, nu, trial_idx)
    cats, test, vals, flat, _block = generate_stream(nk, nu, seed, dataset=dataset)

    # Values of the queried key in STREAM order. flat_nolabel asks for the k-th
    # occurrence as encountered while reading, which is what the model can count;
    # vals[test] is chronological update order and is not the same list.
    stream_vals = [i["value"] for i in flat if i["category"] == test]

    pre = _prefix(flat)

    def _build(suffix):
        return cached_blocks(pre, suffix) if cache else pre + suffix

    queries = {}
    for k in positions:
        if k <= len(stream_vals):
            queries[str(k)] = (_build(_suffix_ordinal(test, k)), stream_vals[k - 1])
    queries["last"] = (_build(_suffix_last(test)), stream_vals[-1])

    return {
        "seed": seed,
        "test_category": test,
        "categories": cats,
        "stream_vals": stream_vals,
        "values_by_category": vals,
        "queries": queries,
    }


__all__ = [
    "ANSWER_INSTRUCTION", "SEED_VERSION", "build_trial", "cached_blocks",
    "make_seed", "prompt_last", "prompt_ordinal", "query_positions",
]
