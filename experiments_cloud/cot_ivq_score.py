"""Answer extraction and error classification for the CoT / non-CoT IVQ sweep.

Two deliberate departures from `ucurve_sweep.classify`:

1. NO CONTAINMENT. That scorer accepts `expected in predicted`, so any response
   that enumerates the stream contains the target value and is credited whatever
   it concluded. Harmless for one-word answers, fatal once reasoning is in play.
   Here the answer is an exact span pulled from <answer>...</answer> and compared
   by equality.

2. FAILURE MODES ARE NAMED, NOT SILENT. A truncated or tag-less response is
   `no_answer`, not a wrong answer. That distinction matters directionally: only
   the thinking arm can exhaust its token allowance, so folding truncation into
   the error bucket would bias the comparison against the arm under test.
"""
import re

# Tolerant of whitespace inside the tags; case-insensitive; DOTALL so a value
# that wrapped across lines is still captured.
_ANSWER_RE = re.compile(r"<answer\s*>(.*?)</\s*answer\s*>", re.DOTALL | re.IGNORECASE)
_OPEN_RE = re.compile(r"<answer\s*>", re.IGNORECASE)

# Signals the model echoed the instruction template instead of substituting a
# value. Looks like a wrong answer unless tested for explicitly.
_PLACEHOLDER_MARKERS = ("the exact value", "{", "}", "your answer")

TIERS = ("tagged", "unclosed", "whole_block", "none")
OUTCOMES = ("correct", "in_sequence", "out_of_context", "garbage", "no_answer")


def normalize(s: str) -> str:
    """Casefold, trim wrapping quotes and trailing punctuation, collapse spaces.

    Whitespace collapsing is required, not cosmetic: SEMANTIC_MULTI values are
    multi-token ('greek revival', 'art nouveau').
    """
    if s is None:
        return ""
    # Iterate: a single pass leaves '"turquoise".' as 'turquoise"' because the
    # trailing quote only becomes outermost after the period is removed.
    prev = None
    while prev != s:
        prev = s
        s = s.strip()
        s = s.strip('"\u201c\u201d\'`')
        s = s.strip(".!,;:")
    s = re.sub(r"\s+", " ", s)
    return s.casefold()


def extract_answer(text: str | None, expected: str | None = None) -> dict:
    """Pull the committed answer span out of the model's text block.

    Cascade, most-trusted first. The tier is recorded so the primary metric can
    be restricted to well-formed responses while malformed ones stay countable
    rather than being silently absorbed.

    `expected` is used only to decide whether a tag-less response happens to be
    a bare correct value (the `whole_block` tier). It never relaxes the
    comparison itself.
    """
    if text is None:
        return {"answer": None, "tier": "none", "n_matches": 0, "placeholder": False}

    matches = _ANSWER_RE.findall(text)
    if matches:
        # Last match: a model that restates its answer commits with the final one.
        raw = matches[-1]
        norm = normalize(raw)
        return {
            "answer": norm or None,
            "tier": "tagged" if norm else "none",
            "n_matches": len(matches),
            "placeholder": any(m in norm for m in _PLACEHOLDER_MARKERS),
            "answer_raw": raw,
        }

    # Opening tag with no close: near-certainly truncation. Salvage the tail but
    # flag it, so it can be excluded from the headline number.
    m = _OPEN_RE.search(text)
    if m:
        raw = text[m.end():]
        norm = normalize(raw)
        return {
            "answer": norm or None,
            "tier": "unclosed" if norm else "none",
            "n_matches": 0,
            "placeholder": any(mk in norm for mk in _PLACEHOLDER_MARKERS),
            "answer_raw": raw,
        }

    # No tags at all. Accept only if the entire block IS the value — still an
    # exact match, not containment.
    norm = normalize(text)
    if expected is not None and norm and norm == normalize(expected):
        return {"answer": norm, "tier": "whole_block", "n_matches": 0,
                "placeholder": False, "answer_raw": text}

    return {"answer": norm or None, "tier": "none", "n_matches": 0,
            "placeholder": False, "answer_raw": text}


def classify(answer: str | None, expected: str, stream_vals: list,
             values_by_category: dict, test_category: str) -> dict:
    """Bucket one response.

    `in_sequence` (right key, wrong position) is kept separate from
    `out_of_context` (wrong key entirely) because with per-category semantic
    pools those are different failures: the first is a positional-addressing
    error, the second is a retrieval error. Collapsing them, as the original
    three-way classifier does, discards the more informative signal.
    """
    if answer is None or answer == "":
        return {"outcome": "no_answer", "predicted_idx": None,
                "predicted_relative_pos": None, "predicted_category": None}

    exp_norm = normalize(expected)
    if answer == exp_norm:
        idx = _index_of(answer, stream_vals)
        return {"outcome": "correct", "predicted_idx": idx,
                "predicted_relative_pos": _rel_pos(idx, len(stream_vals)),
                "predicted_category": test_category}

    idx = _index_of(answer, stream_vals)
    if idx is not None:
        return {"outcome": "in_sequence", "predicted_idx": idx,
                "predicted_relative_pos": _rel_pos(idx, len(stream_vals)),
                "predicted_category": test_category}

    for cat, vals in (values_by_category or {}).items():
        if cat == test_category:
            continue
        if any(normalize(v) == answer for v in vals):
            return {"outcome": "out_of_context", "predicted_idx": None,
                    "predicted_relative_pos": None, "predicted_category": cat}

    return {"outcome": "garbage", "predicted_idx": None,
            "predicted_relative_pos": None, "predicted_category": None}


def _index_of(norm_answer: str, values: list):
    for i, v in enumerate(values):
        if normalize(v) == norm_answer:
            return i
    return None


def _rel_pos(idx, n):
    if idx is None:
        return None
    return round(idx / max(n - 1, 1), 4)


def score_response(text, expected, stream_vals, values_by_category, test_category):
    """extract_answer + classify, merged into one flat record."""
    ex = extract_answer(text, expected=expected)
    cl = classify(ex["answer"], expected, stream_vals, values_by_category, test_category)
    out = {**ex, **cl}
    out["correct"] = out["outcome"] == "correct"
    # Well-formed responses only. The headline accuracy is computed over these;
    # `tier` lets you recompute either way without re-running.
    out["well_formed"] = out["tier"] == "tagged"
    return out
