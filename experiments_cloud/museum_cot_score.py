"""Scoring for the museum CoT sweep: `cot_ivq_score` plus narrative position.

Extraction and the five-way outcome taxonomy are imported unchanged from
`cot_ivq_score` — same `<answer>` span, same equality comparison, same
`no_answer` handling. That file is not modified; the museum-specific part is
purely additive.

What is added
-------------
Where in the narrative the model's answer came from, counted backwards from the
end (`dist_from_narrative_end`; 0 = the very last record in the narrative,
whoever it belonged to).

This exists because the July museum runs showed that `out_of_context` is not a
uniform bucket. Of the 1,488 cross-visitor errors on the endpoint CVQ query,
1,432 (96.2%) were exactly the final record of the narrative, and on the IVQ
arm's semantic `last` query 284 of 298 (95.3%) were. Ordinal-query errors show
no such concentration. So "wrong entity" and "grabbed whatever was mentioned
last" are different failures that the outcome label alone cannot separate.

Recovering that from the July data required regenerating every error trial from
its seed. Recording it at run time costs one list index and no API calls.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from experiments_cloud.cot_ivq_score import (  # noqa: E402
    OUTCOMES, TIERS, classify, extract_answer, normalize,
)


def locate_in_narrative(answer: str | None, narrative_records: list) -> dict:
    """Find the model's answer among all records, in narrative order.

    Searches from the END so that a value appearing more than once resolves to
    its most recent occurrence — the relevant one for a recency hypothesis.
    Under `disjoint_pool` values are unique anyway, so this only matters if the
    pool mode is ever changed.

    Returns nulls rather than raising when the answer is absent from the
    narrative (a `garbage` outcome), so every record carries the same fields.
    """
    if not answer or not narrative_records:
        return {"narrative_idx": None, "dist_from_narrative_end": None,
                "narrative_entity": None}

    n = len(narrative_records)
    for i in range(n - 1, -1, -1):
        rec = narrative_records[i]
        if normalize(rec["value"]) == answer:
            return {
                "narrative_idx": i,
                "dist_from_narrative_end": n - 1 - i,
                "narrative_entity": rec["entity"],
            }
    return {"narrative_idx": None, "dist_from_narrative_end": None,
            "narrative_entity": None}


def score_response(text, expected, stream_vals, values_by_category,
                   test_category, narrative_records):
    """extract_answer + classify + narrative location, merged into one record.

    Signature mirrors `cot_ivq_score.score_response` with `narrative_records`
    appended, so the two sweeps' scoring calls read the same way.
    """
    ex = extract_answer(text, expected=expected)
    cl = classify(ex["answer"], expected, stream_vals, values_by_category,
                  test_category)
    loc = locate_in_narrative(ex["answer"], narrative_records)

    out = {**ex, **cl, **loc}
    out["correct"] = out["outcome"] == "correct"
    # Well-formed responses only. Headline accuracy is computed over these;
    # `tier` lets you recompute either way without re-running.
    out["well_formed"] = out["tier"] == "tagged"
    # True when the answer is the narrative's final record but not the target.
    # The single most informative derived flag in the July re-analysis.
    out["took_narrative_final"] = (
        out["dist_from_narrative_end"] == 0 and not out["correct"]
    )
    return out


__all__ = ["OUTCOMES", "TIERS", "classify", "extract_answer",
           "locate_in_narrative", "normalize", "score_response"]
