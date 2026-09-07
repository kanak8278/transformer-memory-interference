"""Theme 2 — intermediate-value retrieval (IVQ) on UNMODIFIED models.

Accuracy at every update position, not just first and last. The shape of the
per-position curve is the finding: base models are U-shaped (endpoints only).

Scope (changed 2026-07-22): this theme now holds ONLY standalone base-model IVQ
runs. The open-weight IVQ experiments (behavioral block-vs-LoRA, dense-IVQ, E1
extrapolation) were all run as base-vs-LoRA comparisons, so they live with the
LoRA work in `04_lora/lora_ivq.csv`. What remains here is the one genuinely
standalone interior sweep: the proprietary U-curve, which has no LoRA arm
(you cannot fine-tune an API model).

Position semantics: `position` is the 1-indexed update queried. position 1 ->
FVQ, position N -> CVQ, interior -> IVQ.

Source (RAW)
  experiments_cloud/results/ucurve_proprietary_results.csv
      6 proprietary models x 4 formats (flat_nolabel, flat_verbose, block,
      landmark) x 6 cells (K in 5,10 x N in 10,20,50) x ~15 probed positions,
      on SEMANTIC_MULTI. Identical grid across all 6 models; only per-cell trial
      count varies (Wilson early stopping).
"""

from __future__ import annotations

import csv
from pathlib import Path

from common import ORDINAL, RAW, ROOT, SEMANTIC, emit, pos_key, rel, with_model

THEME = "02_intermediate_ivq"
OUT = Path(__file__).resolve().parent.parent / THEME


def classify(position, num_updates: int) -> str:
    """WHICH value the query targets: first, last, or an interior one."""
    if position == "last" or int(position) == num_updates:
        return "CVQ"
    return "FVQ" if int(position) == 1 else "IVQ"


def from_ucurve() -> list[dict]:
    path = ROOT / "experiments_cloud" / "results" / "ucurve_proprietary_results.csv"
    src = rel(path)
    rows = []
    for r in csv.DictReader(open(path)):
        n = int(r["num_updates"])
        pos = r["position"]
        rows.append(with_model(
            r["model"], theme=THEME, variant="base",
            # SEMANTIC_MULTI: run_ucurve_proprietary.sh passes --dataset SEMANTIC_MULTI.
            dataset="semantic_multi", prompt_format=r["format"],
            num_keys=int(r["num_keys"]), num_updates=n,
            condition=classify(pos, n),
            # numbered positions are ordinal ("the 7th value"); the separate
            # 'last' rows use the semantic *_lastquery templates.
            query_type=SEMANTIC if pos == "last" else ORDINAL,
            position="last" if pos == "last" else int(pos),
            accuracy=float(r["accuracy"]), ci_lower=float(r["ci_lower"]),
            ci_upper=float(r["ci_upper"]), wilson_hw=float(r["wilson_hw"]),
            n_trials=int(r["n_trials"]), provenance_tier=RAW, source_file=src,
            notes=f"upstream source_file: {r['source_file']}",
        ))
    return rows


def main() -> None:
    print(f"[{THEME}] intermediate-position (IVQ) retrieval — standalone base runs")
    rows = from_ucurve()
    rows.sort(key=lambda r: (
        r["model"], r["prompt_format"],
        int(r["num_keys"] or 0), int(r["num_updates"] or 0),
        pos_key(r["position"])))
    emit(rows, OUT, "ivq.csv")


if __name__ == "__main__":
    main()
