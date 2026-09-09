"""Theme 8 — does the effect survive naturalistic prose?

Every other theme uses a synthetic key-value stream: `visual art: turquoise`,
one update per line. This theme re-asks the same questions against the museum
M0 narrative — continuous prose about visitors moving through galleries, where
an "update" is a sentence rather than a labelled line. If first-value bias, the
interior collapse and the CoT rescue all reproduce there, they are not artefacts
of the synthetic format.

Three sweeps, one stimulus, three of the questions asked elsewhere:

  endpoint.csv   FVQ vs CVQ over the widest grid in the corpus (58 cells)
                 -- the theme-1 question
  ivq.csv        per-position accuracy, K in {5,10} x N in {10,20,50}
                 -- the theme-2 question
  cot_ivq.csv    nocot vs cot_thinking, same cells
                 -- the theme-7 question

**HAIKU ONLY, AND THAT IS THE POINT OF READING THIS FIRST.** claude-4.5-haiku is
the only model with usable data on all three. The other two Claude 4.5 models
were run on the CoT sweep and both failed, differently:

  sonnet  `nocot` completed but **54.1% of its attempts were malformed**
          (9,757 of 18,024) against 0.0% for haiku and 3.9% for opus. Its
          reported 0.350 accuracy rests on the surviving 46%, a self-selected
          population, and by the >=20% off-stream rule that excluded seven
          open-weight models from theme 01 this arm does not qualify. Its
          `cot_thinking` arm died 64 trials into the first of six cells and
          wrote no checkpoint -- while the runner logged `exit 0`.
  opus    `nocot` completed cleanly. `cot_thinking` was never run.

Neither is consolidated here. `PROVENANCE.md` records where they sit and what
would make them usable, so the omission is a decision rather than an oversight.

Two properties of these sources that do not hold elsewhere in the corpus:

1. **Seeds are not reproducible for two of the three sweeps.** `endpoint` and
   `ivq` both record `seed_formula: abs(hash((nk,nu,t)))%(2**31)`, and Python's
   `hash` is PYTHONHASHSEED-salted for the string in that tuple. So those two
   sweeps cannot be re-run trial-for-trial -- the numbers are sound, the exact
   stimuli are not recoverable. `cot_ivq` is fine: `seed_version=museum_cot_v1`
   over blake2b, the same scheme theme 07 uses.
2. **`endpoint` has a real off-stream problem at depth.** 1,596 of 20,152
   observations are `garbage` (7.9% overall), concentrated in the deep cells:
   30% at (20,50), 29% at (15,50), 26% at (10,50), 20% at (20,30). Four cells
   are at or past the exclusion threshold applied in theme 01. They are emitted
   with `n_offstream` and `accuracy_onstream` populated so the filter is the
   reader's to apply, not silently pre-applied.

Sources (all RAW)
  experiments_cloud/results/museum_endpoint/claude-haiku/sweep_20260722_161853.json
      58 cells, K in {1,2,5,10,15,20,25,30,40,45} x N in {5,10,15,20,30,50},
      200 trials max with early stopping. The sibling `sweep_partial.json` is an
      in-progress snapshot of the SAME run and is stat-for-stat identical on all
      58 cells; only the final file carries `end_time`, so only it is read.
      `sweep_20260722_152429.json` is a 3-trial smoke and is skipped.
      `sweep_full_*.json` duplicate these plus per-trial `trial_details`.
  legacy/results/museum_ivq/claude-haiku/ivq_full_20260722_173410_summary.json
      6 cells x 12-18 probed positions, n=200 flat (no early stopping). The
      three `ivq_smoke_*` files are skipped. Despite living under `legacy/`, this
      is a 2026-07-22 run contemporaneous with the others -- the path is
      historical, not a statement about the data.
  experiments_cloud/results/museum_cot/claude-haiku-4-5-20251001__{nocot,cot_thinking}/checkpoint.json
      6 cells x 12-18 positions x 2 arms.

Why theme 08 and not rows added to themes 01 / 02 / 07: the stimulus is the
manipulated variable. Filing these three sweeps under the themes whose questions
they mirror would scatter one comparison across three directories and force the
same haiku-only caveat into each. Read alongside those themes, not merged into
them -- the grids do not match cell-for-cell anyway.
"""

from __future__ import annotations

import json
from pathlib import Path

from common import (DERIVED, ORDINAL, RAW, ROOT, SEMANTIC, cond_from_ri_pi,
                    emit, offstream, pos_key, rel, with_model)

THEME = "08_museum_naturalistic"
OUT = Path(__file__).resolve().parent.parent / THEME

DATASET = "museum_m0"
# M0 is the render mode: continuous narrative prose, disjoint value pools per
# visitor. Named as a prompt_format so it never collides with the synthetic
# formats (`plain`, `block`, `flat_nolabel`, ...) in a groupby.
PROMPT_FORMAT = "narrative_m0"

ENDPOINT_SRC = (ROOT / "experiments_cloud" / "results" / "museum_endpoint"
                / "claude-haiku" / "sweep_20260722_161853.json")
IVQ_SRC = (ROOT / "legacy" / "results" / "museum_ivq" / "claude-haiku"
           / "ivq_full_20260722_173410_summary.json")
COT_DIR = ROOT / "experiments_cloud" / "results" / "museum_cot"
COT_ARMS = ("nocot", "cot_thinking")
COT_MODEL_DIR = "claude-haiku-4-5-20251001__{arm}"

# Both `endpoint` and `ivq` seed from `abs(hash(...))` over a tuple containing a
# string, which Python salts per process. Carried on every row from those two
# sources so it cannot be lost by slicing the CSV.
UNSEEDED = "seed_formula uses PYTHONHASHSEED-salted hash(); stimuli not re-derivable"


def classify(position, num_updates: int) -> str:
    """WHICH value the query targets. `first`/`last` are their own positions here.

    The museum sweeps probe BOTH the worded endpoints (`first`, `last`) and the
    numeric slots (`1` .. `N`), so slot 1 and `first` are separate rows aimed at
    the same item. That is the ordinal-vs-semantic contrast, and `query_type`
    keeps them apart.
    """
    if position == "first":
        return "FVQ"
    if position == "last":
        return "CVQ"
    k = int(position)
    if k == 1:
        return "FVQ"
    return "CVQ" if k == num_updates else "IVQ"


def from_endpoint() -> list[dict]:
    """FVQ vs CVQ, 58 cells. The widest grid anywhere in the corpus."""
    blob = json.loads(ENDPOINT_SRC.read_text())
    src = rel(ENDPOINT_SRC)
    rows = []
    for cell in blob["cells"].values():
        nk, nu = cell["num_keys"], cell["num_updates"]
        for ri_pi, stats in cell["stats"].items():
            n_off, acc_on, err_json = offstream(stats)
            et = stats.get("error_types") or {}
            rows.append(with_model(
                blob["model"], theme=THEME, variant="base", dataset=DATASET,
                prompt_format=PROMPT_FORMAT, num_keys=nk, num_updates=nu,
                condition=cond_from_ri_pi(ri_pi),
                # The endpoint sweep words the query as "first"/"last"; it never
                # asks for a numbered update. Cross-check: sweep config carries
                # no position list.
                query_type=SEMANTIC,
                position="first" if ri_pi == "RI" else "last",
                accuracy=stats["accuracy"], ci_lower=stats["ci_lower"],
                ci_upper=stats["ci_upper"], n_trials=stats["n"],
                n_correct=et.get("correct"),
                n_offstream=n_off, accuracy_onstream=acc_on,
                error_types_json=err_json, regime=cell.get("regime"),
                provenance_tier=RAW, source_file=src,
                notes=f"stopped_early={cell['stopped_early']}; "
                      f"n_observations={cell['n_observations']}; {UNSEEDED}",
            ))
    return rows


def from_ivq() -> list[dict]:
    """Per-position accuracy. n=200 flat -- no early stopping in this sweep."""
    blob = json.loads(IVQ_SRC.read_text())
    meta = blob["metadata"]
    src = rel(IVQ_SRC)
    rows = []
    for r in blob["summary"]:
        pos = str(r["pos_label"])
        nu = r["num_updates"]
        # This sweep's taxonomy: in_sequence = right visitor, wrong slot;
        # in_context = a value belonging to another visitor. Both name a value
        # that IS in the narrative, so only `garbage` is off-stream -- the same
        # convention theme 07 uses.
        et = {"correct": r["n_correct"], "in_sequence": r["n_in_sequence"],
              "in_context": r["n_in_context"], "garbage": r["n_garbage"]}
        on = r["n"] - r["n_garbage"]
        rows.append(with_model(
            meta["model"], theme=THEME, variant="base", dataset=DATASET,
            prompt_format=PROMPT_FORMAT, num_keys=r["num_keys"], num_updates=nu,
            condition=classify(pos, nu),
            query_type=SEMANTIC if pos in ("first", "last") else ORDINAL,
            position=pos if pos in ("first", "last") else int(pos),
            accuracy=r["acc"], ci_lower=r["ci_low"], ci_upper=r["ci_high"],
            wilson_hw=r["wilson_hw"], n_trials=r["n"], n_correct=r["n_correct"],
            n_offstream=r["n_garbage"],
            accuracy_onstream=round(r["n_correct"] / on, 4) if on else "",
            error_types_json=json.dumps(et, sort_keys=True),
            provenance_tier=RAW, source_file=src,
            notes=f"model_id={meta['model_id']}; comparator={meta['comparator']}; "
                  f"{UNSEEDED}",
        ))
    return rows


def from_cot() -> list[dict]:
    """nocot vs cot_thinking on the narrative. Haiku only -- see module docstring."""
    rows = []
    for arm in COT_ARMS:
        path = COT_DIR / COT_MODEL_DIR.format(arm=arm) / "checkpoint.json"
        ck = json.loads(path.read_text())
        src = rel(path)
        cfg = ck["config"]
        for cell in ck["cells"].values():
            nk, nu = cell["num_keys"], cell["num_updates"]
            for pos, s in cell["positions"].items():
                # `query_style` is recorded per position by the sweep, so the
                # ordinal/semantic split is read rather than inferred.
                qs = s.get("query_style")
                query_type = SEMANTIC if qs in ("semantic", None) else ORDINAL
                if pos in ("first", "last"):
                    query_type = SEMANTIC
                notes = [
                    f"model_id={ck['model_id']}",
                    f"converged={s['converged']}",
                    f"n_malformed={s['n_malformed']}",
                    f"seed_version={ck['seed_version']}",
                    f"cache={ck['cache']}",
                ]
                # Budget truncation is silent in the API response, so the sweep
                # derives it per call. Where it is non-zero the CoT accuracy is
                # a LOWER BOUND: a 28-call probe at budget 16000 found 50% of
                # calls exceeding 3000 on deep positions of K10/N50.
                hit = s.get("n_thinking_budget_hit") or 0
                if hit:
                    notes.append(f"n_thinking_budget_hit={hit} "
                                 f"(accuracy is a LOWER BOUND at this position)")
                if cfg.get("thinking_budget"):
                    notes.append(f"thinking_budget={cfg['thinking_budget']}")
                rows.append(with_model(
                    ck["model_id"], theme=THEME, variant=arm, dataset=DATASET,
                    prompt_format=PROMPT_FORMAT, num_keys=nk, num_updates=nu,
                    condition=classify(pos, nu), query_type=query_type,
                    position=pos if pos in ("first", "last") else int(pos),
                    accuracy=s["accuracy"], ci_lower=s["ci_lower"],
                    ci_upper=s["ci_upper"], wilson_hw=s["wilson_hw"],
                    n_trials=s["n"], n_correct=s["n_correct"],
                    provenance_tier=RAW, source_file=src,
                    notes="; ".join(notes),
                ))
    return rows


def qa_budget_truncation(rows: list[dict]) -> None:
    """Report how much of the CoT arm is a lower bound rather than a measurement."""
    cot = [r for r in rows if r["variant"] == "cot_thinking"]
    hit = [r for r in cot if "n_thinking_budget_hit=" in r["notes"]]
    print(f"     QA thinking-budget truncation: {len(hit)} of {len(cot)} "
          f"cot_thinking rows have >=1 truncated call (accuracy = lower bound)")


def qa_offstream(rows: list[dict]) -> None:
    """Flag cells that would fail theme 01's >=20% off-stream exclusion."""
    bad = []
    for r in rows:
        if not r["n_offstream"] or not r["n_trials"]:
            continue
        frac = int(r["n_offstream"]) / int(r["n_trials"])
        if frac >= 0.20:
            bad.append((frac, r["num_keys"], r["num_updates"], r["condition"]))
    bad.sort(reverse=True)
    print(f"     QA off-stream >=20% (theme-01 exclusion threshold): "
          f"{len(bad)} row(s)")
    for frac, nk, nu, cond in bad[:6]:
        print(f"       ! K={nk} N={nu} {cond}: {frac:.0%} off-stream")


def main() -> None:
    print(f"[{THEME}] museum M0 narrative — claude-4.5-haiku only")

    endpoint = from_endpoint()
    endpoint.sort(key=lambda r: (int(r["num_keys"]), int(r["num_updates"]),
                                 r["condition"]))
    qa_offstream(endpoint)
    emit(endpoint, OUT, "endpoint.csv", bymodel="by_model_endpoint")

    ivq = from_ivq()
    ivq.sort(key=lambda r: (int(r["num_keys"]), int(r["num_updates"]),
                            pos_key(r["position"]), r["query_type"]))
    emit(ivq, OUT, "ivq.csv", bymodel="by_model_ivq")

    cot = from_cot()
    cot.sort(key=lambda r: (r["variant"], int(r["num_keys"]),
                            int(r["num_updates"]), pos_key(r["position"]),
                            r["query_type"]))
    qa_budget_truncation(cot)
    emit(cot, OUT, "cot_ivq.csv", bymodel="by_model_cot")


if __name__ == "__main__":
    main()
