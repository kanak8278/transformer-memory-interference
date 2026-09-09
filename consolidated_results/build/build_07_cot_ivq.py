"""Theme 7 — does chain-of-thought rescue the interior?

Theme 2 established that base models collapse in the interior of an update
stream when the format hides indices (`flat_nolabel`). This theme asks whether
letting the model reason first repairs that collapse, on the three Claude 4.5
models. The stimulus is byte-identical across arms; the only manipulated
variable is whether extended thinking is enabled on the API call, so the
contrast is not confounded by prompt wording.

Output layout: ONE CSV PER ARM, `cot_ivq_nocot.csv` and
`cot_ivq_cot_thinking.csv`, each with its own `by_model_<arm>/` split. They
partition the 846 rows with no overlap. The cost of this split is that the
theme's headline contrast — same cell, same stream, arm on vs off — now spans
two files; `sort_key` is arm-independent so the shared rows align positionally,
and `check_arm_alignment` proves that on every build rather than assuming it.

Two datasets, and they are NOT symmetric:

  SEMANTIC_MULTI     both arms (`nocot`, `cot_thinking`), K in {5,10} x
                     N in {10,20,50}. This is the actual CoT-vs-no-CoT contrast.
  ARBITRARY_SINGLE   `cot_thinking` only, K=5, N in {50,100,200,300,400,460}.
                     A depth ladder with no paired no-CoT arm — it answers "how
                     deep does CoT hold", not "does CoT help".

Denominator convention, inherited from the sweep and preserved here.
`n_trials` counts WELL-FORMED responses only: the answer arrived inside
<answer>...</answer> tags (`extraction_tier == "tagged"`). Responses with no
usable tag (`no_answer` — truncation, or the model never committed a span) are
excluded from the denominator and counted in `n_malformed`, which this builder
carries in `notes`. That exclusion is deliberate upstream: only the thinking arm
can exhaust its token allowance, so folding truncation into the error bucket
would bias the comparison against the arm under test.

`n_offstream` therefore counts `garbage` only — a response that named no value
in the stream at all. `out_of_context` (a real value, but belonging to a
different key) is ON-stream by this repo's definition and stays in the error
bucket, as does `in_sequence` (right key, wrong position).

Per-position convergence is uneven and matters. Each position retires on its own
Wilson half-width, so within a single cell `n_trials` ranges from ~51 to 200.
**Never pool positions inside a cell** — a mean over them weights the
early-retiring easy positions and the max-trials hard ones as if they were
equally sampled. Read `n_trials` per row.

Sources (RAW)
  experiments_cloud/results/cot_ivq/<model>__<arm>/checkpoint.json
  experiments_cloud/results/cot_ivq_arbitrary_single/<model>__<arm>__ARBITRARY_SINGLE/checkpoint.json
      Per-position accuracy, n, n_correct, Wilson half-width and CI. Authoritative
      for every number emitted here.
  .../trials.jsonl (same directories)
      Per-trial records. Read only to fill the output-taxonomy columns
      (`n_offstream`, `accuracy_onstream`, `error_types_json`) and to cross-check
      that the checkpoint denominators match a recount from the raw trials.

Not used as a source, deliberately:
  results/cot_ivq_arbitrary_single/_figs/ladder_summary.csv — carries no `model`
      column and holds a stale haiku-only snapshot (93 of 102 cells match the
      finished haiku checkpoint). The per-run copies under each model directory
      are correct but cover only opus and sonnet. Checkpoints are the one
      complete, model-labelled source.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

from common import (ORDINAL, RAW, ROOT, SEMANTIC, emit, pos_key, rel,
                    with_model)

THEME = "07_cot_ivq"
OUT = Path(__file__).resolve().parent.parent / THEME

# Both sweeps drive ucurve_prompts.prompt_flat_nolabel (see
# experiments_cloud/cot_ivq_prompts.py) — a flat interleaved stream with no
# order markers, so position k is only recoverable by counting occurrences.
PROMPT_FORMAT = "flat_nolabel"

RUN_DIRS = (
    ROOT / "experiments_cloud" / "results" / "cot_ivq",
    ROOT / "experiments_cloud" / "results" / "cot_ivq_arbitrary_single",
)

# One canonical CSV per arm, following the theme-04 pattern (two combined CSVs,
# each with its own per-model split). Insertion order matters: it fixes the
# argument order of the alignment report below. Together the two files partition
# every row exactly once.
ARM_FILES = {
    "nocot": "cot_ivq_nocot.csv",
    "cot_thinking": "cot_ivq_cot_thinking.csv",
}

# Outputs that named no value in the stream. Narrower than common.OFFSTREAM_TYPES
# because this sweep's taxonomy is finer: `out_of_context` IS a stream value (it
# belongs to another key), so it is not off-stream.
OFFSTREAM_OUTCOMES = ("garbage",)


def classify(position, num_updates: int) -> str:
    """WHICH value the query targets: first, last, or an interior one."""
    if position == "last" or int(position) == num_updates:
        return "CVQ"
    return "FVQ" if int(position) == 1 else "IVQ"


def trial_records(run_dir: Path) -> dict:
    """(num_keys, num_updates, position) -> [(trial_idx, well_formed, outcome)].

    Sorted by trial_idx, which is what lets `taxonomy` reconstruct the exact
    subset of trials the checkpoint counted. See the docstring there.
    """
    path = run_dir / "trials.jsonl"
    out: dict = collections.defaultdict(list)
    with open(path) as fh:
        for line in fh:
            t = json.loads(line)
            key = (t["num_keys"], t["num_updates"], t["position"])
            out[key].append((t["trial_idx"], t["well_formed"], t["outcome"]))
    for recs in out.values():
        recs.sort()
    return out


def taxonomy(records: list, n: int) -> collections.Counter:
    """Outcome counts over the `n` well-formed trials the checkpoint counted.

    Why a prefix rather than the whole list. `trials.jsonl` is append-only and
    survives a resume, so for a cell that was resumed it holds the UNION of
    attempts, while the checkpoint's `n` counts only the final attempt's
    observations. One position in the corpus is affected (sonnet,
    ARBITRARY_SINGLE, K=5 N=400, position 6: checkpoint n=56, JSONL 64 distinct
    trial_idx). Counting the union there would put a 64-trial error taxonomy
    next to a 56-trial denominator in the same row.

    The sweep issues trials in ascending `trial_idx` waves and appends to `obs`
    in that order, so the first `n` well-formed records by trial_idx ARE the
    counted set. The `n_correct` cross-check in `from_run` verifies that on
    every row; it is not an assumption left untested.

    Malformed records inside the consumed prefix are tallied under "_malformed",
    matching the checkpoint's `n_malformed`, which is likewise excluded from `n`.
    """
    counts: collections.Counter = collections.Counter()
    seen = 0
    for _idx, well_formed, outcome in records:
        if seen >= n:
            break
        if well_formed:
            counts[outcome] += 1
            seen += 1
        else:
            counts["_malformed"] += 1
    return counts


def from_run(run_dir: Path) -> tuple[list[dict], list[str]]:
    """Rows for one model x arm x dataset run, plus any cross-check failures."""
    ckpt_path = run_dir / "checkpoint.json"
    ck = json.load(open(ckpt_path))
    src = rel(ckpt_path)
    records = trial_records(run_dir)

    dataset = ck["dataset"].lower()          # SEMANTIC_MULTI -> semantic_multi
    arm = ck["arm"]                          # nocot | cot_thinking
    cfg = ck["config"]
    budget = cfg.get("thinking_budget")

    rows, problems = [], []
    for cell in ck["cells"].values():
        nk, nu = cell["num_keys"], cell["num_updates"]
        for pos, s in cell["positions"].items():
            n = s["n"]
            recs = records.get((nk, nu, pos), [])
            counts = taxonomy(recs, n)

            # Cross-checks. The prefix must be long enough to supply `n`
            # well-formed trials, and the correct count over that prefix must
            # equal the checkpoint's `n_correct` -- which is what proves the
            # prefix really is the subset the checkpoint counted. A `n_correct`
            # mismatch means the JSONL and the checkpoint disagree about the same
            # run, so neither can be trusted.
            recount = sum(v for k, v in counts.items() if k != "_malformed")
            if recount != n:
                problems.append(
                    f"{src} {nk}k{nu}u pos={pos}: checkpoint n={n} but only "
                    f"{recount} well-formed trials in trials.jsonl")
            if counts.get("correct", 0) != s["n_correct"]:
                problems.append(
                    f"{src} {nk}k{nu}u pos={pos}: n_correct="
                    f"{s['n_correct']} but {counts.get('correct', 0)} correct "
                    f"trials in trials.jsonl")

            off = sum(counts.get(k, 0) for k in OFFSTREAM_OUTCOMES)
            on = n - off
            acc_on = round(s["n_correct"] / on, 4) if on else ""
            errors = {k: v for k, v in sorted(counts.items()) if k != "_malformed"}

            notes = [
                # The dated snapshot. Theme 02's claude rows carry only undated
                # aliases, so this is the only place the exact snapshot survives.
                f"model_id={ck['model_id']}",
                f"converged={s['converged']}",
                f"n_malformed={s['n_malformed']}",
                f"seed_version={ck['seed_version']}",
                f"sdk={ck['anthropic_sdk_version']}",
            ]
            if budget is not None:
                notes.append(f"thinking_budget={budget}")

            rows.append(with_model(
                ck["model_id"], theme=THEME, variant=arm, dataset=dataset,
                prompt_format=PROMPT_FORMAT, num_keys=nk, num_updates=nu,
                condition=classify(pos, nu),
                # Numbered positions are ordinal ("the 7th value of X"); the
                # separate `last` row asks for "the last value of X". Same target
                # item at position N, different phrasing, and they diverge.
                query_type=SEMANTIC if pos == "last" else ORDINAL,
                position="last" if pos == "last" else int(pos),
                accuracy=s["accuracy"], ci_lower=s["ci_lower"],
                ci_upper=s["ci_upper"], wilson_hw=s["wilson_hw"],
                n_trials=n, n_correct=s["n_correct"],
                n_offstream=off, accuracy_onstream=acc_on,
                error_types_json=json.dumps(errors, sort_keys=True),
                provenance_tier=RAW, source_file=src,
                notes="; ".join(notes),
            ))
    return rows, problems


def sort_key(r: dict) -> tuple:
    """Row order, shared by both arm files.

    `variant` is deliberately NOT part of it: with the same key applied to both
    files, the rows common to both arms appear in the same relative order, so
    the paired comparison needs no sort or merge on the reader's side.
    """
    return (r["dataset"], r["model"],
            int(r["num_keys"] or 0), int(r["num_updates"] or 0),
            pos_key(r["position"]), r["query_type"])


def check_arm_alignment(rows: list[dict]) -> None:
    """Report whether the arm files can be compared row-for-row.

    Splitting by arm turns the theme's headline contrast into a cross-file
    operation, so this states in the build log exactly how safe that is: which
    (dataset, model, cell, position) keys exist in both arms, and whether the
    shared ones come out in the same order in each file.
    """
    key = lambda r: (r["dataset"], r["model"], r["num_keys"],  # noqa: E731
                     r["num_updates"], r["position"], r["query_type"])
    per_arm = {arm: [key(r) for r in rows if r["variant"] == arm]
               for arm in ARM_FILES}
    a, b = (per_arm[arm] for arm in ARM_FILES)
    shared = set(a) & set(b)
    aligned = [k for k in a if k in shared] == [k for k in b if k in shared]
    arms = " / ".join(ARM_FILES)
    print(f"     arm alignment ({arms}): {len(shared)} keys in both, "
          f"{len(set(a) - shared)} nocot-only, {len(set(b) - shared)} "
          f"cot_thinking-only, shared rows in identical order: {aligned}")


def main() -> None:
    print(f"[{THEME}] CoT vs non-CoT interior retrieval — Claude 4.5")
    rows, problems = [], []
    for base in RUN_DIRS:
        for run_dir in sorted(p for p in base.iterdir()
                              if p.is_dir() and not p.name.startswith("_")):
            r, p = from_run(run_dir)
            rows.extend(r)
            problems.extend(p)
            print(f"     {run_dir.name}: {len(r)} position-rows")

    print(f"     cross-check vs trials.jsonl: {len(problems)} mismatch(es)")
    for line in problems[:10]:
        print(f"       ! {line}")

    # One file per arm. The two are a clean PARTITION of the 846 rows -- no row
    # appears in both -- so there is no de-duplication hazard if they are
    # concatenated. `sort_key` is applied to both, and it deliberately excludes
    # `variant`, so the arm files list the same (dataset, model, cell, position)
    # in the same order. The 270 semantic_multi rows therefore line up
    # positionally between the two files, which is what makes the paired
    # comparison a zip rather than a join. Verified in `check_arm_alignment`.
    rows.sort(key=sort_key)
    for arm, filename in ARM_FILES.items():
        arm_rows = [r for r in rows if r["variant"] == arm]
        emit(arm_rows, OUT, filename, bymodel=f"by_model_{arm}")

    check_arm_alignment(rows)


if __name__ == "__main__":
    main()
