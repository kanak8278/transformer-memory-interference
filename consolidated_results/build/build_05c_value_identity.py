"""Theme 5c — closed-pool value-identity probe: is the answer VALUE decodable?

`probing.csv` in this theme probes *correctness* — a binary "will the model get
this right" readout, which degenerates wherever accuracy sits at 0% or 100%.
This experiment probes *value identity* instead: at the answer position, which
of the 50 pool words is the ground-truth answer? 50-way classification, chance
2%, and the label comes from the trial definition rather than the model's
output, so it stays defined at 0% and 100% accuracy alike.

The headline is the **wrong-answer subset**. Decodable there means the value is
tracked and not surfaced — a much stronger version of theme 05's "tracked but
suppressed", which only ever showed the *current* value surviving to late layers.

Why the pool is closed and exactly K*N. `generate_values_for_trial` draws
`rng.sample(pool, K*N)` without replacement, so setting `len(pool) == K*N` makes
every stream a permutation of the whole pool: all 50 values appear exactly once
in every trial. That removes the presence confound. If only some pool values
appeared per trial a probe could beat chance by detecting which values are in
context at all, which says nothing about retrieval. It also forces the small
cells -- K*N <= 50 -- which is why only (5,10) and (10,5) exist.

Read the controls before the result. Four label sets are fitted on the SAME
activations, so the controls are free:

    expected  = cv[k-1]   the result
    shuffled              labels permuted across trials; must fall to chance
    cv_first  = cv[0]     primacy control
    cv_last   = cv[-1]    recency control

Two of these are DEGENERATE by construction and must not be read as controls:
`cv_last` at a condition where k == N is the same label as `expected`, and
`cv_first` at k == 1 likewise. The builder flags those rows in `notes`.

Where the controls actually land: `shuffled` is at chance everywhere (0.025 to
0.057), so there is no fold leakage. But `cv_first` BEATS `expected` at every
interior slot -- gemma (5,10) k2 reads cv_first 0.961 against expected 0.638.
So what the residual stream robustly carries at the answer position is the FIRST
value, not the queried one. The design anticipated a recency confound and built
`cv_last` for it; the confound that actually appeared is primacy. State that
alongside any "tracked but not surfaced" claim.

Design doc: `lora_intervention/experiments/linear_probing/PROBE50_DESIGN.md`.

Sources (RAW)
  lora_intervention/experiments/linear_probing/results_probe50/<model>_<cell>/
      manifest.json          behavioural accuracy, the 6-bucket output audit,
                             the value pool, trial counts
      probe_fits_ref.json    the `correct` and `all` subsets, `expected` only
      probe_fits_wrong.json  the `wrong` subset, all four label sets
      probe_fits_wrong_topup.json / _missing.json
                             later shards that extend or SUPERSEDE the above

Shard resolution, and why it cannot be skipped. The wrong-subset fits are spread
over up to three files per cell, and four (cell, condition) pairs are claimed
twice with different trial counts -- Qwen (10,5) CVQ appears at n=616 and at
n=1231. The larger is the completed run; the smaller is an under-powered first
pass that was topped up. This builder keeps the largest n per
(cell, condition, subset) and reports how many rows it dropped. Averaging the
two, or letting file order decide, would silently mix sample sizes.

Not used as a source: `.../results/probe_layer_values_raw.csv`. It is a
different experiment (correctness/retrieval AUC on the v3 grid), it carries no
`model` column, and `results/extract_and_plot.py:41` hardcodes Qwen -- so it is
a Qwen-only view of a two-model grid. The 17 GB of `reps_*.npy` activations are
also left in place; they are inputs to the fit, not results.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from common import ORDINAL, RAW, ROOT, SEMANTIC, model_meta, write_csv

THEME = "05_mechanistic"
SUBDIR = "value_identity_probe"
OUT = Path(__file__).resolve().parent.parent / THEME / SUBDIR

SRC_ROOT = (ROOT / "lora_intervention" / "experiments" / "linear_probing"
            / "results_probe50")

# Directories that are pilots or smoke tests, not results.
SKIP_DIRS = ("_pilot", "_smoke")

LAYER_COLUMNS = [
    "model", "model_family", "variant", "num_keys", "num_updates",
    "condition",          # FVQ | IVQ | CVQ -- WHICH value is targeted
    "query_type",         # semantic ("first"/"last") | ordinal ("2nd")
    "condition_label",    # the raw run label: CVQ | k2 .. k10 | FVQ | k1
    "slot",               # the queried slot k, 1-indexed
    "subset",             # wrong | correct | all
    "label_set",          # expected | shuffled | cv_first | cv_last
    "C", "layer", "rel_depth",
    "top1", "top1_std", "mean_rank", "mrr",
    "n_folds", "hit_max_iter",
    "n_probe",            # trials the probe was fitted on
    "chance",
    "provenance_tier", "source_file", "notes",
]

SUMMARY_COLUMNS = [
    "model", "model_family", "variant", "num_keys", "num_updates",
    "condition", "query_type", "condition_label", "slot", "subset",
    "label_set", "C", "n_probe", "chance",
    "best_layer", "best_top1", "best_rel_depth", "mean_top1",
    "behavioral_acc", "estimable",
    "provenance_tier", "source_file", "notes",
]

BEHAVIORAL_COLUMNS = [
    "model", "model_family", "variant", "num_keys", "num_updates",
    "condition", "query_type", "condition_label", "slot", "query_word",
    "n_trials", "accuracy_single_token", "accuracy_generate",
    "scoring_agreement", "n_generate", "n_wrong", "wrong_frac",
    "topped_up_from",
    # The 6-bucket output audit. Every pool value is in every stream, so
    # `off_pool` is a genuine instruction violation: the token was nowhere in
    # context. `wrong_slot_same_key` is the retrieval error proper.
    "exact", "wrong_slot_same_key", "other_key_value", "off_pool",
    "multi_word", "empty",
    "provenance_tier", "source_file",
]

POOL_COLUMNS = ["model", "model_family", "num_keys", "num_updates",
                "pool_seed", "pool_size", "idx", "value",
                "provenance_tier", "source_file"]


def parse_cell(dirname: str) -> tuple[str, int, int]:
    """'Qwen2.5-3B-Instruct_5k_10u' -> ('Qwen2.5-3B-Instruct', 5, 10)."""
    model, k_part, n_part = dirname.rsplit("_", 2)
    return model, int(k_part.rstrip("k")), int(n_part.rstrip("u"))


def classify(slot: int, num_updates: int) -> str:
    """WHICH value the query targets."""
    if slot == 1:
        return "FVQ"
    return "CVQ" if slot == num_updates else "IVQ"


def query_type_of(label: str) -> str:
    """`CVQ`/`FVQ` use the words "last"/"first"; `k<n>` uses an ordinal.

    The pair matters: FVQ and k1 target the SAME slot and differ only in
    wording, as do CVQ and k<N>. That is the ordinal-vs-semantic contrast this
    repo tracks everywhere else, available here for free.
    """
    return SEMANTIC if label in ("FVQ", "CVQ") else ORDINAL


def degenerate_label(label_set: str, slot: int, num_updates: int) -> str:
    """Note text when a control label is identical to `expected`, else ''."""
    if label_set == "cv_last" and slot == num_updates:
        return "DEGENERATE: cv_last == expected at k=N, not a control here"
    if label_set == "cv_first" and slot == 1:
        return "DEGENERATE: cv_first == expected at k=1, not a control here"
    return ""


def resolve_shards(cell_dir: Path) -> tuple[dict, int, dict]:
    """Pick the winning shard per (condition, subset); count what was dropped.

    Returns ({(condition, subset): (record, cond_record, src)},
             n_superseded_layer_rows,
             {(condition, subset): [(src, n), ...]}  # the full claim list)
    """
    winners: dict = {}
    claims: dict = {}
    for path in sorted(cell_dir.glob("probe_fits_*.json")):
        fits = json.load(open(path))
        for label, cond in fits["conditions"].items():
            for subset, rec in cond["subsets"].items():
                key = (label, subset)
                claims.setdefault(key, []).append((path.name, rec["n"]))
                if key not in winners or rec["n"] > winners[key][0]["n"]:
                    winners[key] = (rec, cond, path)

    superseded = 0
    for key, rec_list in claims.items():
        if len(rec_list) < 2:
            continue
        keep_n = winners[key][0]["n"]
        for name, n in rec_list:
            if n != keep_n:
                superseded += 1
    return winners, superseded, claims


def rows_for_cell(cell_dir: Path) -> tuple[list, list, list, list, list]:
    man = json.load(open(cell_dir / "manifest.json"))
    dirname = cell_dir.name
    raw_model, nk, nu = parse_cell(dirname)
    model, family, _prop = model_meta(raw_model)
    n_layers = man["n_layers"]
    pool = man["pool"]
    chance = round(1.0 / len(pool), 6)
    man_src = str((cell_dir / "manifest.json").resolve().relative_to(ROOT))

    winners, superseded, claims = resolve_shards(cell_dir)

    layer_rows, summary_rows, notes_log = [], [], []

    for (label, subset), (rec, cond, path) in sorted(winners.items()):
        src = str(path.resolve().relative_to(ROOT))
        slot = cond["k"]
        base = dict(
            model=model, model_family=family,
            # These are all unmodified base models. LoRA is deferred: it scores
            # 1.000 at CVQ and shallow k in every cell the closed pool permits,
            # so the wrong subset is empty at any T. See PROBE50_DESIGN.md.
            variant="base",
            num_keys=nk, num_updates=nu,
            condition=classify(slot, nu), query_type=query_type_of(label),
            condition_label=label, slot=slot, subset=subset,
        )
        n_probe = rec["n"]
        shard_note = ""
        if len(claims[(label, subset)]) > 1:
            others = ", ".join(f"{nm}:n={n}" for nm, n in claims[(label, subset)]
                               if n != n_probe)
            shard_note = f"superseded shard(s) {others}"

        if not rec.get("label_sets"):
            # Below min_n -- not estimable. Emit ONE summary row so the gap is
            # visible as a row rather than as an absence, and no layer rows.
            summary_rows.append({**base, "label_set": "", "C": "",
                                 "n_probe": n_probe, "chance": chance,
                                 "best_layer": "", "best_top1": "",
                                 "best_rel_depth": "", "mean_top1": "",
                                 "behavioral_acc": cond.get("accuracy_single_token"),
                                 "estimable": False,
                                 "provenance_tier": RAW, "source_file": src,
                                 "notes": "; ".join(
                                     x for x in [rec.get("note", ""), shard_note] if x)})
            notes_log.append(f"{dirname} {label}/{subset}: {rec.get('note')}")
            continue

        for label_set, by_c in rec["label_sets"].items():
            for c_val, fit in by_c.items():
                degen = degenerate_label(label_set, slot, nu)
                note = "; ".join(x for x in [degen, shard_note] if x)
                for layer, m in fit["layers"].items():
                    # A per-layer fit can be degraded rather than absent: 288
                    # records across the corpus carry top1=None plus a note
                    # ("min class support 1 < 2 folds"), all in the `correct` /
                    # `all` subsets where one class had too few examples to
                    # stratify. Keep the row so the layer curve has no silent
                    # hole, but carry the note instead of inventing a number.
                    layer_note = "; ".join(
                        x for x in [note, m.get("note", "")] if x)
                    layer_rows.append({
                        **base, "label_set": label_set, "C": float(c_val),
                        "layer": int(layer),
                        "rel_depth": round(int(layer) / n_layers, 6),
                        "top1": m["top1"], "top1_std": m.get("top1_std"),
                        "mean_rank": m["mean_rank"], "mrr": m["mrr"],
                        "n_folds": m.get("n_splits"),
                        "hit_max_iter": m.get("hit_max_iter"),
                        "n_probe": n_probe, "chance": chance,
                        "provenance_tier": RAW, "source_file": src,
                        "notes": layer_note,
                    })
                # `best_layer` is None when every layer in the fit was
                # degraded, so there was no layer to pick.
                best_layer = fit.get("best_layer")
                n_degraded = sum(1 for m in fit["layers"].values()
                                 if m.get("top1") is None)
                summary_note = "; ".join(x for x in [
                    note,
                    (f"{n_degraded}/{len(fit['layers'])} layers not estimable "
                     "(class support < folds)") if n_degraded else "",
                ] if x)
                summary_rows.append({
                    **base, "label_set": label_set, "C": float(c_val),
                    "n_probe": n_probe, "chance": chance,
                    "best_layer": best_layer,
                    "best_top1": fit.get("best_top1"),
                    "best_rel_depth": (round(best_layer / n_layers, 6)
                                       if best_layer is not None else ""),
                    "mean_top1": fit.get("mean_top1"),
                    "behavioral_acc": cond.get("accuracy_single_token"),
                    "estimable": best_layer is not None,
                    "provenance_tier": RAW, "source_file": src,
                    "notes": summary_note,
                })

    behavioral_rows = []
    for label, cd in man["conditions"].items():
        audit = (cd.get("audit_single_token") or {}).get("counts", {})
        slot = cd["k"]
        behavioral_rows.append({
            "model": model, "model_family": family, "variant": "base",
            "num_keys": nk, "num_updates": nu,
            "condition": classify(slot, nu), "query_type": query_type_of(label),
            "condition_label": label, "slot": slot,
            "query_word": cd.get("query_word"),
            "n_trials": cd.get("n"),
            "accuracy_single_token": cd.get("accuracy_single_token"),
            "accuracy_generate": cd.get("accuracy_generate"),
            "scoring_agreement": cd.get("scoring_agreement"),
            "n_generate": cd.get("n_generate"),
            "n_wrong": cd.get("n_wrong"),
            "wrong_frac": cd.get("wrong_frac_single_token"),
            "topped_up_from": cd.get("topped_up_from"),
            **{b: audit.get(b, "") for b in
               ("exact", "wrong_slot_same_key", "other_key_value", "off_pool",
                "multi_word", "empty")},
            "provenance_tier": RAW, "source_file": man_src,
        })

    pool_rows = [{
        "model": model, "model_family": family, "num_keys": nk,
        "num_updates": nu, "pool_seed": man["pool_seed"],
        "pool_size": len(pool), "idx": i, "value": v,
        "provenance_tier": RAW, "source_file": man_src,
    } for i, v in enumerate(pool)]

    return layer_rows, summary_rows, behavioral_rows, pool_rows, [
        (dirname, superseded, notes_log)]


def qa_shuffled_at_chance(layer_rows: list) -> None:
    """The control task must not beat chance. If it does, the fit leaks."""
    shuf = [r for r in layer_rows if r["label_set"] == "shuffled"]
    if not shuf:
        return
    chance = shuf[0]["chance"]
    worst = max(shuf, key=lambda r: r["top1"])
    over = [r for r in shuf if r["top1"] > 4 * chance]
    print(f"     QA shuffled control: {len(shuf)} layer-rows, max top-1 "
          f"{worst['top1']:.3f} (chance {chance:.3f}), "
          f"{len(over)} above 4x chance")


def qa_scoring_agreement(behavioral_rows: list) -> None:
    """single_token scoring must agree with greedy generate on the audited subset."""
    vals = [r["scoring_agreement"] for r in behavioral_rows
            if r["scoring_agreement"] is not None]
    if not vals:
        return
    print(f"     QA single_token vs generate agreement: {len(vals)} conditions, "
          f"min {min(vals):.3f}, {sum(1 for v in vals if v < 1.0)} below 1.000")


def main() -> None:
    print(f"[{THEME}/{SUBDIR}] closed-pool value-identity probe (base models)")
    layer_rows, summary_rows, behavioral_rows, pool_rows = [], [], [], []
    total_superseded = 0

    cell_dirs = sorted(d for d in SRC_ROOT.iterdir()
                       if d.is_dir() and not d.name.endswith(SKIP_DIRS)
                       and (d / "manifest.json").exists())
    for cell_dir in cell_dirs:
        lr, sr, br, pr, info = rows_for_cell(cell_dir)
        layer_rows += lr
        summary_rows += sr
        behavioral_rows += br
        pool_rows += pr
        dirname, superseded, notes_log = info[0]
        total_superseded += superseded
        print(f"     {dirname}: {len(lr)} layer-rows, {len(sr)} summary rows"
              + (f", {len(notes_log)} not estimable" if notes_log else ""))

    print(f"     shard resolution: {total_superseded} superseded "
          f"(condition, subset) fits dropped in favour of the larger n")
    qa_shuffled_at_chance(layer_rows)
    qa_scoring_agreement(behavioral_rows)

    def sort_key(r):
        return (r["model"], int(r["num_keys"]), int(r["num_updates"]),
                r.get("subset", ""), r["slot"], r.get("query_type", ""),
                r.get("label_set", ""), float(r.get("C") or 0),
                int(r.get("layer", 0)))

    layer_rows.sort(key=sort_key)
    summary_rows.sort(key=sort_key)
    behavioral_rows.sort(key=lambda r: (r["model"], r["num_keys"],
                                        r["num_updates"], r["slot"],
                                        r["query_type"]))

    OUT.mkdir(parents=True, exist_ok=True)
    for rows, cols, name in (
        (layer_rows, LAYER_COLUMNS, "probe_layers.csv"),
        (summary_rows, SUMMARY_COLUMNS, "summary.csv"),
        (behavioral_rows, BEHAVIORAL_COLUMNS, "behavioral.csv"),
        (pool_rows, POOL_COLUMNS, "pools.csv"),
    ):
        _write(rows, cols, OUT / name)
        print(f"  -> {SUBDIR}/{name}: {len(rows)} rows")


def _write(rows: list, columns: list, path: Path) -> None:
    """Writer for this subfolder's own schemas (not the tidy-long COLUMNS)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="raise")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c, ""))
                        for c in columns})


if __name__ == "__main__":
    main()
