"""Shared schema, model registry and writers for the consolidation build.

Every consolidated CSV uses ONE tidy-long schema (see COLUMNS): one row is one
measurement. Columns a given source cannot supply are left empty rather than
invented.

Naming: the paper's FVQ / CVQ terminology is canonical here. Raw JSONs and
comparison files still use the older RI / PI keys; the mapping is
    RI -> FVQ  (first-value query)
    PI -> CVQ  (current-value query, i.e. the last value)
and is applied at read time by `cond_from_ri_pi`.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_ROOT = Path(__file__).resolve().parent.parent

# ── Tidy-long schema ────────────────────────────────────────────────────────
# accuracy and gap are deliberately separate columns: some sources (the
# back-extracted .tex tables) preserve only a gap, and mixing a difference into
# an accuracy column would silently corrupt any aggregation.
COLUMNS = [
    "theme",             # 01_fvq_cvq | 02_intermediate_ivq | 03_prompt_format | 04_lora
    "model",             # canonical model id (see MODELS)
    "model_family",
    "is_proprietary",
    "variant",           # base | lora | lora_arith_control | block | ...
    "dataset",           # arbitrary_single | semantic_multi | gsm8k
    "prompt_format",     # chat_template | plain | block | flat_nolabel | ...
    "num_keys",
    "num_updates",
    "condition",         # FVQ | CVQ | IVQ | FVQ-CVQ (gap rows)  -- WHICH value
    "query_type",        # semantic | ordinal                    -- HOW it was asked
    "position",          # first | last | 1..N  (update index queried)
    "accuracy",
    "gap",               # FVQ - CVQ, only on condition=FVQ-CVQ rows
    "ci_lower",
    "ci_upper",
    "wilson_hw",
    "n_trials",
    "n_correct",
    # Output-validity columns. Only the stage-1 sweeps classify what the model
    # actually emitted; everything else leaves these empty. `n_offstream` counts
    # trials whose output was NOT any value in the stream (garbage / empty /
    # oom) -- i.e. failures that are not retrieval errors at all.
    "n_offstream",
    "accuracy_onstream",   # correct / (n_trials - n_offstream)
    "error_types_json",
    "regime",            # AB | B | C (from stage-1 sweeps) where recorded
    "split",             # train | held_moderate | held_hard | held_out | ood
    "provenance_tier",   # raw | derived
    "source_file",       # repo-relative path the number came from
    "notes",
]

RAW = "raw"
DERIVED = "derived"


# ── Excluded models ─────────────────────────────────────────────────────────
# Dropped 2026-07-22 on output-validity grounds: every model whose off-stream
# rate (garbage + empty + oom -- outputs that are not any value in the stream)
# reached 20% of trials. Above that threshold a cell measures whether the model
# can follow the output format at all, not whether it can retrieve a value, so
# its accuracies are not comparable with the rest of the corpus.
#
# Evidence and the full ranking: 01_fvq_cvq/VALIDITY.md, reproducible via
# build/check_output_validity.py. Rationale recorded in PROVENANCE.md.
#
# These are filtered out of every emitted CSV. The raw source files are
# untouched and the QA cross-checks still run over the full data, so removing a
# name from this set and rebuilding restores it.
EXCLUDED_MODELS = {
    "gemma-3-270m-it":           "72.6% off-stream",
    "pythia-410m":               "63.5% off-stream",
    "Qwen2.5-0.5B-Instruct":     "51.7% off-stream",
    "mamba-1.4b-hf":             "25.8% off-stream (also only 3 cells)",
    "stablelm-2-1_6b-chat":      "24.6% off-stream",
    "TinyLlama-1.1B-Chat-v1.0":  "21.3% off-stream",
    "Qwen3.5-0.8B":              "20.3% off-stream",
}


# ── Model registry ──────────────────────────────────────────────────────────
# canonical_id -> (family, is_proprietary)
MODELS = {
    "Qwen2.5-0.5B-Instruct":   ("Qwen2.5", False),
    "Qwen2.5-1.5B-Instruct":   ("Qwen2.5", False),
    "Qwen2.5-3B":              ("Qwen2.5", False),
    "Qwen2.5-3B-Instruct":     ("Qwen2.5", False),
    "Qwen3.5-0.8B":            ("Qwen3.5", False),
    "Qwen3.5-2B":              ("Qwen3.5", False),
    "Qwen3.5-4B":              ("Qwen3.5", False),
    "Qwen3.5-9B":              ("Qwen3.5", False),
    "gemma-3-270m-it":         ("Gemma-3", False),
    "gemma-3-1b-it":           ("Gemma-3", False),
    "gemma-3-4b-it":           ("Gemma-3", False),
    "TinyLlama-1.1B-Chat-v1.0": ("TinyLlama", False),
    "stablelm-2-1_6b-chat":    ("StableLM-2", False),
    "pythia-410m":             ("Pythia", False),
    "mamba-1.4b-hf":           ("Mamba", False),
    "SmolLM3-3B":              ("SmolLM3", False),
    # Theme 06. GPT-2-small *architecture* trained on the synthetic KV task; the
    # `variant` column says which run (scratch_cosine / scratch_plateau /
    # pretrained). Never web-pretrained in the scratch variants, so "model" here
    # names an architecture, not a released checkpoint.
    "gpt2-small":              ("GPT-2", False),
    "gpt-4.1":                 ("GPT-4.1", True),
    "gpt-4.1-mini":            ("GPT-4.1", True),
    "claude-4.5-haiku":        ("Claude-4.5", True),
    "claude-4.5-sonnet":       ("Claude-4.5", True),
    "claude-4.5-opus":         ("Claude-4.5", True),
    "gemini-2.5-flash":        ("Gemini-2.5", True),
    "gemini-2.5-pro":          ("Gemini-2.5", True),
}

# Every spelling seen across raw JSONs, CSVs and .tex labels -> canonical id.
_ALIASES = {
    "qwen/qwen2.5-0.5b-instruct": "Qwen2.5-0.5B-Instruct",
    "qwen/qwen2.5-1.5b-instruct": "Qwen2.5-1.5B-Instruct",
    "qwen/qwen2.5-3b": "Qwen2.5-3B",
    "qwen2.5-3b (base)": "Qwen2.5-3B",
    "qwen/qwen2.5-3b-instruct": "Qwen2.5-3B-Instruct",
    "google/gemma-3-270m-it": "gemma-3-270m-it",
    "google/gemma-3-1b-it": "gemma-3-1b-it",
    "google/gemma-3-4b-it": "gemma-3-4b-it",
    "gemma-3-270m-it": "gemma-3-270m-it",
    "gemma-3-1b-it": "gemma-3-1b-it",
    "gemma-3-4b-it": "gemma-3-4b-it",
    "tinyllama-1.1b": "TinyLlama-1.1B-Chat-v1.0",
    "tinyllama/tinyllama-1.1b-chat-v1.0": "TinyLlama-1.1B-Chat-v1.0",
    "stablelm-2-1.6b": "stablelm-2-1_6b-chat",
    "stabilityai/stablelm-2-1_6b-chat": "stablelm-2-1_6b-chat",
    "pythia-410m": "pythia-410m",
    "eleutherai/pythia-410m": "pythia-410m",
    "state-spaces/mamba-1.4b-hf": "mamba-1.4b-hf",
    "claude-sonnet": "claude-4.5-sonnet",          # ucurve CSV spelling
    "claude-4.5-sonnet": "claude-4.5-sonnet",
    "claude-4.5-haiku": "claude-4.5-haiku",
    "claude-haiku": "claude-4.5-haiku",
    # Dated API snapshots, as recorded by the CoT sweep (theme 07). The undated
    # aliases above come from the earlier ucurve runs, whose CSVs did not
    # preserve a snapshot date -- so these three ids collapse into the same
    # canonical model as rows whose snapshot is unrecorded. Theme 07 keeps the
    # full dated id in its `notes` column.
    "claude-haiku-4-5-20251001": "claude-4.5-haiku",
    "claude-sonnet-4-5-20250929": "claude-4.5-sonnet",
    "claude-opus-4-5-20251101": "claude-4.5-opus",
    "gpt-4.1": "gpt-4.1",
    "gpt-4.1-mini": "gpt-4.1-mini",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "smollm3-3b": "SmolLM3-3B",
}


def canon_model(name: str) -> str:
    """Map any observed model spelling to its canonical id."""
    if name in MODELS:
        return name
    key = name.strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    # bare HF repo path not otherwise listed -> take the leaf
    leaf = name.strip().split("/")[-1]
    if leaf in MODELS:
        return leaf
    if leaf.lower() in _ALIASES:
        return _ALIASES[leaf.lower()]
    raise KeyError(f"unknown model spelling: {name!r} — add it to _ALIASES")


def model_meta(name: str) -> tuple[str, str, bool]:
    c = canon_model(name)
    fam, prop = MODELS[c]
    return c, fam, prop


def cond_from_ri_pi(key: str) -> str:
    """RI -> FVQ, PI -> CVQ. Accepts already-translated names unchanged."""
    return {"RI": "FVQ", "PI": "CVQ", "FVQ": "FVQ", "CVQ": "CVQ"}[key]


# `condition` says WHICH value is targeted; `query_type` says HOW it was asked.
# They are independent, and the difference matters enormously: the U-curve sweep
# asks for the final update both ways and gets 0.005 (ordinal) vs 0.69
# (semantic) from the same model on the same cell.
SEMANTIC = "semantic"   # "What was the first/last value of X?"
ORDINAL = "ordinal"     # "What was the 7th value of X?" / "...X in Update 7?"


# Outputs that are not any value in the stream: the model failed to produce a
# candidate at all (wrong-token / filler / refusal / truncation), which is a
# different failure from retrieving the wrong value.
OFFSTREAM_TYPES = ("garbage", "empty", "oom")


def offstream(stats: dict) -> tuple:
    """(n_offstream, accuracy_onstream, error_types_json) for a stage-1 cell.

    Returns empties when the source did not classify outputs.
    """
    import json as _json
    et = stats.get("error_types")
    if not et:
        return "", "", ""
    n = stats.get("n") or sum(et.values())
    off = sum(et.get(k, 0) for k in OFFSTREAM_TYPES)
    on = n - off
    acc_on = round(et.get("correct", 0) / on, 4) if on else ""
    return off, acc_on, _json.dumps(et, sort_keys=True)


# ── Row construction ────────────────────────────────────────────────────────

def row(**kw) -> dict:
    """Build a schema-complete row; unspecified columns are empty strings."""
    unknown = set(kw) - set(COLUMNS)
    if unknown:
        raise KeyError(f"not in schema: {sorted(unknown)}")
    r = {c: "" for c in COLUMNS}
    r.update({k: ("" if v is None else v) for k, v in kw.items()})
    return r


def with_model(name: str, **kw) -> dict:
    m, fam, prop = model_meta(name)
    return row(model=m, model_family=fam, is_proprietary=prop, **kw)


def pos_key(position) -> int:
    """Sort order for the `position` column: first < 1..N < last < blank."""
    if position == "first":
        return -1
    if position == "last":
        return 10_000
    if position == "" or position is None:
        return 10_001
    return int(position)


def rel(path) -> str:
    """Repo-relative path string, for the source_file column."""
    return str(Path(path).resolve().relative_to(ROOT))


# ── Writers ─────────────────────────────────────────────────────────────────

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def write_by_model(rows: list[dict], theme_dir: Path,
                   subdir: str = "by_model") -> int:
    """Emit one CSV per model into <theme_dir>/<subdir>/, sliced from `rows`.

    Regenerated wholesale on every build — the combined CSV is canonical, these
    are a convenience view and must never be hand-edited. `subdir` lets a theme
    with more than one combined CSV give each its own split without the two
    wiping each other.
    """
    bm = theme_dir / subdir
    if bm.exists():
        for old in bm.glob("*.csv"):
            old.unlink()
    bm.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["model"], []).append(r)
    for model, grp in groups.items():
        write_csv(grp, bm / f"{_SAFE.sub('_', model)}.csv")
    return len(groups)


def emit(rows: list[dict], theme_dir: Path, filename: str,
         bymodel: str = "by_model") -> None:
    """Write the combined CSV + the per-model split, and report.

    Applies EXCLUDED_MODELS centrally so no theme builder can bypass the
    output-validity exclusion. `bymodel` names the per-model subdir; pass a
    distinct name when a theme emits more than one combined CSV.
    """
    theme_dir.mkdir(parents=True, exist_ok=True)
    kept = [r for r in rows if r["model"] not in EXCLUDED_MODELS]
    n_dropped = len(rows) - len(kept)
    if n_dropped:
        names = sorted({r["model"] for r in rows if r["model"] in EXCLUDED_MODELS})
        print(f"     excluded {n_dropped} rows from {len(names)} model(s) "
              f"on output-validity grounds: {', '.join(names)}")
    rows = kept
    write_csv(rows, theme_dir / filename)
    n_models = write_by_model(rows, theme_dir, bymodel)
    tiers = {}
    for r in rows:
        tiers[r["provenance_tier"]] = tiers.get(r["provenance_tier"], 0) + 1
    tier_txt = ", ".join(f"{k}={v}" for k, v in sorted(tiers.items()))
    print(f"  -> {theme_dir.name}/{filename}: {len(rows)} rows, "
          f"{n_models} models ({tier_txt})")


# ── LaTeX table back-extraction ─────────────────────────────────────────────

def strip_tex(cell: str) -> str:
    """Remove colour/formatting macros from a table cell, keep the number."""
    s = cell.strip()
    s = re.sub(r"\\cellcolor\{[^}]*\}", "", s)
    s = re.sub(r"\\(?:textbf|emph|scriptsize|small)\b", "", s)
    s = re.sub(r"\\begin\{tabular\}(\[[^\]]*\])?\{[^}]*\}", "", s)
    s = s.replace(r"\end{tabular}", "")
    s = s.replace(r"\\", " ").replace("{@{}c@{}}", "")
    s = s.replace("$-$", "-").replace(r"\,", " ").replace(r"\%", "%")
    s = s.replace("{", " ").replace("}", " ").replace("$", "")
    return re.sub(r"\s+", " ", s).strip()


def tex_body_rows(path: Path) -> list[list[str]]:
    """Yield the data rows of a generated tabular as lists of cleaned cells.

    Skips comments, rules, \\multicolumn section headers and the tabular
    envelope, so only model/value rows come back.
    """
    out = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if (not line or line.startswith("%")
                or line.startswith(r"\begin{tabular}")
                or line.startswith(r"\end{tabular}")
                or line.startswith((r"\toprule", r"\midrule", r"\bottomrule",
                                    r"\cmidrule"))):
            continue
        if line.startswith(r"\multicolumn"):
            continue
        # Header rows in these generated tables are always \textbf-wrapped.
        if r"\textbf" in line.split("&")[0]:
            continue
        line = line.rstrip("\\").rstrip()
        if line.endswith(r"\\"):
            line = line[:-2]
        cells = [strip_tex(c) for c in line.split("&")]
        if len(cells) < 2 or not cells[0]:
            continue
        out.append(cells)
    return out


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def nums(text: str) -> list[float]:
    """All numbers in a cleaned cell, in order (percent signs stripped)."""
    return [float(m) for m in _NUM.findall(text.replace("%", ""))]


def pct(x: float) -> float:
    """Percent -> proportion, rounded to 4dp."""
    return round(x / 100.0, 4)
