# 01 — First value vs last value (FVQ vs CVQ)

**Question.** Given a key whose value is updated N times inside the context, is
the model better at recalling the *first* value it was given (FVQ) or the
*current* one, i.e. the last update (CVQ)?

| | |
|---|---|
| file | `fvq_cvq.csv` — 2,012 rows |
| per-model slices | `by_model/` — 14 files (generated; do not edit) |
| models | 8 open-weight + 6 proprietary (**7 models excluded** — see below) |
| datasets | `arbitrary_single` (1,140 rows), `semantic_multi` (872 rows) |
| tiers | 1,968 `raw`, 44 `derived` |
| output-validity data | 1,140 rows carry `n_offstream` / `accuracy_onstream` / `error_types_json` |

---

## 7 models excluded on output-validity grounds

Models whose **off-stream rate reached 20%** — where the output is not any value
in the stream at all (key name, filler, refusal), an instruction-following
failure rather than a retrieval one — are **dropped from this file entirely**:

| dropped | off-stream |
|---|---|
| gemma-3-270m-it | 72.6% |
| pythia-410m | 63.5% |
| Qwen2.5-0.5B-Instruct | 51.7% |
| mamba-1.4b-hf | 25.8% (also only 3 cells) |
| stablelm-2-1_6b-chat | 24.6% |
| TinyLlama-1.1B-Chat-v1.0 | 21.3% |
| Qwen3.5-0.8B | 20.3% |

Above ~20% a cell measures whether the model can follow the output format, not
whether it can retrieve a value. There is a clean break in the data: #7 is at
20.3%, #8 at 16.2%. The 8 surviving open-weight models run 3.4%–16.2%.

Evidence and full ranking: [`VALIDITY.md`](VALIDITY.md). The exclusion list is
`EXCLUDED_MODELS` in `../build/common.py`; raw sources are untouched, so removing
a name and rebuilding restores it.

**Even in the retained models, use `accuracy_onstream` rather than `accuracy`
when precision matters** — the remaining off-stream rates still depress absolute
accuracies by a few points.

## Headline

Across **990 paired (K, N) cells** on the 14 retained models, FVQ beats CVQ in
**679 of them (69%)**. Averaged within each model×dataset series, the mean gap is
**+0.19**, and it is positive for **19 of 20** series.

(Before the exclusion: 1,357 cells, 72%, +0.23, 25/27 series. Dropping the seven
weakest models moves the headline by ~3pp and does not change its direction —
the effect was never carried by them.)

The one negative series is the finding, not an artefact: **Qwen2.5-3B inverts at
high key counts, and Qwen2.5-3B-Instruct sits near zero for the same reason.** Qwen2.5-3B-Instruct averages
+0.02 overall but ranges from **−0.55 to +0.75** across its 79 cells — it is
first-value-dominant at low K and current-value-dominant at high K. Any summary
that averages over K erases this. Report per cell, or state the K range.

## Coverage and per-model gap

Mean / min / max of (FVQ − CVQ) over all paired cells for that model.

| model | dataset | cells | mean gap | min | max | off-stream |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | arbitrary_single | 79 | +0.46 | −0.01 | +0.92 | 7.8% |
| Qwen2.5-3B *(base)* | arbitrary_single | 73 | **−0.12** | −0.54 | +0.73 | 12.5% |
| Qwen2.5-3B-Instruct | arbitrary_single | 79 | **+0.02** | −0.55 | +0.75 | 4.8% |
| Qwen3.5-2B | arbitrary_single | 79 | +0.07 | −0.38 | +0.67 | 16.2% |
| Qwen3.5-4B | arbitrary_single | 50 | +0.19 | −0.22 | +0.38 | 5.6% |
| Qwen3.5-9B | arbitrary_single | 79 | +0.21 | +0.05 | +0.40 | 7.1% |
| gemma-3-1b-it | arbitrary_single | 55 | +0.54 | +0.30 | +0.80 | 14.6% |
| gemma-3-4b-it | arbitrary_single | 76 | +0.46 | +0.10 | +0.87 | 3.4% |
| claude-4.5-haiku | semantic_multi | 69 | +0.22 | −0.01 | +0.56 | n/a |
| claude-4.5-sonnet | semantic_multi | 69 | +0.08 | −0.06 | +0.35 | n/a |
| gpt-4.1 | semantic_multi | 69 | +0.12 | −0.03 | +0.36 | n/a |
| gpt-4.1-mini | semantic_multi | 69 | +0.08 | −0.02 | +0.34 | n/a |
| gemini-2.5-flash | semantic_multi | 69 | +0.13 | +0.00 | +0.71 | n/a |
| gemini-2.5-pro | semantic_multi | 69 | +0.03 | +0.00 | +0.65 | n/a |

The proprietary sweeps never recorded an output taxonomy and their per-trial
JSONs are not local, so their off-stream rate is unknown (`n/a`). They are
**assumed clean without screening** — off-stream rate tracked capability
monotonically in the open-weight corpus and all six sit well above that range.
This is a stated assumption, not a check; see `../PROVENANCE.md` §Excluded
models, consequence 3.

Five retained open-weight models also have a single `semantic_multi` cell
(K=5, N=30), back-extracted from the paper — see "Derived rows" below.

## Sweep grid

**Full per-model configuration detail — exact cells, engine settings, trial
counts, and two confounds that affect cross-model claims — is in
[`CONFIGURATIONS.md`](CONFIGURATIONS.md).** Two things from it to know before
reading the table above:

- **Prompt format is not constant.** Four models ran `chat_template`, four ran
  `completion_few_shot` (base checkpoints). Given theme 3 shows formatting moves
  the gap by up to 0.94, cross-format comparisons are confounded.
- **Cells are not shared.** Only 42 of 81 cells were run by all eight models.
  Restricted to those, **every model's mean gap is positive**, including
  Qwen2.5-3B's −0.12 → +0.06. The negatives come from high-K cells only some
  models reached.


Open-weight `arbitrary_single` (from the stage-1 vLLM sweeps):
K ∈ {2, 3, 5, 7, 10, 15, 20, 25, 30} × N ∈ {5, 7, 10, 15, 20, 30, 50, 75, 100},
100 trials per cell, greedy decoding. Cell counts below 79 mean the sweep
stopped early for that model (garbage output or OOM), not that the cell failed.

Proprietary `semantic_multi`:
K ∈ {1, 2, 5, 10, 15, 20, 25, 30, 40, 45} × N ∈ {1, 5, 10, 15, 20, 30, 50},
up to 200 trials per cell.

## Derived rows (44)

The open-weight `semantic_multi` sweep is **not local** — it ran on a GPU box
(`v3/results_vllm/semantic_multi/`, see `AAAI_PREP_PLAN.md` §1.3). Two committed
paper tables preserve part of it:

- `paper/figures/tab2_openweight.tex` → 5 retained models at K=5, N=30, **with**
  FVQ and CVQ separately (10 rows).
- `paper/figures/tab_full_sem.tex` → retained models at 4 representative cells,
  but **only the gap survived** the table. Those 34 rows carry
  `condition=FVQ-CVQ`, an empty `accuracy`, and a populated `gap`.

Filter them out with `df[df.accuracy.notna()]` when you want accuracies, or
`df[df.provenance_tier == "raw"]` when you want only re-derivable numbers.

## Verification

`paper/figures/tab_full_arb.tex` independently restates the `arbitrary_single`
gaps we compute from raw JSON. The builder checks them: **47 cells checked, 0
mismatched.** This is what licenses trusting the `derived` rows elsewhere.

The QA cross-checks deliberately run over the **full, pre-exclusion** data, so
dropping models never weakens parser validation.

## Caveats

- **The state-space architectural control is now gone.** `mamba-1.4b-hf` was the
  only non-attention model and it is excluded (26% off-stream, 3 cells). If that
  control matters for the paper, it needs a fresh run — ideally with generations
  persisted so its off-stream rate can be diagnosed rather than guessed.
- Gaps here are point estimates. `ci_lower`/`ci_upper` are per-condition, not
  per-gap — a gap's own CI is not in the source data and was not invented.
- `Qwen2.5-3B` (no suffix) is the **base**, non-instruct model, and is a
  different row from `Qwen2.5-3B-Instruct`.

## Sources

See `../PROVENANCE.md` for the full file-by-file list, or
`../build/build_01_fvq_cvq.py` — its docstring names every input.
