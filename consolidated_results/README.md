# Consolidated Results

One place for every result behind the seven stories this project tells, each
with a traceable line back to the file every number came from.

Built 2026-07-22 on branch `aaai-prep`; theme 07 added 2026-09-09. Nothing here
is a new experiment — it is a re-presentation of results that already existed,
scattered across `v3/`, `lora_intervention/`, `experiments_cloud/` and
`paper/figures/`.

```
consolidated_results/
├── README.md                   ← you are here
├── DATASETS.md                 what the keys and values are; why the grids end where they do
├── PROMPTS.md                  every prompt template, verbatim, + which models saw it
├── PROVENANCE.md               every output file → every source file
├── 01_fvq_cvq/                 first value vs last value
├── 02_intermediate_ivq/        intermediate positions, standalone base models (proprietary)
├── 03_prompt_format/           block / boundary / plain / labelled formatting
├── 04_lora/                    LoRA: adapters, held-out evals, per-position base-vs-LoRA IVQ
├── 05_mechanistic/             HOW LoRA closes the gap: logit lens, probing, attention routing, causal ablation
│   └── entropy_lens/           per-layer uncertainty + calibration, one subfolder per arm (Qwen base/lora, from-scratch)
├── 06_from_scratch/            GPT-2-small trained from random init on a synthetic KV task
├── 07_cot_ivq/                 does chain-of-thought rescue the interior? one CSV per arm (nocot / cot_thinking)
└── build/                      re-runnable builders (see "Rebuilding")
```

Each behavioural theme (01–04) holds a canonical combined CSV, a `by_model/`
split, and a `README.md`. Theme 07 holds one CSV per arm with a `by_model_<arm>/`
split each, as theme 04 holds one per unit of measurement. Themes 05 (mechanistic) and 06 (from-scratch) hold one
CSV per unit of measurement and their own `README.md`.

---

## Which experiment is which

**Semantic-Multi is the main experiment** — meaningful multi-token keys and
values. **Arbitrary-Single is the supporting one**: single-token values, which
the mechanistic work needs (logit lens, probing, head ablation are only
well-defined with a single-token answer), run to show the phenomenon is not an
artefact of semantic content.

That bridge is measured, not assumed. Across 29 (model, K, N) cells where both
datasets have a gap: **r = 0.926, r² = 0.858, 93% sign agreement, mean |Δ| =
0.114**, with Arbitrary reading **+0.08 hotter**. Reproduce:
`build/check_arb_vs_sem.py`.

Consequence for reading the CSVs: `dataset` is never incidental. The abundant
raw data (570 open-weight model-cells) is the *supporting* experiment; the main
experiment's open-weight side is the thin, derived one (32 model-cells). See
`01_fvq_cvq/README.md`.

## The seven stories, in one paragraph each

**1 — First value vs last value.** Across 14 models and 990 paired (K, N)
cells, the first value is easier to retrieve than the current one in 69% of
cells; averaged per model×dataset series the gap is **+0.19**, positive for 19
of 20 series. The exception is not noise: Qwen2.5-3B *inverts* at high key counts
(the reversal regime), which is why the sign is reported per cell and never
averaged away. Seven further models were **excluded** on output-validity
grounds — see below.

**2 — Intermediate positions.** The endpoints hide the real failure. Asked for
an *interior* update, models collapse to the floor while scoring near-ceiling at
the first and last. The retrieval curve is U-shaped: models index the boundaries
of an update chain, not positions within it. This theme now covers the
**standalone proprietary** interior sweep (6 models, all 4 formats); the
open-weight interior data was all base-vs-LoRA and lives in theme 4
(`04_lora/lora_ivq.csv`), where the base-model floor is Qwen **0.03** / Gemma
**0.02**.

**3 — Prompt format.** Formatting moves the gap more than scale does. At
K=10, N=50 on open-weight models the FVQ−CVQ gap is **+0.78** under plain
running text and **−0.00** under block formatting; proprietary models go from
+0.26 to +0.00. Block formatting also lifts *interior* accuracy (0.79–0.81 vs
0.02–0.03 plain) — it is the only intervention here that produces general
position indexing.

**4 — LoRA.** Task-specific LoRA closes the endpoint gap completely (Qwen
28/28 held-out cells, CVQ 0.98; Gemma 28/28, CVQ 0.97) and transfers to the
semantic distribution it never trained on. But it does *not* buy position
indexing: interior accuracy only reaches 0.29 (Qwen) / 0.20 (Gemma). An
arithmetic-trained control adapter demonstrably learned its own task (GSM8K
0.16 → 0.70) and fixed **0** cells — the effect is task-specific, not a
by-product of any fine-tuning.

Read together: **block formatting and LoRA reach the same endpoint readout by
different routes, but only formatting generalises to the interior.**

**5 — Mechanism (how LoRA does it).** The base model *computes* the current
value mid-stack (logit-lens P(v_last) peaks ~0.5 at L33) then **suppresses it to
0.25 by the readout** — "tracked but suppressed." A linear probe recovers it,
so the information is present; attention just doesn't route it out. LoRA's fix
is ~8 late-layer heads that re-point attention at the last update (P(attend)
L33: 0.07→0.53); ablating exactly those heads removes the LoRA gain (paired
n=200, CIs exclude zero). Qwen primary, Gemma replicates. Files in
`05_mechanistic/`. A fifth method, **Entropy-Lens**
(`05_mechanistic/entropy_lens/`), adds the uncertainty axis: Qwen **expands then
prunes** (entropy peaks 0.78 at 89% depth, collapses to ~0 at readout) while the
from-scratch model is **flat then cliff** (no expansion at all, one drop at the
last layer). LoRA leaves that geometry untouched — its effect is
representational, not a change in entropy dynamics. And only the from-scratch
model is *calibrated*: its output entropy tracks correctness (gap +0.28 to
+0.49), whereas base Qwen is **confidently wrong** (gap +0.005 to +0.035).
Qwen-only, like `attention_routing`.

**6 — From scratch: the bias is not inherited from pretraining.** A GPT-2-small
*architecture* trained from random init on a 51-symbol synthetic KV task (no
pretrained weights, no natural language) reproduces the signature — and does so
while being **good** at the task: iid **0.800**, held-out **0.633**, against a
chance floor of 0.10. Two things make it more than a replication. First, the
three roles are acquired as **separate phase transitions in FVQ → CVQ → IVQ
order**: after 13,000 flat steps, first-value breaks out alone (0.30→0.61 in one
1,000-step window), current-value follows ~6,000 steps later, interior counting
arrives last — so the ordering the whole project documents is here the *order in
which the capability is learned*. Second, the ordering **survives competence**
(0.91 / 0.86 / 0.75 at 0.80 overall), so it is a structural bias, not a
capability deficit. Files in `06_from_scratch/`; **only** the `h100_cosine` run
was consolidated and that README lists exactly what was left behind.

**7 — Chain-of-thought: a compute limit, not a missing representation.** Give
the same three Claude 4.5 models the byte-identical prompt and turn extended
thinking on, and the interior floor lifts from **0.17–0.30 to 0.97–1.00** on
Semantic-Multi. Nothing about the stimulus or the weights changed, so what the
single forward pass lacked was the serial computation to count occurrences, not
access to the value — which constrains how theme 2's collapse can be described.
It is not free, though: on the Arbitrary-Single depth ladder (K=5, N up to 460)
CoT accuracy decays monotonically with depth (opus 0.998 at N=50 → **0.756** at
N=460) and **the U-curve reappears** — at N=460 opus holds 1.00 across positions
1–8 and scores 0.19 at position 395 while the final update reads 0.77. So CoT
flattens the middle at moderate depth without abolishing the shape. Files in
`07_cot_ivq/`, one CSV per arm; the CoT-vs-no-CoT contrast exists on
Semantic-Multi **only**.

### Seven models are excluded — do not re-add them without reading why

Across the original corpus, **21.6% of FVQ trials were `off-stream`**: the model
emitted something that is not any value in the stream at all (key name, filler,
refusal). That is an instruction-following failure, not a retrieval failure, and
it was concentrated almost entirely in the smallest models.

Every model at **≥20% off-stream** has been **dropped from all consolidated
results**:

| dropped | off-stream | | dropped | off-stream |
|---|---|---|---|---|
| gemma-3-270m-it | 72.6% | | stablelm-2-1_6b-chat | 24.6% |
| pythia-410m | 63.5% | | TinyLlama-1.1B-Chat-v1.0 | 21.3% |
| Qwen2.5-0.5B-Instruct | 51.7% | | Qwen3.5-0.8B | 20.3% |
| mamba-1.4b-hf | 25.8% | | | |

The data breaks cleanly at the threshold — #7 is 20.3%, #8 is 16.2% — and the
retained models run 3.4%–16.2%. Only theme 1 was affected (752 rows); themes
2–4 never used these models.

**This does not change any finding.** Pre-exclusion the theme-1 headline was 72%
of cells / +0.23 mean gap; post-exclusion it is 69% / +0.19. The effect was never
carried by the weak models.

Enforced centrally by `EXCLUDED_MODELS` in `build/common.py`, applied in `emit()`
so no builder can bypass it. Raw sources are untouched and the QA cross-checks
still run over the full pre-exclusion data, so removing a name and rebuilding
restores it. Full audit — including the false-positive risk from the lenient
substring matcher and what still cannot be checked from disk:
`01_fvq_cvq/VALIDITY.md`.

`fvq_cvq.csv` also carries `n_offstream`, `accuracy_onstream` and
`error_types_json` per row; prefer `accuracy_onstream` when precision matters.

### One thing consolidation surfaced that the scattered files hid

The U-curve sweep queries the *same target value* two ways — "the last value of
X" and "the 50th value of X" — and the answers diverge enormously. GPT-4.1 under
plain text averages **0.012** ordinal against **0.832** semantic across six
cells (71×); at K=10, N=50 alone it is **0.005** against **0.69** (138×).
Because the two were never in one table before, the ordinal rows had been read
as if they were comparable to the semantic ones. They are not. The split is now an explicit
`query_type` column, and it separates a *counting* failure from a *retrieval*
failure — see "condition vs query_type" below. Worth a look before the next
draft; it may deserve to be a result rather than a caveat.

---

## Schema

Every measurement CSV uses the same tidy-long schema — one row is one
measurement. Columns a source cannot supply are left empty rather than invented.

| column | meaning |
|---|---|
| `theme` | which theme folder this row belongs to (`01_fvq_cvq` … `07_cot_ivq`) |
| `model` | canonical model id (see `build/common.py` `MODELS`) |
| `model_family`, `is_proprietary` | derived from `model` |
| `variant` | `base`, `lora`, `lora_arith_control`, `nocot`, `cot_thinking`, post-training stage, … — the model *version or inference mode* measured |
| `dataset` | `arbitrary_single`, `semantic_multi`, `gsm8k` |
| `prompt_format` | `plain`, `block`, `flat_nolabel`, `flat_verbose`, `landmark`, `chat_template`, `completion_few_shot`, … — **every template verbatim in `PROMPTS.md`** |
| `num_keys`, `num_updates` | the (K, N) cell |
| `condition` | **which** value is targeted: `FVQ`, `CVQ`, `IVQ`, `FVQ-CVQ` (gap-only rows), `TASK_ACC` |
| `query_type` | **how** it was asked: `semantic` ("the last value of X") or `ordinal` ("the 7th value of X") — see below |
| `position` | `first`, `last`, or the 1-indexed update queried |
| `accuracy` | proportion correct |
| `gap` | FVQ − CVQ; **only** on `condition=FVQ-CVQ` rows |
| `ci_lower`, `ci_upper`, `wilson_hw` | whichever the source recorded |
| `n_trials`, `n_correct` | trial counts (see the warning below) |
| `regime` | `AB` / `B` / `C`, where the stage-1 sweeps recorded it |
| `split` | `train`, `held_moderate`, `held_hard`, `held_out`, `ood`, `extrapolation` |
| `provenance_tier` | `raw` or `derived` — **read the next section** |
| `source_file` | repo-relative path the number came from |
| `notes` | run name, cell key, early-stop flag, caveats |

`accuracy` and `gap` are deliberately separate columns: some sources preserve
only a difference, and mixing a difference into an accuracy column would
silently corrupt any aggregation.

### `condition` vs `query_type` — do not conflate them

`condition` says which value the query targets; `query_type` says how it was
phrased. **They are independent, and the difference is large enough to change
conclusions.** The U-curve sweep asks for the final update *both ways* on the
same cell, and GPT-4.1 under plain text scores **0.012 asked ordinally** ("the
50th value") against **0.832 asked semantically** ("the last value").

The ordinal/semantic split appears only under formats where the model must count
occurrences itself (`flat_nolabel`, `landmark`) and vanishes under formats that
number the updates for it (`block`, `flat_verbose`) — so the interior failure
decomposes into a *counting* failure and a *retrieval* failure. Full table in
`02_intermediate_ivq/README.md`.

Each source's phrasing was read off its prompt builder, not assumed; the
file:line references are in `PROVENANCE.md`. **Never pool the two types into one
mean.**

### Naming

The paper's **FVQ / CVQ** terminology is canonical here:

| paper | this folder | raw JSON / older code |
|---|---|---|
| First-Value Query | `FVQ` | `RI` |
| Current-Value Query (the last value) | `CVQ` | `PI` |

The raw files and `*_comparison.txt` still use `RI`/`PI`. Translation happens at
read time in `build/common.py::cond_from_ri_pi`; nothing was renamed on disk.

---

## `provenance_tier`: raw vs derived — read this before plotting

- **`raw`** — computed from a results file still in the repo. Re-derivable.
- **`derived`** — back-extracted from a committed LaTeX table in
  `paper/figures/`, because the raw run **no longer exists locally** (it lived
  on an L40S / SageMaker box; see `AAAI_PREP_PLAN.md` §1.3). These numbers are
  as trustworthy as the paper, but they are **rounded to 2dp**, often carry no
  per-condition CI, and cannot be recomputed without re-running the experiment.

Derived rows are a small minority (118 of the 9,770 rows in themes 01–04;
themes 05, 06 and 07 add 906, 458 and 846 rows, of which themes 06 and 07 are
all `raw`) and are confined to:
open-weight `semantic_multi` (theme 1), the open-weight format sweep, the
SmolLM3 post-training formats, and the **Gemma** LoRA semantic-OOD baseline.

> **Corrected 2026-09-07.** This section previously said 134 rows and included
> the LoRA semantic-OOD baselines generally. The **Qwen** semantic-OOD block —
> 4 cells × {base, LoRA} × {FVQ, CVQ} = 16 rows — was tagged `derived` on the
> stated grounds that its raw run was not local. That was **wrong**: the run
> survives as `lora_sem_validation_results.json` (git-tracked since well before
> this folder was built). Those 16 rows are now `raw` and carry `n_trials` /
> `n_correct`; the values were unchanged by the switch (16/16 exact). Only the
> Gemma baseline is genuinely gone. See `04_lora/README.md`.

**Never mix tiers inside a single mean or a single plotted series without
saying so.** Filter explicitly:

```python
import pandas as pd
df = pd.read_csv("01_fvq_cvq/fvq_cvq.csv")
raw_only = df[df.provenance_tier == "raw"]
```

### How much of this is verified

Three of the four builders cross-check their output against an *independent*
restatement of the same numbers. All currently pass exactly:

| check | independent? | result |
|---|---|---|
| theme 1 gaps vs `paper/figures/tab_full_arb.tex` | yes | 47 cells, **0 mismatched** |
| theme 3 `.tex` back-extraction vs the raw U-curve CSV | yes | 48 values, **0 mismatched** |
| theme 4 rows vs `lora_intervention/results/*_comparison.txt` | yes | 564 values, **0 mismatched**, 4 unavailable |
| theme 4 Qwen SEM raw vs `paper/figures/tab_lora_sem.tex` | **no** — see below | 16 values, **0 mismatched** |

The theme-3 check is the important one: where the raw data *does* still exist,
the LaTeX back-extraction reproduces it exactly — which is the evidence that the
`derived` tier is sound where the raw data is gone.

The theme-4 Qwen SEM check is **not** independent and must not be quoted as if
it were. `paper/scripts/generate_lora_tables.py` generates `tab_lora_sem.tex`
*from* `lora_sem_validation_results.json`, the same file the builder now reads
directly — so it closes the loop on the tex parse and the 2dp rounding, and
nothing more. It cannot detect an error in the run itself.

---

## Known gaps and caveats

- **Trial counts vary and are often small.** LoRA evals early-stop at 40–80
  trials per cell (`stopped_early` is recorded in `notes`); dense-IVQ uses 20
  per position; the SmolLM3 format rows are means over 32 cells. Always read
  `n_trials` before comparing two numbers.
- **The Qwen semantic-OOD LoRA cells have unequal n.** 20 trials on (7,20) and
  (15,20) but only **10** on (15,30) and (20,30) — the original script was
  killed mid-run and `lora_sem_finish.py` re-ran the remainder at a lower
  budget. All four read LoRA FVQ = CVQ = 1.00, so the two 10-trial cells look
  identical to the two 20-trial ones. This was invisible while the rows were
  sourced from the 2dp paper table; it is now in `n_trials` / `n_correct`.
  A 10/10 does not license the same claim as a 20/20.
- **There is no longer a state-space architectural control.** `mamba-1.4b-hf`
  was the only non-attention model and it is excluded (26% off-stream, 3 cells).
  If that control matters, it needs a fresh run.
- **Gemma has no baseline at (25,75) and (30,75)** — the stage-1 sweep never
  covered those cells, which is why `gemma_main_comparison.txt` shows `---`
  there. Those 4 values are the "unavailable" entries in the theme-4 QA.
- **`plain` ≠ `flat_nolabel`.** Both are unformatted prompts but they come from
  different experiments with different templates. Do not merge them.
- **Deliberate overlap across themes 2, 3, 4.** The proprietary U-curve appears
  in both `02_intermediate_ivq/ivq.csv` and `03_prompt_format/format.csv`; the
  open-weight per-position sweeps appear in both `04_lora/lora_ivq.csv` and
  `format.csv`. Same measurements, different analysis axis — each theme is
  self-sufficient. Do not concatenate these files without de-duplicating.
- **Interior IVQ is split by provenance, not arbitrarily.** Standalone
  base-model interior data (proprietary) is in theme 2; interior data that was
  run as a base-vs-LoRA comparison (all open-weight) is in
  `04_lora/lora_ivq.csv`. If you want "the U-curve for open-weight models," it's
  in theme 4.
- **Theme 07 has no `nocot` arm on Arbitrary-Single.** The CoT-vs-no-CoT
  contrast exists on `semantic_multi` only; the `arbitrary_single` rows are a
  CoT-only depth ladder, and they sit in `cot_ivq_cot_thinking.csv` with no
  counterpart in `cot_ivq_nocot.csv`. Do not compute an unqualified "CoT effect"
  across both datasets, and do not read the two arm files' row counts (270 vs
  576) as an arm imbalance — the 306-row difference is that ladder.
- **Theme 07's arm files are a partition, not an overlap.** Unlike the themes
  2/3/4 overlap below, no row is in both, so they concatenate safely. Pairing
  them back up needs the CoT file filtered to `semantic_multi` first; see
  `07_cot_ivq/README.md`.
- **Theme 07 positions retire on individual Wilson half-widths.** Within one
  cell `n_trials` ranges 51–200, so a mean across positions inside a cell
  silently over-weights the easy ones. This is the sharpest instance of the
  general `n_trials` warning above.
- **The same canonical `claude-4.5-*` id spans recorded and unrecorded
  snapshots.** Theme 07 preserves the dated API id per row in `notes`; themes 02
  and 03 have no snapshot date at all. A theme-02 vs theme-07 difference is not
  purely an arm effect.
- **`by_model/` files are generated.** They are wiped and rewritten on every
  build. Edit nothing there; the combined CSV is canonical.

### Deliberately out of scope

- **Mechanistic per-layer results are now in scope** — they are `05_mechanistic/`.
  This bullet previously said they were excluded and that the folder covered
  behavioural retrieval only; that was left stale when theme 05 was added, and
  is corrected here (2026-09-07).
- **Most of `synthetic_scratch_training/` is still out of scope.** Theme 06
  consolidates the `05_gpt2_scratch_h100/h100_cosine` run only. Experiments
  01–04, the `h100_plateau` control arm, and the unlaunched 06/07 are left in
  place; `06_from_scratch/README.md` has a row per item saying what it is and
  why. The `h100_plateau` arm is the one to re-add first — it is the control for
  the LR-schedule claim.
- **The legacy ACL-era PI/RI study** (39 API models, top-level `results/`) is a
  different task definition and prompt design and is excluded. Its location is
  recorded at the end of `PROVENANCE.md` so it stays findable.
- **`mechanistic_probing_v2/`** is excluded as a results source. Note it is
  still a live *code* dependency (`core/dataset_configs.py`) — do not delete it.

---

## Rebuilding

```bash
cd consolidated_results/build
python3 build_all.py          # all seven themes, with QA checks
python3 build_03_formats.py   # or one at a time
```

Pure stdlib, no dependencies, reads only from the repo. Every builder prints its
row count, its raw/derived split, and its QA result. **A non-zero mismatch count
means a source file moved underneath the build — investigate before using the
output.**

| file | role |
|---|---|
| `build/common.py` | schema, model registry, RI→FVQ mapping, CSV writers, LaTeX back-extraction helpers |
| `build/build_01_fvq_cvq.py` … `build_04_lora.py` | one per theme; the module docstring lists its exact sources |
| `build/build_all.py` | runs all four |

To add a new experiment: extend the relevant builder, re-run it, and add a row
to `PROVENANCE.md`.
