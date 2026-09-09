# Provenance

Every consolidated file, and every source file it was built from. Row counts are
as of the 2026-07-22 build (theme 04 rebuilt 2026-09-07; theme 07 added
2026-09-09). The CSVs reference their sources in place, via the `source_file`
column on each row.

**One exception to that.** `04_lora/sources/` holds byte-identical copies of
`lora_sem_validation_results.json` and `lora_sem_finish_results.json`. Those two
are loose JSONs at the repo root — the kind of file a cleanup removes — and they
are the only local record of 16 rows and of that experiment's trial counts.
`source_file` still reports the canonical root path; the builder reads the copy
only if the root file has gone, and says so when it does. Do not add further
copies without the same justification.

Regenerate with `cd build && python3 build_all.py`.

---

## Excluded models (applied 2026-07-22)

**Seven models are dropped from every consolidated CSV** and must not be
re-introduced into any future result without re-running them.

**Criterion:** off-stream rate ≥ 20%, where *off-stream* = `garbage` + `empty` +
`oom` — the model emitted something that is not any value in the key-value
stream (the key name, a filler word, a refusal). That is an instruction-following
failure, not a retrieval failure, so above this threshold a cell measures whether
the model can produce the output format at all and its accuracies are not
comparable with the rest of the corpus.

| # | model | off-stream | garbage | empty | of N trials |
|---|---|---|---|---|---|
| 1 | `gemma-3-270m-it` | **72.6%** | 8,844 | 303 | 12,600 |
| 2 | `pythia-410m` | **63.5%** | 5,464 | 0 | 8,600 |
| 3 | `Qwen2.5-0.5B-Instruct` | **51.7%** | 8,073 | 0 | 15,600 |
| 4 | `mamba-1.4b-hf` | **25.8%** | 155 | 0 | 600 |
| 5 | `stablelm-2-1_6b-chat` | **24.6%** | 2,661 | 0 | 10,800 |
| 6 | `TinyLlama-1.1B-Chat-v1.0` | **21.3%** | 2,002 | 0 | 9,400 |
| 7 | `Qwen3.5-0.8B` | **20.3%** | 3,201 | 0 | 15,800 |

The threshold falls on a clean break in the data: #7 is 20.3%, #8
(`Qwen3.5-2B`) is 16.2%. The 8 retained open-weight models span 3.4%–16.2%.

**What it removed:** 752 rows from `01_fvq_cvq/fvq_cvq.csv` (2,764 → 2,012;
21 → 14 models). **Themes 02, 03 and 04 are unaffected** — none of the seven
appears in those experiments.

**Effect on the theme-1 headline:** 1,357 → 990 paired cells; FVQ>CVQ in 72% →
**69%** of cells; mean per-series gap +0.23 → **+0.19**; positive in 25/27 →
**19/20** series. Direction and conclusion unchanged.

**Three consequences to carry forward:**

1. **The state-space architectural control is gone.** `mamba-1.4b-hf` was the
   only non-attention model in the corpus. Any claim that the effect is
   attention-specific now has no negative control and needs a fresh run.
2. **The small end of the scaling curve is gone.** The retained open-weight
   models start at 1.5B. Scaling claims below that are no longer supported.
3. **Proprietary models were NOT screened — this is an assumption, not a check.**
   The two proprietary CSVs are aggregates that carry no output taxonomy, and
   the 18 upstream per-trial JSONs (`experiments_cloud/results/{semantic,
   semantic_k1,ucurve_proprietary}/*/`) are **not local and were never committed
   to git** — so their off-stream rate cannot be computed from anything in this
   repo.

   **Partly superseded for theme 07 (2026-09-09).** The CoT sweep persists a
   per-trial outcome for every response, so its three claude models ARE screened:
   0.12% off-stream corpus-wide, worst cell-arm 0.23%. That covers only theme 07's
   runs — themes 02 and 03 remain unscreened.

   **Decision (2026-07-22): accepted as clean without screening**, on the
   grounds that off-stream rate tracked capability almost monotonically in the
   open-weight corpus (72.6% at 270M → 3.4% at 4B) and every proprietary model
   sits well above that range. The six proprietary models therefore pass no
   validity filter, while the eight open-weight models pass a ≥20% one.
   State this asymmetry if the criterion is presented as a methodological
   standard.

   *Caveat on the inference:* the failure mode found in the one local
   proprietary file was an **API auth failure**, not a capability failure —
   120/120 trials scored `garbage` because the run never reached the model
   (`experiments_cloud/results/arbitrary/claude-haiku/sweep_full_20260316_225220.json`).
   That mode is size-independent, so model scale does not rule it out. Later
   runs of the same sweep are clean and `sweep_semantic.py:293-296` now retries
   on error-prefixed responses, so the guard exists — but it is untested against
   the runs we actually cite.

   *If this is ever revisited:* the sweep scripts do persist generations
   (`sweep_semantic.py:324`, `ucurve_sweep.py:317-318`) and already store
   `error_type` per trial, so recovering those 18 files makes the screen a read
   rather than a re-run. Only a lost machine forces a paid API re-run.

**Mechanism.** `EXCLUDED_MODELS` in `build/common.py`, applied centrally in
`emit()` so no theme builder can bypass it. Raw source files are untouched;
deleting a name from that set and rebuilding restores the model. The QA
cross-checks deliberately run over the full pre-exclusion data, so the exclusion
never weakens parser validation.

**Evidence:** `01_fvq_cvq/VALIDITY.md`; reproduce with
`build/check_output_validity.py`.

---

## 01_fvq_cvq/fvq_cvq.csv — 2,012 rows (after exclusions)

Built by `build/build_01_fvq_cvq.py`.

| rows | tier | source |
|---|---|---|
| — | *excluded* | `v3/results_vllm/arbitrary_single/Qwen2.5-0.5B-Instruct/stage1_sweep_20260408_224841.json` |
| 158 | raw | `v3/results_vllm/arbitrary_single/Qwen2.5-1.5B-Instruct/stage1_sweep_20260408_231134.json` |
| 158 | raw | `v3/results_vllm/arbitrary_single/Qwen2.5-3B-Instruct/stage1_sweep_20260409_000134.json` |
| 146 | raw | `v3/results_vllm/arbitrary_single/Qwen2.5-3B/stage1_sweep_20260409_010049.json` |
| — | *excluded* | `v3/results_vllm/arbitrary_single/Qwen3.5-0.8B/stage1_sweep_20260509_110846.json` |
| 158 | raw | `v3/results_vllm/arbitrary_single/Qwen3.5-2B/stage1_sweep_20260509_114227.json` |
| 100 | raw | `v3/results_vllm/arbitrary_single/Qwen3.5-4B/stage1_sweep_20260509_120216.json` |
| 158 | raw | `v3/results_vllm/arbitrary_single/Qwen3.5-9B/stage1_sweep_20260509_100749.json` |
| — | *excluded* | `v3/results_vllm/arbitrary_single/TinyLlama-1.1B-Chat-v1.0/stage1_sweep_20260409_011936.json` |
| — | *excluded* | `v3/results_vllm/arbitrary_single/gemma-3-270m-it/stage1_sweep_20260409_014928.json` |
| 110 | raw | `v3/results_vllm/arbitrary_single/gemma-3-1b-it/stage1_sweep_20260409_015538.json` |
| 152 | raw | `v3/results_vllm/arbitrary_single/gemma-3-4b-it/stage1_sweep_20260409_022631.json` |
| — | *excluded* | `v3/results_vllm/arbitrary_single/stablelm-2-1_6b-chat/stage1_sweep_20260409_012754.json` |
| — | *excluded* | `v3/results_vllm/arbitrary_single/pythia-410m/stage1_sweep_20260409_013555.json` |
| — | *excluded* | `v3/results_vllm/arbitrary_single/mamba-1.4b-hf/stage1_sweep_20260409_013931.json` — 3 cells; the only non-attention model |
| 828 | raw | `experiments_cloud/results/proprietary_semantic_multi.csv` |
| 10 | **derived** | `paper/figures/tab2_openweight.tex` — 5 retained open-weight models, K=5/N=30 SEM |
| 34 | **derived** | `paper/figures/tab_full_sem.tex` — retained models × 4 cells, **gap only** |

Cross-checked against `paper/figures/tab_full_arb.tex` (not emitted — it
restates the raw sweeps): **47 cells checked, 0 mismatched**.

## Dataset labels — corrected 2026-07-22

An earlier build labelled the proprietary U-curve rows `arbitrary_single`. That
was wrong. Verified against the runners and captions:

| source | dataset | evidence |
|---|---|---|
| `ucurve_proprietary_results.csv` | **`semantic_multi`** | `run_ucurve_proprietary.sh:13,29` passes `--dataset SEMANTIC_MULTI` for all 6 models |
| `paper/figures/tab_format_ci.tex` | **`semantic_multi`** | `paper/main.tex:357` — "Format-intervention results on Semantic-Multi" |
| `paper/figures/tab_smollm3_formats.tex` | `arbitrary_single` | `paper/main.tex:1520` — "evaluated on Arbitrary-Single", K∈{2,3,5,7} × N∈{2,3,5,7,10,15,20,30} |
| `experiments_cloud/sweep_arbitrary.py` | `ARBITRARY_MULTI` | `:68` — a third dataset, distinct from `ARBITRARY_SINGLE`; **not ingested** |

Affected and now fixed: 2,160 rows in theme 02 and 2,192 in theme 03.

**Both model families were run on both datasets** — the apparent
"open-weight = Arbitrary, proprietary = Semantic" split in theme 01 is an
artefact of which raw data survived, not of experimental design:

| | `ARBITRARY_SINGLE` | `SEMANTIC_MULTI` |
|---|---|---|
| open-weight | raw, 8 models × ≤79 cells | ran; raw **not local** → 1 accuracy cell + 4 gap cells per model, derived |
| proprietary | ran only as `ARBITRARY_MULTI` (different dataset, 2 models, 10–20 trials/cell, Mar 2026) — not ingested | raw aggregate, 6 models × 69 cells |

---

## 02_intermediate_ivq/ivq.csv — 2,160 rows, all raw

Built by `build/build_02_ivq.py`. **Scope narrowed 2026-07-22:** this theme now
holds only standalone base-model IVQ. The open-weight experiments (behavioral,
dense-IVQ, E1) were base-vs-LoRA comparisons and moved to
`04_lora/lora_ivq.csv` — see that section. What remains is the proprietary
U-curve, which has no LoRA arm.

| rows | source |
|---|---|
| 2,160 | `experiments_cloud/results/ucurve_proprietary_results.csv` |

`query_type` (ordinal vs semantic) was assigned per source by reading the prompt
builders, not guessed:

| source | file:line |
|---|---|
| U-curve | `experiments_cloud/ucurve_prompts.py` — `prompt_flat_nolabel` (ordinal) vs `prompt_*_lastquery` (semantic) |
| (the dense-IVQ / behavioural / E1 phrasing refs now live in the 04_lora section) | |

## 04_lora/lora_ivq.csv — 1,896 rows, all raw

Built by `build/build_04_lora.py` (`build_lora_ivq()`). Per-position base-vs-LoRA
IVQ, moved here from theme 02 on 2026-07-22 because these are LoRA comparisons.
Qwen2.5-3B-Instruct and gemma-3-4b-it only.

| rows | source | experiment |
|---|---|---|
| 498 | `lora_intervention/experiments/behavioral_block_lora_results/results.json` | behavioral (Qwen) |
| 498 | `lora_intervention/experiments/gemma_results/behavioral/results.json` | behavioral (Gemma) |
| 220 | `lora_intervention/results/dense_ivq_ivq_20260523_101418.json` | dense-IVQ (Qwen; adapter not local) |
| 140 | `lora_intervention/experiments/e1_results/results_multiples.jsonl` | E1 scan A/B (Qwen) |
| 140 | `lora_intervention/experiments/gemma_results/e1/results.jsonl` | E1 scan A/B (Gemma) |
| 200 | `lora_intervention/experiments/bigcell_ivq/out/qwen/results.jsonl` | E1 scan C big-cell interior (Qwen; H100, 200-trial) |
| 200 | `lora_intervention/experiments/bigcell_ivq/out/gemma/results.jsonl` | E1 scan C big-cell interior (Gemma; H100, 200-trial) |

`query_type` phrasing, verified against the prompt builders:

| source | file:line |
|---|---|
| dense IVQ | `lora_intervention/evaluate_ivq.py:121` — `"last" if k == N else ordinal(k)` |
| behavioural, block arm | `lora_intervention/experiments/behavioral_block_lora_sweep.py:76` — `"...in Update {k}"` |
| E1 | `lora_intervention/experiments/e1_extrapolation_frontier.py:138-146` — `"first"` / `"last"` at endpoints |

## 03_prompt_format/format.csv — 3,222 rows

Built by `build/build_03_formats.py`.

| rows | tier | source |
|---|---|---|
| 2,160 | raw | `experiments_cloud/results/ucurve_proprietary_results.csv` |
| 498 | raw | `lora_intervention/experiments/behavioral_block_lora_results/results.json` |
| 498 | raw | `lora_intervention/experiments/gemma_results/behavioral/results.json` |
| 6 | raw | `lora_intervention/experiments/block_readout_results/results_K2N30.json` |
| 6 | raw | `lora_intervention/experiments/block_readout_results/results_K10N50.json` |
| 6 | raw | `lora_intervention/experiments/gemma_results/block/results_K2N30.json` |
| 6 | raw | `lora_intervention/experiments/gemma_results/block/results_K10N50.json` |
| 32 | **derived** | `paper/figures/tab_format_ci.tex` — open-weight rows only |
| 10 | **derived** | `paper/figures/tab_smollm3_formats.tex` |

The `block_readout` files also contain 36-layer logit-lens curves; **only the
behavioural accuracies were extracted**. The per-layer data stays in the source.

The proprietary rows of `tab_format_ci.tex` restate numbers we also hold raw.
Cross-checked: **48 values checked, 0 mismatched** — this is the evidence that
the `.tex` back-extraction is sound where the raw data is gone.

## 04_lora/lora_evals.csv — 480 rows · 04_lora/adapters.csv — 3 rows

Built by `build/build_04_lora.py`.

| rows | tier | source |
|---|---|---|
| 56 | raw | `lora_intervention/results/main_eval_20260523_055309.json` |
| 88 | raw | `lora_intervention/results/gemma_main_eval_20260525_165155.json` |
| 56 | raw | `lora_intervention/results/gemma_smoke_eval_20260525_154202.json` |
| 56 | raw | `lora_intervention/results/dense_ivq_eval_20260523_095657.json` |
| 56 | raw | `lora_intervention/results/qwen_arith_control_full_eval_20260525_212145.json` |
| 10 | raw | `lora_intervention/results/qwen_arith_control_eval_20260525_204131.json` |
| 1 | raw | `lora_intervention/results/gsm8k_task_eval_qwen_base_20260525_210412.json` |
| 1 | raw | `lora_intervention/results/gsm8k_task_eval_qwen_arith_control_20260525_210220.json` |
| 56 | raw | `v3/results_vllm/arbitrary_single/Qwen2.5-3B-Instruct/stage1_sweep_20260409_000134.json` — baselines |
| 52 | raw | `v3/results_vllm/arbitrary_single/gemma-3-4b-it/stage1_sweep_20260409_022631.json` — baselines |
| 16 | raw | `lora_sem_validation_results.json` — Qwen SEM, base + LoRA (copy preserved at `04_lora/sources/`) |
| 32 | **derived** | `paper/figures/tab_lora_gemma_sem.tex` — Gemma SEM baseline only |

`adapters.csv` is built from `lora_intervention/checkpoints/{adapter,
gemma_adapter, qwen_arith_adapter}/run_config.json` + `adapter_config.json` +
the `.safetensors` file size, with the K,N training grid read from
`lora_intervention/data_gen.py` (`TRAIN_KEYS`, `TRAIN_UPDATES`).

Cross-checked against `lora_intervention/results/*_comparison.txt`: **564 values
checked, 0 mismatched, 4 unavailable** (Gemma K=25/N=75 and K=30/N=75 have no
baseline — the stage-1 sweep never covered those cells).

The 16 Qwen SEM rows are additionally round-tripped against
`paper/figures/tab_lora_sem.tex`: **16 values checked, 0 mismatched**. This is a
consistency check, **not** an independent one — `paper/scripts/generate_lora_tables.py`
generates that table from the same JSON the builder reads.

---

## 05_mechanistic/entropy_lens/ — 34,164 profile + 450 confusion + 450 behavioural + 15 summary rows

Built by `build/build_05b_entropy_lens.py`. All `raw`. One subfolder per **arm**
(`qwen_base`, `qwen_lora`, `gpt2_scratch`) — not per model, because base and
lora are the same Qwen2.5-3B-Instruct with and without the adapter.

| rows | tier | source |
|---|---|---|
| 16,200 / 150 / 150 | raw | `lora_intervention/experiments/entropy_lens/results/entropy_base.json` → `qwen_base/{profiles,confusion,behavioral}.csv` |
| 13,140 / 150 / 150 | raw | `.../entropy_lora.json` → `qwen_lora/…` |
| 4,824 / 150 / 150 | raw | `.../entropy_scratch.json` → `gpt2_scratch/…` |
| 15 | raw | `.../entropy_summary.json` → `summary.csv` (geometry scalars; behavioural + confusion pooled from the per-cell records) |

Grid K∈{2,4,6,8,10,12} × N∈{4,6,8,10,12}, 150 trials × 5 conditions =
22,500 scored trials per arm. Profile row counts differ per arm because a
`wrong` subset with n=0 is omitted (LoRA was perfect on 85 of 150 cell×condition
combinations, the scratch model on 48) and because Qwen has 36 layers vs 12.

Cross-check: the `gpt2_scratch` arm re-measures theme 06's model on the same 30
cells with a different seed formula, n=150 vs 200/500, and the checkpoint loaded
from the HF Hub rather than the local `.pt` — **150 cells, mean |diff| 0.0156,
median 0.0083, max 0.1100, 149/150 within 0.10**. The only cross-experiment
replication of theme 06 that exists.

**Two things not to miss:** the grid has **zero** cells in common with
`04_lora/lora_ivq.csv`, so the Qwen rows are a small-K/small-N regime found
nowhere else here; and `query_type` differs *across arms* for the endpoints
(Qwen FVQ/CVQ are semantic, the scratch model's are ordinal step tokens).
`entropy_lens/RUN.md`'s claim that the arms are matched on "query-type" is wrong
for FVQ/CVQ. **No Gemma arm.**

---

## 06_from_scratch/ — from_scratch_ivq.csv (252) · from_scratch_training.csv (120) · from_scratch_summary.csv (86)

Built by `build/build_06_from_scratch.py`. GPT-2-small *architecture* trained
from random init on the synthetic KV task; all rows `raw`, no derived tier.

**Scope: the `h100_cosine` arm only** (iid 0.800 / held-out 0.633) — the run
behind `present/ONE_PAGER.tex` §7. Everything else under
`synthetic_scratch_training/` is left in place and itemised in
`06_from_scratch/README.md`.

| rows | tier | source |
|---|---|---|
| 252 | raw | `synthetic_scratch_training/05_gpt2_scratch_h100/results/h100_cosine/joint_knstep.json` — 204 trained + 48 held-out (K,N,step) cells |
| 120 | raw | `.../h100_cosine/val_agg_step*.json` (30 checkpoints × {ALL,FVQ,CVQ,IVQ}), lr/loss joined from `.../train_log.json` |
| 86 | raw | `.../h100_cosine/final_eval.json` — overall, by_role, by_step, by_cell, by_duplicate |

Figures in `present/` trace to the same two files:
`10_trained_from_scratch_ood.png` ← `joint_knstep.json` via
`05_gpt2_scratch_h100/src/plot_depth_grid.py`;
`11_trained_from_scratch_heatmap.png` ← `final_eval.json` via
`src/plot_results.py`.

Cross-check: `joint_knstep.json` re-aggregated vs `final_eval.json` — **6 values
checked, 0 outside the 5-prediction budget (worst 3.0)**. Both are separate eval
passes over the same frozen set at the same checkpoint and differ by 4
predictions in 64,800 (0.006%), attributable to non-deterministic float
reduction order under TF32/bf16; the check is therefore a flipped-prediction
budget, not a float epsilon.

**Not consolidated, deliberately:** `h100_plateau` (the LR-schedule control,
iid 0.283), experiments `01_baseline_step_ood`–`04_gpt2_scratch_kaggle_t4`, and
the unlaunched `06_first_last_only` / `07_gpt2_pretrained_first_last`. Model
weights are not in the repo (~329 MB, gitignored; HF Hub
`kanak8278/gpt2-small-synthetic-kv-interference`).

---

## Derived-tier summary

118 of 9,770 rows (1.2%) are back-extracted from committed LaTeX because the
raw run is no longer on disk. All four source tables:

| table | what it preserves | raw location (missing) |
|---|---|---|
| `tab2_openweight.tex` | open-weight SEM at K=5/N=30, FVQ + CVQ | `v3/results_vllm/semantic_multi/` |
| `tab_full_sem.tex` | open-weight SEM gaps at 4 cells | `v3/results_vllm/semantic_multi/` |
| `tab_format_ci.tex` | open-weight formats at K=10/N=50 | `experiments_cloud/results/ucurve_vllm/` |
| `tab_smollm3_formats.tex` | SmolLM3 stage × format gaps | `v3/results_vllm/training_dynamics_smollm3/` |
| `tab_lora_gemma_sem.tex` | Gemma LoRA SEM baseline | `v3/results_vllm/semantic_multi/` |

See `AAAI_PREP_PLAN.md` §1.3 for the full missing-data inventory and re-run
costs. Everything else there is either mechanistic (out of scope for this
folder) or already covered above.

---

## 05_mechanistic/ — logit_lens.csv, probing.csv, attention_routing.csv, causal_ablation.csv

Built by `build/build_05_mechanistic.py`. Qwen2.5-3B-Instruct + gemma-3-4b-it,
base vs LoRA. One CSV per method (different units); provenance tiers as elsewhere.

| file | rows | raw sources | derived sources |
|---|---|---|---|
| `logit_lens.csv` | 456 | `v3/results_vllm/logit_lens/gemma-3-4b-it-{baseline-hf,LoRA}/stage2_logit_lens_*.json` (latest timestamp per tag) | `lora_intervention/results/ci_analysis.txt` (Qwen base+LoRA late layers + CIs; raw Qwen logit-lens not local) |
| `probing.csv` | 354 | `v3/results_vllm/probing/probing_{gemma-3-4b-it-baseline-hf,gemma-3-4b-it-LoRA,Qwen2.5-3B-Instruct-LoRA}_2k_5u.json` | `ci_analysis.txt` (Qwen base+LoRA + CIs) |
| `attention_routing.csv` | 45 | — | `lora_intervention/results/attention_routing_comparison.txt` (primacy K2/N30), `attention_routing_reversal_comparison.txt` (reversal K15/N20); raw per-head JSON not local |
| `causal_ablation.csv` | 51 | `v3/scripts/experiments/results/Qwen2.5-3B-Instruct-{baseline-promoters,LoRA-promoters,LoRA-clean5}/stage3_causal_*.json`; `lora_intervention/experiments/ablation_rerun_results/results.json` (Qwen n=200); `lora_intervention/experiments/gemma_results/ablation/results.json` (Gemma n=200) | — |

**Cross-check (not a formal QA gate, but confirms the extraction):** the n=200
paired ablation deltas recomputed from per-trial data reproduce the paper's
headline exactly — Qwen L32 paired (LoRA−base) = −0.082 (K2/N5), −0.230 (K2/N50).

**Overlap note:** Qwen-LoRA probing is in both `raw` (all 36 layers, no CI) and
`derived` (7 late layers, with CI). Filter by `provenance_tier` before
aggregating. `ci_analysis.txt` is also the only source of Qwen **base** probing.

---

## 07_cot_ivq/ — cot_ivq_nocot.csv (270) · cot_ivq_cot_thinking.csv (576)

Built by `build/build_07_cot_ivq.py`. Added 2026-09-09; split into one CSV per
arm 2026-09-09. claude-4.5-{haiku,sonnet,opus} on `flat_nolabel`, `nocot` vs
`cot_thinking`.

**Why two files.** Requested for legibility: an arm per file, each with its own
`by_model_<arm>/` split, following the theme-04 two-CSV pattern. The two are a
partition of the same 846 rows — every row appears exactly once — so unlike the
theme 2/3/4 overlap noted at the end of this file, concatenating them needs no
de-duplication. The trade-off is that the theme's headline contrast is paired
and now crosses a file boundary; `sort_key` in the builder omits `variant` so
the 270 shared rows emerge in the same order in both files, and
`check_arm_alignment` re-proves that on every build (270 keys in both, 0
nocot-only, 306 cot_thinking-only, identical order: True). The 306
cot_thinking-only rows are the ARBITRARY_SINGLE ladder, which has no no-CoT arm.

| rows | tier | source |
|---|---|---|
| 90 × 6 | raw | `experiments_cloud/results/cot_ivq/claude-{haiku-4-5-20251001,sonnet-4-5-20250929,opus-4-5-20251101}__{nocot,cot_thinking}/checkpoint.json` — SEMANTIC_MULTI, K ∈ {5,10} × N ∈ {10,20,50} |
| 102 × 3 | raw | `experiments_cloud/results/cot_ivq_arbitrary_single/claude-{haiku-4-5-20251001,sonnet-4-5-20250929,opus-4-5-20251101}__cot_thinking__ARBITRARY_SINGLE/checkpoint.json` — K=5, N ∈ {50,100,200,300,400,460} |

`checkpoint.json` is authoritative for every emitted number (accuracy, n,
n_correct, wilson_hw, CI). The sibling `trials.jsonl` in each of those 9
directories is read only to fill `n_offstream` / `accuracy_onstream` /
`error_types_json` and to cross-check the denominators — 846/846 rows agree on
`n_correct`. Row split across the two output files: 270 `nocot` (all
SEMANTIC_MULTI) and 576 `cot_thinking` (270 SEMANTIC_MULTI + 306
ARBITRARY_SINGLE).

**No ARBITRARY_SINGLE `nocot` arm exists.** That dataset is a CoT-only depth
ladder, so the arm contrast is available on SEMANTIC_MULTI only.

**Two source files in that tree are deliberately NOT used.**
`cot_ivq_arbitrary_single/_figs/ladder_summary.csv` carries no `model` column
and holds a stale haiku-only snapshot (93 of 102 cells match the finished haiku
checkpoint), so a reader would silently attribute haiku's mid-run numbers to
whichever model they assumed. The per-run copies under
`claude-{opus,sonnet}-*/_figs/` are correct but cover only 2 of 3 models.
`cot_ivq/_figs/cot_ivq_summary.csv` **is** clean and complete (540 rows,
model-labelled) — it is skipped only so there is one code path, not because it
is wrong; it makes a good independent check.

**Resume artifact, one position.** sonnet ARBITRARY_SINGLE K=5 N=400 position 6:
the checkpoint counted 56 trials, `trials.jsonl` holds 64 distinct `trial_idx`
(no duplicates). That cell was resumed, and the JSONL is append-only, so it
holds the union of attempts while the checkpoint holds the final attempt's
observations. The builder counts the checkpoint's prefix; the `n_correct`
cross-check passing on that row is what proves the prefix is the right subset.
`trials.jsonl.predup` next to it is a local-only dedup safety copy, gitignored
via `*.predup`.

**Denominator.** `n_trials` counts well-formed responses only (answer inside
`<answer>` tags). 120 malformed responses across the corpus are excluded from it
and carried as `n_malformed=` in `notes`, matching the upstream sweep — only the
thinking arm can exhaust its token budget, so folding truncation into the error
bucket would bias the arm under test.

**Off-stream taxonomy differs from theme 01.** This sweep classifies five
outcomes, so off-stream here is `garbage` alone: `out_of_context` is a real
stream value belonging to another key, hence on-stream, and `no_answer` is
already outside `n_trials`. 92 garbage in 79,552 trials (0.12%) — far below the
20% exclusion threshold above.

**Model registry.** `claude-4.5-opus` is new to `build/common.py` and appears in
no other theme. The three dated snapshot ids alias onto the canonical
`claude-4.5-*` names; the full dated id is preserved per row in `notes`, because
theme 02's claude rows come from earlier runs whose CSVs did not record a
snapshot. A theme-02 vs theme-07 difference is therefore not purely an arm effect.

**Producers** (in `experiments_cloud/`, not copied here): `cot_ivq_sweep.py`
(driver), `cot_ivq_prompts.py` (stimulus + blake2b seeding), `cot_ivq_score.py`
(extraction + 5-way classification), `cot_ivq_plot.py` /
`cot_ivq_plot_ladder.py` (figures), `run_cot_ivq_full.sh` /
`run_cot_ivq_arbitrary.sh` (invocations).

---

## 05_mechanistic/value_identity_probe/ — 4 arm directories, 6,967 layer-rows

Built by `build/build_05c_value_identity.py`. Added 2026-09-09. All rows `raw`.
Qwen2.5-3B-Instruct and gemma-3-4b-it, **base only**, cells (K=5,N=10) and
(K=10,N=5).

| rows | tier | source |
|---|---|---|
| behaviour, audit, pool | raw | `lora_intervention/experiments/linear_probing/results_probe50/{Qwen2.5-3B-Instruct,gemma-3-4b-it}_{5k_10u,10k_5u}/manifest.json` |
| `correct` + `all` fits | raw | `.../probe_fits_ref.json` (4 files) |
| `wrong` fits, 4 label sets | raw | `.../probe_fits_wrong.json` (4 files) |
| later `wrong` shards | raw | `.../probe_fits_wrong_topup.json` (3), `.../probe_fits_wrong_missing.json` (1) |

**One output directory per arm**, named exactly as its source directory, so the
mapping needs no lookup table:
`Qwen2.5-3B-Instruct_{5k_10u,10k_5u}/` and `gemma-3-4b-it_{5k_10u,10k_5u}/`.
Each holds the three subsets as three files — `probe_all.csv`,
`probe_correct.csv`, `probe_wrong.csv` — plus `summary.csv`, `behavioral.csv`
and `pool.csv` (that arm's 50-word label space). Two stacked cross-arm views sit
at the top level: `summary_all_arms.csv` (216 rows) and
`behavioral_all_arms.csv` (38 rows); they contain no rows the arm files lack.

Subset file sizes differ by design, not coverage: `probe_all` and
`probe_correct` carry the `expected` label set only, `probe_wrong` carries all
four, so it is ~4x larger.

Design doc: `lora_intervention/experiments/linear_probing/PROBE50_DESIGN.md`.
Producers: `probe50_collect.py` (activations), `probe50_fit.py` (probes),
`run_probe50_all.sh` / `run_probe50_pipeline.sh` / `finish_probe50*.sh`.

**Shard resolution is mandatory.** The wrong-subset fits are spread over up to
three files per cell, and four (condition, subset) pairs are claimed twice with
different trial counts: Qwen (10,5) CVQ at n=616 and n=1231, FVQ at 605 and 1330,
k1 at 655 and 1297, gemma (10,5) CVQ at 244 and 1231. The larger n is the
completed run, the smaller an under-powered first pass that was topped up. The
builder keeps the largest n per (condition, subset), drops the other 4, and
names the superseded shard in `notes`. Concatenating the files, or letting glob
order win, would mix sample sizes silently.

**27 fits are not estimable, and appear as rows rather than as absences.**
each arm's `summary.csv`, and `summary_all_arms.csv`, carries
`estimable=False` plus the reason. The two subsets fail at
opposite ends of accuracy: `wrong` empties where the model is too accurate (4
rows — gemma FVQ/k1, n=10..22), `correct` empties where it is too inaccurate
(23 rows — interior slots, n=0..159). A further 288 individual layer records
inside otherwise-valid fits carry null metrics and a
`min class support < folds` note; they are kept as rows so no layer curve has a
silent hole.

**Not used as a source.**
`lora_intervention/experiments/linear_probing/results/probe_layer_values_raw.csv`
is a different experiment (correctness/retrieval **AUC** on the v3 grid, base vs
LoRA), it carries **no `model` column**, and `results/extract_and_plot.py:41,90`
hardcodes `probe_Qwen2.5-3B-Instruct-{state}-v3_*` — so it is a Qwen-only view
of a two-model grid (31 cells x 7 conditions x 36 layers x 2 states = 15,624
rows exactly). See "Deliberately not consolidated" for that grid's status.

**The 17 GB of `reps_*.npy` activations stay in place and are now gitignored**
(`*/reps_*.npy` under `results_probe50/`). They are inputs to the fit, not
results: every number derived from them is in these four CSVs and in the
committed `probe_fits_*.json`. Keeping them matters for one specific reason —
the C robustness sweep was never run (`C_grid` is `[0.1]` everywhere), and with
the activations on disk that is a CPU-only re-fit rather than a GPU re-run.

**Controls.** Four label sets on the same activations: `expected`, `shuffled`,
`cv_first`, `cv_last`. `shuffled` is at chance (max top-1 0.057 vs 0.020, 0 of
1,228 layer-rows above 4x chance), so no leakage. `cv_first` **beats**
`expected` at every interior slot, so the primacy value is more decodable than
the queried one — record that with any interior claim. `cv_last` at k=N and
`cv_first` at k=1 are the same label as `expected` by construction; those rows
are flagged `DEGENERATE:` in `notes` and must not be counted as controls.

---

## 08_museum_naturalistic/ — endpoint.csv (116) · ivq.csv (96) · cot_ivq.csv (288)

Built by `build/build_08_museum.py`. Added 2026-09-09. All rows `raw`.
Dataset `museum_m0`, prompt_format `narrative_m0`. **claude-4.5-haiku on all
three sweeps; claude-4.5-opus on the `nocot` arm of `cot_ivq.csv` only** — so
that file has a two-model no-CoT arm against a one-model CoT arm, and a groupby
on `variant` alone compares different model sets. `build_08_museum.py` prints
the matrix (`qa_arm_coverage`) on every build.

| rows | tier | source |
|---|---|---|
| 116 | raw | `experiments_cloud/results/museum_endpoint/claude-haiku/sweep_20260722_161853.json` — 58 cells x {RI, PI} |
| 96 | raw | `legacy/results/museum_ivq/claude-haiku/ivq_full_20260722_173410_summary.json` — 6 cells x 12-18 positions |
| 192 | raw | `experiments_cloud/results/museum_cot/claude-haiku-4-5-20251001__{nocot,cot_thinking}/checkpoint.json` |
| 96 | raw | `experiments_cloud/results/museum_cot/claude-opus-4-5-20251101__nocot/checkpoint.json` — complete (6/6 cells, 1,200 trials), 3.9% malformed |

**Files in those directories deliberately NOT read.**
`museum_endpoint/.../sweep_partial.json` is an in-progress snapshot of the same
run, stat-for-stat identical on all 58 cells; only the final file carries
`end_time`. `sweep_20260722_152429.json` is a 3-trial smoke.
`sweep_full_*.json` duplicate the same cells plus per-trial `trial_details`.
`museum_ivq/.../ivq_smoke_*` are three smokes. All are kept as sources of
per-trial detail, none is a second source of truth.

**Why a theme and not rows added to 01 / 02 / 07.** The stimulus is the
manipulated variable. Filing these three sweeps under the themes whose questions
they mirror would scatter one comparison across three directories and repeat the
haiku-only caveat in each. The grids do not match cell-for-cell anyway.

**Correction to an earlier note.** A previous version of this file excluded the
museum runs on the grounds that "a different stimulus domain ... its positions
are not comparable with the KV grid". That was wrong as written: the `ivq` run's
own metadata names its comparator as
`plain flat_nolabel in ucurve_proprietary_results.csv` — the theme-02 data that
IS consolidated. It was built to be read against it. The real reasons to be
careful are coverage (one model) and validity (below), not domain.

### Two models were run and are NOT consolidated

| model | arm | status | why excluded |
|---|---|---|---|
| sonnet | `nocot` | completed, 18,024 attempts | **54.1% malformed** (9,757) vs 0.0% haiku / 3.9% opus. Reported accuracy 0.350 rests on the surviving 46%, a self-selected population. Fails the >=20% off-stream rule applied to seven open-weight models in theme 01. |
| sonnet | `cot_thinking` | **crashed** | Died at trial 64/200 of cell 1 of 6 after a leaked-semaphore warning; wrote no `checkpoint.json`; `trials.jsonl` holds 791 lines from that one cell. **`_runner_sonnet.log` records `END sonnet cot_thinking (exit 0)` and `DONE`** — the log claims success. Checkpoints are written per cell (`museum_cot_sweep.py:493`), so a mid-cell crash loses the cell entirely. |
| opus | `nocot` | completed cleanly | **Now consolidated (2026-09-09).** 6/6 cells, 1,200 trials, 3.9% malformed. It is what makes the museum result cross-model on the no-CoT arm, and it splits the finding: haiku reads FVQ 0.827 / CVQ 0.051 on ordinal queries while opus reads 0.631 / 0.642, so the *counting* failure is haiku-specific — but both sit on the interior floor (0.142 / 0.107), so the *interior* failure is not. Same split theme 07 found on the synthetic stimulus. |
| opus | `cot_thinking` | **never run** | A smoke now exists (below) and passes, so the full run is a spend decision, not a technical one. |

**Opus smoke, 2026-09-09.**
`experiments_cloud/results/museum_cot/claude-opus-4-5-20251101__cot_thinking__smoke/`
— `--smoke` on the hardest cell (K=10, N=50), 2 trials x 6 positions. 12/12
calls returned, **0 thinking-budget truncations**, output 310-2,715 tokens
against a 3,000 budget, 88 s, 1 of 12 `no_answer`. Mechanically clean. Projected
full-run cost from haiku's actual 17,160 calls repriced at opus rates is
**~$420 as a floor**; opus emitted 1.44x haiku's output tokens per call in the
smoke, so budget $500-600 and 3-4 h.

### Validity: read `accuracy_onstream`, not `accuracy`, for depth claims

Off-stream failure in `endpoint.csv` is asymmetric by condition — FVQ **1.0%**
(104/10,076) against CVQ **14.8%** (1,492/10,076) — and grows with N. All **17**
rows past the 20% threshold are CVQ, up to **58%** at (20,50). Correcting for it
moves the mean FVQ-CVQ gap from **+0.260 to +0.189** and flips the sign in 2 of
58 cells ((10,50) +0.267 -> -0.087; (5,50) +0.214 -> -0.026). 54/58 cells stay
positive, so the theme-01 replication survives — at +0.189 against theme 01's
+0.19 — but the raw column overstates it by ~27%. Rows are emitted uncorrected
with `n_offstream` and `accuracy_onstream` populated, so the filter is the
reader's to apply.

**7 of 96 `cot_thinking` rows are lower bounds**, not measurements: >=1 call at
that position hit the 3,000-token thinking budget. The API does not report
truncation, so the sweep derives `thinking_budget_hit` per call and
`n_thinking_budget_hit` per position; affected rows say so in `notes`.
Truncation lands on deep positions and never on shallow ones, so it depresses
exactly the numbers under test.

**Seeds are not reproducible for `endpoint` and `ivq`.** Both record
`seed_formula: abs(hash((nk,nu,t)))%(2**31)`; Python salts `hash` per process for
the string in that tuple — the defect `cot_ivq_prompts.py` moved to blake2b to
avoid. Accuracies are sound, individual stimuli are not regenerable. Every row
from those two sources carries the warning in `notes`. `cot_ivq` is clean
(`seed_version=museum_cot_v1`).

**Producers** (in `experiments_cloud/`, not copied here):
`museum_endpoint_sweep.py`, `museum_ivq_haiku.py`, `museum_cot_sweep.py`,
`museum_cot_prompts.py`, `museum_cot_score.py`, `museum_cot_refusals.py`,
`plot_museum_ivq.py`, `run_museum_cot.sh`.

## Deliberately not consolidated

Recorded so these stay findable, not because they are unimportant.

| what | where | why excluded |
|---|---|---|
| Mechanistic per-layer results — logit lens, probing classifiers, attention routing, promoter-head ablation | `v3/results_vllm/{logit_lens,probing,attention_routing}/`, `v3/scripts/experiments/results/`, `lora_intervention/results/*_comparison.txt`, `lora_intervention/experiments/*/ablation/` | This folder covers **behavioural** retrieval only. Where a source file held both, only the behavioural part was extracted and `notes` says so. |
| Legacy ACL-era PI/RI study — 39 API models | `results/` (top level), `results/raw_{ri,pi}_*`, `results/key_results/` | Different task definition and prompt design from the current FVQ/CVQ generation. Mixing them would produce misleading cross-era comparisons. |
| Pre-v3 probing | `mechanistic_probing_v2/` | Superseded. **Do not delete** — `core/dataset_configs.py` is a live import for `lora_intervention/data_gen.py`. |
| Narrative-domain variant (Dota2 / ATC / ICU) | `narrative_generator/`, `data/narrative_interference/` | Abandoned line; see paper §Limitations. |
| Older non-vLLM v3 runs | `v3/results/` | Superseded by `v3/results_vllm/`. |
| v3 correctness / retrieval **AUC** probe grid | `lora_intervention/experiments/linear_probing/results/` — 124 JSONs (Qwen + gemma, base + LoRA, 31 cells x 7 conditions x ~36 layers), plus 36 earlier pilots and `isoaccuracy_v3*_summary.json` | A different probe question (is *correctness* decodable) and a different metric (AUC) from the value-identity probe above, and it degenerates wherever accuracy is 0% or 100% — 339 of 868 condition-cells per `PROBE50_DESIGN.md`. `05_mechanistic/probing.csv` holds a 354-row slice of this family from a different source, covering only the (2,5) cell. Consolidating the full grid is a live option; if taken, build from the 124 JSONs, not from `probe_layer_values_raw.csv`. |
| Museum sweeps — sonnet (both arms) and opus `cot_thinking` | `experiments_cloud/results/museum_cot/claude-sonnet-4-5-20250929__{nocot,cot_thinking}/` | **Supersedes an earlier blanket exclusion of all museum runs.** The haiku arms are now theme 08. Sonnet `nocot` is 54.1% malformed and sonnet `cot_thinking` crashed after one partial cell. Opus `nocot` is now consolidated; opus `cot_thinking` was never run beyond a smoke. Per-arm detail in the theme-08 section above. |
| Training-dynamics checkpoint sweeps (SmolLM2 42 ckpts, SmolLM3 37 ckpts) | raw not local; survives as `paper/figures/tab_smollm2_traj.tex`, `tab_smollm3_traj.tex`, `fig3_training_dynamics.*` | Not one of the four themes. The SmolLM3 *format* slice **is** included (theme 3). |

## Related maps

- `REPO_MAP.md` — which top-level folders belong to which generation of the project
- `AAAI_PREP_PLAN.md` §1.3 — missing raw data and re-run costs; §1.4 — adapter inventory
- `paper/scripts/*.py` — each generator's docstring names the raw inputs behind its table
- `findings/` — the open-issues ledger (002–005 still open)
