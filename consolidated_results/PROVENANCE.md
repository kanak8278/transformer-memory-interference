# Provenance

Every consolidated file, and every source file it was built from. Row counts are
as of the 2026-07-22 build (theme 04 rebuilt 2026-09-07). The CSVs reference
their sources in place, via the `source_file` column on each row.

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

## Deliberately not consolidated

Recorded so these stay findable, not because they are unimportant.

| what | where | why excluded |
|---|---|---|
| Mechanistic per-layer results — logit lens, probing classifiers, attention routing, promoter-head ablation | `v3/results_vllm/{logit_lens,probing,attention_routing}/`, `v3/scripts/experiments/results/`, `lora_intervention/results/*_comparison.txt`, `lora_intervention/experiments/*/ablation/` | This folder covers **behavioural** retrieval only. Where a source file held both, only the behavioural part was extracted and `notes` says so. |
| Legacy ACL-era PI/RI study — 39 API models | `results/` (top level), `results/raw_{ri,pi}_*`, `results/key_results/` | Different task definition and prompt design from the current FVQ/CVQ generation. Mixing them would produce misleading cross-era comparisons. |
| Pre-v3 probing | `mechanistic_probing_v2/` | Superseded. **Do not delete** — `core/dataset_configs.py` is a live import for `lora_intervention/data_gen.py`. |
| Narrative-domain variant (Dota2 / ATC / ICU) | `narrative_generator/`, `data/narrative_interference/` | Abandoned line; see paper §Limitations. |
| Older non-vLLM v3 runs | `v3/results/` | Superseded by `v3/results_vllm/`. |
| Training-dynamics checkpoint sweeps (SmolLM2 42 ckpts, SmolLM3 37 ckpts) | raw not local; survives as `paper/figures/tab_smollm2_traj.tex`, `tab_smollm3_traj.tex`, `fig3_training_dynamics.*` | Not one of the four themes. The SmolLM3 *format* slice **is** included (theme 3). |

## Related maps

- `REPO_MAP.md` — which top-level folders belong to which generation of the project
- `AAAI_PREP_PLAN.md` §1.3 — missing raw data and re-run costs; §1.4 — adapter inventory
- `paper/scripts/*.py` — each generator's docstring names the raw inputs behind its table
- `findings/` — the open-issues ledger (002–005 still open)
