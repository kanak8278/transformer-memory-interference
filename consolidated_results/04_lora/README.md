# 04 — LoRA fine-tuning

**Question.** Can the retrieval gap be trained away? On which models, with what
configuration, and does the fix generalise beyond what it was trained on?

| | |
|---|---|
| files | `adapters.csv` (3 adapters), `lora_evals.csv` (480 rows, endpoint held-out), `lora_ivq.csv` (1,896 rows, per-position base-vs-LoRA) |
| per-model slices | `by_model/` — 2 files (generated; do not edit) |
| models fine-tuned | Qwen2.5-3B-Instruct, Gemma-3-4b-it |
| tiers | 448 `raw`, 32 `derived` |

---

## Which models were fine-tuned, and how

All three adapters share the same hyperparameters: **rank 16, α 32, lr 2e-4,
attention-only targets (`q_proj, k_proj, v_proj, o_proj`)**.

| adapter | base model | role | data | epochs | size |
|---|---|---|---|---|---|
| `checkpoints/adapter` (run `main`) | Qwen2.5-3B-Instruct | §5.2 main result | KV task, K∈{2,3,5,10} × N∈{5,10,15,20} | 2.0 (complete) | 29.5 MB |
| `checkpoints/gemma_adapter` (run `gemma_main`) | Gemma-3-4b-it | cross-family control | same grid, regenerated | 1.42 — **stopped at step 400/564** | 35.7 MB |
| `checkpoints/qwen_arith_adapter` (run `qwen_arith_control`) | Qwen2.5-3B-Instruct | negative control | **GSM8K**, no KV data | 0.95 — **stopped at step 100/212** | 29.5 MB |

Two things worth knowing before citing these:

- **The K,N training grid is not recorded in any `run_config.json`.** It lives
  in code (`lora_intervention/data_gen.py`: `TRAIN_KEYS`, `TRAIN_UPDATES`) and
  is copied into `adapters.csv` from there. The referenced `data/*.jsonl` files
  are not local, though `data_gen.py` regenerates them deterministically.
- **Two of three adapters stopped early**, each with a recorded reason (in
  `stop_reason`). Gemma's eval loss was still improving at step 400.
- A fourth **MLP-only LoRA** (App L) exists in the paper but **not in this repo**
  — it was never copied off the GPU box. It is absent from `adapters.csv`.

## Evaluation results, aggregated over cells

Mean FVQ / CVQ across all evaluated (K, N) cells:

| model | variant | dataset | cells | FVQ | CVQ | gap |
|---|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct | base | arbitrary_single | 28 | 0.24 | 0.53 | −0.29 |
| Qwen2.5-3B-Instruct | **lora** | arbitrary_single | 28 | **1.00** | **0.98** | +0.02 |
| Qwen2.5-3B-Instruct | base | semantic_multi | 4 | 0.36 | 0.64 | −0.28 |
| Qwen2.5-3B-Instruct | **lora** | semantic_multi | 4 | **1.00** | **1.00** | +0.00 |
| Qwen2.5-3B-Instruct | lora_dense_ivq | arbitrary_single | 28 | 0.98 | 0.88 | +0.10 |
| Qwen2.5-3B-Instruct | *lora_arith_control* | arbitrary_single | 28 | 0.38 | 0.24 | +0.14 |
| Gemma-3-4b-it | base | arbitrary_single | 26 | 0.89 | 0.21 | +0.68 |
| Gemma-3-4b-it | **lora** | arbitrary_single | 28 | **0.99** | **0.97** | +0.02 |
| Gemma-3-4b-it | base | semantic_multi | 16 | 0.91 | 0.54 | +0.37 |
| Gemma-3-4b-it | **lora** | semantic_multi | 16 | **1.00** | **0.97** | +0.03 |
| Gemma-3-4b-it | lora_smoke | arbitrary_single | 28 | 1.00 | 0.96 | +0.04 |

**The 28 held-out ARB cells all reach ≥92.5% CVQ** on both models — none of the
28 is left behind, and none of the held-out cells is inside the training grid
(training capped at K=10, N=20; evaluation starts at K=7, N=30).

**It transfers across distributions.** The adapters were trained only on
`arbitrary_single`; they hit 0.97–1.00 on `semantic_multi`, which they never saw.

**It replicates across families.** Gemma starts from a very different place
(+0.68 gap vs Qwen's −0.29) and lands in the same one (+0.02).

## Note on the Qwen baseline sign

Qwen's baseline gap over these 28 cells is **−0.29** — CVQ *above* FVQ. That is
not an error: the 28 held-out cells are all high-K (7–30), and Qwen inverts in
the high-K reversal regime (see `../01_fvq_cvq/README.md`). Gemma at the same
cells is +0.68. **The LoRA result is "both models converge on ~1.00/~0.97",
not "both models' positive gaps shrink".** Any summary phrased as gap-reduction
misreports the Qwen half.

## The negative control is the important row

`qwen_arith_control` was trained on **GSM8K** — real arithmetic reasoning, no
key-value data — with otherwise identical LoRA hyperparameters.

| | GSM8K accuracy | ARB cells fixed |
|---|---|---|
| Qwen2.5-3B-Instruct base | 0.160 | — |
| + arithmetic LoRA | **0.696** | **0 / 28** |
| + task LoRA (`main`) | — | **28 / 28** |

The control adapter demonstrably learned its own task (a 4.4× improvement) and
moved CVQ *down*, from 0.53 to 0.24. **The gap-closing is task-specific, not a
generic consequence of fine-tuning anything with LoRA.** This is the single
most load-bearing row in the folder.

## Per-position IVQ — `lora_ivq.csv` (moved here 2026-07-22)

The interior-retrieval comparisons that used to sit in theme 02. They are
base-vs-LoRA by construction, so they belong here. **2 models, 3 experiments:**

| experiment | Qwen2.5-3B-Instruct | gemma-3-4b-it | what it asks |
|---|---|---|---|
| **behavioral** (base_plain / base_block / lora_plain, 8 cells) | ✅ | ✅ | does the fix flatten the U-curve, and do block-format vs LoRA do it the same way? |
| **E1 scan A/B** (main adapter past its training grid, 9 cells) | ✅ | ✅ | is the LoRA fix general position-indexing or just pushed-out endpoints? |
| **E1 scan C** (big-cell interior sweep, K15–30 × N10–50, 20 cells) | ✅ | ✅ | separate K from N: does interior retrieval decay with N, K, or both? |
| **dense_ivq** (a separate dense-trained adapter, 6 cells, 3 splits) | ✅ | ❌ | does training on interior positions surface interior retrieval? |

behavioral and E1 share **identical grids** across the two models — a clean
cross-family replication. dense_ivq is Qwen-only and its adapter
(`checkpoints/dense_ivq/`) is **not local**; `variant=lora_dense_ivq`
distinguishes it from the main adapter.

**The finding (the money result of the whole project):** endpoints and interior
diverge. Mean interior (IVQ) accuracy, plain prompt:

| | base | **+ block format** | **+ LoRA** |
|---|---|---|---|
| Qwen2.5-3B-Instruct | 0.03 | **0.79** | **0.29** |
| gemma-3-4b-it | 0.02 | **0.81** | **0.20** |

Block formatting and LoRA both drive the *endpoints* to ~1.00, but only block
formatting lifts the *interior*. LoRA produces a boundary-anchored model that
memorised how to hit the two ends — E1 confirms it: interior accuracy collapses
0.56 → 0.07 as N grows to 100 while endpoints hold. **N is the killer, not K**,
and this replicates across both families.

**Scan C isolates that claim.** The big-cell interior sweep (K15–30 × N10–50,
run on an H100 at the canonical 200-trial policy, `bigcell_ivq/out/`) varies K
and N independently. LoRA's endpoints stay at ~1.00 across the whole sweep, but
its interior (IVQ) is **flat in K and collapses in N**: at N=10 the LoRA interior
holds ~0.61–0.84 for every K∈{15,20,25,30}, but at fixed K it decays with N to
~0.12 (Qwen) / ~0.20 (Gemma) by N=50. Base gemma reprises the headline PI story
even out here — FVQ ~0.85–1.00 but CVQ decaying 0.68 → 0.04 as N grows — and LoRA
closes that endpoint gap to ~1.0 while leaving the interior frontier open.

`variant` values here: `base`, `lora` (main adapter), `lora_dense_ivq`.
`split` marks E1 rows (`extrapolation`) and dense-IVQ splits
(`train`/`held_moderate`/`held_hard`). Per-model slices in `by_model_ivq/`.

## What LoRA does not buy

Endpoint accuracy is not position indexing. The same adapters reach only
**0.29 (Qwen) / 0.20 (Gemma)** mean accuracy on interior positions, and E1 shows
interior performance collapsing to 0.05–0.07 as N grows to 100 while the
endpoints hold at ~1.00. See `../02_intermediate_ivq/README.md`. The
`lora_dense_ivq` variant — trained on *all* positions, not just endpoints — is
the attempt to fix this; its per-position results are in `ivq.csv`.

## Row-level notes

- **`variant` distinguishes runs, not just adapters.** `lora` and `lora_smoke`
  share the Gemma adapter but are different evaluations with different numbers;
  `lora_arith_control` (28 cells) and `lora_arith_control_5cell` (the earlier
  preview) share the arithmetic adapter. They were separated after a
  cross-check caught them overwriting each other.
- **`base` rows come from the raw stage-1 sweeps**, not from the integer
  percents in `*_comparison.txt` — so they carry full precision and CIs. The
  `.txt` files are used as an independent check instead: **564 values checked,
  0 mismatched.**
- **Gemma has no baseline at (25,75) and (30,75)** — the stage-1 sweep never
  covered those cells, which is why `gemma_main_comparison.txt` prints `---`.
  Those are the 4 "unavailable" entries in the QA output, and why Gemma's
  baseline row count is 26 while its LoRA row count is 28.
- **Trial counts are 40–80 per cell with early stopping**, not the 100 in
  `max_trials`. `notes` flags early-stopped cells. This is currently
  undocumented in the paper (see `findings/004`).
- **`derived` rows (32)** are the **Gemma** `semantic_multi` baseline,
  back-extracted from `paper/figures/tab_lora_gemma_sem.tex`, because that raw
  sweep is not local. The Gemma SEM *LoRA* side is raw (it came from
  `gemma_main_eval_*.json`); only its baseline is derived.
- **The 16 Qwen `semantic_multi` rows were `derived` until 2026-09-07 and are
  now `raw`.** They were being back-extracted from `tab_lora_sem.tex` on the
  stated grounds that the raw run was not local. It is:
  `lora_sem_validation_results.json`, git-tracked since before this folder
  existed. Switching the builder to read it changed **no value** (16/16 exact
  against the table) but recovered `n_trials` / `n_correct`, which the 2dp table
  could not carry — and those turn out to matter, because the LoRA arm is n=20
  on (7,20) and (15,20) but **n=10** on (15,30) and (20,30). All four cells read
  1.00, so the table made two 10-trial cells look like the other two.
  That file, and the `lora_sem_finish.py` re-run that completed it, are the only
  local record of those counts, and they sit loose at the repo root — so
  byte-identical copies are kept in `sources/`. `source_file` still points at
  the canonical root path; the builder falls back to the copy only if the root
  file disappears, and prints a note when it does.

## Out of scope here

The mechanistic evidence for *how* LoRA closes the gap — promoter-head
ablation, attention routing, per-layer logit lens, probing classifiers — is not
consolidated in this folder. It stays in
`lora_intervention/results/*_comparison.txt`,
`lora_intervention/experiments/*/ablation/`, and `v3/results_vllm/`.

## Sources

See `../PROVENANCE.md`, or the docstring of `../build/build_04_lora.py`.
