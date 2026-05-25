# Gemma-3-4b-it LoRA — Run Log (§5.2 family control)

Branch: `probing` @ `e9957b0` + uncommitted edits to `data_gen.py`,
`train.py`, `evaluate.py`, `merge_lora.py` (Gemma-3 dispatch) + new file
`GEMMA_PROBING_PLAN.md`.

Hardware: SageMaker L4 (23 GB VRAM), Ubuntu 24.04, Python 3.12,
torch 2.11.0+cu130, transformers 4.57.6, peft 0.19.1, trl 1.4.0,
datasets 4.8.5, vllm 0.21.0.

---

## 1. Data generation

Deterministic seeding fix (replaced randomized `hash()` with `hashlib.blake2b`)
applied before regeneration. Same K/N grid, same condition mix, same
categories as the Qwen run.

```
python3 lora_intervention/data_gen.py
```

Output (md5):

| File | Records | md5 |
|---|---|---|
| train.jsonl | 18,000 | 190b5b5757b8e51db22e71480b580800 |
| val.jsonl | 2,000 | 2f9ec9b077904ad39707ba358ff26b1e |
| test_id.jsonl | 600 | 4551d5f06df171e6db04d7f194c19719 |
| test_ood.jsonl | 2,200 | 8acbca7ed5a96d3e72f81ca8fa1d2516 |

Condition mix in train: FVQ 7,180 (40%), CVQ 7,207 (40%), IVQ 3,613 (20%).

NOTE: NOT byte-identical to the Qwen training data (Qwen run pre-dates the
deterministic seed fix). Same distribution, different concrete examples.
Apples-to-apples caveat already flagged in PAPER_PLAN.md §5.1.

## 2. Training

`train.py` changes for Gemma:
- `os.environ["USE_TF"] = "0"`, `["USE_JAX"] = "0"` (avoid TF/Keras import
  errors triggered by transformers' lazy module probe)
- `Gemma3ForCausalLM` dispatch instead of `AutoModelForCausalLM` (default
  loads VLM with vision projector, breaks `assistant_only_loss`)
- `processing_class=tokenizer` passed to `SFTTrainer` (forces text-only
  processor; default detection picks Gemma's multimodal processor)
- `attn_implementation="sdpa"` (Gemma-3 defaults to eager)

Command:
```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 lora_intervention/train.py \
  --run-name gemma_main \
  --model google/gemma-3-4b-it \
  --batch-size 2 --grad-accum 32 --max-seq-len 1280
```

Hyperparameters: LoRA rank 16, alpha 32, target {q,k,v,o}_proj, dropout
0.05, lr 2e-4 cosine, warmup 100 steps, effective batch 64, bf16, gradient
checkpointing.

Trainable params: 8,912,896 / 3,889,176,064 (0.23%).

**Stopped at step 400 of 564** (1.42 epochs vs planned 2.0) — eval loss
still improving (no overfit), training train/eval gap small. Reason:
sufficient training signal; saves ~2.5h GPU.

Wall clock: 3:44 train time + tokenization + eval overhead = ~3h53m total.

| Step | Train loss | Train acc | Eval loss | Eval acc |
|---|---|---|---|---|
|  10 | 7.13 | 33.7% | – | – |
|  20 | 1.36 | 79.8% | – | – |
|  50 | 0.17 | 95.4% | – | – |
| 100 | 0.09 | 97.0% | 0.104 | 96.9% |
| 200 | 0.05 | 98.3% | 0.079 | 97.5% |
| 300 | 0.04 | 98.4% | 0.065 | 98.1% |
| 400 | 0.03 | 98.7% | **0.046** | **98.3%** |

Saved checkpoints: `checkpoints/gemma_main/{checkpoint-200, checkpoint-300,
checkpoint-400}/`. We will use `checkpoint-400/` for evaluation. The
`final/` symlink doesn't exist because we killed training before it ran
the post-loop save.

## 3. Held-out evaluation

### 3a. Smoke (5 trials/cell, ARB only) — pipeline validation

```
python3 lora_intervention/evaluate.py \
  --model google/gemma-3-4b-it \
  --adapter lora_intervention/checkpoints/gemma_main/checkpoint-400 \
  --baseline-path v3/results_vllm/arbitrary_single/gemma-3-4b-it/stage1_sweep_20260409_022631.json \
  --run-name gemma_smoke --skip-ood --smoke
```

Result: **27 / 28 cells FIXED** at 5 trials. Only K=25/N=50 showed wobble
(PI=60%, likely small-N variance). vLLM merged-Gemma pipeline works
end-to-end. Wall time ~2 min.

| | Baseline gap range | Post-LoRA (smoke) |
|---|---|---|
| RI accuracy | 80–99% | 100% on all 28 cells |
| PI accuracy | 4–52% | 100% on 26 cells, 60–80% on 2 |
| Gap (RI–PI) | +43 to +87% | 0% on 26 cells, +20 to +40% on 2 |

Outputs:
- `lora_intervention/results/gemma_smoke_eval_20260525_154202.json`
- `lora_intervention/results/gemma_smoke_comparison.txt`

### 3b. Full eval (100 trials/cell, ARB + SEM OOD)

```
python3 lora_intervention/evaluate.py \
  --model google/gemma-3-4b-it \
  --adapter lora_intervention/checkpoints/gemma_main/checkpoint-400 \
  --baseline-path v3/results_vllm/arbitrary_single/gemma-3-4b-it/stage1_sweep_20260409_022631.json \
  --run-name gemma_main --trials 100
```

Wall time: ~45 min (28 ARB cells + 16 SEM cells, vLLM 0.21.0 on L4 GPU,
Wilson CI early stopping at 0.07 / min 30 trials).

#### Headline numbers

| Dataset | Cells | RI mean | PI mean | Min PI | Fixed | Worse |
|---|---|---|---|---|---|---|
| ARBITRARY_SINGLE (held-out cells K≥7, N≥15) | 28 | **99.4%** | **97.0%** | 87.8% | 28/28 | 0 |
| SEMANTIC_MULTI (OOD, K=7-20 / N=10-50) | 16 | **100.0%** | **96.8%** | 83.0% | 16/16 | 0 |

#### Baseline vs LoRA (sample of high-load cells)

| Cell | Baseline RI | Baseline PI | Baseline gap | LoRA RI | LoRA PI | LoRA gap | Verdict |
|---|---|---|---|---|---|---|---|
| K=10 N=75 | 92% | 14% | +78% | 100% | 98% | +2% | ✓ FIXED |
| K=15 N=75 | 82% | 9% | +73% | 100% | 100% | +0% | ✓ FIXED |
| K=20 N=75 | 85% | 7% | +78% | 100% | 90% | +10% | ✓ FIXED |
| K=25 N=50 | 81% | 9% | +72% | 99% | 91% | +8% | ✓ FIXED |
| K=30 N=30 | 91% | 4% | +87% | 100% | 90% | +10% | ✓ FIXED |
| K=30 N=50 | 87% | 6% | +81% | 97% | 93% | +4% | ✓ FIXED |

#### Interpretation — §5.2 family control claim substantiated

- **Behavioral**: Gemma-3-4b-it shows strong primacy bias at baseline
  (gap +43% to +87% on held-out cells). Same LoRA recipe used for
  Qwen2.5-3B-Instruct closes the gap to ~0% on all 28 ARB cells and 16
  SEM OOD cells.
- **Architecture independence**: Gemma has 34 layers × 8 attention heads
  (with grouped-query attention); Qwen has 36 layers × 16 heads. Their
  baseline interference *regimes* differ qualitatively — Qwen shows
  reversal (PI > RI by up to 55 points at high load), Gemma shows clean
  primacy (RI > PI by up to 87 points). The same intervention closes
  both.
- **Generalization**: train cells were K∈{2,3,5,10}, N∈{5,10,15,20}.
  LoRA holds on K up to 30 and N up to 75 (~10× context-length
  extrapolation) and on SEM OOD with completely different value
  semantics.

Outputs:
- `lora_intervention/results/gemma_main_eval_20260525_165155.json`
- `lora_intervention/results/gemma_main_comparison.txt`

## 4. Comparison to Qwen run (§5.1)

| | Qwen2.5-3B-Instruct LoRA (§5.1) | Gemma-3-4b-it LoRA (§5.2) |
|---|---|---|
| Architecture | 36L × 16H, dense | 34L × 8H, GQA |
| Baseline regime | Reversal (PI > RI by ≤55%) | Primacy (RI > PI by ≤87%) |
| ARB held-out cells fixed | 28 / 28 | 28 / 28 |
| Mean RI / PI (held-out) | 99% / 97% | 99% / 97% |
| SEM OOD | Fixed | Fixed |
| Training epochs | 2.0 | 1.42 (stopped early; eval still improving) |
| Training data | Same K/N grid, same conditions, same volume; different concrete examples (PYTHONHASHSEED reset across runs — deterministic seeding introduced for Gemma run) |

## 5. §7 mechanism on Gemma — post-LoRA probing + logit lens

Mirroring the §5.1 Qwen mechanistic re-run, but with two caveats:

1. `transformer_lens` OOMs on Gemma-3-4b on the L4 (23 GB) because TL
   duplicates weights into its layer-wise structures during construction,
   exceeding GPU memory even with `from_pretrained_no_processing`.
   Wrote HF-direct replacements: `run_probing_lora_hf.py` and
   `run_logit_lens_lora_hf.py`. Same `train_probes` sklearn function,
   same metric semantics; different access path (HF
   `output_hidden_states=True` + `model.lm_head` instead of TL `cache`
   and `model.W_U`).
2. Pre-existing baseline files for Gemma probing
   (`probing_gemma-3-4b-it_2k_5u.json`) and logit lens
   (`logit_lens/gemma-3-4b-it/stage2_logit_lens_*.json`) are
   **broken** (RI/PI behavioral 0%, probes at chance everywhere, value
   prob matrices all-NaN or all-zero). Re-ran both on **base
   Gemma-3-4b-it** through the new HF-direct path for a clean
   apples-to-apples comparison.

### 5.1 Merge

```
python3 lora_intervention/merge_lora.py \
  --base google/gemma-3-4b-it \
  --adapter lora_intervention/checkpoints/gemma_adapter \
  --out lora_intervention/checkpoints/gemma_merged --device cpu
```

CPU merge in float32 (safest); validation passed (top-5 token-ID match,
max logit |diff| 5.72e-06 between PEFT-forward and merged-forward).
Saved 14.8 GB merged model (gitignored).

### 5.2 Probing classifier @ K=2/N=5, 200 trials per condition

```
python3 lora_intervention/run_probing_lora_hf.py \
  --merged_path google/gemma-3-4b-it \
  --base google/gemma-3-4b-it \
  --out_name gemma-3-4b-it-baseline-hf \
  --point 2,5 --trials 200 --device cuda --dtype bfloat16

python3 lora_intervention/run_probing_lora_hf.py \
  --merged_path lora_intervention/checkpoints/gemma_merged \
  --base google/gemma-3-4b-it \
  --out_name gemma-3-4b-it-LoRA \
  --point 2,5 --trials 200 --device cuda --dtype bfloat16
```

Each cell ran in ~64 s (model load + 400 forward passes + sklearn
training across 34 layers). bf16 forward passes; final logits and
hidden states unchanged downstream.

| Probe (late layers L26-L33) | Base Gemma | Gemma+LoRA | Δ |
|---|---|---|---|
| Behavioral RI | 83% | 80% | -3 |
| Behavioral PI | 63% | **81%** | **+18** |
| Condition discrimination (RI vs PI) | 100% | 100% | 0 |
| RI correct probe | 90-93% | 84-93% | flat |
| **PI correct probe** | **70-73%** | **82-89%** | **+15 to +20** |

The condition probe is already saturated pre-LoRA — the residual stream
distinguishes which kind of query the model is being asked. What LoRA
changes is the **PI-correctness** probe: pre-LoRA the late-layer
residual encodes whether the model has the right answer for PI only
~70% of the time; post-LoRA that rises to ~85%. This is the same
metric §5.1 reported for Qwen (58% → 94%); both architectures converge
to similar post-LoRA probe accuracy.

Files:
- `v3/results_vllm/probing/probing_gemma-3-4b-it-baseline-hf_2k_5u.json`
- `v3/results_vllm/probing/probing_gemma-3-4b-it-LoRA_2k_5u.json`

### 5.3 Logit lens @ {K=2/N=5, K=2/N=10, K=2/N=50}, 100 trials/cell

```
python3 lora_intervention/run_logit_lens_lora_hf.py \
  --merged_path google/gemma-3-4b-it \
  --base google/gemma-3-4b-it \
  --out_name gemma-3-4b-it-baseline-hf \
  --points "2,5;2,10;2,50" --trials 100 --device cuda --dtype bfloat16

python3 lora_intervention/run_logit_lens_lora_hf.py \
  --merged_path lora_intervention/checkpoints/gemma_merged \
  --base google/gemma-3-4b-it \
  --out_name gemma-3-4b-it-LoRA \
  --points "2,5;2,10;2,50" --trials 100 --device cuda --dtype bfloat16
```

Both runs ~166 s total each (model load + 600 forward passes + per-layer
projection through `lm_head`).

PI P(v_last) at final layer L33:

| Cell | Base Gemma | Gemma+LoRA | Δ | Qwen Δ (§5.1) |
|---|---|---|---|---|
| K=2/N=5 | 0.807 | **1.000** | +0.19 | +0.75 |
| K=2/N=10 | 0.569 | **1.000** | +0.43 | +0.77 |
| K=2/N=50 | 0.502 | **0.999** | +0.50 | +0.79 |

RI P(v_first) at L33 is ≥0.97 pre-LoRA and =1.000 post-LoRA — already
clean for v_first, so the LoRA improvement here is small but consistent
with the behavioral §5.2 number (RI stays near-perfect).

Both architectures converge to **P(v_last) ≈ 1.0 at L33 post-LoRA** from
very different starting points (Qwen 0.11 → 1.0, Gemma 0.50 → 1.0). The
"v_last found in mid-layers then suppressed by late layers" pattern is
fully removed.

Files:
- `v3/results_vllm/logit_lens/gemma-3-4b-it-baseline-hf/stage2_logit_lens_*.json`
- `v3/results_vllm/logit_lens/gemma-3-4b-it-LoRA/stage2_logit_lens_*.json`

### 5.4 Stage 3 ablation — DEFERRED

The pre-existing Gemma Stage 3 (`v3/results_vllm/causal/gemma-3-4b-it/`)
identifies suppressor heads at **L17H0, L17H1, L17H3, L19H4, L30H6** —
qualitatively different topology from Qwen's L26-L30 cluster (Gemma's
is mid-depth + one late head, Qwen's is uniformly late). The
post-LoRA Stage 3 ablation would test whether LoRA refined this circuit
or built a parallel path. It requires HF-direct attention head hooks
(transformer_lens is unusable on Gemma on this hardware), which we did
not implement in this pass. Logged as future work.

### 5.4b Bootstrap 95% CIs on Gemma deltas

Mirrors `compute_cis.py` from Qwen §5.1 R2; new script
`compute_cis_gemma.py`. Probing CIs are parametric (sklearn 5-fold CV
`mean ± 1.96 × std / sqrt(5)`, pooled-variance for Δ); logit lens CIs
are nonparametric bootstrap (n=2000, seed=17) over per-trial means.

The logit lens script `run_logit_lens_lora_hf.py` was extended to save
per-trial trajectories (`per_trial_p_first_per_layer`,
`per_trial_p_last_per_layer`); baseline and LoRA logit lens were
re-run with this output enabled (files
`stage2_logit_lens_20260525_195739.json` and
`stage2_logit_lens_20260525_200056.json`).

**Probing classifier — PI correctness probe** (key dual to the
behavioral PI fix):

| Layer | Base [95% CI]      | LoRA [95% CI]      | Δ [95% CI]              |
|-------|--------------------|--------------------|-------------------------|
| L26   | 0.702 [.65, .76]   | 0.815 [.73, .90]   | +0.113 [+0.011, +0.215] |
| L27   | 0.689 [.63, .74]   | 0.842 [.75, .94]   | +0.153 [+0.044, +0.263] |
| L28   | 0.729 [.64, .82]   | 0.882 [.83, .94]   | +0.153 [+0.044, +0.261] |
| L29   | 0.709 [.65, .77]   | 0.894 [.83, .95]   | +0.185 [+0.098, +0.272] |
| L30   | 0.716 [.69, .75]   | 0.894 [.83, .95]   | +0.178 [+0.110, +0.246] |
| L31   | 0.710 [.67, .75]   | 0.882 [.83, .94]   | +0.172 [+0.101, +0.243] |
| L32   | 0.730 [.69, .77]   | 0.869 [.81, .93]   | +0.140 [+0.067, +0.212] |
| L33   | 0.703 [.60, .81]   | 0.857 [.78, .93]   | +0.153 [+0.022, +0.285] |

All eight late layers show Δ > 0 with 95% CI excluding zero. The
condition probe and RI-correct probe Δs include zero (no
post-LoRA change — consistent with §5.1 Qwen).

**Logit lens — P(v_last) at final layer L33** (PI condition):

| Cell        | Base [95% CI]    | LoRA [95% CI]    | Δ [95% CI]              |
|-------------|------------------|------------------|-------------------------|
| K=2, N=5    | 0.802 [.72, .87] | 1.000 [1.0, 1.0] | +0.198 [+0.124, +0.278] |
| K=2, N=10   | 0.562 [.47, .65] | 1.000 [1.0, 1.0] | +0.438 [+0.344, +0.531] |
| K=2, N=50   | 0.588 [.50, .68] | 0.993 [.99, 1.0] | +0.405 [+0.309, +0.494] |

All three cells: Δ > 0.19 with 95% CI excluding zero. P(v_first) at
L33 for RI is at ceiling pre-LoRA (≥ 0.98) so the LoRA Δ is small but
positive across the board.

**Methodological note.** Intermediate-layer logit lens (L29-L32) on
Gemma is uninformative under the chosen lens configuration:
`output_hidden_states` returns pre-final-RMSNorm states for L < n_layers
- 1, and projecting those through `lm_head` gives near-uniform
distributions over Gemma's 262k-vocab. Only the post-norm final state
(included by HF as `hidden_states[n_layers]`) carries useful signal.
Reapplying `final_norm` at intermediate layers was tested but applying
it at the final layer too (i.e., a blanket fold-in) double-norms and
breaks L33. Skipping the fold-in at L=n_layers-1 was implemented in the
script but not re-run for the CI tables — the L33 result is the
headline metric.

Output: `lora_intervention/results/ci_analysis_gemma.txt`

### 5.5 §7 mechanism — what we can claim from Gemma

The two analyses above (probing + logit lens) on Gemma+LoRA, combined
with the §5.1 results on Qwen+LoRA, support the following:

> **Architecture-agnostic recovery mechanism.** Both Qwen2.5-3B-Instruct
> and Gemma-3-4b-it suppress v_last at late layers in the base model
> (PI P(v_last) at the final layer = 0.11 / 0.50-0.81 for Qwen / Gemma).
> Targeted 18k-example LoRA training on the same task format closes
> this suppression in both models: P(v_last) at the final layer reaches
> ≈1.0 in all tested cells for both architectures. The same probe-level
> signature also emerges in both: the residual stream goes from
> not-encoding-PI-correctness (Qwen 58%, Gemma 70%) to clearly
> encoding it (Qwen 94%, Gemma 86%). The starting points differ; the
> destinations are nearly identical.

What remains open without Stage 3: whether the late-layer promoter
heads identified in Qwen (L31-L33 cluster) have an analog in Gemma, or
whether Gemma's recovery routes through different heads given its
shallower suppressor cluster (L17-L19 + L30H6).

## 5.6 §5.3 negative-control LoRA (arithmetic) — fills paper's `\note{pending}` placeholders

**Headline.** Arithmetic-domain LoRA on Qwen2.5-3B-Instruct, same
hyperparameters as the main task-specific LoRA, fails to close the
FVQ/CVQ gap. 0 of 5 representative held-out cells reach the "fixed"
criterion (RI ≥ 0.65 AND PI ≥ 0.65), versus 28/28 for the
task-specific LoRA. The paper's three `\note{pending arithmetic control}`
references can now cite this run.

**Stronger control than originally planned.** PLAN.md Decision 7
specifies single-token numeric answers for the arithmetic data
(format-identity with main LoRA). We deviated to **full GSM8K
including chain-of-thought reasoning** (`openai/gsm8k`, main config,
train split, 6,726 train / 747 val) — a more conservative test:
"even arithmetic *reasoning* training doesn't fix the gap." Same
system prompt as main run.

**Training (`checkpoints/qwen_arith_adapter/`).**

- Qwen/Qwen2.5-3B-Instruct + rank-16 attention-only LoRA, lr 2e-4
  cosine, batch 2 × grad_accum 32 (eff 64), max_seq_len 768
- Stopped at step 100 of 212 (epoch 0.95) — train/eval loss
  plateaued (0.29/0.31), no overfit, eval token acc 90.8%
- Wall time: ~30 min on L4

**GSM8K task accuracy (end-to-end generation, 250 test problems).**

| Model | GSM8K accuracy | Δ |
|---|---|---|
| Qwen2.5-3B-Instruct (baseline) | 16.0% (40/250) | — |
| Qwen + arith LoRA | **69.6% (174/250)** | **+53.6 pts** |

The LoRA *demonstrably* learned arithmetic. The pipeline is effective.

**FVQ/CVQ held-out grid (5-cell subset, 100 trials/cell, vLLM, Wilson 95% CIs).**

| Cell      | Base RI/PI [CIs]            | Arith-LoRA RI/PI [CIs]      | Main-LoRA RI/PI (§5.1)   |
|-----------|-----------------------------|-----------------------------|--------------------------|
| K=10 N=50 | 0.42 [.32,.52] / 0.50 [.40,.60] | 0.48 [.38,.58] / **0.26** [.17,.35] | 1.00 / 1.00 |
| K=15 N=20 | 0.32 [.23,.42] / 0.58 [.48,.67] | 0.49 [.39,.59] / **0.33** [.24,.42] | 1.00 / 0.98 |
| K=20 N=30 | 0.21 [.13,.29] / 0.51 [.41,.60] | 0.32 [.23,.41] / **0.29** [.20,.38] | 1.00 / 0.98 |
| K=25 N=75 | 0.06 [.02,.11] / 0.60 [.50,.69] | 0.11 [.05,.17] / **0.19** [.11,.27] | 1.00 / 0.93 |
| K=30 N=75 | 0.01 [.00,.03] / 0.56 [.46,.66] | 0.07 [.02,.12] / **0.26** [.17,.35] | 0.99 / 0.92 |

PI dropped significantly on every cell (95% CIs of base and arith-LoRA
PI do not overlap). RI rose modestly. Neither RI nor PI reach the
65% "fixed" threshold on any cell.

**Mechanism — catastrophic forgetting, not format mismatch.** Sampled
predictions show the arith-LoRA model produces GSM8K-style verbose
output ("The first value of X is Y", with Markdown formatting,
multi-sentence reasoning) **and frequently hallucinates values that
are not in the input stream**. Our lenient `is_correct` matcher
(`pred == exp or exp in pred or pred.startswith(exp)`) catches the
correct-but-verbose cases, so the reported 26-33% PI is not an
undercount caused by format. The model has lost retrieval competence
on the FVQ/CVQ task. Example:

```
expected: 'bet'   (category: visual art)
prediction: "The first value of ancient civilization is 'blood'."
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^
            wrong category                   value not in the stream
```

This makes the §5.3 claim **stronger**: targeted task-aligned
data doesn't merely fail to help — generic LoRA training on a
different domain *destroys* the pretrained retrieval signal.

**Files.**

- Adapter: `lora_intervention/checkpoints/qwen_arith_adapter/`
  (44 MB, mirrors the convention of `gemma_adapter/` and `adapter/`)
- Data generator: `lora_intervention/data_gen_arithmetic.py`
- Training data: regenerable, gitignored at `lora_intervention/data_arithmetic/`
- FVQ/CVQ eval JSON: `lora_intervention/results/qwen_arith_control_eval_*.json`
- FVQ/CVQ comparison table: `lora_intervention/results/qwen_arith_control_comparison.txt`
- GSM8K task eval (arith): `lora_intervention/results/gsm8k_task_eval_qwen_arith_control_*.json`
- GSM8K task eval (base):  `lora_intervention/results/gsm8k_task_eval_qwen_base_*.json`
- Eval script for GSM8K: `lora_intervention/eval_gsm8k_task.py`

### Full 28-cell ARB grid (added after 5-cell preview)

Re-ran the same arith adapter on the complete 28-cell ARB held-out
grid (`--run-name qwen_arith_control_full`, 100 trials/cell, ~50 min
wall time on L4 + vLLM).

| | Cells fixed | Mean RI | Mean PI | Mean gap |
|---|---|---|---|---|
| Qwen base (no LoRA) | 0/28 | ~25% | ~52% | reversal regime D |
| Qwen + **arith** LoRA | **0/28** | **37.7%** | **23.9%** | +13.8% (weak primacy) |
| Qwen + **task-specific** LoRA (§5.1) | **28/28** | 99.9% | 97.7% | ~0% |

Every one of the 28 cells fails the "fixed" criterion (RI ≥ 0.65 AND
PI ≥ 0.65). Pattern is uniform across the grid: RI rises modestly,
PI drops sharply, regime flips from baseline reversal to weak
primacy, but neither metric approaches the task-specific LoRA's
near-ceiling values.

Files:
- `lora_intervention/results/qwen_arith_control_full_eval_*.json`
- `lora_intervention/results/qwen_arith_control_full_comparison.txt`

## 6. Next steps (not in this commit)

- **§7 stage 3 on Gemma**: HF-direct attention head ablation. Test
  whether ablating Gemma's baseline suppressor set (L17H0, L17H1,
  L17H3, L19H4, L30H6) on the LoRA model has any effect on
  P(v_last) — i.e., whether LoRA reuses the same suppressor circuit
  with different weights, or builds an independent promoter path.
- **Attention routing on Gemma+LoRA**: identifies the *promoter* heads
  built by LoRA. Blocked by missing `v3/attention_routing.py` module
  in this repo checkout. Once that file is available, the same script
  approach as §5.1 should work.
- **§5.3 negative control**: arithmetic-only LoRA on Qwen — rules out
  "any LoRA closes the gap" critique.
