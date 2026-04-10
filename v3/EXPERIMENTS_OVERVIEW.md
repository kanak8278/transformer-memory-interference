# Experiments Overview

All experiments reproduced using vLLM (v0.19.0) on NVIDIA L40S (46GB).
Results in `v3/results_vllm/`. Total: 300+ JSON files across 15 experiment types.

---

## Stage 1: Behavioral Sweep — ARBITRARY_SINGLE (11 models)

**What & Why**: Measures RI (recall first value) and PI (recall last value) accuracy across a grid of key counts × update counts. This is the core experiment establishing the PI > RI asymmetry — models are consistently better at retrieving the first value than the last.

**What we found**: 8/11 models show PI > RI. Largest gaps: StableLM (+65.8%), Mamba (+64.0%), Gemma-1b (+54.3%). Base models without instruction tuning (Qwen 3B-Base, Pythia) show reversed asymmetry. Gemma 4b has the highest raw RI (95.1%).

## Stage 1: Behavioral Sweep — SEMANTIC_MULTI (11 models)

**What & Why**: Same experiment but with multi-token semantic values (real category members like "alexandrite" for gemstone) instead of arbitrary single-token words. Tests whether the asymmetry depends on value type.

**What we found**: Same PI > RI pattern holds. Semantic values show slightly different magnitudes but the same direction. Confirms the effect is about positional retrieval, not value characteristics.

## Stage 2: Logit Lens (10 models)

**What & Why**: Tracks P(v_i) for ALL values across ALL layers using TransformerLens. Shows WHERE in the network the model commits to the wrong value — does v_last appear then get outcompeted, or is it never found?

**What we found**: Pattern B ("overtaken") dominates — P(v_last) rises at ~85-90% depth then gets crushed in final layers. P(v_first) for RI rises monotonically to near 1.0 with no competition. The asymmetry is visible in the layer-by-layer probability trajectory.

## Stage 3: Causal Analysis (8 models)

**What & Why**: Uses attribution patching (gradient-based) and targeted activation patching to identify which attention heads suppress P(v_last). Tests whether ablating these heads fixes PI.

**What we found**: No single bottleneck head. The mechanism is distributed — top heads each contribute 2-7pp. Ablating confirmed heads actually REDUCES P(v_last) further, meaning they were helping retrieval, not suppressing it. PI failure is a network-wide phenomenon, not a localized circuit.

## Probing Classifiers (8 models)

**What & Why**: Trains linear probes on residual stream representations to test whether RI vs PI conditions are distinguishable, and whether correct vs incorrect answers are encoded differently per condition.

**What we found**: Condition discrimination is near-perfect (96-100%) — the model clearly encodes whether it's an RI or PI task. RI correctness is well-encoded (68-83%), PI correctness is weakly encoded (46-73%). The asymmetry exists in the representation space, not just behavior.

## Jacobian at Initialization (8 models × 2 modes)

**What & Why**: Computes input-output Jacobian norms at each position for both untrained (random init) and pretrained models. Tests whether positional bias is architectural (present at init) or learned.

**What we found**: Untrained models show primacy bias (Qwen 0.5B: 2.1× first/middle ratio). All pretrained models show recency dominance — training builds recency processing that exceeds the initial primacy bias. U-shape (both first and last positions have high influence) is universal.

## Training Dynamics — SmolLM2-1.7B (42 checkpoints)

**What & Why**: Runs the behavioral sweep across 42 intermediate training checkpoints (125K to 5.125M steps + final) to track WHEN the PI > RI asymmetry emerges during pretraining.

**What we found**: PI > RI gap present from step 125K (+17%), peaks at 625K (+44%), fluctuates through mid-training, and settles at +25% for the final model. Both RI and PI improve during training, but RI improves more consistently. The asymmetry is not an artifact of late training.

## Training Dynamics — SmolLM3-3B Base (35 checkpoints)

**What & Why**: Same as SmolLM2 but for a 3B model across three pretraining stages (stage1/stage2/stage3) to see how the asymmetry evolves with continued pretraining and data mixing changes.

**What we found**: Gap starts at +59% (step 40K) and oscillates between +6% and +55% through pretraining. Stage2/3 (continued pretraining) stabilizes the gap around +20-35%. The pattern is robust across the full training pipeline.

## Training Dynamics — SmolLM3-3B IT Checkpoints (5 checkpoints × 2 formats)

**What & Why**: Tests instruction-tuned checkpoints (SFT, mid-training, soup-APO, LC-expert, final) in both completion format (max_tokens=100) and chat template format (max_tokens=1024, with thinking). Tests whether instruction tuning changes the asymmetry and whether chain-of-thought helps.

**What we found**: All IT checkpoints show PI > RI in completion format (gap +17-27%). Chat+thinking shows LARGER gaps for most checkpoints — reasoning doesn't fix PI, it makes it harder. The `it-mid-training` checkpoint outputs Python code instead of answers in chat mode (treats retrieval as a coding task). The asymmetry is pre-alignment.

## Dota 2 Narrative Experiment (7 models)

**What & Why**: Tests PI > RI on naturalistic Dota 2 match commentaries instead of synthetic KV-pair streams. Validates that the asymmetry transfers to real-world narrative text where "first" and "last" values are embedded in story context.

**What we found**: PI > RI transfers to narratives for 5/7 models. Weaker models show clear gaps (+17-21%). Stronger models (Qwen 3B, Gemma 4b) show small or reversed gaps — they handle recency better on naturalistic text. 94% of cells show PI > RI for Qwen 1.5B.

## Remedy: Control (3 models)

**What & Why**: Baseline KV stream format (`color: red`) with no intervention. Establishes the gap that remedies try to reduce.

**What we found**: Qwen 3B gap=+4.2%, Gemma 4b gap=+46.3%, SmolLM3 gap=+0.3%. The gap varies dramatically by model — Gemma 4b has the most room for improvement.

## Remedy: Numbered Updates (3 models)

**What & Why**: Prefixes each KV pair with `Update 1 —`, `Update 2 —` etc. Targets positional confusability — makes v_{N-1} distinguishable from v_{N-2} by explicit numbering.

**What we found**: Minimal effect for Qwen 3B (+1.4pp) and SmolLM3 (-2.5pp). Modest PI improvement for Gemma 4b (PI: 49%→58%, gap narrowed by 9.4pp). Numbering alone doesn't solve the problem.

## Remedy: Landmark Separators (3 models)

**What & Why**: Adds `---` separators between update rounds. Targets capacity saturation — prevents representational over-mixing by creating structural boundaries.

**What we found**: The strongest intervention. FLIPS the asymmetry for Qwen 3B (gap: +4%→-13%) and SmolLM3 (gap: +0.3%→-23%). Narrows gap for Gemma 4b by 12.5pp. Separators help last-value retrieval dramatically but can hurt first-value retrieval.

## Remedy: Recency Cue (3 models)

**What & Why**: Appends explicit instruction: "The MOST RECENT value (listed LAST) is the current value." Targets cumulative reinforcement — directly tells the model to prioritize recency.

**What we found**: Highly model-dependent. Destroys both RI and PI for Qwen 3B (-20pp each). Makes PI WORSE for Gemma 4b (PI: 49%→37%). Creates massive gap for SmolLM3 (+27%). Extra instruction text acts as noise for most models.

## Remedy: Combined (3 models)

**What & Why**: Landmark separators + round numbers (`--- round 2 ---`). Tests whether combining structural and numbering cues is better than either alone.

**What we found**: Similar to landmark but slightly weaker. Flips asymmetry for SmolLM3 (gap=-19%). Best single intervention for Gemma 4b (PI: 49%→63%, gap narrowed by 16.6pp). Round numbers add modest value on top of landmarks.

## Scaling Law Fits (11 models × 2 datasets)

**What & Why**: Fits PI(N) = a·exp(-b·N) + c (exponential decay) and RI(N) = d - e·log(N) (logarithmic decay) to characterize how accuracy changes with update count N.

**What we found**: Exponential decay fits PI well for most models (R²=0.83-0.97 at low key counts). Fits degrade at high key counts where floor effects dominate. RI follows a slower logarithmic decay. The functional forms match the theoretical prediction from associative memory models.

## Error Position Analysis (11 models × 2 datasets)

**What & Why**: Analyzes WHERE PI failures land in the value sequence — does the model output the first value (primacy intrusion), the penultimate value (near-last error), or something random (garbage)?

**What we found**: Three distinct failure modes: (1) near-last for strong models (Qwen 3B: mean position 0.74, 66% at ≥0.7), (2) primacy fallback for Gemma/StableLM (mean position 0.21), (3) diffuse for weak models (Qwen 0.5B: mean 0.34). The failure mode depends on model architecture and strength.

## Cross-Model Scaling (11 models)

**What & Why**: Tests whether RI and PI accuracy correlate with model size at fixed operating points. The theory predicts RI should scale with model size (attention sink strengthens) while PI should not.

**What we found**: At 5 keys, 10 updates: RI correlates strongly with model size (r=0.876, p=0.001). PI also correlates but more weakly (r=0.734, p=0.016). Both scale with size, but RI more consistently — partial support for the attention sink hypothesis.

## SAE Features (1 model)

**What & Why**: Uses Google's Gemma Scope sparse autoencoders on Gemma-3-1b-it to find interpretable features that distinguish RI from PI conditions at specific layers.

**What we found**: Late layers show RI activates 18% more SAE features than PI. Preliminary evidence that the asymmetry has an interpretable feature-level signature, but needs more comprehensive analysis.

---

## Summary Statistics

| Experiment | Models | Files | Key Metric |
|---|---|---|---|
| Stage 1 ARBITRARY_SINGLE | 11 | 22 | 8/11 show PI > RI |
| Stage 1 SEMANTIC_MULTI | 11 | 22 | Same pattern, different magnitudes |
| Stage 2 Logit Lens | 10 | 10 | P(v_last) peaks then drops |
| Stage 3 Causal | 8 | 8 | Distributed, no bottleneck |
| Probing | 8 | 8 | RI encoded 68-83%, PI 46-73% |
| Jacobian | 8×2 | 16 | Primacy at init, recency from training |
| SmolLM2 Training Dynamics | 42 | 42 | Gap from step 125K |
| SmolLM3 Training Dynamics | 35+10 | 46 | Gap through pretraining+IT |
| Dota2 Narrative | 7 | 7 | Transfers to naturalistic text |
| Remedy (5 styles × 3 models) | 3 | 3 | Landmark separators flip asymmetry |
| Scaling Laws | 11 | 2 | PI = exp decay (R²≈0.9) |
| Error Positions | 11 | 2 | 3 failure modes |
| Cross-Model Scaling | 11 | 2 | RI~size r=0.88 |

---

## Results Directory Structure

All results in `v3/results_vllm/`. Each experiment saves a summary JSON (`*_sweep_*.json` or `*_<timestamp>.json`) and optionally a trials JSON (`*_trials_*.json`) with per-trial raw data.

```
v3/results_vllm/
│
├── arbitrary_single/                    # Stage 1 behavioral sweep — single-token values
│   ├── Qwen2.5-0.5B-Instruct/          # Per-model subdirectory
│   │   ├── stage1_sweep_*.json          #   Summary: config, per-cell RI/PI accuracy, regime map
│   │   ├── stage1_trials_*.json         #   Full trial data: every prediction, error type, raw output
│   │   └── stage1_checkpoint.json       #   Checkpoint (intermediate save, can resume from)
│   ├── Qwen2.5-1.5B-Instruct/
│   ├── ... (11 models total)
│
├── semantic_multi/                      # Stage 1 behavioral sweep — multi-token semantic values
│   └── (same structure as arbitrary_single, 11 models)
│
├── logit_lens/                          # Stage 2 — per-layer P(v_i) probability tracking
│   ├── Qwen2.5-0.5B-Instruct/
│   │   ├── stage2_logit_lens_*.json     #   Per-trial: value_probs_by_layer [n_values × n_layers]
│   │   └── stage2_checkpoint.json       #   Checkpoint after each operating point
│   ├── ... (8 models: Qwen x4, Gemma 270m/1b/4b, Pythia)
│
├── causal/                              # Stage 3 — attribution patching + targeted patching + ablation
│   ├── Qwen2.5-0.5B-Instruct/
│   │   ├── stage3_causal_*.json         #   3A attribution scores, 3B patching deltas, 3C ablation results
│   │   └── stage3_checkpoint.json
│   ├── ... (8 models)
│
├── probing/                             # Probing classifiers — linear probes on residual stream
│   ├── probing_Qwen2.5-0.5B-Instruct_2k_5u.json    # Per-layer probe accuracy (condition + correctness)
│   ├── probing_Qwen2.5-1.5B-Instruct_2k_5u.json
│   ├── ... (8 models)
│
├── jacobian/                            # Jacobian at initialization — positional sensitivity
│   ├── jacobian_Qwen2.5-0.5B-Instruct_untrained.json  # Random init: per-position gradient norms
│   ├── jacobian_Qwen2.5-0.5B-Instruct_pretrained.json # Trained: same metric after pretraining
│   ├── ... (8 models × 2 modes = 16 files)
│
├── training_dynamics/                   # SmolLM2-1.7B — 42 intermediate checkpoints
│   ├── step-125000.json                 #   Per-checkpoint: config, per-cell stats, summary (RI/PI/gap)
│   ├── step-125000_trials.json          #   Per-trial raw data for this checkpoint
│   ├── step-250000.json
│   ├── ...
│   ├── step-5125000.json
│   ├── final.json                       #   Final pretrained model
│   └── training_dynamics_summary.json   #   Combined trajectory: all checkpoints in one file
│
├── training_dynamics_smollm3/           # SmolLM3-3B — 35 base + 10 IT (dual format)
│   ├── stage1-step-40000.json           #   Base pretraining checkpoints (completion format)
│   ├── stage1-step-160000.json
│   ├── ...
│   ├── stage3-step-4720000.json         #   Final pretraining stage
│   ├── it-SFT_completion.json           #   IT checkpoint in completion format (max_tokens=100)
│   ├── it-SFT_chat_1024.json           #   Same checkpoint in chat format (max_tokens=1024, with thinking)
│   ├── it-SFT_chat_1024_trials.json    #   Full raw response saved (includes <think>...</think>)
│   ├── it-mid-training_completion.json
│   ├── it-mid-training_chat_1024.json
│   ├── it-soup-APO_completion.json
│   ├── it-soup-APO_chat_1024.json
│   ├── it-LC-expert_completion.json
│   ├── it-LC-expert_chat_1024.json
│   ├── final_completion.json
│   └── final_chat_1024.json
│
├── narrative_dota2/                     # Dota 2 narrative interference — naturalistic text
│   ├── Qwen2.5-0.5B-Instruct/
│   │   ├── narrative_dota2_*.json       #   Per-cell RI/PI accuracy on Dota 2 match commentary
│   │   └── narrative_dota2_*_trials.json
│   ├── ... (7 models: Qwen x4, Gemma x3)
│
├── narrative/                           # Old narrative results (copied from v3/results/)
│   ├── narrative_Qwen2.5-1.5B-Instruct.json         # Dota 2 (Qwen 1.5B)
│   ├── narrative_wildlife_Qwen2.5-1.5B-Instruct.json # Wildlife domain
│   ├── narrative_icu_Qwen2.5-1.5B-Instruct.json     # ICU domain
│   ├── narrative_atc_Qwen2.5-1.5B-Instruct.json     # ATC domain
│   ├── narrative_api_*_claude-haiku.json              # Claude Haiku on all domains
│   └── narrative_claude-haiku.json                    # Haiku Dota 2 (150 trials)
│
├── remedy/                              # Remedy interventions — 5 prompt styles
│   ├── Qwen2.5-3B-Instruct/
│   │   └── remedy_*.json               #   All 5 styles: control, numbered, landmark, recency_cue, combined
│   ├── gemma-3-4b-it/                  #   Per-style: per-cell RI/PI + comparison table
│   └── SmolLM3-3B/
│
├── sae/                                 # SAE sparse autoencoder features
│   └── gemma_scope_sae_analysis.json    #   Gemma-3-1b-it feature activation differences (RI vs PI)
│
├── bidirectional/                       # Bidirectional model control (old results)
│   └── bidir_probe_bert-base-uncased.json  # BERT MLM — failed (0% on both RI and PI)
│
├── analysis/                            # Cross-model analysis computed from Stage 1 data
│   ├── scaling_law_fits_arbitrary_single.json    # PI(N) = a*exp(-b*N)+c fits per model per key count
│   ├── scaling_law_fits_semantic_multi.json
│   ├── error_positions_arbitrary_single.json     # WHERE PI failures land (position histograms)
│   ├── error_positions_semantic_multi.json
│   ├── cross_model_scaling_arbitrary_single.json # RI/PI vs model size correlations
│   └── cross_model_scaling_semantic_multi.json
│
├── summary/                             # Consolidated cross-model summaries
│   ├── all_models_arbitrary_single.json  # Per-model per-cell stats for all 11 models
│   ├── all_models_semantic_multi.json
│   └── cross_model_comparison.json       # Standard operating point comparison
│
├── RUN_LOG.md                           # Full chronological run history with issues and decisions
└── EXPERIMENT_SUMMARY.md                # Quick reference summary of all results
```

### File naming conventions

| Pattern | Meaning |
|---|---|
| `stage1_sweep_YYYYMMDD_HHMMSS.json` | Stage 1 behavioral sweep summary (no trial details) |
| `stage1_trials_YYYYMMDD_HHMMSS.json` | Stage 1 full trial data (every prediction + raw output) |
| `stage1_checkpoint.json` | Resumable checkpoint (overwritten during run) |
| `stage2_logit_lens_*.json` | Stage 2 logit lens with per-layer probability arrays |
| `stage3_causal_*.json` | Stage 3 attribution + patching + ablation results |
| `probing_<model>_<keys>k_<updates>u.json` | Probing classifier results at one operating point |
| `jacobian_<model>_<mode>.json` | Jacobian norms (mode = untrained or pretrained) |
| `narrative_dota2_*.json` | Dota 2 narrative experiment summary |
| `remedy_*.json` | Remedy experiment (all 5 styles in one file per model) |
| `step-<N>.json` | Training dynamics checkpoint result |
| `*_completion.json` | Run using completion format (few-shot) |
| `*_chat_1024.json` | Run using chat template with max_tokens=1024 |
