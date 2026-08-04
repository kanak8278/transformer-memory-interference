# Experiment Status Map
Last updated: 2026-05-20 (session 2)

## Active Model Set

### Local models (vLLM)
| Model | Family | Params |
|-------|--------|--------|
| gemma-3-270m-it | Gemma-3 | 270M |
| gemma-3-1b-it | Gemma-3 | 1B |
| gemma-3-4b-it | Gemma-3 | 4B ← primary figures |
| Qwen2.5-0.5B-Instruct | Qwen2.5 | 0.5B |
| Qwen2.5-1.5B-Instruct | Qwen2.5 | 1.5B |
| Qwen2.5-3B | Qwen2.5 | 3B base |
| Qwen2.5-3B-Instruct | Qwen2.5 | 3B instruct |
| Qwen3.5-0.8B | Qwen3.5 | 0.8B |
| Qwen3.5-2B | Qwen3.5 | 2B |
| Qwen3.5-4B | Qwen3.5 | 4B |
| Qwen3.5-9B | Qwen3.5 | 9B |

### Excluded (too sparse / near-random / incomplete)
- TinyLlama-1.1B: 41/81 semantic, 47/81 arbitrary — dropped
- StableLM-1.6B: 55/81 semantic, 54/81 arbitrary — dropped
- Pythia-410m: 39/81 semantic, 43/81 arbitrary, often reversed — dropped
- Mamba-1.4B: 0/81 semantic, 3/81 arbitrary — dropped

### Proprietary API models
| Model | Provider | Notes |
|-------|----------|-------|
| claude-4.5-haiku | Anthropic | 200 trials, clear gap |
| claude-sonnet | Anthropic | 40–200 trials, small gap |
| gemini-2.5-flash | Google | 50–200 trials, moderate gap |
| gemini-2.5-pro | Google | 40–200 trials, near-zero gap |
| gpt-4.1 | OpenAI | 40–200 trials, moderate gap |
| gpt-4.1-mini | OpenAI | 50–175 trials, small gap |
| gpt-5-nano | OpenAI | different grid (K up to 45) — separate |

---

## 1. Behavioral Sweep (K×N grid, FVQ vs CVQ)

**Status: DONE for active model set.**

### Local models
- Dataset: semantic_multi + arbitrary_single
- Grid: K∈{2,3,5,7,10,15,20,25,30} × N∈{5,7,10,15,20,30,50,75,100} = 81 cells
- Trials: 100/cell (fixed, no adaptive stopping)
- CSV: `v3/results_vllm/local_models_behavioral.csv`

| Model | Semantic | Arbitrary |
|-------|---------|-----------|
| gemma-3-270m-it | 57/81 | 63/81 |
| gemma-3-1b-it | 62/81 | 55/81 |
| gemma-3-4b-it | 62/81 | 76/81 |
| Qwen2.5-0.5B-Instruct | 63/81 | 78/81 |
| Qwen2.5-1.5B-Instruct | 63/81 | 79/81 |
| Qwen2.5-3B | 61/81 | 73/81 |
| Qwen2.5-3B-Instruct | 63/81 | 79/81 |
| Qwen3.5-0.8B | 63/81 | 79/81 |
| Qwen3.5-2B | 63/81 | 79/81 |
| Qwen3.5-4B | 63/81 | 50/81 |
| Qwen3.5-9B | 63/81 | 79/81 |

### Proprietary API models
- Dataset: semantic_multi ONLY
- Grid: K∈{2,5,10,15,20,25,30,40,45} × N∈{1,5,10,15,20,30,50} = 63 cells
- Trials: up to 200/cell (adaptive Wilson CI stopping, min ~40)
- CSV: `experiments_cloud/results/proprietary_semantic_multi.csv`

| Model | Cells | Avg trials | Mean gap |
|-------|-------|-----------|---------|
| claude-4.5-haiku | 62/62 | 175 | +0.230 |
| claude-sonnet | 62/62 | 89 | +0.078 |
| gemini-2.5-flash | 62/62 | 97 | +0.131 |
| gemini-2.5-pro | 62/62 | 55 | +0.031 |
| gpt-4.1 | 62/62 | 110 | +0.126 |
| gpt-4.1-mini | 62/62 | 81 | +0.077 |

---

## 2. Positional + Format Analysis (Ucurve)

This experiment serves double duty: it measures **per-position accuracy** (positional analysis)
AND tests **format conditions** (block, flat_verbose, flat_nolabel, boundary) in one sweep.
The old remedy experiments (numbered, landmark, recency_cue, combined) are deprecated —
inconsistent results and superseded by the ucurve format sweep.

**What the formats test:**
- `flat_nolabel` — plain interleaved stream, no structural cues (baseline failure condition)
- `flat_verbose` — labeled stream with explicit update indices
- `block` — round-grouped format (all K updates per round together)
- `boundary` — explicit section boundary markers between rounds

**Position scheme:** Positions 1–7 always included (fixed anchors), then evenly-spaced fill
positions between 8 and N-1, plus position N (final). Total = min(N, n_points) positions.
- Old Haiku runs: n_points=11 → evenly spaced across 1–N
- New runs (proprietary + planned local): n_points=16 → positions 1–7 fixed + 8 fill + final

**Key finding so far (Claude Haiku, flat_nolabel):** cliff at position 3–4; positions 1–2
near-perfect, position 5+ near-zero. All structured formats hold ~1.0 at every position.

### Completed runs (Claude Haiku, ARBITRARY_SINGLE, n_positions=11)

All old Haiku runs used evenly-spaced positions across 1–N.

| Experiment | Dir | Grid | Formats | Status |
|-----------|-----|------|---------|--------|
| ucurve | `results/ucurve/` | K=5,7,10 × N=10,30,50 | block, flat_short, flat_verbose, flat_nolabel | ✓ Done |
| ucurve_wide | `results/ucurve_wide/` | K=5,7,10 × N=10–100 | flat_nolabel | ✓ Done |
| ucurve_hard | `results/ucurve_hard/` | K=15/N=100 | flat_verbose | ✓ Done |
| ucurve_single_key | `results/ucurve_single_key/` | K=1 × N=10–100 | flat_nolabel, flat_verbose (+ _last variants) | ✓ Done |

### Completed: K=1 on SEMANTIC_MULTI, all 6 proprietary models

Dir: `results/ucurve_single_key_sem/{model}/checkpoint.json`
Grid: K=1 × N∈{10,15,20,30,50} — Formats: flat_nolabel, flat_nolabel_last, flat_verbose, flat_verbose_last
n_positions=11 — All 6 models: 20/20 format-cells each ✓ Done (2026-05-19)

### In progress: ucurve_proprietary (SEMANTIC_MULTI, n_positions=16)

Dir: `results/ucurve_proprietary/{model}/checkpoint.json`
Grid: K∈{5,10} × N∈{10,20,50} — Formats: flat_nolabel_last, flat_verbose_last, block_last, landmark_last
Note: `_last` formats run position queries AND a semantic "last value" query per trial.

| Model | Completed | Status |
|-------|-----------|--------|
| claude-haiku | 24/24 | ✓ Done (2026-05-19) |
| claude-sonnet | 24/24 | ✓ Done (2026-05-19) |
| gpt-4.1 | 23/24 | 🔄 Almost done |
| gpt-4.1-mini | 23/24 | 🔄 Almost done |
| gemini-2.5-flash | 19/24 | 🔄 Running |
| gemini-2.5-pro | 13/24 | 🔄 Running |

### Local models (DONE — 2026-05-20)

Dir: `experiments_cloud/results/ucurve_vllm/{model}/ucurve_*.json`

| Model | Format-cells with data | Notes | File |
|---|---|---|---|
| Qwen2.5-3B-Instruct | 24/24 | all 200 trials | `ucurve_20260520_124743.json` |
| Qwen3.5-2B | 24/24 | all 200 trials | `ucurve_20260520_122122.json` |
| Qwen3.5-4B | 24/24 | one cell @ 70 trials, rest @ 200 | `ucurve_20260520_154253.json` |
| Qwen3.5-9B | 21/24 | 19 cells @ 200, 5 cells @ 50–140; 3 cells of `10_50` missing | `checkpoint.json` (no final snapshot) |
| gemma-3-4b-it | 24/24 | one cell @ 190 trials, rest @ 200 | `ucurve_20260520_141803.json` |

For Qwen3.5-9B, the `checkpoint.json` IS the data — all 21 format-cells with
saved trials are readable from it the same way as a final file. The 3 missing
cells are `10_50 × {flat_verbose_last, block_last, landmark_last}`. 9B 10_50 of `flat_nolabel_last` is present at 200 trials.

Setup matches `ucurve_proprietary`:
- **Dataset**: SEMANTIC_MULTI
- **Formats**: flat_nolabel_last, flat_verbose_last, block_last, landmark_last
- **Grid**: K∈{5,10} × N∈{10,20,50} — 6 cells × 4 formats = 24 format-cells/model
- **n_positions**: 11 (positions 1–10 + N)
- **Script**: `experiments_cloud/ucurve_vllm.py`

5 local models now directly comparable with the 6 proprietary API models
in `experiments_cloud/results/ucurve_proprietary/`.

---

## 4. Mechanistic Probing

### Existing runs — DO NOT USE (operating point is wrong)

All existing probing runs are at K=2/N=5, which is near-ceiling accuracy for most models
(almost no failures to analyze). Gemma-3-4b at K=2/N=5 showed 0%/0% behavioral accuracy
(experiment broke entirely). These files exist but are not usable for the paper.
- Files: `v3/results_vllm/probing/probing_*_2k_5u.json`

### Planned runs — 4 models, 7 probing points

Operating point selection criterion (ARBITRARY_SINGLE only — single-token values enable
clean logit tracking per value):
- Normal zone: FVQ ∈ [0.75, 0.95] AND CVQ ∈ [0.25, 0.65]
- Reversal zone: CVQ > FVQ, both > 0.25 (not at floor)

**Normal zone (FVQ > CVQ — standard failure mode):**

| Model | K/N | FVQ | CVQ | Gap | Purpose |
|-------|-----|-----|-----|-----|---------|
| Qwen2.5-3B-Instruct | K=2/N=30 | 0.86 | 0.41 | +0.45 | Baseline for reversal comparison |
| Qwen3.5-2B | K=2/N=20 | 0.83 | 0.36 | +0.47 | Cross-family, K-anomaly model |
| Qwen3.5-4B | K=7/N=10 | 0.90 | 0.60 | +0.30 | Larger model, both in range |
| gemma-3-4b-it | K=7/N=30 | 0.95 | 0.52 | +0.43 | Control — never reverses |

**Reversal zone (CVQ > FVQ — mechanism flip?):**

| Model | K/N | FVQ | CVQ | Gap | Purpose |
|-------|-----|-----|-----|-----|---------|
| Qwen2.5-3B-Instruct | K=15/N=20 | 0.32 | 0.58 | −0.26 | Does logit lens show mechanism flip? |
| Qwen3.5-2B | K=15/N=7 | 0.41 | 0.66 | −0.25 | K-driven reversal at LOW N (anomaly) |
| Qwen3.5-4B | K=10/N=50 | 0.43 | 0.65 | −0.22 | Reversal in larger model |

**All operating points verified from ARBITRARY_SINGLE behavioral data (100 trials/cell).**

**What reversal probing tests:** In the reversal zone, does logit lens show v_last being
promoted across layers instead of v_first (mechanism flip: primacy → recency)? Or does
FVQ collapse for unrelated reasons while CVQ succeeds via recency? The logit lens will
distinguish these: a mechanism flip shows v_last rising while v_first decays in early layers.

**Reversal predictor analysis (DONE — see notebooks/reversal_analysis.ipynb):**
- K (num_keys) is the primary driver, not K×N alone (logistic reg coef log_k=+1.5 to +1.9)
- Qwen3.5-2B anomaly: N coef = −1.10 (more N → LESS reversal) — pure K-driven
- Gemma-3-4b never reverses: CVQ structurally near-zero, GQA 4-head architecture
- SFT raises reversal threshold ~3× (Qwen2.5-3B base KN=49 vs instruct KN=150)
- Family analysis: notebooks/family_reversal_analysis.ipynb

---

## 5. Logit Lens

**Status: EXISTS but all at K=2 — same problem as probing.**

- 8 models: Gemma-3 (270m/1b/4b), Qwen2.5 (0.5B/1.5B/3B-base/3B-inst), Pythia-410m
- Points tested: K=2, N∈{5,10,15,20,50,75} depending on model
- Files: `v3/results_vllm/logit_lens/`

**Missing: K≥5 for all models.**

---

## 6. Causal / Head Ablation

**Status: EXISTS but all at K=2.**

- Same 8 models as logit lens, experiments 3A/3B/3C (head ablation)
- Files: `v3/results_vllm/causal/`

**Missing: K≥5 for all models.**

---

## 7. Training Dynamics

**Status: DONE for SmolLM2 and SmolLM3. SmolLM3 is the cleaner story.**

### SmolLM2-1.7B
- 41 checkpoints, step-125k to step-5.125M
- Dataset: arbitrary_single style
- Files: `v3/results_vllm/training_dynamics/step-*.json`
- Quality: noisy, gap oscillates — hard to draw clean conclusions

### SmolLM3-3B
- 46 files: stage1 pretraining (37 checkpoints) + all post-training stages
- Post-training stages: SFT, mid-training, LC-expert, soup-APO, final
- Files: `v3/results_vllm/training_dynamics_smollm3/`
- Key finding at K=5/N=10:
  - step-40k: gap=+0.39 (appears from very early pretraining)
  - All IT stages: gap=+0.14 to +0.68 (persists through all post-training)
- Quality: cleaner than SmolLM2 — **use SmolLM3 as primary**

**Missing:** Ucurve on SmolLM3 checkpoints (would show when first-match cliff first appears in training).

---

## 8. Narrative (Dota2)

**Status: DONE for 11 local models. Weak story — gaps much smaller than synthetic.**

- Dataset: Dota2 narrative (naturalistic, semantically rich)
- Files: `v3/results_vllm/narrative_dota2/`
- Mean gaps: 0.01–0.21 (vs 0.26+ on synthetic)
- Several models show CVQ > FVQ (negative gap)
- Use as "generalization note" only, or drop from paper

| Model | Mean FVQ | Mean CVQ | Gap |
|-------|---------|---------|-----|
| gemma-3-4b-it | 0.768 | 0.692 | +0.076 |
| Qwen3.5-9B | 0.824 | 0.812 | +0.012 |
| Qwen3.5-4B | 0.829 | 0.797 | +0.031 |
| gemma-3-270m-it | 0.373 | 0.201 | +0.172 |
| Qwen2.5-0.5B | 0.405 | 0.192 | +0.213 |

**Missing:** API models on narrative.

---

## Priority Run Queue

| Priority | Experiment | Models | Est. time |
|----------|-----------|--------|-----------|
| 🔴 Critical | Mechanistic probing at correct operating points (see Section 4) | 7 models, 8 points (5 normal + 3 reversal) | ~6h cloud |
| ~~🔴 Critical~~ | ~~Ucurve local models~~ — **DONE 2026-05-20**, see Section 2 | ~~5 local models~~ | ~~done~~ |
| 🟡 High | Complete Mamba behavioral (if keeping arch claim) | Mamba-1.4B | ~3h local |
| 🟡 High | Fill Gemma-3-4b semantic to 81/81 | gemma-3-4b-it | ~1h local |
| 🟠 Medium | Ucurve on SmolLM3 checkpoints (3–4 ckpts) | SmolLM3-3B | ~2h local |
| 🟠 Medium | Ucurve on SmolLM3 checkpoints (3–4 ckpts) | SmolLM3-3B | ~2h local |
| ⚪ Low | Narrative on API models | all API | API cost |

---

## Data Files Reference

| CSV | Description | Rows |
|-----|-------------|------|
| `experiments_cloud/results/proprietary_semantic_multi.csv` | 6 API models, semantic_multi, 62 cells each | 372 |
| `v3/results_vllm/local_models_behavioral.csv` | 15 local models, both datasets | 1755 |
