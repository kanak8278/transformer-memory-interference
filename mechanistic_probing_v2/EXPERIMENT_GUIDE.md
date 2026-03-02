# Experiment Guide: End-to-End Mechanistic Probing Pipeline

Practical guide for running the full experiment suite on any model.
For theoretical background see RESEARCH_PLAN.md. For head selection methodology see HEAD_IDENTIFICATION_METHODS.md.

---

## Overview

The pipeline has two stages:

**Stage 1 — Find operating points** (behavioral sweep)
Run a grid of key×update configurations to find regime B: where RI works but PI fails.
This tells you WHERE to run the mechanistic experiments.

**Stage 2 — Mechanistic chain** (run_all.py)
Load the model once, run 17 experiments across two phases:

- Phase 1 (no heads needed): logit lens, patching, head knockout
- Phase 2 (heads from 25a): instruction sensitivity, forced attention, bias sweep, validation

---

## Models

| Model | Size | Arch | Status | Validated Primacy Heads |
|-------|------|------|--------|------------------------|
| Qwen2.5-0.5B-Instruct | 0.5B | GQA | DONE — distributed effects | No dominant head |
| Qwen2.5-1.5B-Instruct | 1.5B | GQA | DONE — full suite + 25d validated | **L8H3** (Δld=+8.3) |
| Qwen2.5-3B-Instruct | 3B | GQA | Behavioral done, mechanistic TODO | TBD |
| Gemma-3-1b-it | 1B | GQA | DONE — full suite + 25d validated | **L14H2** (Δld=+20.2) |
| Pythia-410m | 410M | MHA | Behavioral + copy suppression done | TBD |
| Mamba/RWKV | — | SSM | TODO — architectural control | N/A |

**Architecture note:** GQA models (Qwen, Gemma) cannot use OV eigenspectrum (exp 30 Test 1) — it auto-skips. MHA models (Pythia) can run all 3 tests.

---

## Dataset Types — Which to Use Where

Four dataset types exist. Choosing correctly is critical for consistency between stages.

| Dataset | Values | Pool | Single-token? | Use for |
|---------|--------|------|---------------|---------|
| `ARBITRARY_MULTI` | Gem42, Art375 (synthetic) | Unlimited | No | Behavioral landscape mapping (large grids) |
| `ARBITRARY_SINGLE` | English words (above, ruby, iron) | 2300 words | **Yes** (verified) | **All mechanistic experiments. Stage 1 recalibration.** |
| `SEMANTIC_MULTI` | alexandrite, baroque (real) | 45-70/cat | No | ACL paper replication |
| `SEMANTIC_SINGLE` | ruby, jade, coral (real) | 4-89/cat | Yes | Semantic + mechanistic (limited pool, max ~34 updates at 5 keys) |

**Why dataset choice matters:** Stage 2 (mechanistic experiments 12-30) always use
`ARBITRARY_SINGLE` internally — they need single-token values so logit lens can track
`P(specific_word_id)` at each layer. If Stage 1 uses a different dataset, the regime B
operating point may not transfer cleanly.

**Recommendation for Stage 1:**

| Goal | Dataset |
|------|---------|
| Quick landscape map across full grid | `ARBITRARY_MULTI` (unlimited, fast) |
| Find operating points matching mechanistic experiments | `ARBITRARY_SINGLE` (same as Stage 2) |
| Replicate ACL paper conditions | `SEMANTIC_MULTI` |
| Narrative behavioral comparison | Run `11e_narrative_behavioral_sweep.py` separately |

**Practical approach:**

1. Run `ARBITRARY_MULTI` first to get the big picture (full 10×8 grid)
2. Run `ARBITRARY_SINGLE` targeted at the regime B region found in step 1
   to confirm the operating point holds with single-token values

---

## Stage 1: Behavioral Sweep

**Purpose:** Find the right (num_keys, num_updates) operating point in regime B,
ideally using `ARBITRARY_SINGLE` to match mechanistic experiments.

**Step 1a — Full landscape (ARBITRARY_MULTI):**

```
behavioral_sweep_base_models.py --model <model> --dataset ARBITRARY_MULTI \
    --trials 30 --gpu N
```

Identifies rough regime B region. Fast because pool is unlimited.

**Step 1b — Recalibration (ARBITRARY_SINGLE):**

```
behavioral_sweep_base_models.py --model <model> --dataset ARBITRARY_SINGLE \
    --trials 50 --gpu N
```

Confirms regime B holds with single-token values (the mechanistic data setup).
Note: pool has 2300 words — instruct grid goes up to 46 keys which may hit limits at
high update counts. Focus sweep on small key counts (2-5) × low-mid updates (1-20).

**What to expect:**

- Regime A (both high): RI≥60%, PI≥50% — too easy
- **Regime B (target)**: RI≥50%, PI<40% — PI fails while RI works
- Regime C (both fail): both <50% — too hard, model can't do the task

**Choosing operating points for Stage 2:**
Pick 1-2 cells in regime B at low update counts (3-10) — smaller context, cleaner signal.
Use cells where baseline PI ≈ 20-50% (partial failure, not total collapse — gives more
informative mechanistic signal than 0% or 100%).
Example good operating points: `2,3`, `1,5`, `2,5`

**Results location:** `results/{model_short}/behavioral_sweep/behavioral_sweep_{ts}.json`

---

## Stage 2: Mechanistic Chain

### Phase 1 — No heads needed (run first)

Run with: `run_all.py --model <model> --points "2,3" --trials 100 --gpu N --phase 1`

| Exp | Name | What it does | What to expect |
|-----|------|-------------|----------------|
| 12 | Logit lens + DLA | Tracks P(init) and P(final) layer-by-layer; per-head DLA | RI shows P(final) rising correctly in late layers; PI shows P(final) suppressed/failing to rise. Whether P(init) is elevated is measured but only interpretable after exp 14 confirms what failure mode dominates. |
| 13 | Positional gradient | Tracks P(v_i) for each value position across layers | Sharp cliff: P(v0) >> P(v1) >> ... → "absolute primacy" not gradual decay |
| 14 | PI mass distribution | Where does probability land on PI failures? | Primacy intrusion rate: model outputs v0 on PI failures |
| 15 | Activation patching | Causal: patch clean→corrupted at each (layer, position) | Hockey stick: near-zero recovery early layers, sharp jump in final third |
| 22 | Query patching granular | Component-level patching at query position | Late-layer residual stream patching works; supports Hypothesis B (accumulated corruption) |
| 25b | Attribution patching | Fast (1 forward + 1 backward) head attribution | Finds late-layer heads; disagreement with 25a reveals indirect effects |
| 25c | Observational metrics | 5 attention-based metrics from single forward pass | Shows WHY exp-16-style metrics fail: observational ≠ causal |
| **25a** | **Per-head knockout** | **Zero each head, measure Δlogit_diff on PI** | **Gold standard primacy head rankings. Run LAST in phase 1.** |

**Critical:** Run 25a last. Its output feeds Phase 2.

**Expected from 25a:**

- `top_primacy_heads`: heads where knockout helps PI (positive causal_effect)
- `top_retrieval_heads`: heads where knockout hurts PI (negative causal_effect)
- For concentrated models (Gemma, Qwen 1.5B): one dominant head with Δld >> rest
- For distributed models (Qwen 0.5B): spread across 10+ heads with similar magnitude

---

### Phase 2 — Needs heads from 25a (run after Phase 1)

Run with: `run_all.py --model <model> --points "2,3" --trials 100 --gpu N --phase 2`

Heads auto-extracted from 25a results (top 5 by default). Override with `--heads "8,3 0,7"`.

| Exp | Name | What it does | What to expect |
|-----|------|-------------|----------------|
| 21a | Logit lens under ablation | Logit lens with primacy heads zeroed out | If ablating raises P(final) trajectory → heads were suppressing final-value retrieval. If ablating lowers P(init) → heads were promoting initial value. Both are possible and distinguishable here. |
| 17b | Instruction sensitivity | Do primacy heads attend to "first"/"last"? | Should attend <2% to query word — heads ignore instruction |
| 18b | Forced attention | Force primacy heads to attend to final value | Mechanism A (QK routing): forcing helps. Mechanism B (OV corruption): forcing doesn't help |
| 19 | Positional bias sweep | Add linear recency bias to primacy heads; sweep λ | No sweet spot found → bias can't fix PI without oracle knowledge of value positions |
| 19b | Bias attention proof | Where do heads attend under blind vs oracle bias? | Blind bias diverts to instruction tokens; oracle correctly redirects to final value |
| 20 | Minority override | Why do N primacy heads beat M recency heads? | W_OV norms: primacy heads 1.5-2× larger → bigger output per unit attention |
| 23 | Ablation + patching | Query patching with primacy heads ablated | If ablation removes need for patching → heads caused query corruption |
| 25d | Knockout validation | V1: primacy-specific? V2: stable? V3: CIs? | All three tests should pass for paper-quality heads |
| 30 | Copy suppression | V1-promotion vs V2-suppression via DLA | DLA decomposition should show V1-promotion dominant for validated primacy heads |

---

## Head Selection for Paper-Quality Results

Do NOT use auto-extracted top-5 for the paper. Follow two-gate criterion:

**Gate 1:** Run 25a at 2 different operating points (both regime B) with 100+ trials.
**Gate 2:** Select heads that rank in top-10 at BOTH points (stability).
**Gate 3 (optional):** Run 25d V3 with 200+ trials + bootstrap CIs.

**Use `--heads "8,3 0,7"` in run_all.py when you have validated heads.**

### Validated heads (paper-ready)

| Model | Heads (use with --heads) | Δld | Validated |
|-------|--------------------------|-----|-----------|
| Qwen2.5-1.5B-Instruct | `"8,3 0,7 0,3"` | +8.3, +3.1, +2.4 | ✓ 25d all 3 tests |
| Gemma-3-1b-it | `"14,2 4,1 8,2"` | +20.2, +8.1, +6.3 | ✓ 25d all 3 tests |
| Qwen2.5-0.5B-Instruct | Distributed — no dominant head | — | Not validated |
| Qwen2.5-3B-Instruct | TBD | — | TODO |
| Pythia-410m | TBD | — | TODO |

---

## Running in Parallel on Multi-GPU Machine

Each run_all.py call uses one GPU. Run different models simultaneously:

```bash
# 8-GPU SageMaker: run 4 models in parallel
python experiments/run_all.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --points "1,3 1,5" --trials 100 --gpu 0 &

python experiments/run_all.py --model google/gemma-3-1b-it \
    --points "2,2 2,5" --trials 100 --gpu 1 &

python experiments/run_all.py --model Qwen/Qwen2.5-3B-Instruct \
    --points "1,3" --trials 100 --gpu 2 &

python experiments/run_all.py --model EleutherAI/pythia-410m \
    --points "2,3 3,5" --trials 100 --gpu 3 &
```

---

## What's Still TODO

| Task | Priority | Notes |
|------|----------|-------|
| Qwen2.5-3B mechanistic suite | HIGH | Behavioral done; run Phase 1+2 with run_all.py |
| Pythia mechanistic suite | HIGH | Base model control; confirms effect is not SFT-caused |
| Mamba behavioral sweep | MEDIUM | Architectural control; prediction: different interference profile |
| Qwen2.5-3B head validation (25d) | HIGH | Need stable heads for paper |
| Pythia head validation (25d) | HIGH | Need stable heads for paper |
| Paper figures | CRITICAL | All data exists for 0.5B, 1.5B, Gemma; need plotting scripts |

---

## Key Findings Summary (For Reference)

**Universal (all models):**

- PI > RI at all scale (Cohen's d = 1.73 from ACL paper)
- Attention-based head classification (exp 16 method) fails: identifies wrong heads
- Causal knockout (25a) is the only reliable method
- Late-layer query position patching recovers PI accuracy (hockey stick pattern)
- Primacy heads ignore the query word "first"/"last" (<2% attention)

**Architecture-specific:**

- Gemma 1B: concentrated mechanism, L14H2 dominant, mixed QK+OV (Mechanism A+B)
- Qwen 1.5B: concentrated mechanism, L8H3 dominant, pure OV corruption (Mechanism B)
- Qwen 0.5B: distributed, no single dominant head
- Pythia (MHA): V1-promotion confirmed via OV eigenspectrum (not V2-suppression)
- GQA models (Qwen, Gemma): OV eigenspectrum test not applicable

**Mechanism B (pure OV corruption) on Qwen 1.5B:**
Forcing correct attention has zero effect → the head corrupts representations
regardless of where it attends. The damage is in what it writes, not where it looks.
