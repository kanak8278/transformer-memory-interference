# Implementation Plan: Comprehensive Head Identification (Exp 25a/25b/25c)

## Goal

Replace ad-hoc head classification (pi_primacy > 0.6 threshold) with multiple established methods. Run all methods on all 3 models, compare results, pick the approach where methods agree.

## The Problem

Our current exp 16 uses attention-based thresholds to classify heads. Results are inconsistent:
- Qwen 0.5B: 8 heads (loose), 1 head (strict)
- Qwen 1.5B: 5 heads (loose), 1 head (strict)
- Gemma 1B: 1 weak head (pi_primacy=0.662, barely over threshold)

Different thresholds give different heads. Downstream experiments (18, 19b, 20, 23) depend on this classification and give contradictory results when wrong heads are selected.

## Three New Experiments

### 25a: Per-Head Causal Knockout (`25a_per_head_knockout.py`)

**What:** Zero out each head individually, measure PI logit_diff change.
**Why:** Gold standard causal method (Wang et al. 2022 IOI). No threshold — produces a continuous ranking.
**Algorithm:**
1. Run baseline (no ablation) on 100 PI trials → get baseline logit_diff per trial
2. For each of N_heads heads: zero out that head's output (hook on `attn.hook_z`), run same 100 trials
3. Compute: `causal_effect[head] = mean(ld_knockout - ld_baseline)` across trials
4. Positive effect = head was hurting PI (primacy head). Negative = head was helping PI (retrieval head).

**Output:** `per_head_knockout.json` with causal_effect for all heads, ranked.
**Speed:** N_heads × 100 trials forward passes. Gemma: 104 × 100 = 10,400. Qwen 0.5B: 336 × 100 = 33,600. Slow but definitive.
**Optimization:** Can reduce to 50 trials for initial scan, 100 for top-20 heads.

**What already exists:** Exp 16 does knockout for top-5 retrieval heads + groups. This sweeps ALL heads.

---

### 25b: Attribution Patching per Head (`25b_attribution_patching.py`)

**What:** Gradient-based approximation of per-head causal effect. ~100x faster than 25a.
**Why:** Syed et al. 2023 showed this approximates full activation patching well. Fast enough to sweep all heads.
**Algorithm:**
1. Run clean forward pass (1 value per key, no interference), cache activations
2. Run corrupted forward pass (N updates), compute gradient of PI logit_diff w.r.t. each head's output
3. For each head: `attribution[head] = (clean_output - corrupted_output) · gradient`
4. This is a first-order Taylor approximation of the full patching effect.

**Output:** `attribution_patching_heads.json` with attribution score per head, ranked.
**Speed:** 2 forward passes + 1 backward pass per trial. ~100 trials total. Very fast.
**Validation:** Compare top-20 ranking against 25a. If they agree → use 25b for future models.

**What already exists:** Nothing — entirely new algorithm for us.

---

### 25c: Observational Head Metrics (`25c_observational_head_metrics.py`)

**What:** Five observational metrics from a single forward pass with cache.
**Why:** Fast, complementary to causal methods. Shows different aspects of head behavior.

**Methods bundled:**

| # | Metric | Formula | Source |
|---|--------|---------|--------|
| 2 | **DLA per head** | `head_output @ (W_U[:, init] - W_U[:, final])` | Wang et al. 2022 |
| 4 | **Attention primacy score** | `attn_to_init / (attn_to_init + attn_to_final)` in PI | Our exp 16 |
| 5 | **Attention entropy** | `-Σ p_i log p_i` at answer position | Xiao et al. 2023 |
| 6 | **Copy score** | Frequency where max-attended token = output token | Wu et al. 2024 |
| 7 | **Condition sensitivity** | `|primacy_RI - primacy_PI|` | Our exp 16 |

**Algorithm:**
1. Run 100 RI + 100 PI trials with cache (same prompts, different query word)
2. For each trial, extract all 5 metrics per head
3. Aggregate across trials: mean ± std per head per metric

**Output:** `observational_head_metrics.json` with all 5 scores per head.
**Speed:** 200 forward passes total (same as exp 12). Fast.

**What already exists:** Methods 2, 4, 7 are computed in exp 12/16 but scattered across files. Method 5 (entropy) and 6 (copy score) are new. This unifies everything.

---

## Models to Run On

| Model | Heads | Operating Point | Estimated Time |
|-------|-------|----------------|----------------|
| Qwen 0.5B (24L, 14H) | 336 | B (1k,5u) | 25a: ~1hr, 25b: ~3min, 25c: ~5min |
| Qwen 1.5B (28L, 12H) | 336 | B (1k,3u) | 25a: ~1hr, 25b: ~3min, 25c: ~5min |
| Gemma 1B (26L, 4H) | 104 | B (2k,2u) | 25a: ~20min, 25b: ~3min, 25c: ~5min |

Run at Point B (PI cracking edge) for each model — that's where head effects are most visible.

## Analysis After Running

1. For each model, rank heads by all 7 methods
2. Compute rank correlation (Spearman) between methods
3. Identify heads that rank in top-10 across multiple methods → robust primacy heads
4. Compare identified heads across models → architecture-general vs model-specific
5. Re-run exp 18 (forced attention) and exp 23 (ablation+patching) with the newly identified heads

## Success Criteria

- [ ] At least 2 causal methods (25a knockout, 25b attribution) agree on top-5 heads
- [ ] Observational methods (25c) correlate with causal methods (Spearman > 0.5)
- [ ] Identified heads are consistent across operating points (run at B and C to verify)
- [ ] Cross-model comparison shows structural pattern (e.g., "last 1/3 of network")

## Implementation Order

1. **25c first** — fastest, reuses most existing code, gives us the observational baseline
2. **25a second** — slow but gold standard, validates/contradicts 25c
3. **25b last** — new algorithm, validate against 25a

## Status

- [ ] 25a: Not started
- [ ] 25b: Not started
- [ ] 25c: Not started
