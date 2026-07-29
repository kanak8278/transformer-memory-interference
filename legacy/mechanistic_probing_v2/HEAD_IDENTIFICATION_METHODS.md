# Head Identification Methods — Literature Survey & Implementation Plan

## Context

We need to identify which attention heads cause PI > RI in associative recall. Our current ad-hoc threshold (pi_primacy > 0.6) gives inconsistent results across models. This document catalogs all published methods and selects which to implement.

## Methods Catalog

### Method 1: Per-Head Knockout (Causal, Gold Standard)

**Source:** Wang et al. 2022 (IOI Circuit)

**Algorithm:**
1. For each head (layer, head_idx), zero out that head's output
2. Run forward pass on PI trials
3. Measure change in logit_diff: `Δld = ld_knockout - ld_baseline`
4. Positive Δld = head was HURTING PI (promoting init over final) → primacy head
5. Negative Δld = head was HELPING PI → recency/retrieval head

**Output:** Continuous score per head. No threshold needed — rank by causal impact.

**Pros:** Direct causal evidence. No arbitrary threshold. Same metric (logit_diff) as our patching experiments.
**Cons:** Slow — N_heads × N_trials forward passes. But feasible (104 heads × 100 trials on Gemma).

**Status:** We already do this partially in exp 16 (ablation section) but only for top-5 retrieval heads. Need to sweep ALL heads.

---

### Method 2: DLA per Head (Observational, Fast)

**Source:** Wang et al. 2022, Neel Nanda tutorials

**Algorithm:**
1. Run forward pass with cache
2. For each head: `dla = head_output @ (W_U[:, init_tid] - W_U[:, final_tid])`
3. Positive DLA in PI = head promotes initial value when it shouldn't → primacy bias

**Output:** Continuous score per head per trial. Average across trials.

**Pros:** Fast (1 forward pass per trial, then matrix ops). Already implemented in exp 12.
**Cons:** Observational — doesn't account for nonlinear downstream effects. Can miss indirect contributions.

**Status:** Already computed in exp 12/14. Need to aggregate into a proper head ranking.

---

### Method 3: Attribution Patching (Causal, Fast Approximation)

**Source:** Syed et al. 2023, Nanda 2024

**Algorithm:**
1. Run clean forward pass, cache activations
2. Run corrupted forward pass, compute gradients of logit_diff w.r.t. each head's output
3. For each head: `attr = (clean_output - corrupted_output) · gradient`
4. First-order Taylor approximation of full activation patching

**Output:** Continuous score per head. Approximates Method 1 but ~100x faster.

**Pros:** 2 forward passes + 1 backward pass total (not per head). Fast enough for sweeping.
**Cons:** Linear approximation — may miss nonlinear effects. Need to validate against Method 1 for top heads.

**Status:** Not implemented. Would be new.

---

### Method 4: Attention Pattern Classification (Observational)

**Source:** Voita et al. 2019, our current exp 16

**Algorithm:**
1. Extract attention weights at answer position for each head
2. Compute: `retrieval_score = attn_to_any_value_position`
3. Compute: `primacy_score = attn_to_initial / (attn_to_initial + attn_to_final)`
4. Classify by threshold

**Output:** Binary classification with arbitrary threshold.

**Pros:** Simple, interpretable. Shows WHERE the head looks.
**Cons:** Our current method. Arbitrary threshold. Doesn't tell you if attention translates to output impact. A head can attend to the initial value but output garbage.

**Status:** Current exp 16. Keep as supplementary but don't use as primary classifier.

---

### Method 5: Attention Entropy / Kurtosis (Observational, Task-Agnostic)

**Source:** Xiao et al. 2023, Zhai et al. 2023

**Algorithm:**
1. For each head, compute entropy of attention distribution: `H = -Σ p_i log p_i`
2. Or kurtosis: `K = E[(p - μ)^4] / σ^4 - 3`
3. Low entropy / high kurtosis = head concentrates on few positions → potential sink/primacy head

**Output:** Continuous score per head. Task-agnostic.

**Pros:** Fast, no task data needed. Identifies structural attention patterns.
**Cons:** Doesn't distinguish primacy from other concentrated patterns (e.g., BOS attention sinks).

**Status:** Not implemented. Easy to add.

---

### Method 6: Copy Score (Retrieval Head Specific)

**Source:** Wu et al. 2024 "Retrieval Head Mechanistically Explains Long-Context Factuality"

**Algorithm:**
1. For each head, check if the token receiving highest attention is the same as the model's output token
2. Copy score = frequency of this match across trials
3. Retrieval head = copy score > 0.1

**Output:** Continuous score per head. Threshold at 0.1 (from paper).

**Pros:** Well-established metric. Identifies heads that do copy-paste retrieval.
**Cons:** Only identifies retrieval heads, not primacy bias specifically. A head can copy correctly (no primacy bias) or copy the wrong thing (primacy bias).

**Status:** Not implemented. Would complement Method 1.

---

### Method 7: Condition Sensitivity (Our Task-Specific Metric)

**Source:** Our exp 16 design

**Algorithm:**
1. Run RI and PI trials with same prompts (only query word differs)
2. For each head: `sensitivity = |primacy_RI - primacy_PI|`
3. Low sensitivity = head ignores the instruction → candidate primacy head

**Output:** Continuous score. Paired with primacy_score for direction.

**Pros:** Directly measures what we care about: does the head adapt to "first" vs "last"?
**Cons:** Observational. A head that doesn't adapt might not matter for the output.

**Status:** Already computed in exp 16. Keep as supplementary.

---

## Implementation Plan

### What to build: `experiments/25_head_identification_comprehensive.py`

Runs all feasible methods on a single model + operating point, producing a unified head ranking.

**Methods to implement (ordered by priority):**

| # | Method | Type | Speed | New code? |
|---|--------|------|-------|-----------|
| 1 | Per-head knockout | Causal | Slow (N_heads × trials) | Partial — extend exp 16 |
| 2 | DLA per head (PI condition) | Observational | Fast | Extract from exp 12 |
| 3 | Attribution patching | Causal approx | Fast | New |
| 4 | Attention primacy score | Observational | Fast | Extract from exp 16 |
| 5 | Attention entropy | Observational | Fast | New (trivial) |
| 6 | Copy score | Observational | Fast | New |
| 7 | Condition sensitivity | Observational | Fast | Extract from exp 16 |

**Output:** Single JSON with all scores per head, plus a combined ranking.

**The key question we're answering:** Do different methods agree on which heads matter? If yes → robust finding. If no → we report the disagreement honestly.

## Empirical Findings (2026-02-25)

### Cross-Model Results (25a Gold Standard Knockout)

| | Gemma 1B (2k,2u) | Qwen 0.5B (1k,5u) | Qwen 1.5B (1k,3u) |
|---|---|---|---|
| **Baseline PI** | 17% | 69% | 44% |
| **#1 primacy head** | **L14H2** (Δld=+20.2) | L12H0 (Δld=+1.8) | **L8H3** (Δld=+8.3) |
| **KO effect** | 17%→86% (+69%) | 69%→70% (+1%) | 44%→92% (+48%) |
| **Distribution** | Concentrated | Distributed | Concentrated |
| **Primacy head location** | L4-L14 (early/mid) | L3-L19 (spread) | L0-L8 (very early) |
| **Old exp 16 "primacy head"** | L20H1 (wrong) | L16H3 (wrong) | L19H1 (wrong) |

### Key Observation: Methods Disagree

**25a vs 25b:** Attribution patching (25b) consistently finds late-layer heads (L17-L27). Per-head knockout (25a) finds early/mid-layer heads (L0-L14). This happens on ALL three models, not just Gemma.

**Root cause:** Early-layer heads affect PI through indirect, nonlinear effects — they corrupt representations that downstream heads read. Attribution patching's first-order Taylor approximation cannot capture this multi-hop chain. This is a known limitation (Syed et al. 2023).

**25a vs 25c:** Observational metrics (DLA, primacy score, entropy, copy score, condition sensitivity) also miss the real primacy heads. L14H2 on Gemma has pi_primacy=0.396 (recency-biased by attention!) yet is the strongest primacy head by causal impact. L8H3 on Qwen 1.5B doesn't appear in any observational top-10.

**Practical implication:** For our paper, 25a (knockout) is the primary method. 25b is a useful fast screen for direct late-layer effects but cannot identify upstream root causes. Observational metrics (25c) should be reported as supplementary but cannot replace causal evidence.

### Attention Patterns Are Misleading

This is the strongest methodological finding from exp 25:
- A head can have **recency-biased attention** (pi_primacy < 0.5) yet be the **strongest primacy head** by causal impact
- A head can have **pi_primacy = 0.992** (our old L16H3) yet have **negligible causal effect** (Δld = +0.6)
- Attention-based head classification (our old exp 16 method, Voita et al. 2019) is insufficient for identifying causal primacy heads

The mechanism is not "head attends to initial value → promotes it." Instead, primacy heads operate **indirectly** — they corrupt intermediate representations that downstream retrieval heads read. The damage propagates through the residual stream, not through direct attention-to-value patterns.

### Exp 17b/18b: Primacy Mechanism Differs Across Architectures (2026-02-25)

Re-ran instruction sensitivity (17b) and forced attention (18b) with 25a-identified heads.

**17b results:** All primacy heads on both models ignore the query word "first"/"last" (0-1.8% attention). This finding replicates from old exp 17 and is now confirmed on the causally-correct heads.

**18b results — Two distinct mechanisms:**

| | Gemma L14H2 | Qwen 1.5B L8H3 |
|---|---|---|
| Knockout effect | PI: 14→71% (+57) | PI: 20→47% (+27) |
| Force correct effect | PI: 14→50% (+36) | PI: 20→21% (+1) |
| Force wrong effect | PI: 14→6% (-8) | PI: 20→21% (+1) |
| **Mechanism** | **A+B (mixed)** | **B (pure indirect)** |
| RI knockout effect | 72→40% (hurts RI) | 56→24% (hurts RI) |

**Gemma (A+B):** ~60% of damage through attention routing (forcing helps), ~40% through output corruption (persists after forcing). Forcing to wrong position hurts PI → attention direction matters.

**Qwen 1.5B (B):** 100% of damage through output corruption. Forcing to correct OR wrong position has zero effect. Attention direction is irrelevant. The head writes primacy-biased information regardless of what it reads.

**Both models:** The head is essential for RI (knockout hurts RI substantially). The head is instruction-insensitive (ignores "first"/"last"). The same head helps RI and hurts PI — it's not broken, it's a feature that doesn't adapt to the query instruction.

**Implication for the paper:** PI > RI is caused by heads that are "good at RI, bad at PI" — they reliably promote early-position values, which is correct for "first" queries and wrong for "last" queries. The asymmetry is architectural: causal attention naturally creates early-position bias, and some heads crystallize this bias into a fixed behavior that can't be overridden by the instruction.

### Exp 25d: Knockout Validation on Gemma — All Tests Passed (2026-02-25)

Three-way validation of top 3 primacy heads (L14H2, L4H1, L8H2) on Gemma 1B:

| Test | Question | Result |
|------|----------|--------|
| **V1** (200 trials, RI+PI) | General capability loss or primacy-specific? | **Primacy-specific:** all 3 heads help RI, hurt PI |
| **V2** (50 trials, Point C) | Same heads at different operating point? | **Yes:** all 3 rank top 6 at both points |
| **V3** (200 trials, CIs) | Statistically significant? | **Yes:** L14H2 PI=86% [81-91%] vs baseline 28% |

**H3 (general capability loss) is rejected.** If knockout caused general degradation, RI would drop AND PI would drop. Instead RI drops and PI rises — the effect is primacy-specific.

**Method 25a (per-head knockout) is validated** as a reliable primacy head identification technique on Gemma. Remaining question: does it validate equally well on Qwen models? (Qwen 0.5B showed distributed effects — validation may be weaker there.)

### Exp 25d: Knockout Validation on Qwen 1.5B — All Tests Passed (2026-02-26)

Three-way validation of top 3 primacy heads (L8H3, L0H7, L0H3) on Qwen 1.5B:

| Test | Question | Result |
|------|----------|--------|
| **V1** (200 trials, RI+PI) | General capability loss or primacy-specific? | **Primacy-specific:** L8H3 and L0H7 confirmed. L0H3 is PI-specific only. |
| **V2** (50 trials, 1k,5u) | Same heads at different operating point? | **Yes:** all 3 rank #1, #2, #3 out of 336 heads |
| **V3** (200 trials, CIs) | Statistically significant? | **Yes:** L8H3 PI=91% [86-94%] vs baseline 38% |

**Method 25a now validated on 2 of 3 models** (Gemma 1B + Qwen 1.5B). Both show concentrated primacy with one dominant head that is primacy-specific (not general capability loss).

**New finding: Two types of primacy heads.** L0H3 on Qwen 1.5B helps PI when knocked out but doesn't hurt RI — a PI-specific head. L8H3 and L0H7 are dual-purpose (help RI, hurt PI). The primacy circuit has heterogeneous components. Not seen on Gemma where all 3 validated heads were dual-purpose.

### Cross-Model 25d Summary

| | Gemma 1B | Qwen 1.5B | Qwen 0.5B |
|---|---|---|---|
| Validated? | Yes | **Yes** | Not run (distributed) |
| Top head stable across points? | Yes (top 6) | **Yes (top 3/336)** | N/A |
| All heads primacy-specific? | Yes (all 3) | **2/3 (L0H3 is PI-only)** | N/A |
| CIs tight? | 86% [81-91%] | **91% [86-94%]** | N/A |

---

## Head Selection Protocol for Real Experiments (Paper-Quality Runs)

### The Problem
`run_all.py` uses top-5 heads from 25a by default. This is fine for exploration
but insufficient for paper-quality claims. Different operating points, trial counts,
and random seeds can shift the rankings. We need a principled selection criterion.

### What the literature does (summary)

| Paper | Method | Criterion |
|-------|--------|-----------|
| Wang et al. (IOI) | Visual + functional validation | No formal threshold — visual inspection then confirm function |
| Wu et al. (Retrieval Heads) | Behavioral threshold | Copy score ≥ **0.1** (retrieves needle in 10%+ of trials) |
| Conmy et al. (ACDC) | Swept threshold τ | No fixed value — report ROC curve across τ values |
| Common practice | Top-K | Top 5–10 by effect size magnitude |

None use a universal threshold. The field leans toward "top-K + functional validation."

### Our Protocol (Two-Gate Criterion)

**Gate 1: Top-N by magnitude at primary operating point**
- Run 25a with 100+ trials at regime B operating point
- Take top N heads by `|causal_effect|` (sorted descending)
- N = 10 initially (generous); narrow down after Gate 2
- Keep primacy (positive causal_effect) and recency (negative) separate

**Gate 2: Stability across operating points**
- Re-run 25a (or use 25d V2) at a second operating point (same regime)
- Include only heads that appear in top-10 at BOTH points
- If the same head ranks #1–3 at both points → high confidence primacy head
- If ranking shifts significantly → head may be operating-point-specific noise

**Gate 3 (optional, for paper): Statistical significance**
- Run 25d V3 with 200 trials + bootstrap CIs on final selected heads
- Confirm: knockout helps PI AND hurts RI (primacy-specific, not general loss)
- Only then do they get cited in the paper

### Selecting Heads for run_all.py (Practical Guide)

**For exploration (any run):**
```bash
python experiments/run_all.py --model ... --points "2,3" --trials 50
# Uses top-5 auto-extracted from 25a — good enough for seeing patterns
```

**For paper-quality (real results):**
```bash
# Step 1: Run 25a at multiple operating points with 100+ trials each
python experiments/run_all.py --model ... --points "2,3 1,5" --trials 100 --exps "25a"

# Step 2: Inspect rankings — find heads stable across BOTH points
python3 -c "
import json
for pt in ['2k_3u', '1k_5u']:
    d = json.load(open(f'results/ModelName/{pt}/per_head_knockout_*.json'))
    print(pt, [(h['layer'], h['head'], h['causal_effect']) for h in d['top_primacy_heads'][:10]])
"

# Step 3: Manually select stable heads, pass to Phase 2
python experiments/run_all.py --model ... --points "2,3" --trials 100 \
    --phase 2 --heads "8,3 0,7 0,3"  # hardcoded from stability analysis
```

### Per-Model Validated Heads (Paper-Quality)

The following heads have passed both gates (25a + 25d validation):

| Model | Primary Point | Stable Primacy Heads | Causal Effect | Validated? |
|-------|--------------|---------------------|---------------|-----------|
| Gemma-3-1b-it | 2k,2u | **L14H2**, L4H1, L8H2 | +20.2, +8.1, +6.3 | ✓ 25d V1/V2/V3 |
| Qwen2.5-1.5B-Instruct | 1k,3u | **L8H3**, L0H7, L0H3 | +8.3, +3.1, +2.4 | ✓ 25d V1/V2/V3 |
| Qwen2.5-0.5B-Instruct | — | Distributed — no dominant head | — | Not run yet |
| Pythia-410m | — | TBD — run 25a first | — | Not run yet |
| Qwen2.5-3B-Instruct | — | TBD — run 25a first | — | Not run yet |

**When running Phase 2 on Gemma or Qwen 1.5B, always use the validated heads above,
not the auto-extracted top-5.** Use `--heads "14,2 4,1 8,2"` for Gemma,
`--heads "8,3 0,7 0,3"` for Qwen 1.5B.

### How Many Heads?

Rule of thumb from our data:
- **Concentrated model** (Gemma, Qwen 1.5B): 1–3 heads dominate. Use top-3.
- **Distributed model** (Qwen 0.5B, Pythia): Effects spread across 10+ heads.
  Use top-5 to top-10 for aggregate analysis. Hard to claim single "primacy circuit."
- **Indicator**: if #1 head has Δld > 3× the #2 head → concentrated. Otherwise distributed.

---

## References

- Wang et al. 2022 "Interpretability in the Wild" (IOI circuit) — per-head patching protocol
- Wu et al. 2024 "Retrieval Head Mechanistically Explains Long-Context Factuality" — copy score
- Syed et al. 2023 "Attribution Patching Outperforms Automated Circuit Discovery" — fast attribution
- Olsson et al. 2022 "In-context Learning and Induction Heads" — prefix matching score
- Voita et al. 2019 "Analyzing Multi-Head Self-Attention" — positional head classification
- Xiao et al. 2023 "Efficient Streaming Language Models with Attention Sinks" — attention sinks
- Heimersheim & Nanda 2024 "How to use and interpret activation patching" — patching best practices
- Conmy et al. 2023 "Towards Automated Circuit Discovery" (ACDC) — automated circuit discovery
