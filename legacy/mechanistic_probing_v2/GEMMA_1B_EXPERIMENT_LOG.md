# Gemma 3 1B IT — Experiment Log

## Model Info

- **Model:** google/gemma-3-1b-it
- **Params:** 1.3B
- **Architecture:** 26 layers, 4 heads (8 query / 4 KV, GQA 2:1), d_model=1152
- **Attention:** Sliding window (1024 tokens) + global, 5:1 interleaving. Global layers: L5, L11, L17, L23.
- **Context:** 32K (TransformerLens n_ctx=2048 for probing)
- **TransformerLens:** v2.17.0, natively supported (reports n_heads=4)
- **Single-token pool:** 1850/2300 words valid (Gemma tokenizer filters out 450)
- **Note:** Sequences are ~60-70 tokens at our operating points, well under 1024-token window. Sliding window is NOT a factor for our experiments.

## Role

Cross-family validation. Different architecture (Google vs Alibaba), different tokenizer, different training. If the same mechanistic findings replicate here → architecture-general, not Qwen-specific.

## Behavioral Sweep (Colab, synthetic values)

Run: 2026-02-22, 100 trials/cell, 10×8 pruned grid.

Key pattern: PI collapses at 3 updates for most key levels. Same PI > RI asymmetry as Qwen.

Notable: Regime D at 10k,1u (RI=39%, PI=69%) — artifact of garbage errors on RI, not real reversal.

## Recalibration (single-token values, 30 trials)

Run: 2026-02-25, local MPS.

Single-token values are harder (same as Qwen). PI collapses at 2 updates.

```
          upd= 1  upd= 2  upd= 3  upd= 5  upd=10  upd=15  upd=20
k= 2 RI    87%    77%    67%    73%    40%    67%    53%
     PI    77%    13%     0%     3%    13%     7%     7%
    gap  +10%  +63%  +67%  +70%  +27%  +60%  +47%

k= 3 RI    63%    80%    67%    70%    43%    33%    37%
     PI    73%    37%    13%     3%     0%     0%     3%
    gap  -10%  +43%  +53%  +67%  +43%  +33%  +33%

k= 5 RI    77%    70%    67%    60%    40%    60%    30%
     PI    77%    27%    13%     3%     3%     0%     0%
    gap   +0%  +43%  +53%  +57%  +37%  +60%  +30%
```

## Operating Points

| Point | Config | RI | PI | Gap | Regime | Rationale |
|-------|--------|----|----|-----|--------|-----------|
| A | 2k, 1u | 87% | 77% | 10% | Both work | Baseline |
| B | 2k, 2u | 77% | 13% | 63% | PI cracking | Primary probing point |
| C | 2k, 5u | 73% | 3% | 70% | Deep asymmetry | PI nearly dead |
| D | 5k, 2u | 70% | 27% | 43% | Multi-key | Scaling with keys |

### Comparison to Qwen operating points

| Model | Point B | RI | PI | Gap |
|-------|---------|----|----|-----|
| Qwen 0.5B | 1k, 5u | 60% | 17% | 43% |
| Qwen 1.5B | 1k, 3u | 68% | 19% | 49% |
| Gemma 1B | 2k, 2u | 77% | 13% | 63% |

Gemma has sharper PI collapse (fewer updates needed) but higher RI — wider gap.

## Experiment Status

| Exp | Name | A (2k,1u) | B (2k,2u) | C (2k,5u) | D (5k,2u) |
|-----|------|-----------|-----------|-----------|-----------|
| 12 | Logit lens + attention + DLA | DONE | DONE | DONE | DONE |
| 13 | Positional gradient | CRASH¹ | DONE | DONE | DONE |
| 14 | PI mass distribution | DONE | DONE | DONE | DONE |
| 15 | Activation patching | DONE | DONE² | DONE | DONE |
| 16 | Head identification | DONE | DONE | DONE | DONE |
| 17 | Instruction sensitivity | DONE³ | DONE | DONE | DONE |
| 18 | Forced attention | DONE | DONE | DONE | DONE |
| 19 | Positional bias sweep | DONE | DONE | DONE | DONE |
| 19b | Bias attention proof | CRASH⁴ | DONE | DONE | DONE |
| 20 | Minority override | CRASH⁴ | DONE | DONE | CRASH⁴ |
| 22 | Query patching granular | DONE | DONE² | DONE | DONE |
| 23 | Ablation patching interaction | DONE | DONE² | DONE | DONE |

¹ IndexError: 1 update = 1 value, no v1 to compute ratio
² Re-run with logit_diff fix (2026-02-25). Original used P(correct) which exploded to 500,000%.
³ 0 primacy heads found at this operating point — no heads to analyze
⁴ ZeroDivisionError / empty array when 0 primacy heads found

---

## Results by Operating Point

### Point A (2k, 1u) — Baseline

Trivial operating point: RI=83%, PI=83%. Both conditions work.

- **Exp 12:** P(init) and P(final) both ~0.98 at last layer. No differentiation needed.
- **Exp 14:** PI peak at last position 100% of the time (only 1 value, so it's trivially correct).
- **Exp 15:** Late-layer patching recovers ~39% (small effect — baseline already good).
- **Exp 16:** 0 primacy-biased heads found. All heads neutral at this trivial level.
- **Exp 17:** 0 primacy heads to analyze.
- **Exp 18:** "OV broken" — but misleading. At 1 update there's no interference, so forcing heads can't improve on an already-working baseline.

**Verdict:** Too easy for meaningful mechanistic analysis. Crashes in exp 13/19b/20 confirm this point doesn't have enough signal.

### Point B (2k, 2u) — Primary Probing Point

RI=82%, PI=32% (from exp 12 actuals).

**Exp 12 — Logit lens + attention + DLA:**
- RI: P(init)=0.92 at last layer. Strong.
- PI: P(init)=0.58, P(final)=0.41 at last layer. Initial value still dominates even when asked for last.
- Top retrieval head: L17H0 (retr=0.78, pi_primacy=0.59). Dominant retriever but only weakly primacy-biased.

**Exp 13 — Positional gradient:**
- RI: sharp primacy cliff (v1/v0=0.000). P concentrated entirely on v0.
- PI: v1/v0=0.627. More spread than Qwen (Qwen was ~0.000 in PI too). Gemma distributes PI probability more evenly.

**Exp 14 — PI mass distribution:**
- RI peak at v0 (rel=0.02). Correct.
- PI peak at v0.2 (rel=0.22). 78% of PI peaks in first 25% of positions. Model retrieves "an early value" not "the last value."

**Exp 15 — Activation patching (re-run with logit_diff):**
- PI recovery: 0% at L0-L12, **81% at L18, 98% at L24, 100% at L25**.
- Late-layer patching fully recovers PI. Same pattern as Qwen.

**Exp 16 — Head identification:**
- 104 total heads (26 layers × 4 heads).
- 14 retrieval heads (>10% value attention). All in L14-L23.
- **Only 1 "primacy-biased" head: L20H1** (pi_primacy=0.662, barely over 0.6 threshold).
- 1 recency-responsive head: L14H2 (pi_primacy=0.396).
- Ablation of L20H1 alone: zero effect (PI stays at 25%).
- Ablation of all retrieval heads: RI=0%, PI=0%.
- **Key difference from Qwen:** No strongly primacy-locked head. Qwen had L16H3 with pi_primacy=0.992 and condition_sensitivity=0.008. Gemma's "worst" is pi_primacy=0.662.

**Exp 17 — Instruction sensitivity:**
- Primacy heads put ~0.003 attention on query word "first"/"last". Same as Qwen — ignores instruction.
- Attention to initial value identical in RI and PI (0.1455).

**Exp 18 — Forced attention:**
- "Forcing attention is no better than knockout → head output is harmful regardless."
- BUT: this is at Point B where interference is low. At Points C and D (see below), forcing DOES work. The head barely matters at this operating point.

**Exp 19/19b — Positional bias:**
- No sweet spot for linear correction (same as Qwen).
- Blind bias diverts 93.4% more attention to non-value tokens. Oracle keeps attention on values.

**Exp 20 — Minority override:**
- W_OV norms similar (1.23x). Gain not the mechanism. Same negative result as Qwen.

**Exp 22 — Query patching granular (re-run with logit_diff):**
- Early layers (L0-L12): -23% (hurts).
- Late layers (L13+): **+54%** recovery.
- Neither attn_out nor mlp_out alone helps — cumulative residual stream.
- **Confirms hypothesis B:** query word processed correctly, accumulated value routing is corrupted.

**Exp 23 — Ablation + patching interaction (re-run with logit_diff):**
- Normal PI: 49%, Ablated PI: 49%. Ablating L20H1 has zero effect.
- Late-layer recovery: +48% in both modes.
- **Confirms distributed bias:** no single head to ablate.

### Point C (2k, 5u) — Deep Asymmetry

RI=55% (from exp 12 P(init)=0.55), PI=6%.

**Exp 12:** RI P(init)=0.55, PI P(init)=0.10, P(final)=0.08. Both are low — deep interference.

**Exp 13:** v1/v0=1.412 — **primacy cliff disappears** at 5 updates. Probability spread across positions. Different from Qwen where cliff persists.

**Exp 15:** PI recovery: 89% at L18, 98% at L21, **100% at L24**. Late-layer patching still works even at deep interference.

**Exp 17:** Heads ignore query word (same pattern). Attention to initial value constant (0.0642) regardless of instruction.

**Exp 18:** **"Forcing attention works BETTER than knockout → head is useful when correctly aimed."** OV circuit works at this operating point. The head matters more when interference is higher.

**Exp 19b:** Oracle bias works, blind fails. Same pattern.

### Point D (5k, 2u) — Multi-key

PI=26%.

**Exp 12:** PI=26% correct.

**Exp 13:** v1/v0=0.596 (moderate cliff). More positions to spread across with 5 keys × 2 values = 10 values.

**Exp 17:** Ignores query word. Attention to initial value constant (0.0883).

**Exp 18:** **"Forcing attention works BETTER than knockout."** OV works here too. Consistent with Point C.

---

## Key Findings — Gemma vs Qwen

### What replicates (architecture-general):

1. **PI > RI asymmetry** — confirmed across all operating points.
2. **Primacy cliff in RI** — sharp at low interference (v1/v0=0.000 at Point B).
3. **Late-layer patching recovers PI ~100%** — L18-L25 on Gemma (L21-L23 on Qwen). Same relative position (~70-100% through network).
4. **Heads ignore query word** — ~0.003 attention to "first"/"last" in both conditions.
5. **Query corruption is accumulated, not early** — early patching hurts, late patching helps. Hypothesis B confirmed on both.
6. **Linear positional bias doesn't help** — blind bias shifts to non-value tokens. Oracle works.

### What differs (architecture-specific):

1. **Head-level primacy bias is distributed on Gemma, concentrated on Qwen.**
   - Qwen: L16H3 has pi_primacy=0.992, condition_sensitivity=0.008. Ablating 8 heads closes the gap.
   - Gemma: Worst head is L20H1 with pi_primacy=0.662. Ablating it does nothing. No single head drives PI failure.

2. **Primacy cliff disappears at higher interference on Gemma.**
   - Qwen Point C: cliff persists (v1/v0 stays near 0).
   - Gemma Point C: v1/v0=1.412 — no cliff. Probability spreads across all positions.

3. **Forced attention is operating-point-dependent on Gemma.**
   - Points A/B (low interference): OV "broken" — forcing doesn't help.
   - Points C/D (high interference): OV works — forcing helps.
   - Qwen: OV works at all operating points (when correct head is forced).
   - Interpretation: Gemma's retrieval mechanism engages more strongly under higher interference.

### Implication for the paper:

~~PI > RI arises from two distinct mechanisms across architectures~~ **REVISED below after exp 25a/25b/25c.**

---

## Exp 25a/25b/25c: Comprehensive Head Identification (2026-02-25)

Ran 3 independent methods to identify primacy heads. Results challenge our earlier "distributed bias" conclusion.

### Results at Point B (2k, 2u)

**25a — Per-head causal knockout (gold standard, Wang et al. 2022):**

| Rank | Head | Δld | KO accuracy | Baseline |
|------|------|-----|-------------|----------|
| 1 | **L14H2** | **+20.2** | **86%** | 17% |
| 2 | L4H1 | +14.9 | 70% | 17% |
| 3 | L8H2 | +13.7 | 55% | 17% |
| 4 | L5H2 | +12.9 | 69% | 17% |
| 5 | L8H3 | +12.5 | 68% | 17% |

**L14H2 is the smoking gun:** knocking it out alone boosts PI from 17% to 86%. Massive single-head effect.

**Critical observation:** The primacy heads are in **early/mid layers (L4-L14)**, NOT in the late retrieval layers (L17-L23) where exp 16 was looking. Our old threshold-based classification completely missed these.

**25b — Attribution patching (Syed et al. 2023):**

Top heads: L17H0 (+15.0), L18H3 (+4.0), L20H2 (+2.9) — all late-layer.
L14H2 does NOT appear in top 15.

**25c — Observational metrics (5 methods):**

| Method | Top head | L14H2 rank |
|--------|----------|------------|
| DLA | L0H3 (+13.9) | Not in top 10 |
| Primacy score | L5H2 (0.833) | Not in top 10 (0.396 — recency!) |
| Low entropy | L22H2 | Not in top 10 |
| Copy score | L21H1 (0.02) | Not in top 10 |
| Retrieval score | L17H0 (0.788) | ranked 7th (0.185) |

Spearman correlations between methods are **weak** (most < 0.3).

### Key Observations

**1. Methods disagree — and that's the finding.**

The causal method (25a) found early/mid-layer heads (L4-L14). The observational methods (25c) and fast approximation (25b) found late-layer heads (L17-L23). These are measuring different things:

- **25a (knockout):** captures indirect effects. L14H2 corrupts the representation that downstream heads read. Its effect is nonlinear and propagates through many layers.
- **25b (attribution):** first-order Taylor approximation. Only captures direct, linear effects on the output. Finds late-layer heads because they're the last in the chain. Misses upstream root causes.
- **25c (observational):** attention patterns and DLA measure what heads do locally, not their downstream impact.

**2. Attribution patching ≠ knockout for upstream heads.**

This is a known limitation (Syed et al. 2023 acknowledged it). For a paper, 25a (knockout) is the trustworthy result. 25b is useful as a fast screen for late-layer direct effects but misses the mechanistic root cause.

**3. L14H2 has primacy_score = 0.396 (recency-biased by attention!).**

This is the most surprising result. L14H2 does NOT attend to the initial value — its attention pattern suggests it's recency-biased. Yet knocking it out massively helps PI. This means **its damage is NOT through direct value retrieval** but through some other mechanism (corrupting downstream representations, suppressing correct retrieval, etc.). Attention patterns alone cannot identify this head's role.

**4. "Distributed bias" conclusion was WRONG.**

Our earlier analysis (exp 16) found no strongly primacy-locked head on Gemma and concluded the bias was distributed. In reality, exp 16 was looking at the wrong metric (attention primacy at late layers). The real primacy heads are upstream and operate through indirect effects.

### Revised Understanding

| | Old (exp 16) | New (exp 25a) |
|---|---|---|
| Primary primacy head | L20H1 (weak, pi_primacy=0.662) | **L14H2** (strong, Δld=+20.2) |
| Location | Late layers (L17-L23) | **Early/mid layers (L4-L14)** |
| Mechanism | Direct attention to init value | **Indirect — corrupts downstream retrieval** |
| Single-head knockout effect | 0% (PI unchanged) | **+69% (PI: 17%→86%)** |
| Attention pattern | Weakly primacy-biased | **Recency-biased** (!) |

---

## Exp 17b/18b: Re-running with Causally-Identified Heads (2026-02-25)

Using L14H2 (from exp 25a) instead of L20H1 (from old exp 16).

### 17b — Instruction Sensitivity (L14H2 + top 4 primacy heads)

All 5 heads IGNORE the query word "first"/"last" (attn ≈ 0.003-1.3%).

**L14H2 specifically:**
- 82% attention to instruction tokens
- 7.7% to initial value, 7.9% to final value — nearly equal
- 0.03% to query word — completely blind to instruction
- Zero change between RI and PI conditions

L14H2 is instruction-insensitive but NOT primacy-biased by attention. It attends roughly equally to init and final values. Its primacy effect is through what it OUTPUTS, not where it LOOKS.

### 18b — Forced Attention (L14H2 only)

| Config | RI | PI | Gap | PI logit_diff |
|--------|----|----|-----|---------------|
| Baseline | 72% | 14% | +58% | -9.89 |
| **Knockout** | **40%** | **71%** | **-31%** | **+13.36** |
| **Force correct** | **78%** | **50%** | **+28%** | **+4.44** |
| Force wrong | 62% | 6% | +56% | -14.22 |

**Mechanism: A+B (mixed)**

- Forcing to correct position: PI 14% → 50% (+36%). OV circuit extracts useful signal when aimed right.
- But knockout is better: PI 14% → 71% (+57%). Remaining 21% gap = damage persists even with correct attention.
- Forcing to wrong position: PI drops to 6%. Head has real causal power through attention.
- **Knockout HURTS RI**: 72% → 40%. L14H2 is essential for RI — it's not broken, it's instruction-insensitive.

**Two pathways of damage:**
1. ~60% of damage through attention routing (fixable by forcing)
2. ~40% through representation corruption (persists regardless of attention)

### Cross-Model Comparison (18b forced attention)

| | Gemma L14H2 | Qwen 1.5B L8H3 |
|---|---|---|
| Baseline PI | 14% | 20% |
| Knockout PI | 71% (+57) | 47% (+27) |
| Force correct PI | 50% (+36) | 21% (+1) |
| Force wrong PI | 6% (-8) | 21% (+1) |
| **Mechanism** | **A+B (mixed)** | **B (pure indirect)** |

Different primacy mechanisms across architectures:
- **Gemma:** Head partially acts through attention routing, partially through output corruption
- **Qwen 1.5B:** Head acts ENTIRELY through output corruption — attention direction is irrelevant

Both heads are instruction-insensitive (ignore "first"/"last"). Both are essential for RI. Both hurt PI when present. But the causal pathway differs.

---

## Exp 25d: Knockout Validation — Three-Way Confirmation (2026-02-25)

The "are we sure?" experiment. Three independent validation tests for the top 3 primacy heads identified by 25a.

### V1: Knockout on BOTH RI and PI (200 trials, primary point 2k,2u)

**Question:** Is the knockout effect primacy-specific (helps PI, hurts RI) or general capability loss (hurts both)?

| Head | RI baseline | RI knockout | RI Δld | PI baseline | PI knockout | PI Δld | Verdict |
|------|------------|-------------|--------|------------|-------------|--------|---------|
| **L14H2** | 89% | 50% | **-17.94** | 28% | **86%** | **+20.93** | **PRIMACY** |
| **L4H1** | 89% | 50% | **-14.52** | 28% | **61%** | **+13.16** | **PRIMACY** |
| **L8H2** | 89% | 48% | **-13.87** | 28% | **56%** | **+12.72** | **PRIMACY** |

All three: **RI drops, PI rises.** This is primacy-specific, NOT general capability loss. The heads help RI (correctly promoting first value) and hurt PI (incorrectly overriding last value). Hypothesis H3 (general capability loss) is **rejected**.

### V2: Full sweep at SECOND operating point (50 trials, Point C: 2k,5u)

**Question:** Do the same heads rank high at a different interference level?

Baseline PI at Point C: 8%, mean_ld=-8.73.

| Rank at Point C | Head | Δld | Also ranked at Point B? |
|-----------------|------|-----|------------------------|
| **1** | **L4H1** | +19.9 | Yes — rank 2 at Point B |
| **2** | **L14H2** | +16.1 | Yes — rank 1 at Point B |
| 3 | L9H1 | +15.4 | New |
| 4 | L10H2 | +14.4 | New |
| 5 | L5H2 | +14.1 | Rank 4 at Point B |
| **6** | **L8H2** | +13.6 | Yes — rank 3 at Point B |

**All 3 target heads rank in top 6 at the secondary point.** L4H1 actually becomes #1 (was #2 at Point B). The ranking is **stable across operating points.**

Note: L4H1 and L14H2 swap positions — L4H1 is stronger at higher interference. This makes sense: at deeper interference (5 updates vs 2), early-layer heads have more corrupted signal to propagate.

### V3: Targeted knockout with bootstrap CIs (200 trials)

| Head | Condition | KO accuracy | 95% CI | Baseline |
|------|-----------|-------------|--------|----------|
| L14H2 | RI | 50% | [43%-57%] | 89% |
| **L14H2** | **PI** | **86%** | **[81%-91%]** | **28%** |
| L4H1 | RI | 50% | [43%-56%] | 89% |
| **L4H1** | **PI** | **61%** | **[54%-67%]** | **28%** |
| L8H2 | RI | 48% | [42%-55%] | 89% |
| **L8H2** | **PI** | **56%** | **[49%-63%]** | **28%** |

CIs are tight. L14H2 PI knockout: 86% [81-91%]. The lower bound (81%) is still massively above baseline (28%). No overlap with baseline CI. This is not noise.

### Validation Summary

| Test | Result | Status |
|------|--------|--------|
| V1: Primacy-specific? | All 3 heads: RI drops, PI rises | **CONFIRMED** |
| V2: Stable across points? | All 3 heads rank top 6 at both points | **CONFIRMED** |
| V3: Tight CIs? | L14H2 PI: 86% [81-91%] vs baseline 28% | **CONFIRMED** |

**These are the most validated findings in our entire study.** Three independent tests, 200 trials with CIs, stable across operating points, primacy-specific (not general loss). The head identification methodology (25a per-head knockout) produces real, reproducible, causal results.

---

## Bug Fixes Applied

### Recovery metric explosion (exp 15, 22, 23)

**Problem:** Recovery formula `(patched_p - corrupted_p) / (clean_p - corrupted_p)` using P(correct) exploded to 26,566% and 502,725% when denominator was small.

**Fix:** Switched to logit difference metric following Wang et al. 2022 (IOI) and Heimersheim & Nanda 2024:
- `logit_diff = logit(correct) - logit(incorrect)`
- `recovery = (patched_ld - corrupted_ld) / (clean_ld - corrupted_ld)`
- `compute_recovery()` returns `None` when `|denom| < 0.01` (true numerical zero only)
- `aggregate_recovery()` filters None values and reports `n_filtered`
- No silent clamping — degenerate trials are dropped and counted transparently

Added to `core/analysis_utils.py`: `compute_logit_diff()`, `compute_recovery()`, `aggregate_recovery()`.

### Edge case crashes (exp 13, 19b, 20)

**Not yet fixed.** These crash at Point A (1 update) or when 0 primacy heads are found:
- Exp 13: IndexError when only 1 value (no v1 to compute ratio)
- Exp 19b: ZeroDivisionError when 0 primacy heads
- Exp 20: ValueError on empty array when 0 primacy heads

Low priority — these are trivial operating points. The data that matters (B, C, D) works.
