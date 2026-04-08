# Honest Assessment: Architecture-Dependent Error Pattern Claim

**Date:** 2026-03-18
**Issue:** The claim of 3 distinct architecture-dependent error patterns is not well-supported.

---

## The Claim We Were Making

> "We observe three distinct architecture-dependent error patterns:
> (1) recency imprecision in multi-head models (Qwen),
> (2) primacy fallback in few-head models (Gemma),
> (3) off-by-one lock in base models (Pythia)."

---

## What the Actual Data Shows

### Qwen 1.5B — "Recency imprecision / off-by-one"

| N | Top error position | % at that position | Pattern |
|---|---|---|---|
| 5 | penultimate (idx=3) | 63% | Strong off-by-one |
| 10 | penultimate (idx=8) | 37% | Off-by-one still dominant |
| 20 | mid-range (diffuse) | ~5% each | Spreading out |

**Verdict:** There IS a real pattern here. At low N, errors strongly cluster at
penultimate. At high N they spread. The "recency imprecision" framing is reasonable.
**This pattern is real and claimable.**

---

### Gemma 1B — "Primacy fallback"

| N | idx=0 (first) | idx=1 | idx=2 | Pattern |
|---|---|---|---|---|
| 5 | 9% | 36% | 27% | Spread across early values |
| 10 | 3% | 26% | 26% | Concentrated at early indices |
| 20 | 15% | 12% | 15% | Some primacy (idx=0,1,2 = 42%) |
| 30 | **42%** | 22% | 10% | Strong primacy at high N |

**Verdict:** At low N (5), Gemma errors are SPREAD across early positions (idx 1,2,3),
not specifically at idx=0. The "primacy fallback" only clearly appears at N=30 where
42% of errors are at idx=0. At N=5, the near_first=9% is not "primacy fallback."

The claim that Gemma shows "primacy fallback" is somewhat true at high N but overstated
at low N. It's better described as: "Gemma errors cluster at early positions generally,
with the first position becoming dominant only at high N."

**Verdict: Partially claimable — weaker than stated, needs qualification.**

---

### Pythia 410M — "Off-by-one lock"

| N | idx=0 (first) | penultimate | Pattern |
|---|---|---|---|
| 5 | 41% | 14% | Primacy dominant! |
| 10 | 37% | 30% | Mixed — both primacy and penultimate |
| 20 | 17% | 50% | Penultimate dominant |
| 30 | 0% | 90% (idx=27,28) | Strong penultimate |

**Verdict:** This is actually the OPPOSITE of "off-by-one lock." At low N, Pythia shows
PRIMACY bias (41% at idx=0). At high N, it shifts to penultimate. The error pattern is
not consistent — it flips depending on N. The "off-by-one lock" label is wrong.

A better description: "Pythia shows mixed errors at low N (primacy and near-last
roughly equal), transitioning to near-last at high N."

**Verdict: Claim is WRONG as stated. Do not use.**

---

## Why This Is Problematic

1. **Post-hoc taxonomy** — we looked at the data and invented labels to fit what we saw.
   No pre-registered predictions.

2. **Small samples** — Pythia at N=30 has only 10 error trials. Gemma at N=10 has 39.
   These are not enough to reliably characterize distributions.

3. **The "architecture determines error mode" claim lacks evidence** — we have one
   model per "type." Qwen is one model, Gemma is one model, Pythia is one model.
   We'd need multiple models of each type to claim it's the architecture causing it,
   not just that model's specific training.

4. **The actual connection to architecture is speculative** — we claim Gemma's 4 heads
   cause primacy fallback, and Qwen's 16 heads cause recency imprecision. But we have
   no causal evidence for this. The head count is correlated with many other differences
   (model size, training data, etc.).

---

## What We Can Honestly Claim

### ✅ Safe to claim:
- "Qwen 1.5B PI errors cluster at the penultimate position at low N, spreading at high N"
  (this is clearly visible in the data with 78-116 error trials)
- "Error distributions differ across models" (observational, no causal claim)
- "At low N, errors are near-last; at high N, errors spread" (for Qwen at least)

### ❌ Do not claim:
- The three-pattern taxonomy as architecture-determined
- "Primacy fallback" for Gemma (at low N it's not primacy)
- "Off-by-one lock" for Pythia (it's primacy at low N, not off-by-one)
- Any interpretation connecting head count to error mode (no causal evidence)

---

## Recommendation

**Drop the three-pattern taxonomy entirely from the paper.**

Replace with: "Error distributions vary across models and operating points.
For Qwen 1.5B we observe that errors at low N (5 updates) cluster strongly at the
penultimate position (63%), consistent with positional confusability between v_{N-1}
and v_{N-2} at close relative positions."

That's it. Don't overgeneralize to a taxonomy that the data doesn't support.
