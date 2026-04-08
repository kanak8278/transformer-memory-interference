# Honest Assessment: PI(N) Scaling Law Claim

**Date:** 2026-03-18
**Issue:** The claim "PI(N) = a·exp(-b·N) + c" is not supported by most of our data.

---

## What the Data Actually Shows

### Model-by-model verdict:

| Model | Data points (2k) | Actual pattern | Curve claimable? |
|---|---|---|---|
| Qwen 1.5B-Inst | 3 (N=5,10,20) | 60%→42%→23% | ❌ 3-point, 3-param fit = R²=1.0 always |
| Qwen 3B-Base | 11 (N=1–50) | 100%→90%→75%→50%→34% | ✅ Real decay visible |
| Qwen 3B-Inst | 3 (N=5,10,20) | 73%→54%→46% | ❌ Same as 1.5B — 3 points |
| Gemma 1B | 9 (N=5–50) | 10%→6%→10%→12%→6%→2%→0% | ❌ Flat noise at ~5-8% floor |
| TinyLlama 1.1B | 3 (N=5,7,10) | 4%→3%→0% | ❌ Already at floor by N=5 |
| Mamba 1.4B | 3 (N=5,7,10) | 7%→8%→10% | ❌ Too few, not even decaying |
| Pythia 410M | 10 (N=5–50) | 18%→8%→8%→6%→8%→20%→5% | ❌ Noise around ~8% floor |
| StableLM 1.6B | 3 (N=5,7,10) | 25%→9%→— | ❌ 3 points |
| Qwen 0.5B | 10 (N=2–50) | 62%→52%→34%→28%→14%→— | ⚠️ Possible decay but noisy |

**Only Qwen 3B-Base genuinely shows a decaying curve with enough data points.**

---

## Why the R² Numbers Are Misleading

We reported R²=0.64–1.0 across models. Here's why that's wrong:

- **R²=1.0 for Qwen 1.5B and 3B-Inst:** Fitting a 3-parameter model (a, b, c) to exactly 3 data points is always a perfect fit. It proves nothing. Any smooth function fits 3 points exactly.

- **R²=0.98 for Qwen 3B-Base:** This IS real — 11 data points, genuine decay. This one is valid.

- **R²=0.32 for Gemma:** Low R² because the data is flat noise. The model hits ~5-10% floor at N=5 and stays there. There's no decay curve — just a floor.

- **R²=0.88 for Mamba:** Only 3 points. Same problem as Qwen 1.5B.

The honest R² range for models with ≥6 data points AND actual decay: **R²=0.98 (Qwen 3B-Base only).**

---

## What We Can and Cannot Claim

### ✅ Can claim:
- "PI accuracy decays with N" — direction is consistent across all models
- "Qwen 3B-Base shows a clear exponential decay with floor: PI(N) ≈ 34% + 66%·exp(-0.08·N), R²=0.98"
- "Most models hit their performance floor by N=5–10, making the decay region too narrow to characterize"

### ❌ Cannot claim:
- "PI(N) = a·exp(-b·N) + c fits well across models (R²=0.64–1.0)"
- Any R²=1.0 fit as evidence
- That the exponential form is empirically validated across architectures

---

## Options Going Forward

### Option A: Re-run with more N values per model (recommended)
Add N=3,7,15,30,50 to TinyLlama, StableLM, Mamba, Qwen 1.5B. This would give 6-8 points per model and make the fits meaningful. Estimated effort: ~4 hours GPU time.

Key cells to add for each model:
- Currently missing: N=1,2,3 (to see where it starts) and N=15,20,30 (to see the tail)
- Target: at least 6 data points spanning the full range before floor

### Option B: Drop the scaling law claim for most models
Only claim the exponential fit for Qwen 3B-Base (the one model with real data).
Reframe as: "We illustrate the decay pattern on Qwen 3B-Base where we have 11 data points; other models reach their performance floor too quickly to characterize the decay."

### Option C: Change the claim to qualitative
Remove the functional form entirely. Say: "PI accuracy decreases as N increases and converges to a model-dependent floor." This is fully supported by all models and doesn't require curve fitting.

---

## Recommendation

**Option C is the safest for immediate submission.** Drop the equation and R² claims. Keep:
- The empirical observation that PI degrades with N
- Show the Qwen 3B-Base curve as a representative illustration
- Acknowledge other models hit floor too quickly to fit

**Option A if we have time.** A focused re-run at 6-8 N-values for TinyLlama, StableLM, Mamba, Qwen 1.5B would take ~4 hours and would let us make the claim properly across 5+ models.

---

## Files Affected

The following files make the incorrect scaling law claim and need updating:

- `paper/drafts/PAPER_RESULTS_DRAFT.md` — Section 4.2
- `paper/drafts/ABSTRACT_DRAFT.md` — all 3 drafts mention "exponential decay"
- `paper/drafts/INTRODUCTION_DRAFT.md` — contribution 3
- `paper/drafts/PAPER_OUTLINE.md` — Section 4.2
- `paper/drafts/ONE_PAGER_FULL_PROJECT.md` — "PI decays exponentially" claim
