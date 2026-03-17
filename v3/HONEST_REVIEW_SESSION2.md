# Honest Self-Review: Session 2 (Updated Post-Theory + Full Senior Review)

**Date:** 2026-03-17
**Goal:** Every weak point, unsubstantiated claim, and gap — with current status.

---

## Claim-by-Claim Audit (13 claims reviewed)

### CLAIM 1: "PI > RI across 11 models, 7 architecture families"
**Problem:** Actually 9 usable models. "7 families" overcounts — 5 are transformer variants.
**Severity:** MODERATE
**Status:** ⚠️ NOTED — fix wording to "9 models, 2 architecture classes (5 transformer variants + 1 SSM + 2 API)"

### CLAIM 2: "PI(N) = a * exp(-b*N) + c with R²=0.64-1.0"
**Problem:** R²=1.0 from 3-point fit is meaningless. Poor fits for Gemma (0.32), Pythia (0.18).
**Severity:** HIGH
**Status:** ✅ FIXED — noted in PAPER_RESULTS_DRAFT.md. Report only ≥6-point fits (R²=0.64-0.98).

### CLAIM 3: "PI vs model size: r=0.886, p=0.019"
**Problem:** p=0.057 after Bonferroni for 3 N-values tested. Only 6 data points.
**Severity:** HIGH
**Status:** ✅ FIXED — reframed as "suggestive" in PAPER_RESULTS_DRAFT.md.

### CLAIM 4: "Error patterns are architecture-dependent (3 modes)"
**Problem:** Post-hoc taxonomy. Only Qwen 3B-Base has enough data for significance.
**Severity:** MODERATE
**Status:** ⚠️ NOTED — present as "observed patterns" not "predictive model."

### CLAIM 5: "Probing shows 100% condition discrimination"
**Problem:** Probe may detect query word "first"/"last."
**Severity:** HIGH (initially).
**Status:** ✅ RESOLVED — the KEY finding is the within-condition correctness probes (87% vs 61%), which can't be explained by query word. The condition probe is expected and correctly interpreted.

### CLAIM 6: "PI correctness at chance (50-61%)"
**Problem:** 61% is p<0.01 above chance. Not "at chance."
**Severity:** LOW
**Status:** ⚠️ NOTED — change wording to "weakly above chance."

### CLAIM 7: "Mamba-1.4B shows PI > RI"
**Problem:** 30 trials, CI ±18%, 17-30% garbage.
**Severity:** MODERATE
**Status:** 🔄 IN PROGRESS — 200-trial rerun running.

### CLAIM 8: "Narrative transfer: gap=+18%"
**Problem:** 30 trials/cell, CIs overlap at 95% level.
**Severity:** HIGH
**Status:** ❌ NOT FIXED — need 100+ trials. Deprioritized below 200-trial model reruns.

### CLAIM 9: "Jacobian at init: Mamba 295x primacy"
**Problem:** Extreme comparison (position 0 vs 49). Quartile ratio is 1.24×.
**Severity:** LOW
**Status:** ⚠️ NOTED — present both numbers. Lead with quartile ratio.

### CLAIM 10: "Bidirectional models show no PI > RI"
**Problem:** BERT can't do task (0% both). T5-base has 77% garbage.
**Severity:** HIGH
**Status:** ⚠️ NOTED — downgrade to "preliminary, appendix" not main claim. T5-large failed to run.

### CLAIM 11: "Formal proof of PI > RI" (NEW — from theory revision)
**Problem 11a:** C1 proven for single-layer only. Multi-layer extension is conjecture.
**Problem 11b:** C1 for HiPPO SSMs contradicted by own transient growth data (||A^49||>1).
**Problem 11c:** Convergence I_∞ > 0 has gap — C2 alone doesn't prevent harmonic-series divergence.
**Problem 11d:** Part (ii) capacity argument needs independence assumption made explicit.
**Problem 11e:** C2 for SSMs is verbal argument, not formal.
**Problem 11f:** d=256 HiPPO result (10^12) is numerical instability, not real physics.
**Severity:** HIGH (collectively)
**Status:** ⚠️ IDENTIFIED — each needs specific fix (see SENIOR_REVIEW_THEORY.md). Total ~65 min of work.

### CLAIM 12: "Component elimination table proves autoregressive + gating + capacity" (NEW)
**Problem:** The table shows what components ARE eliminated (attention, softmax, RoPE) but doesn't PROVE the remaining three are necessary. A model could violate our pattern and still have autoregressive processing.
**Severity:** MODERATE
**Status:** ⚠️ NOTED — reframe as "narrows the candidate mechanisms" not "proves the cause."

### CLAIM 13: "200-trial Wilson CIs" (NEW)
**Problem:** Only 3 of 9 models actually have 200 trials. Others have 30-50.
**Severity:** HIGH — if we claim "200-trial statistical rigor" but only 3 models have it.
**Status:** 🔄 Mamba running. TinyLlama/StableLM pending. Qwen 0.5B/Gemma/Pythia at 50.

---

## Resolution Summary

| # | Claim | Severity | Status |
|---|---|---|---|
| 1 | "11 models, 7 families" | MOD | ⚠️ Fix wording |
| 2 | R²=0.64-1.0 | HIGH | ✅ Fixed (report ≥6-point only) |
| 3 | PI vs size p=0.019 | HIGH | ✅ Fixed (Bonferroni noted) |
| 4 | Three error modes | MOD | ⚠️ Present as observation |
| 5 | 100% condition disc. | HIGH→LOW | ✅ Resolved (correctness probes are key) |
| 6 | PI "at chance" | LOW | ⚠️ Fix wording to "weakly above" |
| 7 | Mamba PI > RI | MOD | 🔄 200-trial running |
| 8 | Narrative +18% | HIGH | ❌ Need 100+ trials |
| 9 | 295× primacy | LOW | ⚠️ Report quartile ratio too |
| 10 | Bidirectional control | HIGH | ⚠️ Move to appendix |
| 11 | Formal theory | HIGH | ⚠️ 6 sub-issues identified, ~65 min to fix |
| 12 | Component elimination | MOD | ⚠️ Reframe wording |
| 13 | 200-trial claim | HIGH | 🔄 3/9 models done, Mamba running |

**Resolved:** 3/13 ✅
**Noted (need wording fixes):** 6/13 ⚠️
**In progress:** 2/13 🔄
**Not fixed:** 2/13 ❌ (narrative, some theory fixes)

---

## Priority Actions (Updated)

### Must-do before submission:
1. **Fix theory gaps** — apply 5 fixes from SENIOR_REVIEW_THEORY.md (~65 min)
2. **200-trial Mamba** — running, wait for results
3. **200-trial TinyLlama + StableLM** — queue after Mamba
4. **Apply wording fixes** — claims 1, 4, 6, 9, 10, 12 (30 min)
5. **LaTeX draft** — all content exists in markdown

### Should-do:
6. **Narrative 100+ trials** — strengthen narrative transfer claim
7. **Fix d=256 in verify_theory.py** — use proper discretization or drop
8. **Improve Fig 7 and Fig 8** — scaling law and error position figures

### Can skip for submission:
9. Q7 (fixing PI) — future work
10. Q9 (exact circuit) — wrong question for distributed mechanism
11. T5-large bidirectional — T5-base is sufficient for appendix
