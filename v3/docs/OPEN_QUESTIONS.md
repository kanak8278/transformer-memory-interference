# Open Questions for NeurIPS Paper

**Updated: 2026-03-17 (Post-Theory Revision + Full Senior Review)**

---

## Questions ANSWERED with Strong Evidence

### Q1: Does PI > RI appear across architectures? ✅ DONE
**Answer: YES.** 9 usable models across 2 broad architecture classes (5 transformer variants + 1 SSM). PI > RI in 91% of cells (51/56), mean gap +50.3%.

**Caveat (from senior review):** "7 architecture families" overcounts — Qwen/TinyLlama/Pythia are all GPT-style transformers. Honest framing: "5 transformer variants + 1 SSM."

### Q2: Is the asymmetry in representation space? ✅ DONE
**Answer: YES.** Probing classifiers across 3 models show RI correctness well-encoded (60-87%), PI correctness weakly above chance (61%).

**Caveat:** 61% is NOT "at chance" — it's p<0.01 above 50%. Say "weakly above chance."

### Q3: Where in the network does PI fail? ✅ DONE
**Answer: Last ~15% of layers.** Logit lens: v_last peaks at ~90% depth then suppressed. Consistent Pattern B ("Overtaken") across all 4 models. This is our strongest mechanistic result.

### Q4: Is the asymmetry architectural or learned? ✅ DONE
**Answer: BOTH.** Jacobian at init shows primacy in untrained models (Qwen 1.47×, Mamba 295×). Training amplifies both primacy and recency.

**Caveat:** 295× is position 0 vs position 49 (extremes). Quartile ratio is 1.24×. Present both numbers.

### Q5: Is there a concentrated "interference circuit"? ✅ DONE
**Answer: NO.** Max patching delta 0.03-0.07 per head. Ablating top heads hurts retrieval. Distributed mechanism = architectural property.

---

## Questions PARTIALLY Answered

### Q6: Scaling law ⚠️ PARTIAL (done but with caveats)
**What we have:**
- PI(N) = a·exp(-b·N)+c fits well for models with ≥6 data points (R²=0.64-0.98)
- PI vs model size at N=20: r=0.886

**Caveats (from senior review):**
- R²=1.0 for Qwen 1.5B/3B-Inst is MEANINGLESS (3-point, 3-parameter fit). Don't report these.
- PI vs size p=0.057 after Bonferroni correction. Frame as "suggestive," not "significant."
- Gemma R²=0.32, Pythia R²=0.18 — fits poor for some models.

**Still need:** More models at ≥6 N-values each for robust fits.

### Q8: Is autoregressive direction the key? ⚠️ PARTIAL (weak evidence)
**What we have:**
- BERT MLM: 0% both conditions (can't do KV retrieval at all — uninformative)
- Flan-T5-base: RI=13%, PI=22%, gap=-9%. No primacy bias but 77% garbage.

**Caveats (from senior review):**
- T5-base result is inconclusive (CIs overlap, high garbage)
- BERT result tells us nothing about bidirectional encoding — it just can't do MLM for this task
- Cannot claim "confirms autoregressive is the cause"

**Honest status:** Preliminary evidence pointing toward autoregressive encoding as factor. Move to appendix, don't make it a main claim.

### Q10: Error position formula ⚠️ PARTIAL (observation, not formula)
**What we have:** Three architecture-dependent failure modes:
1. Qwen: recency imprecision → diffuse (entropy increases, p=0.008)
2. Gemma: primacy fallback (entropy decreases, p<0.001)
3. Pythia: off-by-one lock (penultimate increases)

**Caveats:** Post-hoc taxonomy. Only Qwen 3B-Base has enough data points (8+) for significant correlations. Present as "observed patterns" not "predictive model."

### Q11 (NEW): Is the formal theory complete? ⚠️ PARTIAL
**What we have:**
- C1 PROVEN for single-layer softmax (algebraic proof, 10K numerical verification)
- C2 DERIVED from Chowdhury's influence density (ρ_H strictly decreasing)
- Proposition: under C1+C2, I(h_query; v_1) > I(h_query; v_N) for large N
- 8 predictions, 7 confirmed empirically

**Gaps identified by senior review:**
1. **C1 does NOT extend to multi-layer** — single-layer proof doesn't cover indirect information flow via residual connections. Empirically verified but formally conjectured.
2. **C1 for SSMs (HiPPO) is contradicted** — our own transient growth data shows ||A^49||>1, violating the ||A||≤1 assumption. Empirically C1 holds but the proof breaks.
3. **Convergence gap: I_∞ > 0 not guaranteed** — C2 gives Δ_j non-increasing but Σ Δ_j could diverge (harmonic series). Need additional condition.
4. **Part (ii) needs independence assumption** — must assume values are independently sampled.
5. **C2 for SSMs is hand-waving** — not derived, only verbal argument.

---

## Questions NOT Answered

### Q7: Can PI > RI be fixed? ❌ NOT STARTED
**What we need:** Intervention experiments (landmark tokens, temperature, retrieval training).
**Effort:** High. Deprioritized for this submission.

### Q9: What is the exact circuit? ❌ DEPRIORITIZED
**Honest assessment:** The mechanism is distributed. Searching for a circuit is the wrong question. The finding IS that there's no circuit. Edge attribution would be interesting for a workshop paper but isn't needed for this submission.

---

## Central Theoretical Question: Status

**Q: Why does autoregressive processing + continuous gating + fixed-capacity state create PI > RI?**

**Answer (partial):** We formalize this as C1 (monotone overwrite) + C2 (diminishing marginal overwrite) + fixed capacity. Under these conditions, first-value information converges to a positive limit while last-value information is bounded by shrinking capacity.

**Proof chain:**
- C1 proven for single-layer softmax ✓
- C2 derived from Chowdhury ✓
- Proposition proven given C1+C2 ✓
- Multi-layer extension: conjectured, empirically verified ⚠️
- SSM extension: C1 proof broken for HiPPO, empirically holds ⚠️

**For the paper:** Frame as "theoretical framework with partial formal backing" not "complete proof." The empirical evidence (7/8 predictions confirmed) is the strongest argument.

---

## What Would Make This NeurIPS-Ready

| Requirement | Status | Blocking? |
|---|---|---|
| Cross-architecture evidence | ✅ Done (9 models) | No |
| Mechanistic evidence (4 techniques) | ✅ Done | No |
| Theoretical framework | ⚠️ Partial (C1 proven, gaps remain) | No — frame correctly |
| Statistical rigor (200 trials) | ⚠️ 3 models at 30 trials | YES — need Mamba/TinyLlama/StableLM reruns |
| Figures (publication quality) | ⚠️ 6/8 good, 2 need work | No — minor fixes |
| LaTeX paper draft | ❌ Not started | YES — all content exists in markdown |
| Honest claim scoping | ⚠️ Several overclaims identified | YES — apply senior review fixes |
