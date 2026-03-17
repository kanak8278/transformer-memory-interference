# Open Questions for NeurIPS Paper

## Questions We CAN Answer (Evidence Exists)

### Q1: Does PI > RI appear across architectures?
**Answer: YES.** Tested 11 models across 7 families (Qwen, Gemma, Llama, StableLM, Pythia, Mamba, API). All show PI > RI (except garbage-dominated Mamba-130M and RWKV-430M).

**Strength:** Very strong. 200-trial CIs are non-overlapping. Cross-architecture consistency is compelling.

### Q2: Is the asymmetry in representation space, not just output?
**Answer: YES.** Probing classifiers show 96-100% RI/PI condition discrimination across 3 models. RI correctness is well-encoded (76-89%), PI correctness is near-chance (50-72%).

**Strength:** Strong. Three models, new MI technique, independent from logit lens.

### Q3: Where in the network does PI fail?
**Answer: Last ~15% of layers.** Logit lens shows v_last appears at ~90% depth then gets suppressed by competitor (usually penultimate value). Consistent across 4 models.

**Strength:** Strong for transformers. Not tested for Mamba (no TransformerLens support).

### Q4: Is the asymmetry architectural or learned?
**Answer: BOTH.** Jacobian at initialization shows primacy bias exists in untrained models (Qwen: 1.47× middle, Mamba: 295× first/last). Training amplifies both primacy and recency, but the primacy foundation is architectural.

**Strength:** Strong. Novel experiment comparing untrained vs pretrained across architectures.

### Q5: Is there a concentrated "interference circuit"?
**Answer: NO.** Causal patching on 3 models shows max delta +0.032-0.071 per head. Ablating top heads HURTS retrieval. The mechanism is distributed.

**Strength:** Moderate. The "distributed mechanism" finding is real but hard to publish as a positive result. Need to frame as: "PI > RI is an architectural property, not a fixable bug in a few heads."

## Questions We CANNOT Yet Answer (Gaps)

### Q6: What is the formal scaling law?
**What we have:** RI scales with model size, PI doesn't. At high N, PI converges to ~15-20%.
**What we need:** A formal expression: PI(N, d_model) = f(N, d_model) with a fitted function.
**Effort:** Moderate (need to fit curves to existing data).

### Q7: Can PI > RI be fixed?
**What we have:** Nothing.
**What we need:** Test interventions (landmark tokens, adaptive temperature, retrieval training).
**Effort:** High (each intervention is a separate experiment).

### Q8: Is the autoregressive direction the key?
**What we have:** Inconclusive BERT probe (detected query word, not value encoding).
**What we need:** Properly designed bidirectional probe, or seq2seq model comparison.
**Effort:** Moderate (redesign probe or test T5/BART).

### Q9: What is the exact circuit (for transformers)?
**What we have:** Head-level attribution and patching.
**What we need:** Edge attribution patching (head-to-head connections).
**Effort:** 2-3 days of implementation.

### Q10: Does the error position distribution have a predictive formula?
**What we have:** Descriptive statistics (off-by-one at low N, diffuse at high N).
**What we need:** A formal model predicting error position as a function of N.
**Effort:** Moderate (derive from Chowdhury's influence density).

## The Central Theoretical Question

**Why does autoregressive processing + continuous gating + fixed-capacity state create PI > RI?**

This is the paper's core contribution. We need EITHER:
(a) A formal proof that any model with these properties exhibits PI > RI, OR
(b) Enough empirical evidence across architectures that the pattern is undeniable

We currently have (b) very strongly. For (a), we have informal arguments (THEORETICAL_DERIVATION.md) but no formal theorem. The closest existing work is Pasten et al.'s continuity theorem (2505.10606), but they don't specifically address PI/RI.

## What Would Make This a NeurIPS Paper vs ACL Paper

| Aspect | ACL (current) | NeurIPS (needed) |
|---|---|---|
| Behavioral evidence | 39 models → 11+39 models | ✓ Have this |
| Cross-architecture | Not in ACL | ✓ Have this (7 families) |
| Mechanistic evidence | None in ACL | ✓ Have this (4 techniques) |
| SSM comparison | None in ACL | ✓ Have this (Mamba) |
| Theoretical contribution | None in ACL | **Gap** (need formal result) |
| Narrative transfer | None in ACL | ✓ Have this |
| Jacobian analysis | None in ACL | ✓ Have this |
| Statistical rigor | Moderate | ✓ 200 trials with CIs |

**The single biggest gap is the formal theoretical contribution.** Everything else is strong enough.
