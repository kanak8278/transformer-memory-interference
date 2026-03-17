# Senior Researcher Full Review: All Claims, Experiments, Theory

**Date:** 2026-03-17
**Perspective:** NeurIPS Area Chair reviewing the complete submission

---

## A. THEORY (Grade: B+)

See SENIOR_REVIEW_THEORY.md for detailed line-by-line analysis. Summary:

| Component | Verdict | Action |
|---|---|---|
| C1 single-layer softmax | PROVEN, correct | Keep |
| C1 multi-layer transformer | NOT proven, only empirical | Downgrade to conjecture |
| C1 SSMs (HiPPO) | CONTRADICTED by own transient growth data | Fix: restrict proof to diagonal SSMs |
| C2 transformers | DERIVED from Chowdhury | Keep |
| C2 SSMs | Hand-waving | Acknowledge honestly |
| Convergence I_∞ > 0 | GAP (harmonic series counterexample) | Fix: add empirical condition |
| v_N capacity bound | PARTIAL | Strengthen with independence assumption |
| Predictions (8 total) | 7/8 confirmed | Strong |

**Bottom line:** The theory is a good framework but NOT a self-contained proof.
Frame as: "We formalize two conditions, prove C1 for the single-layer case,
derive C2 from Chowdhury, verify both empirically, and show the consequences
match all observations." Don't overclaim rigor for the parts that aren't rigorous.

---

## B. BEHAVIORAL EXPERIMENTS (Grade: A-)

### B1. Cross-architecture universality

| Claim | Evidence | Problem | Severity |
|---|---|---|---|
| "9 models, 7 families" | Stage 1 data | "7 families" is generous — 5 are transformer variants, 1 is SSM. Say "6 models across 2 broad architecture classes" | LOW |
| "91% of cells show PI > RI" | 51/56 cells | The 5 negative cells are Pythia at extreme N (both collapse). Honest: "91% where at least one condition succeeds" | LOW |
| "Mean gap +50.3%" | Aggregated data | Mixing models of different capability. The gap depends heavily on model quality. Better: report per-model gaps | LOW |
| PI > RI in Mamba SSM | 30 trials, 17-30% garbage | CIs are ±18%. Direction robust but numbers imprecise. 200-trial rerun running | MODERATE |
| PI > RI in TinyLlama | 30 trials | RI=87%, PI=3% at N=5. Gap is huge (83%) but CI is ±18% | MODERATE |
| PI > RI in StableLM | 30 trials | Same CI issue | MODERATE |

**Key weakness:** 3 of 9 models (Mamba, TinyLlama, StableLM) have only 30 trials. For a NeurIPS paper, this is borderline. The DIRECTION is clear but the exact numbers aren't reliable.

**Action:** 200-trial reruns for all three. Mamba is running now.

### B2. Scaling law

| Claim | Evidence | Problem | Severity |
|---|---|---|---|
| PI(N) = a·exp(-b·N)+c | Curve fits | R²=1.0 from 3-point fit (Qwen 1.5B, 3B-Inst) is meaningless. Honest R² range with ≥6 points: 0.64-0.98 | HIGH — noted, fix in text |
| PI vs size: r=0.886, p=0.019 | 6 models at N=20 | p=0.057 after Bonferroni for 3 N-values tested | HIGH — reframe as suggestive |
| PI floor varies 0-44% | Fit parameters | Not explained by theory. Model-specific, likely training-dependent | MODERATE — acknowledge |

### B3. Error positions

| Claim | Evidence | Problem | Severity |
|---|---|---|---|
| Three architecture-dependent modes | Entropy correlations | Post-hoc taxonomy, not pre-registered | MODERATE — present as observation |
| Qwen: entropy increases (p=0.008) | 8 data points | Only 1 model shows this significantly. 1.5B and 3B-Inst have only 3 points each | MODERATE |
| Gemma: primacy fallback | r=-0.945 | Compelling statistic, but could be weak-model artifact (only 4 heads) | LOW |
| Pythia: off-by-one lock | 10 data points | Strong pattern but base model with 30-50% garbage | MODERATE |

### B4. Narrative transfer

| Claim | Evidence | Problem | Severity |
|---|---|---|---|
| PI > RI on Dota 2 data | 30 trials/cell, Qwen 1.5B | RI=51%±18%, PI=33%±17% — CIs OVERLAP at 95% level | HIGH |
| Gap widens with N | 4 data points | Too few for a trend claim | MODERATE |

**Action:** Need 100+ trials/cell OR use paired test (same seed, RI vs PI).

### B5. API models

| Claim | Evidence | Problem | Severity |
|---|---|---|---|
| Haiku: RI=100%, PI→10% | 20 trials/cell, ARBITRARY_MULTI | Different dataset than local models. 20 trials = ±21% CI | MODERATE |
| GPT-4.1-mini: same pattern | 20 trials/cell | Same issues | MODERATE |

**Action:** Either rerun on ARBITRARY_SINGLE or clearly note different dataset.

---

## C. MECHANISTIC EXPERIMENTS (Grade: A-)

### C1. Logit lens

| Claim | Evidence | Problem | Severity |
|---|---|---|---|
| v_last found then suppressed | 4 models | Strong, consistent Pattern B across models. Peak at 90% depth, crash to <5% | NONE — solid |
| v_first rises monotonically | 4 models | Clean, no exceptions | NONE |
| Suppression magnitude varies | Table 3 | Gemma peak is only 0.03 (vs 0.14-0.24 for Qwen). Different models, different scales — still same pattern | LOW |

**This is our strongest mechanistic result.** No issues.

### C2. Probing classifiers

| Claim | Evidence | Problem | Severity |
|---|---|---|---|
| Condition discrimination 97-100% | 3 models | May detect query word "first"/"last" — but this is EXPECTED (different questions → different representations) | LOW — resolved |
| RI correctness 60-87% | 3 models | Genuine within-condition comparison | NONE |
| PI correctness 50-61% | 2 models (Gemma N/A) | 61% is above chance (p<0.01), not "at chance". Say "weakly above chance" | LOW — fix wording |

**Action:** Fix "at chance" to "weakly above chance (61%, barely exceeding chance)."

### C3. Causal analysis

| Claim | Evidence | Problem | Severity |
|---|---|---|---|
| Distributed mechanism | 3 models | Max delta +0.03-0.07. These are genuinely small effects | NONE — this IS the finding |
| No suppression circuit | Ablation hurts PI | Ablating retrieval heads hurts → correctly interpreted | NONE |
| Attribution heads at 60-80% depth | 3 models | Consistent pattern | NONE |

**Good framing:** "The absence of a localizable circuit IS the result."

### C4. Jacobian at init

| Claim | Evidence | Problem | Severity |
|---|---|---|---|
| Qwen: 1.47× primacy | 20 random inputs | Small sample but Jacobian is deterministic given input. 20 is fine for a smooth function | LOW |
| Mamba: 295× primacy | 20 inputs | Same — but the 295× is position 0 vs position 49. Quartile ratio is 1.24× | LOW — present both |
| Primacy is architectural | Untrained models | Strong argument. Novel comparison | NONE |
| Training amplifies both | Pretrained comparison | Clean before/after | NONE |

### C5. Bidirectional control

| Claim | Evidence | Problem | Severity |
|---|---|---|---|
| BERT: can't do KV task | 0% both conditions | Doesn't prove anything about bidirectional encoding — BERT just can't do MLM for KV retrieval | MODERATE — reinterpret |
| T5-base: no PI > RI | gap=-9%, 77% garbage | Too much garbage to be conclusive. p not significant | HIGH |

**Action:** Downgrade to "preliminary evidence" or appendix. Don't claim "confirms autoregressive is the cause."

---

## D. FIGURES (Grade: B+)

| Figure | Quality | Problem | Action |
|---|---|---|---|
| Fig 1 (4-panel teaser) | Good | Panel (b) x-axis starts at 0.6 — crops context. Legend small | Minor fix |
| Fig 2 (PI vs N, 9 models) | Good | CIs shown via shading — good. Some models have noisy lines (0.5B) | Keep |
| Fig 3 (error positions, 6 models) | OK | x-axis labels unclear ("v0", "v1"...). Could use relative position instead | Minor fix |
| Fig 4 (logit lens, 4 models) | OK | Text too small at paper size. 8-panel layout cramped | Could be 2-panel with overlay |
| Fig 5 (probing, 3 models) | Good | Lines noisy but story clear | Keep |
| Fig 6 (Jacobian) | Strong | Clean, dramatic. Paper-ready | Keep |
| Fig 7 (scaling law fitted) | OK | Panel (b) RI fits are poor. Mamba/Pythia go to 0 (collapse) | Consider dropping panel (b) |
| Fig 8 (error position model) | Weak | Panel (a) all R²≈0 (fits failed). Panel (c) uses 0.5B not best model | Redo or drop |

---

## E. TABLES (Grade: A-)

| Table | Quality | Problem |
|---|---|---|
| Table 1 (model inventory) | Good | Clean |
| Table 2 (behavioral with CIs) | Good | LaTeX ready. Marks <100 trial models with * |
| Table 3 (logit lens) | Good | Clean numbers |
| Table 4 (causal patching) | Good | Small effects honestly reported |
| Table 5 (component elimination) | Strong | Novel contribution |

---

## F. CLAIMS vs EVIDENCE ALIGNMENT

### Claims we CAN make confidently:
1. PI > RI is universal across autoregressive models (9/9 usable models)
2. PI > RI exists in SSMs (Mamba), eliminating attention as sole cause
3. v_last is found then suppressed (logit lens, 4 models)
4. RI correctness is well-encoded, PI correctness is not (probing, 2 models)
5. Mechanism is distributed, no bottleneck heads (causal, 3 models)
6. Primacy bias exists at initialization (Jacobian, 2 architectures)
7. PI accuracy decays approximately exponentially with N
8. At N≤2, PI ≈ RI (minimal interference)

### Claims we SHOULD NOT make (or should heavily qualify):
1. ~~"7 architecture families"~~ → "5 transformer variants + 1 SSM"
2. ~~"R²=0.64-1.0"~~ → "R²=0.64-0.98 for models with ≥6 data points"
3. ~~"PI correctness at chance"~~ → "PI correctness weakly above chance (61%)"
4. ~~"Bidirectional models confirm autoregressive is the cause"~~ → "Preliminary bidirectional test shows no primacy but is inconclusive due to high garbage"
5. ~~"PI vs size is significant (p=0.019)"~~ → "Suggestive positive correlation (r=0.886, p=0.057 after Bonferroni)"
6. ~~"Narrative transfer confirms PI > RI in realistic text"~~ → "Narrative test shows same direction but CIs overlap at 30 trials"
7. ~~"Formal proof of PI > RI"~~ → "Formal framework with one proven lemma and empirically verified conditions"

---

## G. OVERALL VERDICT

**Is this a NeurIPS paper?** YES, but as an empirical mechanistic study with theoretical framing, NOT as a theory paper.

**Strongest contributions (in order):**
1. Cross-architecture universality (9 models, including SSM) — **novel**
2. Component elimination table — **novel**
3. Logit lens: value found then suppressed — **clean, compelling**
4. Jacobian at initialization comparison — **novel**
5. Probing: RI correctness vs PI correctness asymmetry — **novel technique application**
6. Three architecture-dependent error modes — **interesting observation**
7. C1 proof for single-layer softmax — **clean lemma**

**Weakest points (in order of risk):**
1. Bidirectional control is too weak to draw conclusions
2. Narrative transfer CIs overlap
3. 3 models with only 30 trials
4. Theory has gaps (multi-layer C1, convergence I_∞>0)
5. Scaling law correlation fragile after Bonferroni

**Reviewer's likely decision:** Accept with minor revisions, IF:
- The claims are honestly scoped
- The 30-trial models get rerun at 200
- The theory is correctly positioned as "framework with partial proofs"
- The bidirectional control is moved to appendix as "preliminary"
