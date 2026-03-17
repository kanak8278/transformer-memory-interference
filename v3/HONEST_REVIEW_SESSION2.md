# Honest Self-Review: Session 2

**Date:** 2026-03-17, Session 2
**Goal:** Find every weak point, unsubstantiated claim, and gap before NeurIPS submission.

---

## Claim-by-Claim Audit

### CLAIM: "PI > RI across 11 models, 7 architecture families"
**Evidence:** Stage 1 data from 9 usable models (Mamba-130M and RWKV-430M are garbage-dominated).
**Problem:** We actually have 9 usable models, not 11. The "11" includes Claude Haiku and GPT-4.1-mini (API models), which were tested on a DIFFERENT dataset (ARBITRARY_MULTI, not ARBITRARY_SINGLE).
**Severity:** MODERATE. The API models confirm the direction, but comparing across datasets weakens the claim.
**Fix:** Be precise: "9 local models on ARBITRARY_SINGLE + 2 API models on ARBITRARY_MULTI" or run API models on ARBITRARY_SINGLE too.

### CLAIM: "7 architecture families"
**Evidence:** Qwen (GQA), Gemma (MHA), TinyLlama (Llama), StableLM, Pythia (GPT-NeoX), Mamba (SSM), API models.
**Problem:** Qwen, TinyLlama, and Pythia are all essentially "GPT-style transformer with different training." They share the same fundamental architecture. Counting them as different "families" is generous.
**Severity:** LOW. The paper should acknowledge that transformer variants are still transformers. The TRUE architectural diversity is: Transformer (5 variants) + SSM (Mamba). Two broad families.
**Fix:** Say "5 transformer variants + 1 SSM" or "6 distinct models across 2 architecture classes."

### CLAIM: "PI(N) = a * exp(-b*N) + c with R²=0.64-1.0"
**Evidence:** Curve fits from fit_scaling_law.py.
**Problem 1:** R²=1.0 for Qwen 1.5B and 3B-Inst comes from having only 3 data points — you can perfectly fit 3 points with a 3-parameter model. This is MEANINGLESS.
**Problem 2:** Gemma R²=0.32 is poor. Pythia R²=0.18 is terrible.
**Severity:** HIGH. The "R²=0.64-1.0" claim cherry-picks. Honest range is R²=0.18-1.0.
**Fix:** Report only models with >=6 data points. Or acknowledge that 3-point fits are overfitting.

### CLAIM: "PI vs model size: r=0.886, p=0.019"
**Evidence:** 6 models at N=20, log(size) vs PI accuracy.
**Problem:** 6 data points for a correlation is borderline. The p=0.019 is not corrected for multiple testing (we tried N=5, N=10, N=20 — that's 3 tests). Bonferroni correction: p_corrected=0.057, not significant.
**Severity:** HIGH. The scaling claim may not survive proper multiple testing correction.
**Fix:** Either report all 3 N values honestly (with correction), or frame as "suggestive" not "significant."

### CLAIM: "Error patterns are architecture-dependent (3 modes)"
**Evidence:** Entropy correlations across N for different models.
**Problem 1:** Gemma entropy decreases (r=-0.945) — but this is based on a model with 4 heads that produces 59% primacy errors. The "primacy fallback" mode may just be a weak model failing differently.
**Problem 2:** The three modes (recency imprecision, primacy fallback, off-by-one lock) were identified post-hoc by looking at the data. No pre-registered predictions.
**Severity:** MODERATE. Post-hoc taxonomy is OK for NeurIPS if honestly presented, but should not be overclaimed.
**Fix:** Present as "observed patterns" not "predictive model." Acknowledge post-hoc nature.

### CLAIM: "Probing shows 100% condition discrimination"
**Evidence:** Cross-validated logistic regression on residual stream.
**Problem 1:** The probe may be detecting trivial features (e.g., the query word "first" vs "last" in the input). The probe sees the ENTIRE residual stream including the query tokens.
**Problem 2:** We haven't tested what the probe is detecting. A simple ablation (mask query tokens, keep only KV stream) would test this.
**Severity:** HIGH. If the probe just detects the query word, the 100% result is meaningless.
**Fix:** Run a control: probe trained on representations at KV-stream positions only (not the query position). If discrimination is still high, it's genuine.

### CLAIM: "PI correctness at chance (50-61%)"
**Evidence:** Logistic regression probe accuracy.
**Problem:** 61% is not at chance. With 200 trials, p < 0.01 for 61% accuracy. It's above chance, just weak.
**Severity:** LOW. Should say "weakly above chance" not "near chance."
**Fix:** Compute exact p-values for probe accuracy. Report honestly.

### CLAIM: "Mamba-1.4B shows PI > RI"
**Evidence:** Stage 1 data, 30 trials per cell, 17-30% garbage.
**Problem:** 30 trials gives Wilson CI of ±18%. At 80% RI and 20% PI, the CIs don't overlap — BUT the garbage rate (17-30%) means many trials are uninformative. Effective sample size may be as low as 20 per cell.
**Severity:** MODERATE. The direction is robust but the numbers are imprecise.
**Fix:** Rerun at 200 trials. Or report with the CI honestly.

### CLAIM: "Narrative transfer: gap=+18%"
**Evidence:** 30 trials/cell, 12 cells, Qwen 1.5B on Dota 2 narratives.
**Problem:** 30 trials/cell is thin. RI=51% with CI=±18%. PI=33% with CI=±17%. The CIs overlap!
**Severity:** HIGH. The narrative transfer claim may not be statistically significant at 95% level.
**Fix:** Need 100+ trials per cell for this to be robust. Or use a paired test (same trial, RI vs PI).

### CLAIM: "Jacobian at init: Mamba 295x primacy"
**Evidence:** jacobian_at_init.py results with 20 random inputs.
**Problem:** 20 inputs is fine for the Jacobian calculation. The 295x is a dramatic number but it's comparing position 0 vs position 49 — the extreme ends. The RELATIVE comparison (first quarter vs last quarter) is more meaningful (1.24x for Mamba, 1.47x for Qwen).
**Severity:** LOW. The number is real, just need to present it alongside the quartile comparison.
**Fix:** Lead with quartile ratios, use 295x as illustration of the extreme case.

### CLAIM: "Bidirectional models show no PI > RI"
**Evidence:** BERT MLM (0% both) and Flan-T5-base (RI=13%, PI=22%, gap=-9%).
**Problem 1:** BERT can't do the task at all — this doesn't tell us about bidirectional processing.
**Problem 2:** T5-base has 77% garbage — the model barely follows instructions.
**Problem 3:** The gap is -9% (PI better than RI), but with 100 trials at 13%, CI=±6.6%. Not significant.
**Severity:** HIGH. The bidirectional control is currently too weak to draw conclusions.
**Fix:** Need Flan-T5-large or Flan-T5-XL with lower garbage. Running T5-large now.

### CLAIM: "Formal bound: P(v_0) >= Omega(L * alpha_sink)"
**Evidence:** FORMAL_BOUND.md derivation.
**Problem:** The "proof" is a sketch, not a proof. Multiple hand-waving steps. The claim that alpha_sink is "additive to content-based attention, effectively exempt from 1/n bound" is not proven.
**Severity:** HIGH for NeurIPS. Reviewers will ask for the actual proof.
**Fix:** Either make it rigorous or label it "Conjecture" not "Proposition."

---

## Summary: Top 5 Most Urgent Issues

1. **Probing classifier may detect query word, not value encoding** — Need control experiment
2. **Scaling law R²=1.0 is from 3-point overfitting** — Need honest reporting
3. **Narrative transfer not statistically significant** — Need more trials or acknowledge
4. **Bidirectional control too weak** — T5-base has 77% garbage
5. **Formal bound is hand-wavy** — Not a real proof

## Top 5 Most Impactful Things To Do

1. **Probing control experiment** — mask query position, reprobe on KV-only representations
2. **200-trial rerun** for Mamba, StableLM, TinyLlama (statistical rigor)
3. **T5-large/XL bidirectional test** (running)
4. **Narrative experiment with 100+ trials** on Qwen 1.5B
5. **Tighten formal bound** or relabel as conjecture
