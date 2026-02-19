# Open Review Items — NeurIPS Submission

*Created 2026-02-20. Items from critical review of mechanistic probing work.*
*Only includes unresolved items. Addressed items removed.*

---

## CRITICAL (Paper will be rejected without these)

### R1: Single-Model Evidence

**Problem:** ALL mechanistic findings (experiments 12-18, 19, 20) are on Qwen2.5-0.5B-Instruct only. One model is a case study, not a finding. NeurIPS reviewers will write: "The mechanistic analysis is conducted on a single 0.5B model. How do we know this generalizes to the 39 models in the behavioral study?"

**Complication:** SmolLM2 v1 work showed DIFFERENT mechanisms at different scales — 135M had no final-value retrieval at any layer, 360M had condition-dependent heads that flip. These are qualitatively different stories. If Qwen-1.5B also shows a different mechanism, the "8 primacy heads" narrative is model-specific, not universal.

**Action:**
- [ ] Run experiments 12-18 on Qwen2.5-1.5B-Instruct
- [ ] Run experiments 12-18 on one non-Qwen model (Gemma-2-2B-IT preferred)
- [ ] If mechanisms differ across scale: reframe narrative as "mechanism evolves with scale" rather than "8 heads cause PI > RI"
- [ ] Minimum: consistent story across 2 models of different sizes

**Effort:** 4-6 days compute + analysis
**Risk:** High — if the story diverges across models, the paper framing must change fundamentally

---

### R2: Trial Counts Are Pilot-Study Size

**Problem:** Current trial counts:
- Head identification: 30 trials
- Ablation: 20 trials
- Forced attention: 20 trials
- Activation patching: 4-5 trials
- Positional gradient: 5 trials
- Instruction sensitivity: 20 trials
- Positional bias sweep: 20 trials

With 20 trials and binary accuracy, 95% CI on 40% is [19%, 64%]. The "gap shrinks from 35% to 5%" claim has overlapping confidence intervals.

**Action:**
- [ ] Scale ALL key experiments to 100+ trials minimum
- [ ] 200+ for the main claims (head identification, ablation, forced attention, oracle bias)
- [ ] Report bootstrap 95% CIs on every accuracy number in the paper
- [ ] Re-verify that results hold at larger N (the override mechanism puzzle may partly be noise)

**Effort:** 2-3 days compute (can parallelize across experiments)
**Risk:** Medium — results likely hold directionally but exact numbers will shift

---

### R3: SSM Architectural Control Missing

**Problem:** The paper claims PI > RI is caused by attention architecture. Without testing a non-attention architecture, this is unfalsifiable. SSM control is the easiest experiment with the highest marginal value.

**Prior work to leverage:**
- Airlangga et al. (2025): Mamba shows U-shaped recall (both primacy AND recency)
- Wang et al. (ICLR 2025): SSMs have intrinsic recency bias (opposite of transformer primacy)

**Prediction:** Mamba/RWKV should show RI > PI or symmetric interference (opposite pattern from transformers).

**Action:**
- [ ] Run behavioral sweep (Phase 1 only, no mechanistic probing) on Mamba-130M or RWKV-169M
- [ ] Use same prompt format, same categories, same grid subset
- [ ] Compare interference profile: does PI > RI reverse?

**Effort:** 1-2 days
**Risk:** Low — existing literature strongly suggests the prediction will hold

---

### R4: QK Fine-Tuning Intervention Not Done (Step 2.11)

**Problem:** The "latent recency locked by routing" novelty framing requires demonstrating the fix. "Forcing attention works" is proof-of-concept; "fine-tuning 0.14% of parameters fixes PI" is the publishable result.

**Action:**
- [ ] Generate 100-200 PI training examples at failing operating points
- [ ] Fine-tune only W_Q and W_K of 8 identified primacy-biased heads (~917K params, 0.14%)
- [ ] Freeze everything else
- [ ] Evaluate: PI accuracy, RI accuracy, generalization to unseen operating points
- [ ] Check: do fine-tuned heads now attend to query word? Does primacy score shift?

**Effort:** 2-3 days
**Risk:** Medium — if it doesn't work cleanly (breaks RI, or doesn't generalize), the framing weakens but doesn't collapse

---

### R5: The Minority Override Mechanism Is Unexplained

**Problem:** Core claim is "8 heads override 53 heads." But:
- All three simple hypotheses rejected (DLA magnitude, OV gain, last-word effect)
- DLA shows recency heads win overwhelmingly (sum=-11.00 vs +0.83)
- Yet PI accuracy is only 55%

The numbers contradict the narrative. Something nonlinear/indirect is happening.

**Leading hypothesis: Information flow corruption.** Primacy heads at L16 write initial-value signal into residual stream. Downstream recency heads at L19-23 read from corrupted positions. The recency heads "correctly" retrieve from the wrong location. DLA doesn't capture this indirect path.

**Action:**
- [ ] **Experiment 21a: Logit lens WITH vs WITHOUT 8 primacy heads.** Track P(initial) and P(final) layer-by-layer under ablation. If ablating changes trajectory at L16+ → confirms information flow corruption.
- [ ] **Experiment 21b: Split DLA by trial outcome.** On the ~45% PI trials that fail, is primacy DLA larger? If so → override is stochastic (depends on specific token positions), not universal.
- [ ] **Experiment 21c: Classify failure outputs.** On PI failures, what does the model actually output? Initial value (primacy intrusion), intermediate value, or garbage? Connect to Step 2.6 peak-at-80th-percentile finding.

**Effort:** 2-3 days
**Risk:** High — if the information flow corruption hypothesis is also wrong, you have a descriptive result without a mechanism. The ablation effect is real but unexplained.

---

## IMPORTANT (Significantly strengthens the paper)

### R6: Competitive Landscape Differentiation

**Problem:** Multiple overlapping papers exist. Reviewers will ask "how is this different?"

**Papers requiring explicit differentiation:**

| Paper | Their claim | Needed response |
|-------|------------|-----------------|
| Wang & Sun (2025) "Unable to Forget" | LLMs fail at PI | They describe the disease, we diagnose the pathology. They study PI only; we show PI and RI are independent processes. Cite and differentiate in Related Work. |
| Wu et al. (2025, ICML) "Emergence of Position Bias" | Causal masking → primacy bias (formal proof) | We build on their result. Our contribution: primacy bias manifests as PI > RI, 82% of heads overcome it, and the failure traces to a sparse routing defect. Cite as foundation, not competitor. |
| Kim et al. (2025) "LayerNorm Induces Recency Bias" | LayerNorm → recency in transformers | **Needs reconciliation.** If LayerNorm creates recency, why does primacy dominate under interference? Likely answer: interference regime creates a different dynamic where causal masking primacy overrides LayerNorm recency. Must test or at least discuss. |
| Singh et al. (2024) / ICLR 2025 Selective Induction Heads | Mature understanding of match-and-copy circuits | Frame retrieval heads as induction-head variants. The functional split (82% instruction-sensitive vs 12% instruction-blind) within the same circuit class is new. |

**Action:**
- [ ] Write Related Work section explicitly differentiating from each paper above
- [ ] Run one experiment to reconcile with Kim et al.: test whether LayerNorm removal changes the interference asymmetry (or at minimum, discuss theoretically)
- [ ] Frame retrieval heads in induction head terminology throughout the paper

**Effort:** 2 days (mostly writing + one small experiment)

---

### R7: Query Routing Framing Needs Tighter Evidence

**Problem:** Activation patching at query position replaces the ENTIRE residual stream at that position across all layers above the patch point. That includes accumulated value information already routed there via attention in earlier layers.

So "patching at query works" doesn't uniquely prove "query routing failure." It could equally mean "the accumulated representation at the query position is what's corrupted by interference."

**Action:**
- [ ] Patch specific components at query position: attention output vs MLP output separately
- [ ] Patch at specific layers at query position (not just answer position)
- [ ] If attention patching at query restores more than MLP patching → routing evidence strengthens
- [ ] If both contribute equally → the claim is "representation at query position" not "query routing"

**Effort:** 1 day (extend existing patching script)

---

### R8: No Paper-Ready Figures

**Problem:** `figures/` directory is empty. NeurIPS is visual. No figures = no paper draft = no submission.

**Minimum figure set:**

| # | Figure | Data source | Status |
|---|--------|-------------|--------|
| 1 | Behavioral asymmetry heatmap (keys x updates, RI vs PI) | Phase 1 sweep | Needs Phase 1 completion |
| 2 | Logit lens trajectories: P(initial) vs P(final) across layers | Exp 12 | Data exists, needs plot |
| 3 | Head classification scatter (primacy score RI vs PI) | Exp 16 | Data exists, needs plot |
| 4 | Ablation bar chart (gap elimination) | Exp 16 | Data exists, needs plot |
| 5 | Forced attention result (100% accuracy) | Exp 18 | Data exists, needs plot |
| 6 | Positional gradient (sharp cliff vs gradual rise) | Exp 13 | Data exists, needs plot |
| 7 | Oracle bias sweep (PI recovery curve) | Exp 19 | Data exists, needs plot |
| 8 | Cross-model comparison | Pending R1 | Blocked on cross-model runs |
| 9 | SSM control comparison | Pending R3 | Blocked on SSM experiment |

**Action:**
- [ ] Create plotting notebook: `notebooks/paper_figures.ipynb`
- [ ] Generate figures 2-7 from existing JSON results (can be done NOW)
- [ ] Figures 1, 8, 9 blocked on pending experiments

**Effort:** 2-3 days for existing data figures; additional after experiments complete

---

## NICE-TO-HAVE (If time permits)

### R9: Prompt Format Experiment (Step 2.12a)

**Question:** If query appears BEFORE the stream (early position where primacy heads attend), does PI improve?

**Why it matters:** Tests whether the QK failure is positional (heads can't see late-position instructions) vs semantic (heads can't process "last" regardless of position). Changes the mechanistic interpretation.

**Action:**
- [ ] Run query-before-stream variant at 2-3 operating points
- [ ] Compare PI accuracy with standard format

**Effort:** 0.5 days

---

### R10: Training Dynamics (Step 3.2)

**Question:** Does PI > RI emerge at the start of training or develop over time?

**Why it matters:** If present from the earliest checkpoint → architectural. If develops over training → learned behavior.

**Action:**
- [ ] Fix SmolLM2-135M checkpoint loading (previously failed)
- [ ] Run behavioral sweep at each of 8 checkpoints

**Effort:** 1-2 days (debugging + compute)

---

## Priority Execution Order

```
Week 1:
  Day 1-2: Experiment 21a+21b (override mechanism — unblocks theory)
  Day 2-3: Step 2.11 (QK fine-tuning — the killer result)
  Day 3:   SSM control behavioral sweep (R3 — easiest win)

Week 2:
  Day 4-6: Cross-model replication on Qwen-1.5B (R1)
  Day 6-7: Scale trial counts to 100+ (R2)

Week 3:
  Day 8-9: Cross-model replication on Gemma-2B (R1)
  Day 9-10: Paper figures (R8)
  Day 10:  Related Work writing (R6)

Week 4:
  Paper writing + query routing tightening (R7) + prompt format (R9)
```

---

## Novelty Statement (For Reference)

The novelty is NOT:
- "Primacy bias exists in transformers" (Wu et al. 2025)
- "LLMs fail at proactive interference" (Wang & Sun 2025)

The novelty IS:
1. **Dual-process dissociation** — RI and PI are mechanistically independent (R²=0.044), scale with different architectural properties
2. **Latent recency capability** — 82% of retrieval heads correctly adapt, OV circuits work perfectly, forced attention → 100% PI
3. **Sparse QK routing failure** — ~2% of heads with instruction-blind QK circuits block recency retrieval
4. **Surgical fix** — 0.14% parameter correction restores PI (pending Step 2.11)
5. **Information flow corruption** — the override mechanism is indirect, not through direct logit competition (pending Exp 21)

One-line: **"Transformers possess latent recency retrieval capability that is architecturally blocked by a sparse QK routing failure in <2% of attention heads."**
