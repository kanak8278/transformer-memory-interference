# Open Review Items — NeurIPS Submission

*Created 2026-02-20. Major revision 2026-02-22.*
*Updated to reflect: Qwen 0.5B+1.5B complete (96 experiment runs), 3B behavioral in progress, literature review complete, model expansion plan finalized.*

---

## STATUS SUMMARY

| Item | Severity | Status | Blocking? |
|------|----------|--------|-----------|
| R1: Cross-model evidence | CRITICAL | PARTIALLY ADDRESSED — Qwen 1.5B done, need Gemma 3 1B | Yes |
| R2: Trial counts | CRITICAL | ADDRESSED — 100 trials/condition on 0.5B and 1.5B | No |
| R3: SSM control | CRITICAL | NOT STARTED | Yes |
| R4: QK fine-tuning intervention | CRITICAL | NOT STARTED | Yes |
| R5: Minority override unexplained | CRITICAL | REFRAMED — copy-suppression hypothesis replaces old framing | Yes |
| R6: Competitive landscape | IMPORTANT | RESEARCH DONE — needs writing | No |
| R7: Query routing framing | IMPORTANT | ADDRESSED — exp 22 disambiguated (Hypothesis B confirmed) | No |
| R8: Paper figures | IMPORTANT | BLOCKED — data exists, no plots yet | Yes (for submission) |
| R9: Prompt format | NICE-TO-HAVE | NOT STARTED | No |
| R10: Training dynamics | NICE-TO-HAVE | UPGRADED to IMPORTANT — Pythia 410M planned | No |
| R11: Head classification fragility | **NEW, CRITICAL** | Needs decision on framing (Option A/B/C) | Yes |
| R12: Copy-suppression test | **NEW, CRITICAL** | NOT STARTED — highest-payoff single experiment | Yes |
| R13: Theoretical framing | **NEW, IMPORTANT** | Literature review done, needs writing | No |
| R14: Completion model validation | **NEW, IMPORTANT** | Pythia 410M planned | No |

---

## CRITICAL (Paper will be rejected without these)

### R1: Cross-Model Evidence — PARTIALLY ADDRESSED

**Problem:** Mechanistic claims based on a single model family are a case study, not a finding.

**Progress since 2026-02-20:**
- [x] Qwen2.5-1.5B-Instruct: ALL 12 core experiments at 4 operating points COMPLETE
- [x] Qwen2.5-3B-Instruct: Behavioral sweep IN PROGRESS, recalibration DONE
- [ ] Non-Qwen model: Gemma-3-1B-IT planned (see RESEARCH_PLAN.md Step 2B.3)
- [ ] Pythia-410M planned for completion-model validation (see R14)

**What replicated across 0.5B and 1.5B (solid):**

| Finding | 0.5B | 1.5B | Consistent? |
|---------|------|------|-------------|
| Primacy cliff (v1/v0 ≈ 0.000) | Yes | Yes | **Yes** |
| Late-layer query patching recovers PI ~100% | L21+ | L27+ | **Yes** (scaled by depth) |
| Heads ignore query word "first"/"last" | attn ≈ 0.003 | attn ≈ 0.003 | **Yes** |
| Circuit in last 1/3 of layers | L16-23 | L19-27 | **Yes** |
| DLA concentrated in final third | 88% | 91% | **Yes** |
| Linear positional bias doesn't help | No sweet spot | No sweet spot | **Yes** |
| RI P(init) near 1.0 | 0.56-0.97 | 0.98-1.00 | **Yes** (stronger at 1.5B) |

**What DID NOT replicate (problem):**

| Finding | 0.5B | 1.5B | Problem |
|---------|------|------|---------|
| Forced attention fixes PI | +11% to +27% | **-10% to -19%** | **FAILED** — misclassified heads hurt 1.5B |
| Head classification (8 primacy, 53 recency) | Works | Different head set | Threshold-dependent, not robust |

**Resolution for forced attention:** Exp 24b showed forcing ONLY the single strictly-classified head (L16H3 on 0.5B, L19H1 on 1.5B) improves both models +20%. The mechanism IS the same, but the head taxonomy is fragile. See R11 for implications.

**Remaining action:**
- [ ] Gemma-3-1B-IT: behavioral sweep + experiments 12-15, 22 (core mechanistic suite)
- [ ] Verify the 7 solid findings replicate on Gemma (different family = real validation)
- [ ] If Gemma confirms: 3 models, 2 families → strong cross-model evidence

**Risk:** MEDIUM (down from HIGH). The 7 solid findings already replicate within Qwen family. Gemma is needed for cross-family claim but the findings are architecture-level, not model-specific.

---

### R2: Trial Counts — ADDRESSED

**Previous problem:** Pilot-study trial counts (4-30 trials) with overlapping CIs.

**Current status:** All experiments on 0.5B and 1.5B now run at 100 trials per condition across 4 operating points each. Total: 96 experiment runs × 100 trials = ~9,600 trials per model.

**Remaining:**
- [ ] Report bootstrap 95% CIs on every accuracy number in the paper
- [ ] Qwen 3B and new models (Gemma, Pythia) should also use 100+ trials

**Risk:** LOW — addressed.

---

### R3: SSM Architectural Control — NOT STARTED

**Problem:** The paper claims PI > RI is caused by attention architecture. Without testing a non-attention architecture, this is unfalsifiable.

**Literature update (2026-02-22):**
- Airlangga et al. (2025, [arxiv:2506.15156](https://arxiv.org/abs/2506.15156)): Mamba shows U-shaped recall (primacy + recency) through DIFFERENT mechanism — sparse persistent SSM channels, not attention collapse. Primacy in Mamba is ablatable by targeting specific channels.
- Wang et al. (ICLR 2025): SSMs have intrinsic recency bias from exponential decay.

**Prediction refined:** Mamba should show a different PI/RI *correlation structure*, not necessarily RI > PI. Mamba's primacy (persistent channels) and recency (exponential decay) may produce more symmetric interference. The key test is whether PI and RI are UNCORRELATED in Mamba (our dual-process claim) or correlated (different architecture = different mechanism).

**Action:**
- [ ] Run behavioral sweep on Mamba-130M (completion format, few-shot)
- [ ] Same categories, same grid subset, same metrics
- [ ] Compare: (1) PI > RI magnitude, (2) PI-RI correlation, (3) decay curve shape
- [ ] If Mamba shows PI ≈ RI → attention-specificity confirmed
- [ ] If Mamba also shows PI > RI → need to test whether the MECHANISM differs (different ablation targets)

**Effort:** 1-2 days
**Risk:** LOW — literature strongly predicts a different profile

---

### R4: QK Fine-Tuning Intervention — NOT STARTED

**Problem:** "Forcing attention works" is proof-of-concept. "Fine-tuning <1% of parameters fixes PI" is the publishable result.

**Revised design (post head-classification learning):**
- Do NOT fine-tune "8 primacy-biased heads" — the head set is fragile across models
- Instead: fine-tune the W_Q and W_K of the top-N heads ranked by PI DLA (continuous metric, no threshold)
- Or: fine-tune ALL heads in the last 1/3 of layers (layers 16-23 for 0.5B) — avoids head selection entirely
- Compare parameter counts: top-1 head (~115K params) vs top-5 vs last-third-heads vs all heads

**Action:**
- [ ] Generate 200 PI training examples at failing operating points
- [ ] Fine-tune W_Q and W_K only, freeze everything else
- [ ] Sweep: top-1, top-5, last-1/3 heads, all heads
- [ ] Evaluate: PI accuracy, RI accuracy (must not degrade), generalization to unseen operating points
- [ ] Report: minimum parameter count that fixes PI

**Effort:** 2-3 days
**Risk:** MEDIUM — fine-tuning might not generalize, or might break RI

---

### R5: The Minority Override Mechanism — REFRAMED

**Previous problem:** "8 heads override 53 heads" but DLA shows recency wins overwhelmingly (sum=-11.00 vs +0.83). All three simple hypotheses rejected (DLA magnitude, OV gain, last-word effect).

**Why the old framing was wrong:** The question "how does the minority override the majority?" assumes primacy heads are PROMOTING V1 louder than recency heads promote V2. DLA proves this is false. The primacy heads produce negligible logit contribution. So either the ablation result (gap 35%→5%) is noisy, or the mechanism is indirect.

**NEW hypothesis: Copy-suppression (from McDougall et al., BlackboxNLP 2024)**

The minority heads may not be promoting V1 at all. They may be **suppressing V2**. McDougall et al. ([arxiv:2310.04625](https://arxiv.org/abs/2310.04625)) showed GPT-2 head L10H7 implements copy suppression — it detects tokens that earlier layers predict and suppresses that prediction via negative OV eigenvalues.

If our primacy-biased heads are copy-suppression heads:
- They attend to V1 positions but their OV circuit negates the V2 signal in the residual stream
- DLA would show weak positive contribution (suppressing V2 ≠ promoting V1 in logit space) — **matches our data**
- Ablating them removes the suppression → recency heads' V2 signal goes through → PI improves — **matches our ablation result**
- The override is not "8 beating 53" in a tug-of-war. It's "8 heads actively cancelling the majority's correct signal"

This explains the DLA paradox: recency heads DO win the logit-space competition, but their signal is suppressed before it reaches the output.

**This is testable in one experiment.** See R12.

**Previous experiments 21a/21b/21c:** Still worth running as supporting evidence, but the copy-suppression hypothesis is the primary target now.

**Risk:** MEDIUM — if copy-suppression hypothesis also fails, fall back to "information flow corruption" (exp 23 already shows L16-L18 primacy heads corrupt the query-position representation that downstream layers read) or drop the override claim entirely and focus on the 7 solid findings.

---

### R11: Head Classification Fragility — NEW, CRITICAL

**Problem:** This is the single biggest methodological weakness in the paper. We tried three different classification criteria and got different head sets each time:

1. Attention-based (retr>0.1, pi_prim>0.6): 6 heads on 0.5B, 5 on 1.5B
2. Strict filter (retr>0.3, pi_prim>0.7, DLA>0): 1 head each (L16H3, L19H1)
3. DLA-based top-5: different heads again

Forcing all heads from criterion 1 HURT PI on 1.5B. Only the single head from criterion 2 helped both models. We found the right answer by stacking filters — that's post-hoc.

**A reviewer will write:** "How did the authors identify primacy heads? They report trying multiple criteria (Section X). The final criterion appears chosen to produce consistency across models."

**Decision needed — three options:**

**Option A: Drop head-level claims entirely.** (RECOMMENDED)
- Frame the story around the 7 solid findings — none depend on head classification
- Narrative: "PI fails because the final value's representation degrades in late layers due to competing architectural forces (causal masking primacy vs LayerNorm recency). The failure localizes to the QK routing in the last 1/3 of layers. OV circuits retain full recency capability (proved by forced attention at the single strictest-classified head)."
- Report DLA as a continuous heatmap, not a head taxonomy
- Pros: Clean, defensible, no threshold attacks
- Cons: Less mechanistic depth

**Option B: Do head classification properly using established methods.**
- Use Wu et al. (2024) retrieval score (copy-paste frequency >0.1) adapted for our task
- Or use Voita et al. (2019) criterion (>90% of max attention to fixed relative position)
- Define a "primacy bias index" = E[DLA toward initial in PI condition], averaged over 200+ trials
- Validate with shuffled position controls
- Pros: Strongest mechanistic claims
- Cons: High effort, still has threshold choices, reviewer attack surface

**Option C: Use DLA as continuous metric (following IOI circuit methodology).**
- No head classification at all — report continuous DLA values for all heads
- Show the distribution: most heads have negative PI DLA (push toward final = helping PI), a few have positive PI DLA (push toward initial = hurting PI)
- The "primacy" question becomes: "how much total DLA pushes toward initial in PI condition?"
- Identify natural clusters if they exist (bimodal distribution = real categories)
- Pros: Follows Wang et al. (2022) IOI methodology, continuous, no arbitrary thresholds
- Cons: DLA is per-trial noisy, need many trials for stable estimates (we have 100)

**Action:**
- [ ] Decide: Option A, B, or C
- [ ] If A: rewrite mechanistic narrative to avoid head counts
- [ ] If C: re-analyze existing DLA data as continuous distributions, check for bimodality
- [ ] Regardless: the forced-attention result (exp 24b, single head, +20% on both models) is valid and should be reported as a targeted intervention, not as evidence for a head taxonomy

---

### R12: Copy-Suppression Hypothesis Test — NEW, CRITICAL

**The highest-payoff single experiment remaining.**

**What to test:** Are the identified primacy-biased heads implementing copy suppression (V2 negation) rather than V1 promotion?

**Method:**
1. For each head, compute W_OV = W_V @ W_O
2. Compute eigenvalues of W_OV — copy-suppression heads have dominant NEGATIVE eigenvalues (McDougall et al., 2024)
3. Project W_OV onto the unembedding direction for V2 (the final value token): if the projection is strongly negative → head suppresses V2 when it attends to V2-related positions
4. Compare: primacy-biased heads vs recency-responsive heads. Do they differ in OV eigenspectrum?
5. Run on both 0.5B and 1.5B for the strictly-classified heads (L16H3 and L19H1)

**What the outcomes mean:**
- **Negative eigenvalues in primacy heads:** Copy-suppression confirmed. Rewrite the story: "Primacy arises not from heads that retrieve V1, but from heads that suppress V2 — a copy-suppression mechanism that, combined with causal masking's structural advantage for V1, produces PI > RI." This is novel and mechanistically precise.
- **No negative eigenvalues:** Copy-suppression rejected. Fall back to information flow corruption (exp 23 data) or indirect effects. The "why does the minority win?" question remains open — consider Option A (drop the claim).

**Script:** `experiments/28_copy_suppression_test.py` (NEW)
**Effort:** 1 day (static weight analysis + a few forward passes)
**Risk:** LOW effort, HIGH payoff. Even a negative result is informative.

---

## IMPORTANT (Significantly strengthens the paper)

### R6: Competitive Landscape — RESEARCH COMPLETE, NEEDS WRITING

**Full literature map (updated 2026-02-22):**

| Paper | Their claim | Our response | Priority |
|-------|------------|-------------|----------|
| **Wu et al. (2025, ICML)** "Emergence of Position Bias" ([arxiv:2502.01951](https://arxiv.org/abs/2502.01951)) | Causal masking → primacy bias (formal proof). Four competing forces: causal masking (primacy), residual connections (recency), positional biases, content. | **Our theoretical foundation.** Their theorem predicts our primacy cliff. We provide the empirical validation in an interference paradigm. Cite as foundation, not competitor. | MUST CITE |
| **Residual-Aware Position Bias** ([arxiv:2602.16837](https://arxiv.org/abs/2602.16837)) (Feb 2026) | Extends Wu et al.: residual connections produce U-shaped bias (primacy+recency) at finite depth. | Explains "lost in the middle" and connects to our dual-process hypothesis. The U-shape = RI (primacy side) + PI resistance (recency side). | CITE |
| **Kim et al. (2025)** "LayerNorm Induces Recency Bias" ([arxiv:2509.21042](https://arxiv.org/abs/2509.21042)) | LayerNorm + residual connections flip default to recency. | **NOT a contradiction.** Kim explains why 82% of heads are recency-responsive (LayerNorm default). Wu explains why the minority primacy force still wins under interference (causal masking advantage for V1). The TWO forces create the dual-process architecture. This is our most interesting theoretical contribution. | MUST CITE + DISCUSS |
| **Wang & Sun (2025)** "Unable to Forget" ([arxiv:2506.08184](https://arxiv.org/abs/2506.08184)) | LLMs fail at PI. Log-linear decay to zero. ICML 2025 Workshop. | They document the disease, we diagnose the pathology. They study PI only; we study both PI and RI and show they're uncorrelated (R²=0.044). We provide mechanistic explanation they lack. | CITE + DIFFERENTIATE |
| **Wang et al. (2022)** IOI Circuit ([arxiv:2211.00593](https://arxiv.org/abs/2211.00593)) | Gold standard circuit analysis methodology in GPT-2 Small. | Methodological anchor. Our analysis follows their DLA + patching + ablation approach. Frame our work as IOI methodology applied to interference. | CITE (methodology) |
| **Olsson et al. (2022)** Induction Heads ([arxiv:2209.11895](https://arxiv.org/abs/2209.11895)) | Induction heads implement [A][B]...[A]→[B] pattern matching. | Our retrieval heads are induction-head variants. But standard induction heads are position-naive — they don't distinguish first vs last occurrence. Primacy bias means the induction mechanism defaults to first binding when multiple exist. New prediction induction head theory doesn't make. | CITE + EXTEND |
| **Singh et al. (2025, ICLR)** Selective Induction Heads ([OpenReview](https://openreview.net/forum?id=bnJgzAQjWf)) | Circuits that dynamically select correct causal structure in context. 3-layer construction: compute, aggregate, select. | Their framework describes what SUCCESSFUL retrieval looks like. Our work studies the FAILURE mode — when selectivity breaks. The model lacks genuine "first"/"last" selectivity (our exp 17: heads don't read the instruction). PI vulnerability = failed selectivity. | CITE + CONTRAST |
| **McDougall et al. (2024)** Copy Suppression ([arxiv:2310.04625](https://arxiv.org/abs/2310.04625)) | Head L10H7 in GPT-2 implements copy suppression via negative OV eigenvalues. Explains 76.9% of that head's impact and ~39% of self-repair. | **Key to our override puzzle.** If primacy heads suppress V2 rather than promote V1, it explains the DLA paradox. See R12. | CITE IF R12 CONFIRMS |
| **Ramsauer et al. (2021)** Hopfield Networks ([arxiv:2008.02217](https://arxiv.org/abs/2008.02217)) | Attention = modern Hopfield update rule. Energy landscape with competing attractors. | Primacy bias = V1 has deeper energy basin due to causal masking. Sharp cliff = sharp basin boundary. Minority override = pushing system into V1's basin from V2's metastable state. | CITE (theory framing) |
| **Wu et al. (2024)** Retrieval Heads ([arxiv:2404.15574](https://arxiv.org/abs/2404.15574)) | <5% of heads are "retrieval heads" implementing conditional copy-paste. Universal, sparse, intrinsic. | Our primacy-biased heads are likely a subset of retrieval heads. Their finding that retrieval heads are sparse and universal matches our finding. | CITE |
| **Bietti et al. (2025, ICLR Spotlight)** Associative Memories ([arxiv:2412.06538](https://arxiv.org/abs/2412.06538)) | Transformers trade off attention-based vs MLP-based associative memories. Capacity scales linearly with params. | Explains why RI scales with model size (R²=0.49, more capacity) but PI doesn't (R²=0.06, PI is structural not capacity-limited). | CITE |
| **Liu et al. (2023)** Lost in the Middle ([arxiv:2307.03172](https://arxiv.org/abs/2307.03172)) | U-shaped performance curve (primacy + recency, poor middle). | Our PI > RI is a controlled demonstration of the primacy side. Our paradigm isolates the mechanism more precisely than their multi-document QA. | CITE (behavioral) |
| **Airlangga et al. (2025)** Mamba Primacy/Recency ([arxiv:2506.15156](https://arxiv.org/abs/2506.15156)) | Mamba has U-shaped recall via different mechanism: sparse persistent SSM channels (primacy) + exponential decay (recency). | Baseline for SSM control. Mamba's primacy is through persistent channels, not attention collapse. If both architectures show PI > RI via different mechanisms → PI > RI is general to autoregressive models, not attention-specific. | CITE (SSM comparison) |
| **Attention Sink literature** ([arxiv:2504.02732](https://arxiv.org/html/2504.02732v1), ICLR 2025) | First token absorbs disproportionate attention due to softmax normalization. | Related but distinct. Attention sinks = first TOKEN. Our primacy = first VALUE in an associative pair. Mechanisms overlap (causal masking ensures first token's key is seen by all layers) but the phenomena are different. Must distinguish clearly. | CITE + DISTINGUISH |

**Action:**
- [ ] Write Related Work section covering all papers above
- [ ] The Wu et al. + Kim et al. reconciliation is a THEORETICAL CONTRIBUTION — devote a subsection to it
- [ ] Frame the two-forces theory prominently (see R13)

**Effort:** 2-3 days writing

---

### R7: Query Routing Framing — ADDRESSED

**Previous problem:** "Patching at query works" is ambiguous between (A) query representation failure and (B) value routing through query position.

**Resolution (exp 22, completed on both models):**

Hypothesis B confirmed, A rejected:
- Early-layer query patching (L0-8): 0% recovery → query word "last" is processed correctly
- Late-layer resid_post patching (L16+): +66-99% recovery
- Neither attn_out nor mlp_out alone recovers PI → corruption is in CUMULATIVE residual stream
- Corruption starts at L16 (0.5B) / L19 (1.5B) — exactly where DLA concentrates

**Revised framing (already in RESEARCH_PLAN.md):** "The model correctly processes 'last' in early layers. By L16+, primacy-biased heads write initial-value information into the residual stream, which accumulates at the query position. The correct 'last' signal is drowned out by accumulated primacy signals."

**Remaining:** None — this is resolved. Write it up clearly in the paper.

---

### R8: Paper Figures — BLOCKED ON PLOTTING

**Updated figure list (reflecting new model plan and analyses):**

| # | Figure | Data source | Status |
|---|--------|-------------|--------|
| 1 | Behavioral asymmetry heatmap (keys × updates, RI vs PI) | Phase 1 sweeps (0.5B, 1.5B, 3B) | Data exists for 0.5B+3B (Colab), needs plot |
| 2 | Logit lens trajectories: P(initial) vs P(final) across layers | Exp 12, all operating points, both models | **Data ready** |
| 3 | Positional gradient (sharp cliff vs gradual rise) | Exp 13, both models | **Data ready** |
| 4 | DLA heatmap (layer × head), continuous values, RI vs PI side-by-side | Exp 12/20, both models | **Data ready** |
| 5 | Activation patching: layer × component at query position | Exp 22, both models | **Data ready** |
| 6 | Ablation + patching interaction (two-layer corruption) | Exp 23, both models | **Data ready** |
| 7 | Forced attention on single strict head (+20% both models) | Exp 24b, both models | **Data ready** |
| 8 | Cross-model comparison (Qwen 0.5B + 1.5B + Gemma 3 1B) | Pending Gemma experiments | BLOCKED |
| 9 | SSM control comparison | Pending R3 | BLOCKED |
| 10 | Two-forces theory diagram (causal masking vs LayerNorm) | Conceptual figure | Drawing needed |
| 11 | SAE feature analysis (Gemma 3 1B) | Pending Gemma + SAE | BLOCKED |
| 12 | Training dynamics (Pythia checkpoints) | Pending Pythia | BLOCKED |
| 13 | Copy-suppression OV eigenspectrum | Pending R12 | BLOCKED |

**Action:**
- [ ] Create `notebooks/paper_figures.ipynb`
- [ ] Generate figures 2-7 NOW from existing JSON results
- [ ] Figure 10 (theory diagram) can be drawn NOW
- [ ] Figures 8-9, 11-13 blocked on pending experiments

**Effort:** 2 days for existing data; additional after experiments

---

### R13: Theoretical Framing — NEW, IMPORTANT

**Problem:** The paper needs a theoretical section that isn't just "we cite Wu et al." The theory should PREDICT the findings, not just explain them post-hoc.

**The two-forces framework:**

The literature establishes two competing architectural forces in transformer decoders:

1. **Causal masking → primacy** (Wu et al., ICML 2025): In attention-only causal models, cumulative attention must collapse onto early positions as depth increases. This creates a structural advantage for the first-stored value.

2. **LayerNorm → recency** (Kim et al., 2025): LayerNorm + residual connections flip the default behavior to recency bias. This is why most heads naturally attend to recent tokens.

**Our theoretical contribution** is showing these two forces create a dual-process architecture for associative recall:
- **RI resistance** (recalling first value despite later updates) benefits from Force 1 (primacy). It scales with model capacity because deeper/wider models have more residual stream capacity to preserve the first value's representation (R²=0.49 in ACL data).
- **PI resistance** (recalling last value despite earlier entries) requires Force 2 (recency) to overcome Force 1. But Force 1 intensifies with depth while Force 2 is a per-layer correction. Under interference (many conflicting values), the cumulative primacy from causal masking overwhelms the LayerNorm recency. PI resistance does NOT scale with size (R²=0.06) because the structural primacy force is architectural, not capacity-dependent.

**Predictions this framework makes:**
1. PI > RI in all causal transformers → confirmed (39 models, Cohen's d=1.73)
2. RI scales with size, PI doesn't → confirmed (R²=0.49 vs R²=0.06)
3. The primacy circuit lives in late layers (where cumulative masking is strongest) → confirmed (last 1/3)
4. SSMs should show different interference profile (no causal masking in attention) → testable (R3)
5. Primacy should be present from early training (architectural, not learned) → testable (Pythia checkpoints)

**Connection to Hopfield energy landscape** (Ramsauer et al., 2021): The first value occupies a deeper energy basin because causal masking gives it more associative storage updates. Later values are stored in shallower basins. Under interference, the system relaxes to the deepest basin = first value wins. The "sharp cliff" (v1/v0 = 0.000) corresponds to a sharp basin boundary — no gradual interpolation between attractors.

**Action:**
- [ ] Write theory section framing Wu + Kim as two competing forces
- [ ] Show our empirical results as predictions of this framework
- [ ] Add Hopfield energy landscape interpretation
- [ ] Explicitly connect: "RI = primacy force, PI = recency force, dual-process = two forces"

**Effort:** 2 days (writing, no experiments)

---

### R14: Completion Model Validation (Pythia) — NEW, IMPORTANT

**Problem:** All current models are instruction-tuned. A reviewer could argue PI > RI is an artifact of instruction tuning (models are trained to follow "first" better than "last" due to training data distribution).

**Why Pythia matters:**
- Completion-only model, no instruction tuning
- If PI > RI appears in few-shot pattern completion format → the asymmetry is architectural, not a training-data artifact
- 154 training checkpoints → can test when PI > RI emerges (Phase 3.2)
- Native TransformerLens support, MHA (no GQA) → cleanest possible mechanistic analysis
- Pre-trained SAEs available (EleutherAI/sparsify)

**Action:**
- [ ] Setup Pythia-410M in TransformerLens (native, trivial)
- [ ] Implement few-shot completion format for interference task
- [ ] Run behavioral sweep (limited grid — 2048 context constrains it)
- [ ] If PI > RI confirmed: run core mechanistic suite (exp 12-15)
- [ ] Training dynamics: sweep ~20 checkpoints for PI/RI emergence

**Effort:** 3-4 days total (1 day setup + behavioral, 2-3 days mechanistic if confirmed)
**Risk:** MEDIUM — Pythia-410M may not show the effect due to small context (2048 tokens) limiting interference levels

---

## NICE-TO-HAVE (If time permits)

### R9: Prompt Format Experiment (Step 2.12a) — UNCHANGED

**Question:** If query appears BEFORE the stream, does PI improve?

Tests whether QK failure is positional (heads can't see late-position instructions) vs semantic (heads can't process "last" regardless of position).

**Action:**
- [ ] Run query-before-stream variant at 2-3 operating points
- [ ] Compare PI accuracy with standard format

**Effort:** 0.5 days

---

### R10: Training Dynamics — UPGRADED TO IMPORTANT

**Previous plan:** SmolLM2-135M checkpoints (8 available, previously failed to load).

**Updated plan:** Pythia-410M (154 checkpoints, native TransformerLens, well-tested).

See R14 for details. The key question — "does PI > RI emerge at the start of training or develop over time?" — has strong theoretical implications:
- If present from step 0 → purely architectural (causal masking imposes it from initialization)
- If emerges during training → learned behavior (training data distribution matters)
- If emerges sharply (phase transition) → connects to induction head emergence (Olsson et al., 2022)

**Effort:** Included in R14 estimate

---

## Priority Execution Order (Updated 2026-02-22)

```
Week 1 (Unblock the story):
  Day 1:   R12 — Copy-suppression OV eigenspectrum test (1 day, highest payoff)
  Day 1-2: R11 — Decide Option A/B/C for head classification, re-analyze DLA data
  Day 2-3: R4 — QK fine-tuning intervention (the publishable fix)
  Day 3:   R3 — SSM control behavioral sweep (Mamba, easiest win)

Week 2 (Cross-family validation):
  Day 4:   Gemma-3-1B-IT setup + behavioral sweep
  Day 5-6: Gemma-3-1B-IT mechanistic suite (exp 12-15, 22)
  Day 7:   Gemma SAE feature analysis (exp 25)

Week 3 (Completion model + training dynamics):
  Day 8:   Pythia-410M setup + behavioral sweep (R14)
  Day 9:   Pythia mechanistic suite (if PI > RI confirmed)
  Day 10:  Pythia training dynamics checkpoint sweep (R10)

Week 4 (Paper):
  Day 11-12: Paper figures from all data (R8)
  Day 12-13: Theory section writing (R13)
  Day 13-14: Related Work / competitive landscape (R6)
  Day 14:    Full paper draft
```

---

## Novelty Statement (Revised 2026-02-22)

**The novelty is NOT:**
- "Primacy bias exists in transformers" (Wu et al., ICML 2025)
- "LLMs fail at proactive interference" (Wang & Sun, 2025)
- "Lost in the middle" behavioral pattern (Liu et al., 2023)

**The novelty IS:**

1. **Two-forces dual-process theory** — Causal masking (primacy, Wu et al.) and LayerNorm (recency, Kim et al.) create independent RI and PI mechanisms. This explains why they're uncorrelated (R²=0.044) and scale with different architectural properties (RI with capacity R²=0.49, PI with nothing R²=0.06).

2. **Mechanistic localization** — The primacy circuit lives in the last 1/3 of layers (DLA: 88-91%), operates via cumulative residual stream corruption at the query position, and is causally demonstrated by late-layer patching (100% PI recovery).

3. **Latent recency capability** — The OV circuits have full recency retrieval ability. Forcing attention on a single head → +20% PI on both 0.5B and 1.5B. The capacity exists; the QK routing blocks it.

4. **Copy-suppression mechanism** (pending R12) — Primacy-biased heads may suppress V2 rather than promote V1, explaining why DLA shows recency winning but PI still fails.

5. **Cross-architecture dissociation** (pending R3) — Attention-based transformers vs SSMs should show qualitatively different interference profiles, confirming the mechanism is attention-specific.

6. **Surgical fix** (pending R4) — Fine-tuning <1% of parameters (QK weights in late-layer heads) restores PI without degrading RI.

**One-line (revised):** "PI > RI in transformers arises from two competing architectural forces — causal masking creates structural primacy that overwhelms LayerNorm-induced recency in the last third of layers, locking latent recency capability behind a QK routing failure that is surgically fixable."
