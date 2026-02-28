# Paper Draft Outline: Why Transformers Remember First and Forget Last

## Framing Decision

**Closest structural model:** Hybrid of Wu et al. (Position Bias, ICML 2025) and Wu et al. (Retrieval Heads, ICLR 2025 Oral).

- From Position Bias: theory-first narrative where experiments validate predictions
- From Retrieval Heads: property enumeration for mechanistic findings, cross-model comparison as the backbone

**NOT like IOI:** IOI is a single-model circuit deep-dive. We're doing cross-model mechanistic comparison — breadth over depth.

**Storytelling arc:**

```
Empirical puzzle (PI > RI, 39 models)
  → Theoretical framework (two competing architectural forces)
    → Predictions from the framework
      → Mechanistic evidence (5 models, 3 architectures confirm predictions)
        → The attention-ID failure (methodological contribution)
          → Architectural control (base model, SSM)
            → Implications (the primacy problem is architectural, not learned)
```

---

## Target: NeurIPS 2026 (9 pages main body + unlimited appendix)

**Page budget:**

| Section | Pages | Purpose |
|---------|-------|---------|
| 1. Introduction | 1.0 | Hook, gap, contribution |
| 2. Related Work | 0.75 | Position in landscape |
| 3. Two-Forces Theory | 1.5 | Theoretical framework + predictions |
| 4. Behavioral Evidence | 1.0 | 39-model sweep + base model confirmation |
| 5. Mechanistic Evidence | 3.0 | The core — 5 models, universal + architecture-specific |
| 6. Architectural Controls | 0.75 | Pythia (base), Mamba (SSM) |
| 7. Discussion & Limitations | 1.0 | What this means, what's missing |
| **Total** | **9.0** | |

**Figure budget:** 8-10 in main body, 5-10 in appendix.

---

## Section 1: Introduction (1 page)

### Opening hook (2-3 sentences)
When LLMs encounter conflicting information in context — a key updated multiple times — they exhibit a striking asymmetry: recalling the *first* value (retroactive interference resistance) is easy, but recalling the *last* value (proactive interference resistance) fails catastrophically. This pattern holds across 39 models spanning 5 architecture families, with Cohen's d = 1.73.

### The gap
The ACL paper documented this but offered no explanation. Why does primacy dominate? Why does RI scale with model size (R²=0.49) while PI does not (R²=0.06)? Why are RI and PI uncorrelated (R²=0.044)?

### What this paper adds (contribution bullets)

1. **Two-forces theory:** Causal masking creates structural primacy (Wu et al., ICML 2025). LayerNorm induces recency (Kim et al., 2025). Under interference, primacy overwhelms recency in the final third of layers. This predicts PI > RI, RI scaling with capacity, and PI invariance to scale.

2. **Mechanistic evidence across 5 models, 3 architectures:** Per-head causal knockout identifies early-layer primacy heads invisible to standard attention metrics. The behavioral outcome is universal; the circuit differs by architecture (concentrated vs distributed, direct vs indirect).

3. **Methodological finding:** Attention-based head classification fails for primacy circuits on all tested models. Causal knockout is necessary. This has implications beyond our task.

4. **Base model confirmation:** PI > RI appears in Pythia-410M (no instruction tuning), ruling out SFT as the cause.

5. **[If done] Architectural control:** SSMs show a different interference profile, confirming the mechanism is attention-specific.

6. **[If done] Surgical fix:** Fine-tuning <1% of parameters (QK weights in late-layer heads) restores PI without degrading RI.

### Figure 1: The phenomenon
*Two-panel figure.* Left: RI vs PI accuracy heatmap (keys × updates) for one model showing the regime map (A/B/C). Right: RI-PI gap across 39 models showing universal PI > RI (from ACL paper).

---

## Section 2: Related Work (0.75 pages)

### Organize by relationship to our work, not by topic

**Paragraph 1: Position bias in transformers**
- Lost in the Middle (Liu et al., 2024) — behavioral observation, no mechanism
- Serial Position Effects (Guo & Vosoughi, 2024) — decision-making framing, overlaps behaviorally
- Wang & Sun (2025) — PI-LLM, our most direct competitor. They study PI only; we add RI, the asymmetry, dual-process, and mechanism.
- **Our differentiation:** First to study both PI AND RI under interference, show they're uncorrelated, and provide mechanistic explanation.

**Paragraph 2: Architectural sources of position bias**
- Wu et al. (ICML 2025) — formal proof that causal masking → primacy. Our theoretical foundation.
- Kim et al. (2025) — LayerNorm → recency. The counterforce.
- Attention Sinks (ICLR 2025) — first-token attention from softmax normalization. Related but distinct (token vs value primacy).
- Found in the Middle (NeurIPS 2024) — U-shaped bias calibration.
- **Our contribution:** Combining Wu + Kim into a dual-process framework that explains WHY PI > RI, not just THAT position biases exist.

**Paragraph 3: Mechanistic interpretability of retrieval**
- IOI Circuit (Wang et al., ICLR 2023) — gold standard methodology, single model
- Retrieval Heads (Wu et al., ICLR 2025 Oral) — universal retrieval heads, cross-model
- Copy Suppression (McDougall et al., 2024) — negative OV eigenvalues suppress copied tokens
- Circuit Consistency (Lieberum et al., NeurIPS 2024) — circuits stable across scale
- **Our contribution:** First circuit analysis of interference (not just retrieval), cross-architecture comparison showing same behavior from different mechanisms.

**Paragraph 4: Associative memory theory**
- Hopfield Networks (Ramsauer et al., 2021) — attention = modern Hopfield
- Hopfield Capacity (Hu et al., NeurIPS 2024) — tight capacity bounds
- Factual Recall as Associative Memory (Nichani et al., 2024) — attention + MLP trade-off
- **Our contribution:** Applying the Hopfield energy landscape to explain interference asymmetry.

---

## Section 3: Two-Forces Theory (1.5 pages)

### 3.1: The Framework (0.75 pages)

**Force 1: Causal masking → primacy** (cite Wu et al.)
- In causal attention, position i is attended by all positions j > i
- Cumulative attention collapse: deeper layers concentrate on earlier positions
- First-stored value occupies a deeper energy basin in the Hopfield landscape
- Prediction: retrieving the first value (RI) benefits from this structural advantage

**Force 2: LayerNorm + residual connections → recency** (cite Kim et al.)
- LayerNorm rescales activations, amplifying recent tokens
- Residual connections preserve recent information through the network
- Most heads (~80% in our data) are recency-responsive by default
- Prediction: retrieving the last value (PI) relies on this per-layer correction

**The dual-process architecture:**
- RI resistance = Force 1 (primacy). Scales with d_model because larger models have more residual stream capacity to preserve the first value's representation.
- PI resistance = Force 2 (recency) overcoming Force 1. Does NOT scale with size because the structural primacy force is architectural, not capacity-dependent.
- Under interference (multiple conflicting values), cumulative primacy from causal masking overwhelms the per-layer recency correction. PI fails; RI holds.

### 3.2: Formal Result — Hopfield-based Proposition (0.5 pages)

**Strategy:** Two-forces (3.1) is the intuitive narrative. This section gives the formal result
that NeurIPS reviewers require. They are complementary — same phenomenon at different levels.

**Foundation (two existing theorems we build on):**

- **Ramsauer et al. (2021), Theorem 3 — Retrieval error bound:**
  For N stored patterns Ξ = {ξ₁,...,ξₙ}, query x, inverse temperature β = 1/√d_k,
  minimum pattern separation Δ_min, the retrieval error after one Hopfield update is:

  ```
  ||x_new - ξ_target|| ≤ (N-1) · exp(-β(Δ_min - M))
  ```

  where M depends on the query's initial proximity to the target.
  Error grows linearly with N, shrinks exponentially with β·Δ_min.
  Standard Hopfield: all patterns are stored symmetrically → PI = RI by construction.

- **Wu et al. (2025, ICML), Theorem 4.5 — Causal masking breaks symmetry:**
  Define cumulative attention weight A_L(i) = attention position i has received
  from all downstream positions across L layers:

  ```
  A_L(i) ∝ Σ_{j > i} Π_{l=1}^{L} α^(l)_{j→i}
  ```

  Their theorem proves: **A_L(1) > A_L(2) > ... > A_L(N)** for all causal transformers
  with L ≥ 1. Earlier positions accumulate strictly more contextual refinement.
  The gap grows with L — deeper models have stronger primacy.

**Our Contribution — Proposition 1 (Asymmetric Retrieval Under Causal Attention):**

*Substituting Wu et al.'s asymmetric A_L(i) into Ramsauer's retrieval framework:*

Let A_L(i) denote the cumulative attention weight of position i across L layers (Wu et al.).
For N stored key-value pairs sharing key k at positions 1...N, the effective retrieval weights are:

```
w_i = softmax(β · A_L(i) · sim(query, key_i))
```

Then:

```
RI error  ≤  (N-1) · exp(-β · (Δ + A_L(1) - A_L(2)))    [small: A_L(1) is large]

PI error  ≥  1 - (N-1)⁻¹ · exp(β · (A_L(N) - A_L(1)))   [large: A_L(N) is small]
```

Since A_L(1) > A_L(N) for all causal transformers (Wu et al., Thm 4.5):

> **PI error > RI error for all N ≥ 2 in any causal transformer.**

*Proof sketch (full proof in Appendix A):*
1. Cite Ramsauer Thm 3 for symmetric baseline.
2. Cite Wu Thm 4.5 for A_L(1) > ... > A_L(N).
3. Substitute into softmax weights: w_1 >> w_N whenever A_L(1) >> A_L(N).
4. RI query targets w_1 (largest) → small error. PI query targets w_N (smallest)
   → w_1 steals probability mass → large error.
Approximately 1 page of algebra connecting two existing results.

**What this explains in data:**

| Proposition term | Empirical observation |
|---|---|
| Error grows with N | PI collapses at 2-5 updates; RI holds to 30-100 |
| A_L(1) >> A_L(N) | Primacy cliff: v1/v0 ≈ 0.01 in all models |
| Error ∝ (N-1) for PI | PI accuracy follows decay curve matching N |
| A_L grows with L | Deeper models (1.5B > 0.5B) show sharper primacy cliff |
| A_L independent of d_model | PI doesn't scale with size (R²=0.06) |
| Δ grows with d_k = d_model/n_heads | RI scales with model size (R²=0.49) |

**Honest limitation to state in paper:**
"Proposition 1 holds for a single-layer model. Real transformers are multi-layer with
LayerNorm and residual connections. Section 3.1 (two-forces) provides intuition for
the multi-layer case; Section 5 validates empirically across 5 architectures."

**Fallback if proof has gaps:** State as Conjecture 1 (empirically supported).
Weaker but honest — better than a wrong theorem.

**Effort:** 3-4 days. Main body: proposition statement + 3-sentence intuition.
Appendix A: full proof (~1-2 pages).

### 3.2.1: Honest Assessment of the Hopfield Direction (TO EXPLORE)

After working through the substitution in detail, the "clean theorem" is NOT fully provable:

1. **sim_i = c + δ(i) + γ·s_i decomposition is assumed, not derived.** Real q^T k_i depends on learned W_Q, W_K and nonlinear multi-layer processing. The relationship between Wu et al.'s A_L(i) and the actual dot product is not formalized.
2. **Ramsauer's Theorem 3 assumes symmetric patterns.** Adding asymmetric weights is a different problem — can't just substitute.
3. **Single-layer Hopfield vs multi-layer transformer.** The s₁ > sₙ asymmetry only arises from multi-layer processing, but the retrieval framework is single-step.
4. **δ₀ ≈ 0 paradox.** Our data shows heads ignore "first"/"last", which predicts RI~100% — but RI is 50-80%. MLP/LayerNorm corrections not captured.

**Viable paths forward:**

- **Option A (framework + numerical validation):** Present as a "retrieval-theoretic framework" with explicit assumptions (Bietti et al. 2024 style). Validate key assumption numerically on a toy 2-4 layer transformer (~1-2 days). Call it "Proposition" with stated assumptions.

- **Option B (prove for toy model):** Define a minimal model where the proof IS rigorous — e.g., a 2-layer attention-only model (no MLP, no LayerNorm, no residual connections). In this setting: (1) causal masking makes h_1^(2) strictly more contextualized than h_N^(2), (2) the q^T k_i decomposition holds exactly, (3) PI > RI follows as a real theorem. Then state: "Theorem 1 holds for the toy model. We validate empirically that the same qualitative prediction extends to full transformers across 5 architectures (Section 5)." This is the Wu et al. (ICML 2025) playbook — prove for a clean model, validate on real ones.

- **Fallback:** "Conjecture" if neither works. Weaker but honest.

### 3.3: Predictions (0.25 pages)

From this framework (intuitive + formal), we derive 6 testable predictions:

| # | Prediction | Source |
|---|-----------|--------|
| P1 | PI > RI in all causal transformers | Force 1 dominates under interference / Hopfield asymmetry |
| P2 | RI scales with model size, PI does not | RI = capacity (d_model), PI = architectural (s(i) structure) |
| P3 | RI and PI are uncorrelated across models | Independent mechanisms |
| P4 | The primacy circuit lives in late layers | Cumulative masking strongest at depth |
| P5 | PI > RI appears in base models (no SFT) | Architectural, not training-dependent |
| P6 | Non-attention architectures show different profile | Force 1 requires attention / no s(i) asymmetry in SSMs |

### Figure 2: Theory illustration
*Schematic.* Left: Hopfield energy landscape with V1 in deep basin, V_N in shallow basin (from the formal result — deeper basin because higher s(i)). Right: Two forces diagram showing causal masking (primacy arrow, strengthens with depth) vs LayerNorm (recency arrow, per-layer correction). The arrow sizes show primacy accumulating while recency stays constant.

---

## Section 4: Behavioral Evidence (1 page)

### 4.1: 39-Model Sweep (0.5 pages) — from ACL paper

- PI > RI universal (Cohen's d = 1.73) → **confirms P1**
- RI scales with size R²=0.49, PI R²=0.06 → **confirms P2**
- RI and PI uncorrelated R²=0.044 → **confirms P3**
- Error taxonomy: RI = passive retrieval failure, PI = active primacy intrusion

### 4.2: Base Model Confirmation (0.5 pages) — NEW (Pythia)

- Pythia-410M (base, no SFT, completion format) shows PI > RI
- RI: 41%, PI: 12%, gap: 29pp across 45 grid cells
- PI cracking at 2 updates, RI holds to 30+
- Same qualitative pattern as instruction-tuned models → **confirms P5**

### Figure 3: Behavioral results
*Three panels.* (a) 39-model PI vs RI scatter (from ACL). (b) RI/PI accuracy grids for Pythia-410M (completion model). (c) RI scaling with model size (R²=0.49) vs PI flat (R²=0.06).

---

## Section 5: Mechanistic Evidence (3 pages) — THE CORE

**Structure follows Retrieval Heads pattern: enumerate universal properties, then show variation.**

### 5.1: Universal Mechanistic Findings (1.5 pages)

Six findings replicated across 3+ models (Qwen 0.5B, 1.5B, 3B, Gemma 1B, Pythia 410M):

**Finding 1: Primacy cliff.** v1/v0 ratio ≈ 0.01-0.03 in RI condition. The initial value captures 40-90× more probability mass than the first update. Universal across all models. → **Confirms P4 mechanism**

**Finding 2: Late-layer decision.** Activation patching shows near-zero PI recovery through 65-79% of layers, then sharp "hockey stick" jump to 100% in final third. The RI/PI decision is made in the last 20-35% of the network. → **Confirms P4**

**Finding 3: Heads ignore the query word.** Primacy heads put <1% attention on "first"/"last". They are instruction-insensitive — they apply the same positional bias regardless of the query. 15/23 tested heads across 4 models ignore the query word completely.

**Finding 4: MLP-mediated mechanism (Type B).** In the natural state, primacy heads attend 93-99% to instruction/non-value tokens. Oracle forcing shifts them to attend 41-62% to the correct (final) value. The heads CAN retrieve correctly when aimed right — the bottleneck is routing, not capability. Confirmed on Qwen 0.5B, 1.5B, Gemma 1B (SageMaker replication).

**Finding 5: Copy-suppression rejected, V1-promotion confirmed.** DLA decomposition on Pythia-410M (MHA): 17/21 heads show V1-PROMOTION, 0/21 V2-SUPPRESSION. OV eigenspectrum (CPU computation, resolving MPS limitation): no negative eigenvalue enrichment in primacy heads. Copy scores universally zero across all models. The mechanism is not suppressing V2 — it's boosting V1.

**Finding 6: Attention-based head identification fails universally.** On every model tested with both methods, attention-based classification (which tokens the head attends to) identified the WRONG heads. Causal knockout (per-head zero-ablation) found the real primacy heads in early/mid layers (L0-L14), invisible to attention metrics.

| Model | Attention-based found | Causal knockout found | Overlap |
|-------|----------------------|----------------------|---------|
| Qwen 0.5B | L3-L19 (distributed) | L12H0 (L3-L19, distributed) | Partial |
| Qwen 1.5B | L19H1 (late) | **L8H3** (early, L0-L8) | None |
| Qwen 3B | L16-L22 (late) | L31H8 (distributed, L0-L31) | None |
| Gemma 1B | L20H1 (late) | **L14H2** (mid, L4-L14) | None |
| Pythia 410M | L11-L22 (mid/late) | **L5H2** (early, L0-L10) | Minimal |

### Figure 4: Universal findings
*Four-panel.* (a) Primacy cliff: positional gradient showing v1/v0 ≈ 0.01 across 3 models. (b) Hockey stick recovery: activation patching curves for 3 models, overlaid (à la Lieberum). (c) Attention-based vs causal knockout comparison table (the failure). (d) Oracle vs blind attention shift for 3 models.

### Figure 5: Logit lens trajectories
*Two rows (RI/PI), three columns (Qwen 1.5B, Gemma, Pythia).* P(initial) and P(final) across layers. Shows V1 rising and persisting in both conditions, V2 only appearing in PI when it succeeds.

### 5.2: Causal Head Identification (1 page)

**The primacy heads — identified by per-head causal knockout (Wang et al., 2022 methodology):**

| Model | Top Head | Δld | KO: PI boost | KO: RI drop | Location |
|-------|----------|-----|-------------|-------------|----------|
| Gemma 1B | **L14H2** | +21.4 | 24%→88% | 92%→52% | Mid |
| Qwen 1.5B | **L8H3** | +8.3 | 44%→92% | 100%→40% | Early |
| Pythia 410M | **L5H2** | +5.4 | — | — | Early |
| Qwen 3B | L31H8 | +2.7 | ~small | ~small | Late |
| Qwen 0.5B | L12H0 | +1.7 | 58%→69% | 87%→85% | Mid |

**Key observations:**
- Knockout is primacy-specific: helps PI AND hurts RI. Not general capability loss.
- Rankings stable across operating points (validated with 200-trial CIs on Gemma and Qwen 1.5B).
- Concentration varies: Gemma and Qwen 1.5B have dominant heads (Δld=8-21); Qwen 0.5B and 3B are distributed (Δld=1.7-2.7).
- Attribution patching (first-order gradient) misses the real heads on every model — they operate through indirect effects across many layers.

### 5.3: Architecture-Dependent Mechanisms (0.5 pages)

The behavioral outcome (PI > RI) is universal, but the circuit differs:

| | Concentrated | Distributed |
|---|---|---|
| **Mechanism A (QK routing)** | — | Qwen 3B (forcing +47pp) |
| **Mechanism B (OV corruption)** | Qwen 1.5B (forcing 0pp) | Qwen 0.5B (forcing ~0pp) |
| **Mechanism A+B (mixed)** | Gemma 1B (forcing +36pp, KO +57pp) | — |

**Interpretation:** The primacy bias is a convergent outcome, not a single circuit. Different architectures and scales implement it through different pathways — some through attention routing (where heads look), some through output corruption (what heads write), some through both. This is consistent with Lieberum et al. (NeurIPS 2024) finding that "individual heads shift across scale but the algorithm is stable."

### Figure 6: Causal knockout results
*Bar chart.* For each model: baseline PI accuracy, PI after top-head knockout, PI after forcing correct attention. Shows the knockout effect and mechanism type at a glance.

### Figure 7: Cross-model mechanism comparison
*2×2 or 2×3 grid.* Each cell shows the forced-attention experiment result for one model: baseline, knockout, force-correct, force-wrong. Visual pattern immediately reveals Type A vs B vs A+B.

---

## Section 6: Architectural Controls (0.75 pages)

### 6.1: Base Model (Pythia-410M) — DONE

- Completion format, no instruction tuning, two-block few-shot demos
- PI > RI confirmed (RI 41%, PI 12%) → rules out SFT as cause → **confirms P5**
- Causal knockout finds early-layer primacy heads (L5H2, L0-L10) — same pattern as instruction-tuned models
- V1-promotion (not V2-suppression) confirmed on MHA architecture

### 6.2: SSMs (Mamba) — TODO

- Prediction: different interference profile (no causal masking in attention) → **tests P6**
- If Mamba shows PI ≈ RI or RI > PI → confirms attention-specificity
- If Mamba also shows PI > RI → mechanism differs (persistent channels vs attention collapse)
- Cite Airlangga et al. (2025): Mamba shows U-shaped recall through different mechanism

### Figure 8: Architectural comparison
*If Mamba done:* Side-by-side RI/PI grids for Transformer vs SSM. If not done: Pythia behavioral grid only, with SSM as future work.

---

## Section 7: Discussion & Limitations (1 page)

### 7.1: What this means

- PI > RI is a necessary consequence of causal attention architecture, not a bug
- The dual-process architecture (independent RI and PI mechanisms) means fixing PI requires targeted intervention, not general scaling
- Attention-based head classification is unreliable for circuits that operate through indirect effects — the mech interp community should prefer causal knockout

### 7.2: Implications for practice

- **RAG/context updates:** LLMs will preferentially recall stale (first-stored) information over fresh updates. System designers should place the most important/recent information last AND first (exploit both primacy and recency).
- **[If done] Surgical fix:** QK fine-tuning on late-layer heads can restore PI. <1% parameter modification, no RI degradation.

### 7.3: Limitations

1. Single task family (key-value interference). Needs validation on naturalistic tasks (medical records, legal amendments).
2. Small models (0.5B-3B). Mechanistic analysis at 7B+ is computationally expensive but would strengthen claims.
3. [If no formal theorem] The two-forces framework is an argument, not a proof. Future work should formalize it.
4. [If no Mamba] The attention-specificity claim is theoretical, not empirically validated with SSMs.
5. The "distributed vs concentrated" primacy pattern across model scales is not yet explained theoretically.

---

## Appendix Plan

| Appendix | Content |
|----------|---------|
| A | Formal proof of toy theorem (if done) |
| B | Full behavioral sweep grids for all 5 models |
| C | Logit lens trajectories at all operating points |
| D | Per-head knockout full rankings (all 384 heads per model) |
| E | Knockout validation details (200-trial CIs, cross-operating-point stability) |
| F | Forced attention full results per model |
| G | Copy-suppression test details (eigenspectrum, DLA decomposition) |
| H | Prompt format details (completion vs chat, few-shot construction) |
| I | SageMaker replication comparison tables |
| J | Pythia-160M failure analysis (why 160M can't do the task) |
| K | Mamba results (if done) |

---

## Figure Summary

| # | Figure | Section | Status |
|---|--------|---------|--------|
| 1 | The phenomenon (regime map + 39-model scatter) | Intro | Need to plot (data exists) |
| 2 | Two-forces theory schematic | Theory | Need to draw |
| 3 | Behavioral results (39-model + Pythia + scaling) | Behavioral | Need to plot |
| 4 | Universal mechanistic findings (4-panel) | Mechanistic | Need to plot |
| 5 | Logit lens trajectories (3 models × 2 conditions) | Mechanistic | Need to plot |
| 6 | Causal knockout bar charts | Mechanistic | Need to plot |
| 7 | Cross-model mechanism comparison | Mechanistic | Need to plot |
| 8 | Architectural control (Pythia ± Mamba) | Controls | Need to plot / Mamba pending |

---

## What We Still Need to Do (Experiments)

| Priority | Item | Effort | Impact on Paper |
|----------|------|--------|----------------|
| **CRITICAL** | Mamba behavioral sweep (R3) | 2 days | Validates P6, enables Section 6.2 |
| **CRITICAL** | Paper figures (R8) | 3-4 days | No paper without plots |
| **HIGH** | QK fine-tuning fix (R4) | 2-3 days | Turns diagnosis into cure |
| **HIGH** | Toy formal theorem | 2-3 days | Strengthens Section 3 |
| **MEDIUM** | Pythia training dynamics | 1-2 days | Nice addition to Section 6.1 |
| **MEDIUM** | Qwen 3B 17b/18b with correct heads | 1 day | Completes mechanism table |
| **LOW** | Naturalistic benchmark | 3-4 days | Addresses single-task limitation |

---

## Narrative Flow (How a Reader Experiences the Paper)

1. **Hook:** "LLMs can't update their beliefs" — they always recall the first thing they learned, not the latest.

2. **Scale of the problem:** 39 models, all show it, Cohen's d = 1.73. It's not a bug in one model — it's architectural.

3. **Why it happens (theory):** Two forces compete in every transformer — causal masking creates primacy, LayerNorm creates recency. Under interference, primacy wins because it accumulates with depth while recency is a per-layer correction.

4. **Predictions from the theory:** Six testable predictions. The rest of the paper validates them.

5. **Behavioral confirmation:** All 6 predictions confirmed (or 5/6 if no Mamba). Base model confirms it's not SFT.

6. **Inside the model:** Logit lens shows V1 dominating from early layers. Activation patching shows the decision happens in the final third. The heads responsible are in early layers and invisible to standard attention metrics.

7. **The surprise:** The same behavioral outcome arises from different circuits in different architectures. Some models concentrate primacy in one powerful head; others distribute it. Some use attention routing; others use output corruption. Convergent evolution toward the same failure mode.

8. **The methodological lesson:** Don't trust attention patterns for head identification. Use causal knockout. This matters beyond our specific task.

9. **[If done] The fix:** Fine-tuning <1% of parameters restores PI. The capability exists; it's locked behind a routing failure.

10. **Implications:** This is a fundamental architectural constraint, not a scaling problem. More parameters won't fix it. Targeted intervention will.
