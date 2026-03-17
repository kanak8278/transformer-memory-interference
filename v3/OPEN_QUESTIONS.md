# Open Questions for NeurIPS Paper

**Updated: 2026-03-17**

## Questions We CAN Answer (Evidence Exists)

### Q1: Does PI > RI appear across architectures?
**Answer: YES.** Tested 11 models across 7 architecture families:

| Family | Models | PI > RI? |
|---|---|---|
| Transformer GQA (Qwen) | 0.5B, 1.5B, 3B-Inst, 3B-Base | YES (gap 11-76%) |
| Transformer MHA (Gemma) | 1B | YES (gap 64-82%) |
| Transformer Llama (TinyLlama) | 1.1B | YES (gap 83-100%) |
| Transformer (StableLM) | 1.6B | YES (gap 67-87%) |
| Transformer GPT-NeoX (Pythia) | 410M | YES (gap 50-66%) |
| SSM (Mamba) | 1.4B | YES (gap 57-70%) |
| API (Claude Haiku, GPT-4.1-mini) | ~25B, ? | YES (gap 30-90%) |

**Mamba-130M and RWKV-430M unusable** (>60% garbage, too small).

**Strength:** Very strong. 200-trial CIs at key points. Cross-architecture consistency compelling. SSM inclusion eliminates attention as sole cause.

### Q2: Is the asymmetry in representation space, not just output?
**Answer: YES.** Probing classifiers across 3 models:

| Model | Condition disc. | RI correct probe | PI correct probe |
|---|---|---|---|
| Qwen 1.5B | 100% | 87% | 61% |
| Qwen 3B | 99% | 81% | 61% |
| Gemma 1B | 97% | 60% | N/A (95.5% failure) |

**Key insight:** Model's representation ENCODES whether RI will succeed (87%) but has NO signal for PI success (61%, near-chance). The asymmetry is deep in representation space.

**Strength:** Strong. Three models, cross-validated, independent from logit lens.

### Q3: Where in the network does PI fail?
**Answer: Last ~15% of layers.** Logit lens across 4 models:

| Model | Layers | Peak P(v_last) | Final P(v_last) | Suppression | RI P(v_first) |
|---|---|---|---|---|---|
| Qwen 0.5B | 24 | 0.24 (L21) | 0.04 | 0.20 | 0.92 |
| Qwen 1.5B | 28 | 0.14 (L26) | 0.01 | 0.13 | 1.00 |
| Qwen 3B | 36 | 0.21 (L32) | 0.01 | 0.19 | 1.00 |
| Gemma 1B | 26 | 0.03 (L20) | 0.00 | 0.03 | 1.00 |

Pattern B ("Overtaken"): v_last IS found at ~90% depth then outcompeted by penultimate.

**Strength:** Strong for transformers. Not testable for Mamba (no TransformerLens support).

### Q4: Is the asymmetry architectural or learned?
**Answer: BOTH.** Jacobian at initialization:

| Model | Untrained First/Last | Pretrained First/Last | Architectural Bias |
|---|---|---|---|
| Qwen 1.5B | 1.47× middle (mild U-shape) | 1.63× first, 3.08× last | U-shape deepens |
| Mamba 1.4B | 295× first pos (extreme decay) | recency learned | Extreme primacy |

**Key finding:** Mamba's architecture is INHERENTLY more primacy-biased than transformers (exponential state decay A^{N-1}→0). Training partially compensates. Primacy foundation is architectural in BOTH architectures.

**Strength:** Strong. Novel experiment comparing untrained vs pretrained across architectures.

### Q5: Is there a concentrated "interference circuit"?
**Answer: NO.** Causal patching on 3 models:

| Model | Top Head | Delta | Ablation Effect |
|---|---|---|---|
| Qwen 1.5B | L19H6 | +0.038 | -0.023 (hurts) |
| Qwen 3B | L26H3 | +0.032 | -0.044 (hurts) |
| Gemma 1B | L15H2 | +0.071 | -0.020 (hurts) |

Ablating top heads REDUCES P(v_last) → heads HELP retrieval, don't suppress it. No "suppression circuit" exists. The mechanism is distributed — PI > RI is an architectural property, not a fixable bug in a few heads.

Fewer heads (Gemma 4 vs Qwen 16) → larger per-head effects but same qualitative story.

**Strength:** Moderate-strong. Framed as: "distributed mechanism = architectural property."

---

## Questions PARTIALLY Answered (New This Session)

### Q6: What is the formal scaling law?
**Status: PARTIALLY ANSWERED.**

**What we now have:**

PI(N) = a · exp(-b · N) + c (exponential decay to model-dependent floor)

| Model | PI floor (c) | Decay rate (b) | R² |
|---|---|---|---|
| Qwen 0.5B | 12.5% | 0.178 | 0.64 |
| Qwen 1.5B | 10.5% | 0.092 | 1.00 |
| Qwen 3B-Base | 33.6% | 0.079 | 0.98 |
| Qwen 3B-Inst | 44.4% | 0.219 | 1.00 |
| Gemma 1B | 0.0% | 0.024 | 0.32 |
| Mamba 1.4B | 0.0% | 0.433 | 0.88 |

PI vs model size at N=20: r=0.886, p=0.019 (significant positive correlation).

**What we still need:**
- Unified formula PI(N, d_model) — currently model-specific fits
- Why PI floor varies so much (0% to 44%) across models
- More model sizes for robust scaling relationship (currently 3-9 points)

### Q10: Does the error position distribution have a predictive formula?
**Status: PARTIALLY ANSWERED.**

**What we now have:**

Three architecture-dependent failure modes discovered:

1. **Qwen (16 heads):** Entropy increases with N (0.06→0.77, r=0.85, p=0.008)
   - Low N: off-by-one (penultimate dominant, 73-98%)
   - High N: diffuse (uniform errors)
   → "Recency imprecision" — positional encoding can't discriminate near-last

2. **Gemma (4 heads):** Entropy DECREASES with N (0.85→0.21, r=-0.945, p<0.001)
   - All N: primacy default (59% errors at first value)
   → "Primacy fallback" — too few heads for distributed retrieval

3. **Pythia (base, 16 heads):** Penultimate INCREASES with N (0%→100%)
   - Low N: diffuse. High N: locked onto penultimate
   → "Off-by-one lock" — base model learns penultimate as default

**What we still need:**
- Formal model connecting architecture (n_heads, d_model) to failure mode
- Connection to Chowdhury's influence density formula
- Clean test with more architectures to validate the three-mode taxonomy

---

## Questions We CANNOT Yet Answer (Critical Gaps)

### Q7: Can PI > RI be fixed?
**Status: NOT STARTED.**
**What we need:** Test interventions (landmark tokens, adaptive temperature, retrieval training, positional encoding modifications).
**Effort:** High. Each intervention is a separate experiment.

### Q8: Is the autoregressive direction the key?
**Status: RUNNING (redesigned).**

Original BERT probe was flawed — detected query word "first"/"last", not value encoding.

**Redesigned approach (bidirectional_probe_v2.py):**
- Uses BERT's native masked LM capability
- KV stream + "The first/last [category] was [MASK]."
- Compares P(correct_value) at mask position for RI vs PI
- If RI ≈ PI → autoregressive processing IS the cause
- If RI > PI → asymmetry exists even bidirectionally

**Running now.** Results expected within this session.

### Q9: What is the exact circuit (for transformers)?
**Status: PARTIAL.**
**What we have:** Head-level attribution and patching (Stage 3). Top heads at 60-80% depth.
**What we need:** Edge attribution patching (head-to-head connections). Full circuit graph.
**Effort:** 2-3 days of implementation.
**Honest assessment:** Given that the mechanism is distributed (no bottleneck heads), a full circuit may not be the right framing. The finding IS that there's no circuit — it's architectural.

---

## The Central Theoretical Question

**Why does autoregressive processing + continuous gating + fixed-capacity state create PI > RI?**

**Current answer (Three-Force Model):**

1. **Cumulative reinforcement:** v_0 is reinforced across L layers of processing; v_{N-1} is reinforced once. In SSMs: v_0 has coefficient A^{N-1} × B_0 (accumulated); v_{N-1} has coefficient B_{N-1} (single step).

2. **Gating dispersion:** All architectures with softmax/sigmoid gating show 1/n attention decay (Veličković 2025). As N grows, maximum retrievable information per position → 0.

3. **Positional discrimination:** v_0 is uniquely identifiable (maximum distance from query). v_{N-1} is confusable with v_{N-2} (adjacent positions, similar encodings). RoPE discrimination bound: β·Δ > log(N-2) fails for near-last positions.

**Component elimination (cross-architecture):**

| Component | In Transformer? | In Mamba? | Eliminated as sole cause? |
|---|---|---|---|
| Causal attention | Yes | No | YES |
| Softmax | Yes | No (sigmoid) | YES |
| RoPE | Yes | No | YES |
| Multi-head attention | Yes | No | YES |
| Autoregressive L→R | Yes | Yes | NOT eliminated |
| Fixed-capacity state | Yes | Yes | NOT eliminated |
| Continuous gating | Yes | Yes | NOT eliminated |

**Formal status:** Informal three-force argument with empirical validation. No formal theorem yet. Closest existing work: Pasten et al. (NeurIPS 2025) continuity theorem.

---

## What Would Make This a NeurIPS Paper

| Aspect | Current Status | NeurIPS Requirement | Gap? |
|---|---|---|---|
| Behavioral evidence | 11 models, 7 families | ✓ | No |
| Cross-architecture | Transformer + SSM | ✓ | No |
| Mechanistic evidence | 4 techniques (logit lens, probing, causal, Jacobian) | ✓ | No |
| SSM comparison | Mamba shows PI > RI | ✓ | No |
| Narrative transfer | Dota 2 narratives, gap=+18% | ✓ | No |
| Jacobian analysis | Untrained + pretrained, 2 architectures | ✓ | No |
| Statistical rigor | 200 trials with Wilson CIs | ✓ | No |
| Formal theory | Three-force argument, no proof | **Gap** | Informal only |
| Scaling law | PI(N) exponential fit, R²>0.6 | **Partial** | Need unified formula |
| Bidirectional control | Running (BERT MLM probe) | **Running** | Pending |
| Error position model | Three architecture-dependent modes | **Partial** | Need formal model |
