# Paper Results Draft: NeurIPS Submission

**Title:** Primacy is Architectural: Cross-Architecture Mechanistic Analysis of Interference in Sequence Models

---

## Section 4: Behavioral Results

### 4.1 PI > RI is Universal Across Architectures

We evaluate 9 models across 2 broad architecture classes — 5 transformer variants (Qwen GQA, Gemma MHA, TinyLlama, StableLM, Pythia GPT-NeoX) and 1 state-space model (Mamba) — plus 2 API models (Claude Haiku, GPT-4.1-mini) on a different dataset. Results are shown in Table 2 and Figure 2.

**Key findings:**
- PI > RI appears in **91% of cells** (51/56) across all tested conditions
- Mean gap: +50.3 percentage points
- The two negative-gap cells occur only in Pythia-410M at N>=30, where both RI and PI collapse to near-zero (total task failure, not genuine PI advantage)
- At N=10 (moderate difficulty), every model shows PI > RI with gaps ranging from +26% (Qwen 3B-Inst) to +100% (TinyLlama)

**Model-specific observations:**
- **Qwen 3B-Base** has the highest PI accuracy (90% at N=5), demonstrating that base models can retain more recent values than instruction-tuned variants
- **Mamba-1.4B** (SSM, no attention) shows PI > RI with gap=+57-70%, directly contradicting the hypothesis that causal attention is the sole cause
- **TinyLlama** shows the most extreme asymmetry (RI=100%, PI=0% at N=10), suggesting strong primacy bias in the Llama architecture

**Statistical rigor:** Key operating points (Qwen 1.5B, 3B) have 200 trials per cell (Wilson CI at most 7%). Other models have 30-50 trials (CI at most 14-18%), sufficient to establish the direction of the effect.

### 4.2 PI Decays Exponentially with N

PI accuracy follows an exponential decay model:

**PI(N) = a * exp(-b * N) + c**

where c is the model-dependent floor (asymptotic PI accuracy as N -> infinity).

| Model | PI floor (c) | Decay rate (b) | R-squared |
|---|---|---|---|
| Qwen 0.5B | 12.5% | 0.178 | 0.64 |
| Qwen 1.5B | 10.5% | 0.092 | 1.00* |
| Qwen 3B-Base | 33.6% | 0.079 | 0.98 |
| Qwen 3B-Inst | 44.4% | 0.219 | 1.00* |

*R-squared=1.0 from only 3 data points (3-parameter fit). Should be interpreted cautiously.
| Gemma 1B | ~0% | 0.024 | 0.32 |
| Mamba 1.4B | ~0% | 0.433 | 0.88 |

**RI accuracy** is robust: 80-100% across all models and N values, with only weak logarithmic degradation at extreme N (>=50).

**PI vs model size** at N=20: Pearson r=0.886, p=0.019 (6 models; p=0.057 after Bonferroni correction for 3 tested N values). Larger models have higher PI floors, but the exponential decay rate is similar. This correlation is suggestive but should be interpreted cautiously given the small number of data points.

### 4.3 Error Position Analysis

While PI > RI is universal, the distribution of PI errors varies across architectures. We observe three distinct patterns (identified post-hoc from the data, not pre-registered predictions):

**Pattern 1: Recency imprecision** (Qwen, 16 heads)
- At low N (5): 73-98% of errors land on the penultimate value (off-by-one)
- At high N (>=20): errors become diffuse (normalized entropy increases: 0.06 -> 0.77, r=0.85, p=0.008)
- Interpretation: the model attempts late-position retrieval but can't discriminate v_{N-1} from v_{N-2}

**Pattern 2: Primacy fallback** (Gemma, 4 heads)
- At all N: 59% of errors land on the first value (v_0)
- Entropy decreases with N (r=-0.945, p<0.001)
- Interpretation: with only 4 attention heads, the model defaults to the primacy-encoded value

**Pattern 3: Off-by-one lock** (Pythia, base model)
- Penultimate fraction increases with N (0% -> 100%)
- Interpretation: base model without instruction tuning converges on the penultimate position

These patterns suggest that while the PI > RI direction is universal, the failure *mechanism* depends on architecture (head count, training). A formal model connecting architecture parameters to error mode remains an open question.

### 4.4 Transfer to Narrative Data

PI > RI transfers to Dota 2 match narratives (Qwen 1.5B, 150 trials/cell, 12 cells):
- Mean RI=54%, Mean PI=35%, Gap=+19%
- All 12 cells show RI ≥ PI (gap +9% to +33%)
- Gap widens with N: 2k_3u gap=+17% -> 3k_20u gap=+33%
- At 150 trials, key cells have non-overlapping Wilson CIs (e.g., 3k_20u: RI=55%[47-63] vs PI=22%[16-30])

---

## Section 5: Mechanistic Evidence

### 5.1 Logit Lens: Value Found Then Suppressed

Across 4 models, logit lens reveals:
1. P(v_last) emerges at ~90% depth (peaks at 0.03-0.24)
2. P(v_last) immediately outcompeted by penultimate value
3. Final P(v_last) = 0.00-0.04

For RI: P(v_first) rises monotonically to 0.92-1.00 with no competition.

| Model | Peak P(v_last) | Final P(v_last) | Suppression | Final P(v_first) RI |
|---|---|---|---|---|
| Qwen 0.5B (24L) | 0.24 (L21) | 0.04 | 0.20 | 0.92 |
| Qwen 1.5B (28L) | 0.14 (L26) | 0.01 | 0.13 | 1.00 |
| Qwen 3B (36L) | 0.21 (L32) | 0.01 | 0.19 | 1.00 |
| Gemma 1B (26L) | 0.03 (L20) | 0.00 | 0.03 | 1.00 |

### 5.2 Probing: Asymmetry in Representation Space

| Probe Target | Qwen 1.5B | Qwen 3B | Gemma 1B |
|---|---|---|---|
| RI vs PI condition | **100%** | **99%** | **97%** |
| RI correct vs incorrect | **87%** | **81%** | **60%** |
| PI correct vs incorrect | 61% | 61% | N/A |

Key: PI correctness is weakly above chance (50-61%, barely exceeding random) -- the model doesn't "know" it will fail.

### 5.3 Causal Analysis: Distributed Mechanism

| Model | Top Head | Delta P(v_last) | Ablation Effect |
|---|---|---|---|
| Qwen 1.5B | L19H6 | +0.038 (92% pos) | -0.023 |
| Qwen 3B | L26H3 | +0.032 (64% pos) | -0.044 |
| Gemma 1B | L15H2 | +0.071 | -0.020 |

Ablating top heads HURTS retrieval. No suppression circuit. Distributed mechanism.

### 5.4 Jacobian at Initialization

| Model | Untrained (quartile ratio) | Untrained (extreme) | Pretrained |
|---|---|---|---|
| Qwen 1.5B | First quarter 1.47× middle | Position 0 vs mid: 1.47× | Both primacy and recency amplified |
| Mamba 1.4B | First quarter 1.24× middle | Position 0 vs position 49: 295× | Recency learned, primacy preserved |

The extreme ratio (295×) compares the two endpoints of the sequence; the quartile ratio (1.24×) is a more robust aggregate measure. Both confirm that primacy bias is ARCHITECTURAL — present before any training data. Mamba's extreme endpoint ratio reflects the non-normal transient growth of the HiPPO state transition matrix (see LEMMA_SSM_DECAY.md).

---

## Section 6: Component Elimination

| Component | Transformer | Mamba | Eliminated? |
|---|---|---|---|
| Causal attention | Yes | No | YES |
| Softmax | Yes | No (sigmoid) | YES |
| RoPE | Yes | No | YES |
| Autoregressive L->R | Yes | Yes | NOT eliminated |
| Fixed-capacity state | Yes | Yes | NOT eliminated |
| Continuous gating | Yes | Yes | NOT eliminated |

The remaining three candidate mechanisms are shared by all models exhibiting PI > RI. This narrows the search space but does not prove these are individually necessary — a model possessing all three could potentially avoid PI > RI through specific training.

**Bidirectional control (preliminary, appendix):**
- Flan-T5-base (bidirectional encoder + autoregressive decoder): RI=13%, PI=22%, gap=-9%.
- No primacy bias observed, but result is inconclusive due to high garbage rate (64-77%) and overlapping confidence intervals. We include this as preliminary evidence suggesting autoregressive encoding may be a factor, but emphasize it does not constitute proof.

---

## Figures Summary (Updated)

| Figure | Content | File | Status |
|---|---|---|---|
| Fig 1 | 4-panel teaser (6 models) | paper_main_figure.png | Done |
| Fig 2 | PI vs N (9 models, 7 families) | cross_model_pi_vs_n.png | Done |
| Fig 3 | Error positions (6 models x 2 N) | cross_model_error_positions.png | Done |
| Fig 4 | Logit lens (4 models) | cross_model_logit_lens.png | Done |
| Fig 5 | Probing (3 models) | probing_results.png | Done |
| Fig 6 | Jacobian comparison | jacobian_comparison.png | Done |
| Fig 7 | Scaling law fitted | scaling_law_fitted.png | Done |
| Fig 8 | Error position model | error_position_model.png | Done |
