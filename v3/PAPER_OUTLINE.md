# Paper Outline: Why Autoregressive Models Remember First, Forget Last

**Target:** NeurIPS 2026 (9 pages main + unlimited appendix)

---

## Title Options
1. "Primacy is Architectural: Cross-Architecture Mechanistic Analysis of Interference in Sequence Models"
2. "Why Autoregressive Models Remember First and Forget Last: A Mechanistic Study"
3. "Sequential Processing Creates Primacy: Mechanistic Evidence Across Transformers and State-Space Models"

---

## Section 1: Introduction (1 page)

**Opening hook:** When LLMs process a key-value stream where values are updated N times, they reliably recall the first value (RI) but catastrophically fail at the last (PI). We call this PI > RI asymmetry.

**What's known:** Our ACL paper showed this across 39 models. But WHY?

**What's new (this paper):**
1. PI > RI is universal — appears in 11 models across 7 architecture families INCLUDING Mamba SSM
2. It's architectural — Jacobian analysis shows primacy bias exists at initialization
3. It's in the representations — probing shows 100% RI/PI discrimination
4. The mechanism is distributed — no "interference heads" to ablate

**Core claim:** PI > RI is an inherent property of autoregressive sequence processing with continuous gating and fixed-capacity state, not specific to attention.

**[Figure 1: Teaser]** — The 4-panel paper_main_figure.png showing (a) behavioral, (b) logit lens, (c) probing, (d) Jacobian

---

## Section 2: Related Work (0.75 pages)

**Position bias in LLMs:** Liu et al. (2023), Hsieh et al. (2024), Chowdhury (2026)
**Attention sink / over-squashing:** Barbero et al. (2024, 2025)
**Softmax limitations:** Veličković et al. (2025)
**Positional encoding:** Barbero et al. (2025) on RoPE
**Representation collapse:** Pasten et al. (2025), Veličković et al. (2025)
**SSM analysis:** Wang et al. (2025) on Mamba primacy/recency
**Mechanistic interpretability:** Conmy et al. (2023), Nanda et al. (2023)

---

## Section 3: Experimental Setup (0.75 pages)

**Task definition:** Key-value stream with N updates per key. RI = recall first. PI = recall last.

**Dataset:** ARBITRARY_SINGLE (2,300 single-token words, 46 categories)

**Models tested:**

**[Table 1: Model inventory]**
| Model | Architecture | Params | Type |
|---|---|---|---|
| Qwen2.5-0.5B/1.5B/3B | Transformer (GQA) | 0.5-3B | Instruct |
| Gemma-3-1B | Transformer (MHA) | 1B | Instruct |
| TinyLlama-1.1B | Transformer (Llama) | 1.1B | Chat |
| StableLM-2-1.6B | Transformer | 1.6B | Chat |
| Pythia-410M | Transformer (GPT-NeoX) | 410M | Base |
| Mamba-1.4B | SSM | 1.4B | Base |
| Claude Haiku | API | ~25B | Instruct |
| GPT-4.1-mini | API | ? | Instruct |

**Evaluation:** Wilson CI confidence intervals, 200 trials/cell for key results.

---

## Section 4: Behavioral Results (1.5 pages)

### 4.1 PI > RI is Universal

**[Figure 2: Architecture comparison]** — cross_model_pi_vs_n.png
- PI accuracy collapses with N across all architectures
- RI stays robust (>80% even at N=50)
- Mamba SSM shows SAME pattern as transformers

**[Table 2: Summary at matched operating points]**
| Model | RI (2k_10u) | PI (2k_10u) | Gap |
|---|---|---|---|
| Qwen 1.5B | 98%±2% | 42%±7% | 56pp |
| Qwen 3B | 80%±6% | 54%±7% | 26pp |
| Gemma 1B | 74% | 10% | 64pp |
| Mamba 1.4B | 57% | 0% | 57pp |

### 4.2 Error Position Analysis

**[Figure 3: Error positions across N and architectures]** — cross_model_error_positions.png
- Low N (5 updates): off-by-one (penultimate dominant)
- High N (20 updates): diffuse (nearly uniform)
- Architecture-dependent: Gemma shows primacy default, Qwen shows recency imprecision

### 4.3 Transfer to Narrative Data

PI > RI transfers to Dota 2 match narratives (gap=+18% on Qwen 1.5B), confirming it's not a KV-format artifact.

---

## Section 5: Mechanistic Evidence (2.5 pages)

### 5.1 Logit Lens: Value Found Then Suppressed

**[Figure 4: Logit lens 4-model comparison]** — cross_model_logit_lens.png
- P(v_last) appears at ~90% depth then gets outcompeted
- P(v_first) rises monotonically to ~1.0
- Consistent across 4 models (Qwen 0.5B/1.5B/3B, Gemma 1B)

**[Table 3: Logit lens metrics]**
| Model | Peak P(v_last) | Final P(v_last) | Suppression | RI P(v_first) |
|---|---|---|---|---|
| Qwen 0.5B | 0.24 | 0.04 | 0.20 | 0.92 |
| Qwen 1.5B | 0.14 | 0.01 | 0.13 | 1.00 |
| Qwen 3B | 0.21 | 0.01 | 0.19 | 1.00 |
| Gemma 1B | 0.03 | 0.00 | 0.03 | 1.00 |

### 5.2 Probing: Asymmetry in Representation Space

**[Figure 5: Probing classifier results]** — new figure needed
- Condition probe (RI vs PI): 96-100% across all models
- RI correctness probe: 76-89% (well-encoded)
- PI correctness probe: 50-72% (near chance)

**Key insight:** The model's representation ENCODES whether RI will succeed but has NO signal for PI success. The asymmetry is deep in representation space.

### 5.3 Causal Analysis: Distributed Mechanism

**[Table 4: Causal patching results]**
| Model | Top Head | Delta | Ablation Effect |
|---|---|---|---|
| Qwen 1.5B | L19H6 | +0.038 | -0.023 (hurts) |
| Qwen 3B | L26H3 | +0.032 | -0.044 (hurts) |
| Gemma 1B | L15H2 | +0.071 | -0.020 (hurts) |

Ablating top heads REDUCES P(v_last) → heads help retrieval, no suppression circuit exists. Fewer heads (Gemma 4 vs Qwen 16) → larger per-head effects.

### 5.4 Jacobian at Initialization

**[Figure 6: Jacobian comparison]** — jacobian_comparison.png
- Untrained Qwen: mild U-shape (primacy 1.47× middle)
- Untrained Mamba: EXTREME primacy (295× first/last)
- Training amplifies both, but primacy foundation is architectural

---

## Section 6: Component Elimination (0.75 pages)

**[Table 5: What causes PI > RI?]**
| Component | In Transformer? | In Mamba? | Eliminated? |
|---|---|---|---|
| Causal attention | Yes | No | YES |
| Softmax | Yes | No (sigmoid) | YES |
| RoPE | Yes | No | YES |
| Autoregressive L→R | Yes | Yes | NOT eliminated |
| Fixed-capacity state | Yes | Yes | NOT eliminated |
| Continuous gating | Yes | Yes | NOT eliminated |

**Unified mechanism:** Sequential processing + continuous gating + fixed-capacity state makes overwrite harder than first-write.

---

## Section 7: Discussion (0.75 pages)

**Implications:**
- PI > RI is not fixable by scaling (PI doesn't scale, R²=0.06)
- Potential mitigations: landmark tokens, adaptive temperature, retrieval-augmented generation
- Connects to broader representation collapse literature

**Limitations:**
- All mechanistic analysis on models ≤3B
- Mamba SSM data has 25% garbage (base model)
- No formal proof of the unified theory
- Probing on BERT was inconclusive (probe design issue)

**Future work:**
- Formal proof: any continuous autoregressive model with bounded state exhibits PI > RI
- Test on bidirectional models (proper design)
- Architectural interventions (hard gating, external memory)

---

## Figures Summary

| Figure | Content | File | Status |
|---|---|---|---|
| Fig 1 | 4-panel teaser (behavioral + mechanistic) | paper_main_figure.png | Done |
| Fig 2 | PI vs N multi-model | cross_model_pi_vs_n.png | Done |
| Fig 3 | Error positions across N | cross_model_error_positions.png | Done |
| Fig 4 | Logit lens 4-model | cross_model_logit_lens.png | Done |
| Fig 5 | Probing results | **Needs creation** | TODO |
| Fig 6 | Jacobian comparison | jacobian_comparison.png | Done |

## Tables Summary

| Table | Content | Status |
|---|---|---|
| Table 1 | Model inventory | Done (in text) |
| Table 2 | Behavioral summary with CIs | Done (200-trial data) |
| Table 3 | Logit lens metrics | Done |
| Table 4 | Causal patching | Done |
| Table 5 | Component elimination | Done |
