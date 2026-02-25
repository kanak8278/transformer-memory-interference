# NeurIPS Expansion Plan: Transformers Remember First, Forget Last

## Current State

ACL submission — strong empirical paper with clean behavioral findings:
- PI > RI universally across 39 models (Cohen's d = 1.73)
- RI and PI uncorrelated (R² = 0.044) → dual-process hypothesis
- Model size predicts RI resistance (R² = 0.49) but NOT PI (R² = 0.06)
- Reasoning models excel at RI but catastrophically fail at PI
- Error taxonomy: RI = passive retrieval failure (51%), PI = active primacy intrusion (56%)
- Decay patterns vary by architecture (exponential, polynomial, log-quadratic, power-law)

## Gap for NeurIPS

Zero theory. The "mechanistic interpretation" is 2 paragraphs of intuition. NeurIPS needs formal models, mechanistic evidence, or intervention experiments — not just behavioral observations.

---

## Direction 1: Formal Model — Attention as Associative Memory with Interference

### Core Framework: Ramsauer et al. (2021) "Hopfield Networks is All You Need"

Transformer attention IS modern Hopfield retrieval:
```
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
≡
x_new = softmax(β · Ξ^T · x)^T · Ξ
```

### What to Derive

**1. Retrieval error bound:**
For N stored key-value pairs with minimum separation Δ between target key and competitors:
```
||retrieved - target|| ≤ (N-1) · exp(-β(Δ - M))
```
where β = 1/√d_k and M depends on key correlations.

**2. Interference condition:**
Retrieval fails when `β · Δ < log(N-1)`. Explains update-count curves — as N grows, need exponentially better key separation.

**3. RI vs PI asymmetry from positional encoding:**
In causal attention, position i is attended by all positions j > i. Define effective key strength:
```
s(i) = Σ_{j>i} α_{j→i}  (cumulative attention received)
```
Early tokens have higher s(i) → stronger representations → easier to retrieve (RI-resistant) but harder to suppress (PI-vulnerable).

**4. Capacity scales with d (parameters), not context length:**
Hopfield capacity ~exp(d_k/2), depends on embedding dimension (scales with model size), not context window. Directly explains R² = 0.49 for size→RIES but R² = 0.003 for context→RIES.

### Target Theorem
Under the associative memory model: (1) PI > RI is a necessary consequence of causal attention with positional encoding, and (2) RI resistance scales with representational capacity while PI resistance is bounded by attention sharpness.

### Related Formal Frameworks to Draw From
- **Akyurek et al. (2023)** — ICL as ridge regression in attention
- **Von Oswald et al. (2023)** — Transformers learn in-context by gradient descent
- **Bai et al. (2023)** — Transformers as statisticians, algorithm selection
- **Bietti et al. (2024)** — Birth of a Transformer: A Memory Viewpoint
- **Cabannes et al. (2024)** — Scaling Laws for Associative Memories
- **Jelassi et al. (2024)** — Transformers better than SSMs at copying (capacity bounds)

### Alternative Mathematical Lenses
- **Kernel regression:** Attention output as Nadaraya-Watson estimate. Interference = bias when keys non-orthogonal.
- **Information-theoretic:** Channel capacity of attention head: C = d_k · log(1 + SNR). N_max ~ exp(d_k / 2).
- **Signal detection theory:** d' = (μ_signal - μ_noise) / σ_noise. Interference reduces d' by increasing μ_noise.

---

## Direction 2: Mechanistic Validation — Attention Probing on Open-Weight Models

### Experiments on Llama-3.x (8B or 70B)

**(a) Attention pattern analysis:**
- For RI queries: measure attention weight on initial value position vs. update positions across layers
- For PI queries: measure attention on final value position vs. initial value position
- Prediction: RI failures ↔ attention shifting away from initial position. PI failures ↔ attention stuck on initial position.

**(b) Logit lens / tuned lens:**
- At each layer, project residual stream to vocabulary space
- Track when correct answer (initial vs. final value) appears and disappears across layers
- Prediction for RI: initial value appears early, gets progressively overwritten
- Prediction for PI: initial value appears early and persists, blocking final value

**(c) Activation patching (causal tracing à la Meng et al. 2022):**
- Run model on interference prompt, identify which positions/layers are critical for correct retrieval
- Patch activations from clean run (no interference) into interference run
- Identifies which layers are the bottleneck for each interference type

**(d) Direct logit attribution:**
- Per-head contribution to output logits for initial value vs. recent value
- Identify "consolidation heads" (promote initial value) vs. "recency heads" (promote recent value)

**(e) Sparse autoencoders (if using a model with published SAEs):**
- Look for features corresponding to "initial value" vs. "most recent value" for each category
- RI failures ↔ initial-value feature overwritten (low activation)
- PI failures ↔ initial-value feature actively competing with recent-value feature

### Key Comparison: IOI Circuit (Wang et al. 2022)
The Indirect Object Identification circuit in GPT-2 is a retrieval-under-interference circuit:
- Duplicate token heads (early): detect repeated names
- S-inhibition heads (mid): suppress the wrong one
- Name mover heads (late): copy correct name to output

Your PI condition likely engages similar mechanisms. If S-inhibition heads are biased toward suppressing recent vs. early tokens, that explains PI > RI.

### Tools/Libraries
- TransformerLens (Neel Nanda) for activation access
- Baukit / pyvene for activation patching
- SAELens for sparse autoencoder analysis
- ACDC (Conmy et al. 2023) for automated circuit discovery

---

## Direction 3: Intervention Experiments — Can You Flip the Asymmetry?

**(a) Recency-weighted prompting:**
- Restructure prompts to put target info at different positions
- Test "focus on most recent information" instructions
- Reversed ordering (updates first, initial facts last)

**(b) Attention manipulation (inference-time on open-weight models):**
- Attention temperature scaling — increase β for later positions
- Positional re-weighting (ALiBi-style linear bias) at inference
- Prediction: increasing recency bias helps PI, hurts RI

**(c) Attention sink ablation (following Xiao et al. 2024):**
- Mask or downweight attention to first few tokens
- Measure whether PI improves (removing primacy protection)
- Measure whether RI degrades (removing consolidation advantage)

**(d) StreamingLLM-style window experiments:**
- Keep initial tokens + sliding window vs. full context
- Shows whether primacy protection is from attention sinks vs. broader positional effects

---

## Direction 4: Information-Theoretic Analysis

- Define retrievable information I(query; value | context) for both RI and PI
- Show causal attention creates asymmetric channel: early positions have higher MI with query output
- Derive: RI degradation follows capacity-limited curve (determined by d_model), PI degradation follows attention-resolution curve (determined by softmax temperature)
- Connect to observed decay patterns: exponential RI decay ↔ exponential pattern overlap growth; log-linear PI decay ↔ logarithmic attention dilution
- Deliverable: information-theoretic bounds predicting shape of interference curves as f(model parameters)

---

## Direction 5: Extended Empirical Analysis

**(a) State-space models (Mamba, RWKV) — HIGHEST PRIORITY in this category:**
- Fundamentally different memory architecture (no explicit attention, fixed-size state)
- Prediction: SSMs show different interference profile — possibly more human-like (RI > PI)
- Supported by Jelassi et al. (2024): SSMs have fundamentally limited memory capacity vs. transformers
- Clean architectural control that validates attention-specificity of the phenomenon

**(b) Naturalistic benchmarks:**
- Medical records (evolving patient data with contradictory readings)
- Legal documents (amended clauses)
- News timelines (updated facts)
- Addresses "synthetic stimuli" limitation

**(c) Fine-tuned vs. base models:**
- Does RLHF/instruction tuning change interference profile?
- If so, reveals what alignment training does to memory

**(d) Cross-linguistic generalization:**
- Test in non-English languages
- Controls for English-specific training distribution effects

---

## Recommended Priority (Pick 2-3)

| Direction | Impact | Difficulty | Time Estimate |
|-----------|--------|------------|---------------|
| 1. Formal model (Hopfield) | Highest | Hard | 4-8 weeks focused math |
| 2. Mechanistic probing | High | Medium | 2-3 weeks engineering |
| 3. Intervention experiments | Medium-High | Medium | 2-3 weeks |
| 4. Info-theoretic analysis | High | Hard | 3-5 weeks |
| 5a. SSM controls | Medium-High | Easy | 1-2 weeks |
| 5b-d. Extended empirical | Medium | Easy-Medium | 2-4 weeks each |

### Recommended combination:
1. **Formal model (#1)** — biggest gap, highest upgrade
2. **Mechanistic probing (#2)** — essential for "dual-process" to be more than metaphor
3. **SSM controls (#5a)** — easy win, clean architectural validation

### Fallback if time-constrained:
- #2 (mechanistic probing) + #5a (SSMs) + sketch of formal model (#1) without full proofs

---

## Key References to Add

### Associative Memory / Hopfield
- Ramsauer et al. (2021) — Hopfield Networks is All You Need
- Bietti et al. (2024) — Birth of a Transformer: A Memory Viewpoint
- Cabannes et al. (2024) — Scaling Laws for Associative Memories

### Mechanistic Interpretability
- Meng et al. (2022) — ROME: Locating and Editing Factual Associations (NeurIPS)
- Wang et al. (2022) — IOI Circuit in GPT-2 Small (ICLR 2023)
- Olsson et al. (2022) — In-Context Learning and Induction Heads
- Conmy et al. (2023) — ACDC: Automated Circuit Discovery (NeurIPS)
- Elhage et al. (2021) — A Mathematical Framework for Transformer Circuits

### Superposition & Representations
- Elhage et al. (2022) — Toy Models of Superposition
- Scherlis et al. (2022) — Polysemanticity and Capacity
- Bricken et al. (2023) — Towards Monosemanticity
- Templeton et al. (2024) — Scaling Monosemanticity

### Attention Sinks / Positional Bias
- Xiao et al. (2024) — Attention Sinks (ICLR 2024)
- Sun et al. (2024) — Massive Activations in LLMs
- Darcet et al. (2024) — Vision Transformers Need Registers (ICLR 2024)
- Press et al. (2022) — ALiBi: Train Short, Test Long
- Chi et al. (2023) — Dissecting Transformer Length Extrapolation

### ICL Theory
- Akyurek et al. (2023) — ICL with Linear Models
- Von Oswald et al. (2023) — Transformers Learn ICL by Gradient Descent
- Bai et al. (2023) — Transformers as Statisticians
- Garg et al. (2022) — What Can Transformers Learn In-Context?

### Memory in Transformers
- Jelassi et al. (2024) — Transformers Better than SSMs at Copying
- Wu et al. (2024) — Retrieval Head Mechanistically Explains Long-Context Factual Recall

### Cognitive Science (already cited, keep)
- Anderson et al. (1994) — Retrieval-Induced Forgetting
- Wixted (2004) — Consolidation-retrieval distinction
- Underwood (1957), Jenkins & Dallenbach (1924)
