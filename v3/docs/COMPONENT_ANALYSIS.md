# Component Analysis: What Causes PI > RI?

## Evidence from Architecture Comparison

Our 11-model comparison lets us perform elimination analysis:

| Component | Present in Transformer? | Present in Mamba SSM? | PI > RI in both? | Eliminated? |
|---|---|---|---|---|
| Causal attention mask | Yes | No | Yes | **ELIMINATED** as sole cause |
| Softmax attention | Yes | No (uses sigmoid) | Yes | **ELIMINATED** as sole cause |
| RoPE positional encoding | Yes (most) | No | Yes | **ELIMINATED** as sole cause |
| Multi-head attention | Yes | No | Yes | **ELIMINATED** as sole cause |
| Autoregressive (L→R) generation | Yes | Yes | Yes | NOT eliminated |
| Fixed-capacity state | Yes (final token) | Yes (recurrent state) | Yes | NOT eliminated |
| Softmax-like gating | Yes (attention) | Yes (selective scan σ) | Yes | NOT eliminated |
| Token embedding + unembedding | Yes | Yes | Yes | NOT eliminated |
| Layer normalization | Yes | Yes | Yes | NOT eliminated |
| Residual connections | Yes | Yes | Yes | NOT eliminated |

## What Components COULD Cause PI > RI?

### 1. Autoregressive Left-to-Right Processing (MOST LIKELY)

Both transformers and SSMs process tokens sequentially from left to right. The first value binding gets processed through the FULL computational pipeline (all layers). The last value binding gets processed through fewer layers (only the final few).

**Testable prediction:** A bidirectional model (BERT-like) should show LESS PI > RI because both first and last tokens get equal processing depth.

**Status:** Not yet tested.

### 2. Fixed-Capacity State Compression

Both architectures compress a growing sequence into a fixed-dimensional representation:
- Transformer: final token's residual stream (d_model dims)
- Mamba: recurrent hidden state (d_state × d_model dims)

When multiple values compete for the same "slot" in this representation, earlier values have had more opportunity to "burn in" through iterative processing.

**Testable prediction:** Increasing d_model/d_state should help RI (more capacity) but NOT PI (the overwrite problem persists regardless of capacity).

**Status:** Partially confirmed — RI scales with model size (R²=0.49), PI does not (R²=0.06).

### 3. Continuous Gating Functions (Sigmoid/Softmax)

Both architectures use continuous (differentiable) gating:
- Transformer: softmax in attention
- Mamba: sigmoid for B_t (input gate), softplus for Δ_t (step size)

Continuous gating makes "sharp overwrite" fundamentally hard. To overwrite v_0 with v_{N-1}, the gate needs to be near-binary (completely suppress old, fully write new). But continuous functions are inherently soft.

**Testable prediction:** Models with discrete/hard gating (e.g., hard attention, binary gates) should show LESS PI > RI.

**Status:** Not yet tested.

### 4. Training Objective (Next-Token Prediction)

Both architectures are trained with the same objective: predict the next token given all previous tokens. This objective doesn't incentivize "overwrite" behavior — it rewards predicting the MOST LIKELY continuation, which is often consistent with the first binding.

**Testable prediction:** Models trained with explicit retrieval objectives (e.g., "what was the last value of X?") should show less PI.

**Status:** Not yet tested.

### 5. LayerNorm

Barbero et al. show LayerNorm induces recency bias (arXiv 2509.21042). Both transformers and Mamba use LayerNorm variants. LayerNorm normalizes the residual stream, potentially amplifying whichever value dominates.

**Testable prediction:** Removing LayerNorm (e.g., using BatchNorm or no normalization) should change the PI/RI balance.

**Status:** Not yet tested.

## Priority Experiments for Component Isolation

| Experiment | Tests | Effort | Impact |
|---|---|---|---|
| BERT/encoder model on same task | Autoregressive hypothesis | 1 day | Very high |
| Vary d_model at fixed depth | Capacity vs overwrite | 2 days | High |
| Hard sigmoid ablation in Mamba | Continuous gating hypothesis | 2 days | High |
| Compare pre-trained vs fine-tuned on retrieval | Training objective hypothesis | 3 days | Medium |
| RMSNorm vs LayerNorm vs none | Normalization hypothesis | 2 days | Medium |

## What Petar's Papers Tell Us About Components

### "Glasses" (over-squashing):
- Component: **causal attention computational graph**
- Eliminated by Mamba (no attention graph, still shows PI > RI)
- But the SPIRIT of over-squashing (fixed-capacity bottleneck) applies to SSMs too

### Softmax dispersion:
- Component: **softmax normalization in attention**
- Eliminated as sole cause (Mamba uses sigmoid, not softmax)
- But sigmoid-like gating has analogous capacity constraints at long sequences

### RoPE degradation:
- Component: **rotary positional encoding**
- Eliminated (Mamba has no positional encoding)
- This tells us PE is not the root cause

### Continuity/Isolation (Pasten et al.):
- Component: **the continuous nature of the computation**
- NOT eliminated — applies to all differentiable models
- Most promising theoretical explanation

## Conclusion: The Root Cause is Likely AUTOREGRESSIVE + CONTINUOUS + FIXED-CAPACITY

The combination of:
1. Left-to-right sequential processing (gives first binding a head start)
2. Continuous (soft) gating (prevents sharp overwrite)
3. Fixed-capacity representation (first binding "fills" the available capacity)

This combination is present in ALL architectures we tested (transformer, SSM, linear attention). It's NOT present in:
- Bidirectional models (no left-to-right processing)
- Lookup tables (discrete, not continuous)
- External memory systems (unlimited capacity)

**The testable prediction:** Models with ANY of these three components removed should show reduced PI > RI.
