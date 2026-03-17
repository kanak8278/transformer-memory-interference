# Theoretical Derivation: Why P(v_first) > P(v_last)

## Setup

Consider a simplified model of key-value retrieval in an autoregressive sequence model.

**Problem:** A sequence contains N key-value updates for a single key k:
- (k, v_0), ..., (k, v_{N-1})

The model is queried: "What was the first/last value of k?"
- RI query: retrieve v_0 (first)
- PI query: retrieve v_{N-1} (last)

**Goal:** Show that P(v_0 | query="first") > P(v_{N-1} | query="last") under general conditions.

## Key Observation from Experiments

From our probing results:
- A linear probe can perfectly (100%) distinguish RI from PI representations
- RI correctness is well-encoded (87%) but PI correctness is at chance (50-72%)

This suggests the MODEL ITSELF cannot reliably encode the last position.

## Argument 1: Softmax Dispersion (after Veličković et al., 2025)

**Lemma (Veličković Thm 2.2):** For bounded attention logits with spread δ and n tokens:

α_k ∈ [1/n · exp(-δ/θ), 1/n · exp(δ/θ)]

Every attention coefficient is bounded by Θ(1/n). As n → ∞, max attention to any position → 0.

**Application to PI > RI:**

For v_0 retrieval (RI):
- v_0 receives attention from ALL subsequent tokens at every layer
- After L layers of attention, v_0's representation has been reinforced L times
- Even with 1/n per-token attention, the CUMULATIVE attention to v_0 grows as L/n
- For v_0 specifically, the attention sink mechanism (Barbero et al.) adds a bias α_sink >> 1/n

For v_{N-1} retrieval (PI):
- v_{N-1} receives attention from only 1 subsequent token (the query)
- Its representation has been reinforced only 1 time
- The maximum attention it can receive is 1/n · exp(δ/θ)
- No attention sink boost (sinks are at position 0, not N-1)

**Result:** P(v_0) benefits from L×α_sink cumulative reinforcement.
P(v_{N-1}) is bounded by a single 1/n attention coefficient.
As N grows, P(v_{N-1}) → 0 while P(v_0) remains stable.

## Argument 2: Positional Encoding Discrimination

**For RoPE (most transformers):**

The attention score between query q at position n and key k_j at position j is:

a(n, j) = (R_{n-j} q)^T k_j

where R_{n-j} is a rotation matrix. The effective "positional similarity" depends on |n - j|.

For v_0: |n - 0| = n → maximum positional distance → unique encoding
For v_{N-1}: |n - (N-1)| = n - N + 1 → small → similar to v_{N-2}

The discrimination bound (from Hopfield theory):
- To distinguish v_{N-1} from v_{N-2}, need: β · Δ > log(N-2)
- Δ_{last} = |a(n, N-1) - a(n, N-2)| is small (adjacent positions)
- Δ_{first} = |a(n, 0) - a(n, 1)| is larger (distant positions have more spread)

**Result:** v_0 is uniquely discriminable. v_{N-1} is confusable with neighbors.

## Argument 3: Why Mamba Also Shows PI > RI

Mamba's selective state-space model uses:

B_t = σ(W_B x_t)  (input gating, σ = sigmoid/softmax-like)
h_t = A h_{t-1} + B_t x_t  (state update)

The hidden state h_N at position N is:

h_N = Σ_{j=0}^{N-1} A^{N-1-j} B_j x_j

For v_0: coefficient = A^{N-1} B_0
For v_{N-1}: coefficient = A^0 B_{N-1} = B_{N-1}

The matrix A is designed to have eigenvalues |λ| < 1 (stable recurrence).
So A^{N-1} → 0 for large N, which would favor v_{N-1}...

BUT: the selective gating B_t acts like softmax over "importance" of each input.
If B_0 >> B_{N-1} (first input gets higher gating), v_0 dominates.

**Why B_0 might be large:** In autoregressive training, the first tokens establish the context. The model learns to gate heavily for early context-setting information (analogous to attention sink).

**This is testable:** Check if B_0 > B_{N-1} in Mamba by examining the gating values.

## Summary: Three Forces Create PI > RI

1. **Cumulative reinforcement** (transformers: multi-layer attention; SSMs: recurrent accumulation)
   - v_0 reinforced across L layers; v_{N-1} reinforced once

2. **Softmax/gating dispersion** (all architectures with softmax-like gating)
   - Maximum retrievable information per position decays as 1/n

3. **Positional discrimination** (all models with positional encoding)
   - v_0 uniquely identifiable; v_{N-1} confusable with neighbors

## What We Can Prove (Formal)

**Proposition (informal):** For any autoregressive model with:
(a) softmax or sigmoid gating
(b) L processing layers
(c) positional encoding with smooth decay

The retrieval probability satisfies:
P(v_0 | first) ≥ Ω(L · α_sink) — scales with depth
P(v_{N-1} | last) ≤ O(1/N · exp(δ/θ)) — decays with sequence length

**Corollary:** PI accuracy degrades as N increases; RI accuracy is robust.

## What We Can't Yet Prove

1. The exact functional form of PI accuracy vs N
2. Why RI scales with model size (d_model) but PI doesn't
3. The specific N-dependent error distribution (off-by-one → diffuse)
4. Why different architectures show different failure modes (Gemma primacy vs Qwen recency)

These require either tighter analysis or are inherently architecture-dependent.
