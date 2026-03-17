# Section 1: Introduction (Draft)

When a language model processes a key-value stream where values are updated N times, it faces a fundamental retrieval challenge: given a query for the first or last value, can it correctly resolve the temporal ordering? Our prior work (Chowdhury, 2026a) demonstrated across 39 models that retrieval of the first value (retroactive interference resistance, RI) consistently dominates retrieval of the last value (proactive interference resistance, PI) — a phenomenon we term PI > RI. But *why* does this happen, and is it specific to the transformer architecture?

In this work, we present a cross-architecture mechanistic investigation of PI > RI. Our key contributions:

**1. Universality.** We show PI > RI in 9 models across 7 architecture families, including the Mamba state-space model which lacks causal attention, softmax, and positional encoding entirely. This eliminates these components as sole causes and narrows the mechanism to autoregressive processing with continuous gating and fixed-capacity state.

**2. Mechanistic characterization.** Using four complementary interpretability techniques — logit lens, probing classifiers, attribution patching, and Jacobian analysis — we show that:
- The correct last value IS computed at intermediate layers (~90% depth) but is outcompeted in the final layers (Pattern B: "overtaken")
- RI success is well-encoded in the residual stream (87% probe accuracy) while PI success is representationally indistinguishable from failure (61%, near chance)
- The mechanism is distributed — no single "interference head" can be ablated to fix PI
- Primacy bias exists at initialization (1.47x in transformers, 295x in SSMs), confirming an architectural rather than learned origin

**3. Quantitative characterization.** PI accuracy decays exponentially with the number of updates: PI(N) = a * exp(-b * N) + c, where the floor c varies from 0% to 44% depending on architecture. RI remains robust (80-100%) across all models and conditions. We identify three architecture-dependent error modes: recency imprecision (Qwen), primacy fallback (Gemma), and off-by-one lock (Pythia).

**4. Component elimination.** Our cross-architecture comparison eliminates causal attention, softmax, RoPE, and multi-head attention as sole causes. The remaining candidate mechanisms — autoregressive left-to-right processing, continuous gating, and fixed-capacity state — are shared by all models exhibiting PI > RI. A preliminary bidirectional control (Flan-T5) shows no primacy bias, supporting autoregressive encoding as the key factor.

These findings establish PI > RI as a fundamental property of sequential autoregressive processing, not an emergent behavior that disappears with scale or an artifact of specific architectural choices. We connect our empirical results to existing theoretical frameworks (Velickovic et al., 2025; Wu et al., 2025; Chowdhury, 2026b; Pasten et al., 2025) and derive bounds showing why PI must degrade with N while RI is protected by cumulative reinforcement and the attention sink mechanism.

[Figure 1: 4-panel teaser showing (a) behavioral collapse of PI across architectures, (b) logit lens suppression, (c) probing asymmetry, (d) Jacobian at initialization]
