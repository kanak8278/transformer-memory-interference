# Section 1: Introduction (Draft)

When a language model processes a key-value stream where values are updated N times, it faces a fundamental retrieval challenge: given a query for the first or last value, can it correctly resolve the temporal ordering? Our prior work (Chowdhury, 2026a) demonstrated across 39 models that retrieval of the first value (retroactive interference resistance, RI) consistently dominates retrieval of the last value (proactive interference resistance, PI) — a phenomenon we term PI > RI. But *why* does this happen, and is it specific to the transformer architecture?

In this work, we present a cross-architecture mechanistic investigation of PI > RI. Our key contributions:

**1. Universality.** We show PI > RI in 9 models across 5 transformer variants and the Mamba state-space model, which lacks causal attention, softmax, and positional encoding entirely. This eliminates these components as sole causes and narrows the candidate mechanisms to autoregressive processing, continuous gating, and fixed-capacity state.

**2. Mechanistic characterization.** Using four complementary interpretability techniques — logit lens, probing classifiers, attribution patching, and Jacobian analysis — we show that:
- The correct last value IS computed at intermediate layers (~90% depth) but is outcompeted in the final layers (Pattern B: "overtaken")
- RI success is well-encoded in the residual stream (87% probe accuracy) while PI success is only weakly above chance (61%)
- The mechanism is distributed — no single "interference head" can be ablated to fix PI
- Primacy bias exists at initialization: Jacobian analysis of untrained models shows first-quarter influence 1.47× middle in transformers and 1.24× in SSMs (with extreme endpoint ratios up to 295× in Mamba due to HiPPO transient growth), confirming an architectural rather than learned origin

**3. Quantitative characterization.** PI accuracy decays exponentially with the number of updates: PI(N) = a * exp(-b * N) + c, where the floor c varies from 0% to 44% depending on architecture. RI remains robust (80-100%) across all models and conditions. We observe three distinct architecture-dependent error patterns: recency imprecision in multi-head models (Qwen), primacy fallback in few-head models (Gemma), and off-by-one lock in base models (Pythia). These patterns were identified post-hoc from the data and should be treated as empirical observations, not a predictive taxonomy.

**4. Component elimination.** Our cross-architecture comparison eliminates causal attention, softmax, RoPE, and multi-head attention as sole causes. The remaining candidate mechanisms — autoregressive left-to-right processing, continuous gating, and fixed-capacity state — are shared by all models exhibiting PI > RI. A preliminary bidirectional control (Flan-T5) shows no primacy bias, supporting autoregressive encoding as the key factor.

These findings establish PI > RI as a pervasive property of autoregressive sequence processing across architectures. We formalize two empirically-verifiable conditions — monotone overwrite (C1) and diminishing marginal overwrite (C2) — prove C1 for single-layer attention, derive C2 from Chowdhury's (2026) influence density, and show that together they predict all observed phenomena including PI's exponential decay with N and RI's robustness.

[Figure 1: 4-panel teaser showing (a) behavioral collapse of PI across architectures, (b) logit lens suppression, (c) probing asymmetry, (d) Jacobian at initialization]
