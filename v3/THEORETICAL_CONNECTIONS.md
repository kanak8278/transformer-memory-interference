# Theoretical Connections: Papers That Explain PI > RI

## The Most Important Paper: Chowdhury (2026)

**"Lost in the Middle at Birth: An Exact Theory of Transformer Position Bias"**
arXiv: 2603.10123

This paper provides the EXACT closed-form formula for attention distribution across positions after H layers of causal attention with residual connections. It predicts ALL our observations.

### The Influence Density Formula

ρ_H(x) = (1-α)^H δ(1-x) + Σ_{k=1}^{H} C(H,k) α^k (1-α)^{H-k} (1/(k-1)!) (ln(1/x))^{k-1}

This decomposes into three zones:
1. **Primacy tail at x=0:** logarithmic divergence (ln(1/x))^{H-1} from causal masking compounding
2. **Recency delta at x=1:** Dirac spike (1-α)^H from residual connections ("gradient teleportation")
3. **Dead zone in the middle:** O(1/(H-1)!) factorial suppression

### Why This Matches Our Data

**PI > RI:** Primacy has polynomial-times-logarithmic growth, recency has exponential decay (1-α)^H. At realistic depths, primacy >> recency.

**Failure at 0.74-0.89:** These positions sit in the dead zone — too far from position 1 (no causal compounding benefit) and too far from position N (no residual teleportation). The attention at these positions is factorially suppressed: O(1/(H-1)!).

**RI scales with model size:** More layers → more compounding → stronger primacy tail.

**PI doesn't scale:** The 1/n softmax dispersion (Veličković) affects the recency delta equally regardless of d_model.

## Supporting Papers

### Wu et al. (ICML 2025) — "On the Emergence of Position Bias in Transformers"
arXiv: 2502.01951

**Theorem 4.1:** Under causal masking, attention to position j decays exponentially with depth:
P^(t)(z_i = j) ≤ C(1 - (j-1)ε)^t

Position 1 is the unique fixed point: all other positions' effective attention decays geometrically.

**Contributes:** The primacy side of the story. Explains WHY v_first is reliably retrieved.
**Missing:** Says nothing about recency or the specific failure zone.

### Barbero et al. (2025) — "Why do LLMs attend to the first token?"
arXiv: 2504.02732

**Key finding:** Attention sinks are LEARNED defense against representational collapse.
- LLaMA 8B: 46% attention on BOS
- LLaMA 405B: 78% attention on BOS

**Contributes:** Explains RI resistance scaling with model size (R²=0.49). Larger models have stronger sinks at position 1.
**Missing:** Nothing about PI or the failure zone.

### Veličković et al. (ICML 2025) — "Softmax is not Enough"
arXiv: 2410.01104

**Theorem 2.2:** ALL softmax attention heads eventually disperse as context grows. Every attention coefficient decays as Θ(1/n).

**Contributes:** WHY PI doesn't scale with model size (R²=0.06). The model can't concentrate enough attention on any specific late position. This is an architectural limit, not a capacity limit.
**Key:** Position 1 is partially protected from dispersion by the attention sink mechanism, but position N-1 is not.

### Barbero et al. (ICLR 2025) — RoPE Analysis
arXiv: 2410.06205

**Theorem 6.1:** Single-frequency RoPE can't robustly attend to specific tokens at long distances. Irrational rotations become dense on the circle.

**Surprise:** RoPE should actually HELP near-last positions (short relative distance). The fact that PI still fails means the attention allocation problem (causal compounding + softmax dispersion) overpowers positional encoding.

### Pasten et al. (NeurIPS 2025) — Representation Collapse
arXiv: 2505.10606

**Theorem 1 (Continuity):** Sufficiently similar inputs produce bounded output distribution changes, regardless of length. Sensitivity is highest at the END of the sequence, lowest at the START.

**Theorem 2 (Isolation):** Once a key-value association is "learned" in context (first binding), later conflicting bindings fall within its attractor basin.

**Contributes:** Explains WHY the first binding persists — it captures an attractor basin that resists displacement.

### Herasimchyk et al. (2026) — Residual-Aware Theory
arXiv: 2602.16837

**Theorem 5.4:** Residual connections prevent Wu et al.'s infinite-depth collapse to position 1. The actual attention distribution is a balance between causal compounding (favoring early) and residual connections (favoring recent).

## Unified Three-Force Model

Our paper should frame PI > RI as the result of three competing forces:

1. **Causal compounding** (Wu, Chowdhury): exponentially favors position 1 with depth
2. **Residual teleportation** (Chowdhury, Herasimchyk): gives position N an O(1) bypass
3. **Softmax dispersion** (Veličković): decays attention to ANY position as 1/n

For RI (recall first): Force 1 subsidizes retrieval. Force 3 is counteracted by Force 1. → GOOD.
For PI (recall last): Force 2 gives SOME advantage, but Force 1 actively competes. Force 3 prevents overcoming Force 1. → BAD.

The 0.74-0.89 failure zone = Chowdhury's dead zone = factorial suppression between primacy tail and recency delta.

## How Our Empirical Work Validates These Theories

| Theory prediction | Our empirical evidence |
|---|---|
| PI > RI at all depths | ✓ Confirmed across 0.5B–frontier (6 models) |
| Dead zone at 0.74-0.89 | ✓ Error positions cluster there at low N |
| Primacy tail grows with depth | ✓ RI resistance scales with model size |
| Recency delta decays with depth | ✓ PI doesn't scale with model size |
| v_last IS found then suppressed | ✓ Logit lens: peak at ~90% depth then crash |
| Mechanism is architectural | ✓ No bottleneck heads, distributed failure |
| Cross-architecture consistency | ✓ Qwen + Gemma show same pattern |
| Fewer heads → less PI capacity | ✓ Gemma (4 heads) has weaker PI than Qwen (12 heads) |
