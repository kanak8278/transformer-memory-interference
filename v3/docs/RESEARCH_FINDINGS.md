# Research Findings from Literature Deep Dive

## Key Papers Discovered

### 1. "Emergence of Primacy and Recency Effect in Mamba" (arXiv 2506.15156)
**CRITICAL:** This paper ALREADY shows primacy/recency in Mamba through different mechanisms (sparse channel persistence for primacy, exponential decay for recency). Our Mamba-1.4B finding is independently validated.

### 2. "LayerNorm Induces Recency Bias in Transformer Decoders" (arXiv 2509.21042)
Shows LayerNorm + causal attention creates recency bias — the opposing force to causal masking's primacy. This maps directly to our dual-process hypothesis.

### 3. "Lost in the Middle at Birth" (arXiv 2603.10123)
Proves with CLOSED-FORM the exact attention distribution: primacy tail (log divergence) + recency delta (exponential decay) + dead zone (factorial suppression). Exists AT INITIALIZATION — architectural, not learned.

### 4. Continuity/Isolation (Pasten et al., NeurIPS 2025, arXiv 2505.10606)
**THE unified theory:** Any compact continuous model exhibits: small input perturbations → small output changes. Updating a value (PI) is a small perturbation that continuity forces the model to partially ignore. First binding establishes an attractor basin.

## The Unified Mechanism (Our Novel Contribution)

**"Sequential processing with continuous, fixed-capacity state makes overwrite harder than first-write."**

This explains PI > RI across ALL architectures:
1. **Transformers:** Over-squashing (Barbero) + softmax dispersion (Veličković) + continuity (Pasten)
2. **Mamba SSM:** State compression + continuous gating + primacy through sparse channels (2506.15156)
3. **RWKV:** Exponential decay + fixed state

The shared ingredient: autoregressive left-to-right processing where the first binding gets encoded into a fixed-capacity state, and subsequent updates must OVERWRITE it through continuous (soft) gating.

## What Eliminates What (Cross-Architecture Evidence)

| Mechanism | Transformer | Mamba | Eliminated by Mamba? |
|---|---|---|---|
| Causal attention over-squashing | Yes | No attention | YES — not the cause |
| Softmax dispersion | Yes (softmax) | No (sigmoid) | YES — not the sole cause |
| RoPE degradation | Yes | No RoPE | YES — not the cause |
| Continuity (Pasten Thm 1) | Yes | Likely (continuous gates) | No — CANDIDATE |
| Fixed-capacity state compression | Yes (final token) | Yes (recurrent state) | No — CANDIDATE |
| Autoregressive first-write advantage | Yes | Yes | No — CANDIDATE |

## Priority Experiments (from MI Techniques Research)

| Priority | Technique | Effort | Impact | What it proves |
|---|---|---|---|---|
| 1 | **Jacobian at initialization** | 1 day | Very high | PI > RI is geometric, not learned |
| 2 | **Edge attribution patching** | 2-3 days | High | Circuit-level, not just node-level |
| 3 | Probing classifiers | Done! | High | RI/PI discrimination in representation space |
| 4 | Induction head analysis | 1-2 days | Medium | Connects to established MI vocabulary |
| 5 | SAE features | 1 week | Medium | Interpretable names for mechanisms |

## Testable Predictions from Unified Theory

1. **Bidirectional models should show LESS PI > RI** — no first-write advantage
2. **Increasing state dim helps RI (capacity) but NOT PI (gating sharpness)** — matches R²=0.49/0.06
3. **Sharp gating (hard sigmoid) should reduce PI** — overwrite becomes more binary
4. **SSMs should show larger gap than transformers** — stronger compression bottleneck
5. **Training with explicit overwrite objectives should help PI** — teaches inhibition circuit
