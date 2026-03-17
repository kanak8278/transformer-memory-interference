# Competitive Landscape: Related Papers and Our Unique Contributions

## Directly Related Papers (Position Bias / Interference in LLMs)

### Theoretical

| Paper | Venue | Key Claim | What They Do | What We Add |
|---|---|---|---|---|
| **Chowdhury 2026** "Lost in the Middle at Birth" (2603.10123) | Preprint | Closed-form influence density: primacy tail + recency delta + dead zone | Derives exact attention distribution formula. Shows bias exists at initialization. | We validate with Jacobian analysis AND show it extends to SSMs (their theory is transformer-only). |
| **Wu et al. 2025** "On the Emergence of Position Bias" (2502.01951) | ICML 2025 | Position 1 is fixed point of multi-layer attention | Proves causal masking compounds first-position advantage exponentially | We show this is insufficient — Mamba has no attention yet shows PI > RI. Need broader explanation. |
| **Veličković et al. 2025** "Softmax is Not Enough" (2410.01104) | ICML 2025 | Softmax dispersion: attention decays as Θ(1/n) | Proves all softmax models must disperse at long contexts | We extend: dispersion affects retrieval differently for first vs last (RI immune, PI vulnerable). |
| **Barbero et al. 2025** "RoPE analysis" (2410.06205) | ICLR 2025 | RoPE doesn't inherently decay; long-distance errors from frequency aliasing | Shows RoPE effects are nuanced, not just "decay" | Our Mamba result (no RoPE, still PI > RI) eliminates RoPE as cause. |
| **Pasten et al. 2025** "Continuity and Isolation" (2505.10606) | NeurIPS 2025 | Continuous compact transformers can't learn all sequences; small perturbations → small output changes | Proves architectural limit on context sensitivity | We leverage: value update = small perturbation that continuity resists → explains PI failure. |
| **Herasimchyk et al. 2026** "Residual-Aware Theory" (2602.16837) | Preprint | Residual connections prevent collapse to position 1; create balance | Shows Wu et al.'s collapse doesn't happen in practice due to residuals | Consistent with our Jacobian: both primacy and recency present, but asymmetric. |

### Empirical / Mechanistic

| Paper | Venue | Key Claim | What They Do | What We Add |
|---|---|---|---|---|
| **Barbero et al. 2024** "Transformers Need Glasses" (2406.04267) | NeurIPS 2024 | Over-squashing: distinct inputs → arbitrarily close final representations | Proves sensitivity bounds, shows counting/copying failure | We show this manifests as PI > RI in KV retrieval + extends to non-transformer archs. |
| **Barbero et al. 2025** "Why LLMs attend to first token" (2504.02732) | Preprint | Attention sinks = defense against over-mixing; scales with model size | Shows BOS gets 46-78% of attention across model sizes | We connect: attention sink explains RI scaling (R²=0.49) while PI doesn't scale (R²=0.06). |
| **Liu et al. 2023** "Lost in the Middle" (2307.03172) | TACL 2024 | U-shaped retrieval: beginning + end > middle | Behavioral study on long-context models | We go deeper: mechanistic evidence (logit lens, probing) + error characterization + cross-arch. |
| **Hsieh et al. 2024** "Found in the Middle" | ACL 2024 | Some models show flat or middle-biased retrieval | Contradicts Liu et al. for some settings | Our data shows PI > RI is robust; "found in the middle" may be task-specific. |
| **Wang et al. 2025** "Primacy/Recency in Mamba" (2506.15156) | Preprint | Mamba shows U-shaped recall through sparse channels + exp decay | Independent validation of primacy/recency in SSMs | We found the same independently; our Jacobian analysis adds mechanistic depth. |

### LayerNorm / Normalization

| Paper | Venue | Key Claim | What They Do | What We Add |
|---|---|---|---|---|
| **LayerNorm Induces Recency Bias** (2509.21042) | Preprint | LayerNorm + causal attention creates recency bias | Shows normalization effects on position bias | We could test: does removing LayerNorm change PI > RI? (Not yet done.) |

### Perplexity / Evaluation

| Paper | Venue | Key Claim | What They Do | What We Add |
|---|---|---|---|---|
| **Veličković et al. 2025** "Perplexity is gameable" (2601.22950) | Preprint | Models can have low perplexity on wrong answers | Shows perplexity conflates confidence and correctness | Explains why PI > RI was missed: models confidently output wrong (first) value → low perplexity. |

## What We Uniquely Contribute

### 1. Cross-Architecture Universality of PI > RI
**No other paper shows this.** We test 11 models across 7 architecture families including Mamba SSM. Everyone else is transformer-only.

### 2. SSM Comparison (Mamba Shows PI > RI)
**Novel finding.** Wang et al. (2506.15156) show U-shape recall in Mamba, but NOT on a controlled PI/RI task with KV streams. We show Mamba has +49% gap on the same task as transformers.

### 3. Jacobian at Initialization Across Architectures
**Novel experiment.** Chowdhury (2603.10123) proves the theory for transformers. We empirically validate with Jacobian AND extend to Mamba (295× primacy at init). No one has compared untrained transformer vs untrained SSM Jacobian profiles.

### 4. Probing Classifiers for PI/RI
**Novel technique application.** No paper uses linear probing to show that RI correctness is encoded (87%) but PI correctness is at chance (50-72%). This demonstrates the asymmetry is in representation space.

### 5. N-Dependent Error Position Analysis
**Novel characterization.** We show errors transition from off-by-one (85% penultimate at N=5) to diffuse (uniform at N=20), and this differs by architecture (Gemma: primacy default, Qwen: recency imprecision).

### 6. Narrative Transfer
**Novel dataset.** PI > RI on Dota 2 match narratives (gap=+18%), showing the effect isn't artifact of KV format.

### 7. Component Elimination Table
**Novel analysis.** By comparing transformer vs SSM, we eliminate attention, softmax, RoPE, multi-head as sole causes, narrowing to autoregressive + continuous gating + fixed-capacity state.

### 8. 200-Trial Statistical Rigor
**Methodological contribution.** Most papers in this space report results from 10-50 trials. Our 200-trial sweeps with Wilson CIs provide publication-grade statistical evidence.

## Summary: What's New vs What's Known

| Finding | Known? | Our Contribution |
|---|---|---|
| PI > RI in transformers | Partially (Liu 2023, our ACL) | More comprehensive, cross-architecture |
| Mechanistic explanation | NO | Logit lens + probing + causal + Jacobian |
| SSM comparison | NO (Wang 2025 partial) | Direct PI/RI comparison on same task |
| Jacobian at init | NO | Novel experiment across architectures |
| Probing for PI/RI | NO | Novel technique application |
| N-dependent error positions | NO | Novel characterization |
| Narrative transfer | NO | Novel dataset validation |
| Component elimination | NO | Novel cross-architecture analysis |
| Unified theory (autoregressive + continuous + fixed-capacity) | Partially (Pasten 2025) | We provide the empirical evidence for Pasten's theory applied to PI/RI |
