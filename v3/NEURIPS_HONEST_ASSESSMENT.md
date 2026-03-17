# Honest Assessment: Is This NeurIPS-Ready?

## The Hard Truth: No, Not Yet

What we have is a **thorough empirical study with mechanistic characterization**. What NeurIPS wants is **theoretical insight with empirical validation**. We're currently inverted — lots of experiments, thin on theory.

### What NeurIPS Papers Look Like

A strong NeurIPS mechanistic interpretability paper (2024-2025 examples):
1. **States a clear theoretical claim** (not just "we observed X")
2. **Proves or derives predictions** from the theory
3. **Tests predictions mechanistically** (not just behaviorally)
4. **Shows something surprising** that changes how people think

Our work currently reads as: "We ran PI/RI experiments on 7 models and found the same pattern everywhere." That's ACL-level (where the original paper was). For NeurIPS, we need to go from "what" to "why" with mathematical precision.

## What We Actually Have (Strengths)

1. **Universal behavioral finding**: PI > RI across 7 models, 3 architectures, base + instruct
2. **Error characterization**: N-dependent transition from off-by-one to diffuse
3. **Logit lens**: v_last found at ~90% depth then suppressed — clean, reproducible
4. **Causal analysis**: distributed mechanism, no bottleneck heads
5. **Cross-architecture**: Qwen + Gemma show same pattern, different failure modes
6. **Narrative data**: Testing if PI > RI transfers to realistic text (running)

## What We're Missing (Critical Gaps)

### 1. No Original Theoretical Contribution

We cite Wu et al., Barbero et al., Veličković et al., Chowdhury et al. — but we don't DERIVE anything new. We validate their predictions. That's useful but not NeurIPS-level contribution.

**What we need:** A formal result connecting the three forces (causal compounding, softmax dispersion, positional encoding) specifically to the PI > RI asymmetry. Even a simplified model (e.g., single-layer attention with RoPE) showing why P(v_first) > P(v_last) in the infinite-width limit would be valuable.

### 2. Stage 3 (Causal) Is Weak

The patching results show +0.032 to +0.071 per head. That's tiny. The ablation makes things WORSE. This means our causal story is: "we couldn't find the mechanism" rather than "we found the mechanism."

The honest interpretation: PI failure is architectural (not localizable to heads), which IS a finding. But it's a negative result. NeurIPS reviewers may say "you just showed it's hard to find, not that you understand it."

**What we need:** A different causal approach. Instead of head-level patching, try:
- **Layer-level interventions**: Swap entire layer outputs between RI and PI
- **Positional encoding ablation**: Remove RoPE for specific position ranges
- **Attention pattern manipulation**: Force uniform attention over late positions
- **Counterfactual positions**: Move v_last to position 0, see if PI becomes easy

### 3. No Architectural Control (SSM)

The theory predicts SSMs (Mamba, RWKV) should show DIFFERENT interference patterns because they lack causal attention. This is the strongest testable prediction and we haven't tested it.

**What we need:** Run Mamba-130M and RWKV-1.5B through Stage 1. If they show PI ≈ RI (or different asymmetry), it's strong causal evidence. If they show PI > RI too, it falsifies the "causal attention is the cause" theory.

### 4. The Logit Lens Story Could Be Sharper

"v_last is found then suppressed" is descriptive. We need to show WHY it's suppressed — what specific computation in layers L32-35 (for 3B) causes the suppression? Is it:
- Attention re-routing? (heads attend away from v_last position)
- Value vector interference? (OV circuits output competing representations)
- LayerNorm amplification? (normalization amplifies the competitor)

### 5. No Scaling Law

We have 3 model sizes (0.5B, 1.5B, 3B) but haven't derived a scaling relationship. A plot showing "PI accuracy = f(N, model_size)" with a fitted functional form would be much more impressive than a table of numbers.

## Revised Paper Plan for NeurIPS

### Structure (9 pages + appendix)

**Section 1: Introduction (1 page)**
- Opening: "Causal attention creates an inherent asymmetry in positional addressing"
- Not "models have primacy bias" — that's the ACL paper
- Instead: "We derive and empirically validate a theory of WHY"
- Key result: Three-force model predicts all observed patterns

**Section 2: Related Work (0.75 pages)**
- Position bias (Wu, Barbero, Chowdhury)
- Softmax limitations (Veličković)
- Lost in the middle (Liu)
- Mechanistic interpretability (Conmy, Wang, Nanda)

**Section 3: Theory — Three-Force Model (2 pages) ← THE CONTRIBUTION**
- Define the PI/RI task formally
- Force 1: Causal compounding (cite Wu, derive for our setting)
- Force 2: Positional encoding confusion (derive RoPE discrimination bound for v_{N-1} vs v_{N-2})
- Force 3: Softmax dispersion (cite Veličković, show 1/n scaling)
- Combined prediction: PI accuracy ∝ exp(-αN) × (d_model)^0 (doesn't scale with width)
- RI accuracy ∝ 1 - exp(-β × d_model) (scales with width via attention sink)

**Section 4: Behavioral Validation (1 page)**
- Table 1: PI > RI across 7 models
- Figure 1: PI accuracy vs N (multi-model overlay) — the degradation curves
- Figure 2: Error position distribution (off-by-one → diffuse)
- API + local models, instruct + base, Qwen + Gemma + Pythia

**Section 5: Mechanistic Evidence (2.5 pages)**
- Figure 3: Logit lens comparison (4 models) — THE MONEY FIGURE
  - P(v_last) found then suppressed vs P(v_first) clean
- Figure 4: Layer-by-layer competition landscape
  - Show exactly where v_penultimate overtakes v_last
- Figure 5: Causal analysis
  - Attribution heads at 60-80% depth
  - Mechanism is distributed: no bottleneck heads
  - Ablation: heads help retrieval, don't suppress
- Figure 6: Cross-architecture comparison
  - Gemma (4 heads) vs Qwen (12 heads) → fewer heads, weaker PI
  - Gemma shows primacy default, Qwen shows recency imprecision

**Section 6: Architectural Control (0.75 pages)**
- SSM (Mamba/RWKV) predictions and results
- If SSMs show PI ≈ RI → causal attention IS the cause
- If SSMs show PI > RI → need to revise theory

**Section 7: Discussion (1 page)**
- Implications for long-context models
- Connection to representation collapse (Veličković, Pasten)
- Limitations: single dataset family, no training dynamics
- Future: can we fix PI? (architectural modifications)

### Main Figures (6)

| Figure | Content | Status |
|---|---|---|
| Fig 1 | PI vs N multi-model overlay | Done (cross_model_pi_vs_n.png) |
| Fig 2 | Error position distribution (3 N-levels × 4 models) | Done (cross_model_error_positions.png) |
| Fig 3 | Logit lens: P(v_last) vs P(v_first) (4 models) | Done (cross_model_logit_lens.png) |
| Fig 4 | Layer competition landscape (value probability heatmap) | Needs creation |
| Fig 5 | Causal analysis: attribution heatmap + patching | Needs creation |
| Fig 6 | SSM comparison | NOT DONE — need to run |

### Tables (3)

| Table | Content | Status |
|---|---|---|
| Table 1 | PI > RI across all models at matched points | Done (COMPLETE_RESULTS_TABLE.md) |
| Table 2 | Logit lens metrics (peak, suppress, RI final) | Done |
| Table 3 | Causal analysis (top heads, deltas, ablation) | Done |

## New Evidence Since Original Assessment

### Probing Classifier (STRONG new evidence)
- RI vs PI condition: 100% linear probe discrimination
- RI correctness: 84-89% probe accuracy (well-encoded)
- PI correctness: 50-72% probe accuracy (near chance)
- Cross-validated on 1.5B and 3B
- **This shows the asymmetry is in REPRESENTATION SPACE, not just output**

### Mamba SSM (Theory revision needed)
- Mamba-1.4B shows PI > RI (gap=+49%)
- PI > RI is NOT attention-specific — it's universal
- Revised theory: softmax/gating dispersion (Veličković), not causal attention

### Narrative Transfer
- PI > RI persists on Dota 2 narrative data (gap=+18% on 1.5B)
- Not a KV-format artifact

## Revised Priority Actions

1. **Derive formal theory** connecting softmax dispersion to PI > RI (Veličković-based)
2. **Run probing on Gemma** — cross-architecture probing validation (running)
3. **StableLM + Phi-3.5** — two more architectures for breadth
4. **Increase sample sizes** on key models (200→500 trials at regime B)
5. **Paper writing** — LaTeX with revised theory + probing + logit lens + causal
6. **Publication-quality figures** — unified style, proper labels

## What Makes This NeurIPS vs ACL

| Aspect | ACL (current paper) | NeurIPS (target) |
|---|---|---|
| Claim | PI > RI exists | PI > RI is architectural and predictable |
| Theory | None | Three-force model with predictions |
| Evidence | 39-model behavioral sweep | Mechanistic + causal + architectural control |
| Contribution | Phenomenon discovery | Mechanism understanding |
| Novelty | New task + finding | Theoretical framework + validation |
