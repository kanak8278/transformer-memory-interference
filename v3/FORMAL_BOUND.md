# Formal Theory: Asymmetric Interference in Sequential Models

## Overview

This document contains the formal theoretical contribution for the paper.
The previous version (4 "propositions" with hand-wavy proofs) was dishonest
and would be rejected immediately by any NeurIPS reviewer. This version
contains ONE correct proposition with honest axioms, plus architecture-specific
corollaries connecting to existing formal results.

**What we prove:** Given two empirically-verifiable conditions (C1, C2),
first-value retrieval dominates last-value retrieval in any fixed-capacity
sequential model.

**What we DON'T prove:** That C1 and C2 follow from architecture alone.
They are empirical conditions, well-supported by our data and by existing
theoretical results (Chowdhury, Veličković), but not derived from first
principles here.

---

## Setup

A sequence of N key-value updates for key k:

  (k, v_1), (k, v_2), ..., (k, v_N)

embedded in a longer context. An autoregressive model processes this
left-to-right, producing a hidden state h_t at each position t. After
the full sequence, a query token produces h_query.

**Retrieval task:**
- RI: "What was the first value?" → target v_1
- PI: "What was the last value?" → target v_N

**Accuracy:**
- R_first(N) = P(model outputs v_1 | RI query, N updates)
- R_last(N) = P(model outputs v_N | PI query, N updates)

**Goal:** Show R_first(N) > R_last(N) for sufficiently large N.

---

## Axioms (Empirically Verifiable)

We state two conditions that must hold. Both are testable from
experimental data (and we verify them).

**Condition C1 (Monotone Overwrite).** Each new value update partially
overwrites the representation of all previous values. Formally, for any
value v_i with i < t:

  I(h_t; v_i) ≤ I(h_{t-1}; v_i)

where I(·;·) is mutual information. The hidden state's information about
any past value is non-increasing as new values arrive.

*Empirical evidence for C1:* Our logit lens data shows P(v_last) at the
final layer (0.01-0.04) is LOWER than P(v_last) at intermediate layers
(0.14-0.24). Each subsequent layer's computation overwrites v_last's
representation. This holds across all 4 models tested.

**Condition C2 (Diminishing Marginal Overwrite).** The information lost
about v_i due to the arrival of v_{i+1} is non-increasing in i.
Formally, define the overwrite loss:

  Δ_i = I(h_i; v_i) - I(h_{i+1}; v_i)

Then Δ_i ≥ Δ_{i+1} ≥ ... ≥ Δ_{N-1} ≥ 0.

*Intuition:* The first overwrite (v_2 displacing v_1) is strongest
because h_1 is entirely dedicated to v_1. By the time v_N arrives,
the hidden state is already a compressed representation of v_1...v_{N-1},
so v_N's marginal overwrite effect is smaller.

*Empirical evidence for C2:* Our logit lens trajectories show P(v_first)
rises monotonically across layers and stabilizes near 1.0 — consistent
with the information loss about v_1 converging (bounded, decreasing
sequence). If overwrite were constant or increasing, P(v_first) would
eventually decline.

---

## Proposition: Asymmetric Interference Scaling

**Proposition.** Under conditions C1, C2, and the following additional
assumptions:

**(A1) Independent values.** v_1, ..., v_N are drawn independently from
a distribution with entropy H(v) > 0.

**(A2) Positive initial encoding.** I(h_1; v_1) > 0 (the model encodes
the first value with nonzero information).

**(A3) Summable overwrite.** Σ_{j=2}^{∞} Δ_j < I(h_1; v_1), i.e., the
total information loss about v_1 from all subsequent values is strictly
less than the initial encoding.

For any autoregressive model with fixed-capacity hidden state (dim(h_t) = d):

(i) The information about v_1 at the query position converges:

  I(h_query; v_1 | N) → I_∞ > 0 as N → ∞

(ii) The information about v_N at the query position is bounded by:

  I(h_query; v_N | N) ≤ C(d) - I(h_N; v_1, ..., v_{N-1})

where C(d) is the total information capacity of the hidden state,
and I(h_N; v_1, ..., v_{N-1}) is non-decreasing in N (by A1).

(iii) Consequently, for sufficiently large N:

  I(h_query; v_1) > I(h_query; v_N)

and therefore R_first(N) > R_last(N).

**Proof.**

*Part (i):* Under C1, the sequence {I(h_t; v_1)}_{t≥1} is non-increasing
and bounded below by 0. Therefore it converges to some limit I_∞ ≥ 0
(monotone convergence theorem).

To show I_∞ > 0: the total information loss is

  I(h_1; v_1) - I_∞ = Σ_{j=2}^{∞} Δ_j

By assumption A3, Σ Δ_j < I(h_1; v_1), so I_∞ = I(h_1; v_1) - Σ Δ_j > 0. ∎

*Remark on A3:* C2 alone (Δ_j non-increasing) does NOT guarantee Σ Δ_j
converges — the harmonic series Δ_j = c/j is non-increasing but diverges.
A3 is an additional empirical condition. We verify it holds in all tested
models: RI accuracy remains 80-100% even at N=50, implying I_∞ >> 0.
For the single-layer softmax case, A3 IS provable: the SNR decays as
O(1/t²) (from the C1 proof), giving Δ_j = O(1/j²) which is summable.

*Part (ii):* By the data processing inequality, the hidden state h_N
carries at most C(d) = d · log(precision) bits about any input.

By A1 (independence), the total information in (v_1,...,v_N) is
I(v_1,...,v_N) = N · H(v), which grows linearly. The model can capture
at most C(d) bits. By the chain rule:

  I(h_N; v_N) = I(h_N; v_1,...,v_N) - I(h_N; v_1,...,v_{N-1} | v_N)

Since v_N is independent of (v_1,...,v_{N-1}) by A1:

  I(h_N; v_N) = I(h_N; v_1,...,v_N) - I(h_N; v_1,...,v_{N-1})
             ≤ C(d) - I(h_N; v_1,...,v_{N-1})

Now, I(h_N; v_1,...,v_{N-1}) is non-decreasing in N: adding a new value
to the sequence can only increase the total information the state has
about past values (or keep it the same), because h_N is computed from
h_{N-1} plus new input, and h_{N-1} already encodes information about
v_1,...,v_{N-1}.

Under C2 (early values are most persistent), I(h_N; v_1,...,v_{N-1})
increases toward C(d) as N grows, leaving less capacity for v_N.

After write, v_N's information can only decrease (by C1 applied to
subsequent processing). ∎

*Part (iii):* From Part (i): I(h_query; v_1) → I_∞ > 0 (stable).
From Part (ii): I(h_query; v_N) ≤ C(d) - I(h_N; v_1,...,v_{N-1}),
where the bound tightens as N grows.

Since I(h_N; v_1,...,v_{N-1}) → C(d) as N → ∞ (capacity saturates with
enough independent values), eventually I(h_query; v_N) < I_∞, and
R_first(N) > R_last(N). ∎

---

## Corollary 1: Transformers (via Chowdhury's Influence Density)

**Setting:** L-layer causal transformer with residual connections.

Chowdhury (2026) provides the exact influence density:

  ρ_L(x) = (1-α)^L δ(1-x) + primacy_tail(x)

where the primacy tail diverges as (ln(1/x))^{L-1} near x=0.

**Claim:** This implies C1 and C2 for transformers.

*C1 is PROVEN for single-layer softmax attention* (see LEMMA_C1_PROOF.md):
For h_t = Σ α_j v_j with softmax weights, adding a new value v_{t+1}
strictly reduces I(h_{t+1}; v_i) < I(h_t; v_i). The proof uses the
fact that all attention weights shrink by factor r = S_t/(S_t + exp(s_{t+1}))
∈ (0,1) while a new noise term is added. Verified numerically across
10,000 random configurations with zero failures.

**Multi-layer extension (conjecture, empirically verified):** In a
multi-layer transformer, each layer adds attention + residual. The
single-layer proof does NOT directly extend because layer ℓ+1 could
increase I(h_query; v_i) by pulling in v_i's information from OTHER
positions via indirect attention paths. However, our logit lens data
shows C1 holds empirically in ALL 4 tested models (8/8 post-peak
layers show P(v_last) strictly decreasing). We conjecture C1 holds
for multi-layer transformers and leave the formal proof to future work.

*C2 is DERIVED from Chowdhury's influence density* (see LEMMA_C2_FROM_CHOWDHURY.md):
Chowdhury's ρ_H(x) = Σ c_k (ln(1/x))^{k-1} is a positive sum of
strictly decreasing functions, hence strictly decreasing on (0,1).
Since attention weights are proportional to ρ_H, and overwrite loss
is increasing in both signal and competitor strength, earlier positions
(higher ρ) experience larger overwrite than later positions (lower ρ).
Verified numerically for H ∈ {4,12,28,36}, α ∈ {0.3,0.5,0.7}.

**Prediction:** R_first(N) should be robust to N and scale with L
(more layers = more reinforcement). R_last(N) should decay as the
softmax dispersion bound limits attention to v_N to O(1/n). Our data
confirms: RI = 80-100% at all N; PI decays exponentially with N.

---

## Corollary 2: SSMs (via State Transition Eigenstructure)

**Setting:** Selective SSM with h_t = A h_{t-1} + B_t x_t.

The hidden state at position N: h_N = Σ_{j=1}^{N} A^{N-j} B_j x_j.

**Claim:** C1 holds for SSMs under the condition that the input gating
B_t doesn't systematically increase with t.

*C1 for SSMs — nuanced:* For diagonal SSMs (e.g., S4D) where ||A||_2 ≤ 1,
C1 is PROVEN: ||A^{t-i}|| is non-increasing, so the Jacobian norm and
hence I(h_t; v_i) are non-increasing (see LEMMA_C1_PROOF.md).

For HiPPO-initialized SSMs (Mamba), the situation is more subtle:
||A||_2 > 1 is possible due to non-normality (transient growth), meaning
the matrix-level influence can INCREASE temporarily. Our own analysis
shows ||A^{49}|| = 2.61 for d=64 (LEMMA_SSM_DECAY.md). So the simple
||A||≤1 argument does NOT apply to HiPPO.

However, C1 holds EMPIRICALLY for Mamba at the behavioral level: our
Jacobian measurements show the influence of position j on the output
is non-increasing in the distance N-j for the pretrained model. The
transient growth actually creates PRIMACY (amplifying early-position
influence), which is consistent with C1's spirit — early values are
MORE protected, not less. We treat C1 for HiPPO SSMs as an empirical
condition, not a proven property.

*C2 for SSMs (empirically supported, not formally derived):* With
HiPPO initialization, A preserves a polynomial basis of the input
history. Intuitively, v_1 occupies the most stable eigenmodes, and
later values compete for remaining modes. Unlike the transformer case
where C2 follows rigorously from Chowdhury's density, the SSM case
lacks a comparable closed-form influence formula. We verify C2
empirically: our logit lens-equivalent measurements (Jacobian at
init) show the influence profile is monotonically decreasing from
position 0 outward, consistent with C2. A formal derivation from the
HiPPO spectral structure is an open problem.

**Prediction:** SSMs should show even MORE primacy than transformers
(because A^{N-1} decay gives v_1 no more influence, but HiPPO
preserves it). Our data confirms: Mamba Jacobian shows 295× primacy
at initialization vs 1.47× for Qwen.

---

## What This Framework Predicts (Testable)

| Prediction | Source | Verified? |
|---|---|---|
| PI(N) degrades with N | Part (ii) — capacity saturation | YES (R²=0.64-0.98) |
| RI(N) is robust to N | Part (i) — convergence | YES (80-100%) |
| v_1 info stabilizes across layers | Part (i) | YES (logit lens: monotonic rise to 1.0) |
| v_N info peaks then declines | Parts (i)+(ii) | YES (logit lens: peak at ~90% then crash) |
| SSMs show PI > RI too | Corollary 2 | YES (Mamba gap=+57%) |
| SSMs show MORE primacy at init | Corollary 2 | YES (295× vs 1.47×) |
| PI ≈ RI at N=2 | Minimal overwrite | YES (Qwen 3B: 100% vs 100% at N=2) |
| Bidirectional models: PI ≈ RI | C1 violated (not sequential) | PARTIAL (T5: gap=-9%) |

---

## Honest Limitations

1. **C1 is proven only for single-layer attention.** The multi-layer
   extension is conjectured and empirically verified (4 models, 8/8
   post-peak layers) but not formally proven. Indirect information
   flow via residual connections could in principle violate C1.

2. **C1 for HiPPO SSMs relies on empirical verification.** The
   simple ||A||≤1 proof fails because HiPPO's non-normality causes
   transient growth (||A^49|| > 1). C1 holds behaviorally but the
   matrix-level argument breaks.

3. **A3 (summable overwrite) is an empirical condition.** For the
   single-layer case, A3 IS provable (Δ_j = O(1/j²)). For multi-
   layer models, we verify it from RI accuracy remaining 80-100%.

4. **C2 for SSMs is not formally derived.** Unlike the transformer
   case where C2 follows from Chowdhury's density, the SSM case
   has no comparable closed-form result.

5. **The capacity argument (Part ii) is qualitative.** We show
   I(h_query; v_N) decreases with N but don't derive the exact rate.
   The empirical fit PI(N) = a·exp(-b·N)+c is not predicted by the
   theory.

6. **We don't explain PI floor variation** (0% to 44% across models)
   or why different architectures produce different error modes.
   These require understanding training-specific factors.

7. **We don't prove anything about optimization.** A sufficiently
   clever training procedure might partially overcome PI > RI.

---

## Relationship to Prior Work

| Paper | What they prove | What we add |
|---|---|---|
| Chowdhury (2026) | Exact attention distribution → primacy tail | We show this implies C1+C2 → PI > RI on behavioral task |
| Veličković (2025) | α_j ∈ Θ(1/n) | We use this to bound PI capacity |
| Wu (2025) | Position 1 is attention fixed point | We extend to retrieval accuracy, not just attention |
| Pasten (2025) | Continuous models resist perturbation | We connect: value updates = perturbations → first binding persists |
| Barbero (2024) | Over-squashing bounds | We show this is transformer-specific; our result is architecture-general |
