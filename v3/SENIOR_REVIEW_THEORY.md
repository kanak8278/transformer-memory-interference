# Senior Researcher Review: Theory Formalization

**Date:** 2026-03-17, Session 2 (post-theory revision)
**Reviewer perspective:** NeurIPS Area Chair, theoretical ML background

---

## Overall Assessment

The theory has improved DRAMATICALLY from the initial 4-proposition version.
The current structure (axioms → proposition → corollaries → verification) is
the right pattern. But there are still real problems a careful reviewer would
catch.

**Grade: B+.** Publishable as empirical-with-theory, not as theory contribution.

---

## What's Good (Keep These)

### 1. The C1/C2 axiom framework is honest
Stating conditions explicitly instead of pretending to derive everything from
scratch is the right move. The empirical verification (logit lens, 10K random
trials) is thorough. A reviewer can evaluate: "do I buy C1 and C2?" which is
a clean question.

### 2. The C1 proof for single-layer softmax is correct
The SNR argument is algebraically clean. The factoring trick
(α_j^{(t+1)} = r · α_j^{(t)} with r ∈ (0,1)) is elegant. 10K random
configs with zero failures is convincing numerical support. This is a
genuine lemma.

### 3. C2 from Chowdhury's density is clean
Showing ρ_H(x) is a positive sum of decreasing functions → decreasing is
trivially correct. The connection to overwrite loss is reasonable. This
properly BUILDS ON existing published work rather than competing with it.

### 4. The predictions table is strong
8 predictions, 7 confirmed. This is what separates theory from post-hoc
rationalization. The N=2 prediction (PI ≈ RI) is particularly good — it
wasn't obvious and the data confirms it.

---

## What's Wrong (Fix These)

### Problem 1: The C1 proof DOES NOT extend to multi-layer transformers

The text says:
> "each layer's attention at the query position 're-reads' the sequence,
> reproducing the single-layer dilution"

This is WRONG. In a multi-layer transformer:
- Layer ℓ+1 attends to the outputs of layer ℓ, NOT to the original values
- The residual stream at position j after L layers is NOT v_j — it's a
  complex nonlinear transformation of v_j mixed with information from all
  other positions
- Layer ℓ+1's attention could INCREASE I(h_query; v_i) by pulling in
  information about v_i from OTHER positions where it was "stored" via
  earlier attention patterns

**Example of failure:** Suppose at layer ℓ, position j has some information
about v_i (because layer ℓ-1 attended to v_i's position). Layer ℓ+1 at the
query position then attends to position j and recovers v_i information.
This is INDIRECT information flow and it CAN increase I(h_query; v_i) at
layer ℓ+1 relative to layer ℓ.

**Impact:** MODERATE. C1 is verified empirically (logit lens shows monotone
decrease for v_last across all 4 models). But the claim that it FOLLOWS
from the single-layer proof is false. Should say: "proven for single-layer,
empirically verified for multi-layer, extension conjectured."

### Problem 2: The convergence argument has a gap

Part (i) claims I_∞ > 0 but the proof only shows I_∞ ≥ 0. The argument is:

> "I_∞ = I(h_1; v_1) - Σ Δ_j > 0 provided the initial encoding I(h_1; v_1)
> is nonzero and the series converges to less than I(h_1; v_1)"

The condition "series converges to less than I(h_1; v_1)" is exactly what
you're trying to prove! This is circular. You need C2 to BOUND Σ Δ_j, but
C2 only says Δ_j is non-increasing — it doesn't bound the sum.

**Counterexample:** Δ_j = 1/j is non-increasing but Σ 1/j = ∞ (harmonic
series diverges). So C2 alone does NOT guarantee convergence to I_∞ > 0.

**Fix:** You need an additional condition or a stronger version of C2:
- Option A: Require Δ_j = O(1/j^{1+ε}) for some ε > 0 (summable)
- Option B: Directly assume I_∞ > 0 as an empirical condition (C3)
- Option C: Use the fact that Δ_j is bounded by C(d)/j where C(d) is the
  capacity, which gives Σ Δ_j ≤ C(d) · H_N (harmonic number) — but this
  DIVERGES, which means I_∞ = 0 is possible!

**Impact:** HIGH. The proposition's Part (i) is the foundation for everything
else. If I_∞ = 0 is possible, then v_1's information COULD be completely
overwritten, and the theory doesn't guarantee PI > RI.

**Honest resolution:** Our EMPIRICAL data shows I_∞ > 0 (RI stays at 80-100%).
The theory should say: "C1+C2 guarantee convergence (I_∞ ≥ 0), and our data
shows I_∞ > 0 in all tested models. A sufficient condition for I_∞ > 0 is
Σ Δ_j < I(h_1; v_1), which holds when C2 is strict enough (Δ_j → 0 faster
than 1/j)."

### Problem 3: Part (ii) conflates two different things

The bound:
> I(h_N; v_N) ≤ C(d) - I(h_N; v_1, ..., v_{N-1})

uses the chain rule of mutual information. But then the text says:
> "As N grows, I(h_N; v_1, ..., v_{N-1}) cannot exceed C(d), but the number
> of prior values consuming representational capacity grows"

This doesn't actually show I(h_N; v_N) decreases. I(h_N; v_1, ..., v_{N-1})
could STAY at the same value (if later values add no new information), in
which case I(h_N; v_N) = C(d) - constant, which is also constant.

**The real argument needs:** As N grows, the prior values v_1,...,v_{N-1}
ARE informationally distinct (they're sampled independently), so
I(h_N; v_1,...,v_{N-1}) should increase toward C(d). But you need to show
this formally — it's not obvious that the model's representation captures
MORE total information about v_1,...,v_{N-1} as N grows (it could be
compressing).

**Fix:** Add the assumption that each v_j is independently drawn from a
distribution with positive entropy. Then I(v_1,...,v_N) = N · H(v_j) which
grows linearly, while I(h_N; v_1,...,v_N) ≤ C(d). By the chain rule,
I(h_N; v_N | v_1,...,v_{N-1}) ≤ C(d) - I(h_N; v_1,...,v_{N-1}), and since
I(h_N; v_1,...,v_{N-1}) is non-decreasing in N (more data → more captured
information, at least weakly), the bound tightens.

### Problem 4: SSM C1 proof has a subtle error

The SSM proof claims:
> ||A^{t-i}|| ≤ ||A||^{t-i} ≤ 1

But ||A|| is the OPERATOR norm, and for non-normal matrices:
- ||A||_2 can be > 1 even if all eigenvalues are < 1
- Your own HiPPO analysis shows ||A^{49}||_2 = 2.61 for d=64!

So ||A|| > 1 is POSSIBLE for HiPPO, which means ||A^{t-i}|| is NOT
guaranteed to be non-increasing. The C1 proof for SSMs is WRONG for HiPPO
initialization.

**Impact:** HIGH. This directly contradicts your Mamba analysis which shows
||A^49|| > 1 (transient growth). You can't simultaneously say "C1 holds
because ||A|| ≤ 1" and "HiPPO has ||A^49|| > 1."

**Fix:** The C1 proof for SSMs should be restricted to: "C1 holds for SSMs
with ||A||_2 ≤ 1 (e.g., diagonal SSMs like S4D). For HiPPO-initialized
Mamba, C1 is violated at the MATRIX level (transient growth), but holds at
the BEHAVIORAL level (empirically verified from logit lens equivalent).
The transient growth actually HELPS primacy — it amplifies early-position
information."

### Problem 5: C2 for SSMs is hand-waving

The text says:
> "With HiPPO initialization, A preserves a polynomial basis... the
> overwrite effect of new values diminishes"

This is verbal argument, not a derivation. Unlike the transformer case
where C2 follows cleanly from Chowdhury's formula, the SSM case has no
formal backing.

**Fix:** Acknowledge honestly: "C2 for SSMs is empirically supported but
not formally derived. The HiPPO polynomial preservation provides intuition
but not a proof."

### Problem 6: The d=256 number is embarrassing

||A_Δ^{49}|| = 6 × 10^12 for d=256 with first-order discretization.
This is numerical blowup from an inappropriate discretization, not a
real physical effect. Mamba uses ZOH or bilinear discretization which
prevents this. Including this number undermines credibility.

**Fix:** Either use proper ZOH discretization in verify_theory.py, or
drop d=256 from the table and acknowledge the limitation.

---

## Summary: Theory Scoreboard

| Component | Status | Rigor Level |
|---|---|---|
| C1 (single-layer softmax) | **PROVEN** | Rigorous (algebraic + 10K numerical) |
| C1 (multi-layer transformer) | **EMPIRICAL ONLY** | Verified in 4 models but proof doesn't extend |
| C1 (SSMs with ||A||≤1) | **PROVEN** | Rigorous for diagonal SSMs |
| C1 (SSMs with HiPPO) | **CONTRADICTED** by transient growth | Empirically true, formally false |
| C2 (transformers) | **DERIVED** from Chowdhury | Clean formal connection |
| C2 (SSMs) | **HAND-WAVING** | Not derived, only verbal argument |
| Convergence (I_∞ ≥ 0) | **PROVEN** | Standard bounded monotone sequence |
| Convergence (I_∞ > 0) | **GAP** | Requires additional condition beyond C2 |
| v_N bound decreasing | **PARTIAL** | Capacity argument needs strengthening |
| Overall Proposition | **CONDITIONAL** | Correct given C1+C2, but C1 not proven multi-layer |

---

## Recommended Actions (Priority Order)

1. **Fix Part (i) convergence gap** — Add explicit condition for I_∞ > 0
   or acknowledge as empirical. 15 min.

2. **Fix SSM C1 contradiction** — Acknowledge HiPPO transient growth
   violates the simple ||A||≤1 argument. Restrict formal proof to
   diagonal SSMs, note HiPPO is empirically verified. 15 min.

3. **Downgrade multi-layer C1** — Change "extends via argument that..."
   to "conjectured for multi-layer, empirically verified in 4 models." 5 min.

4. **Fix d=256 embarrassment** — Drop from table or use proper
   discretization. 10 min.

5. **Strengthen Part (ii)** — Add independence assumption for values,
   show I(h_N; v_1,...,v_{N-1}) is non-decreasing. 20 min.

**Total time to fix all: ~65 min.** After these fixes, the theory is
honestly positioned and defensible. Not a theory paper, but a solid
theory section for an empirical-with-theory paper.
