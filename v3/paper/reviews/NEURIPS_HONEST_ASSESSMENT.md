# Honest Assessment: Is This NeurIPS-Ready?

**Updated: 2026-03-17 (Session 2)**

## Current Verdict: CLOSE, but 2-3 gaps remain

The paper has matured significantly from "thorough empirical study" to "mechanistic investigation with architectural insights." We now have cross-architecture evidence, novel MI techniques, and a unifying theory. But the formal theoretical contribution is still the weakest link.

## What We Have (Strengths)

### 1. Universal behavioral finding (STRONG)
- 11 models across 7 architecture families (Qwen, Gemma, Llama, StableLM, Pythia, Mamba, API)
- PI > RI in EVERY model with sufficient capacity
- 200-trial Wilson CIs at key operating points
- PI(N) fits exponential decay: PI(N) = a·exp(-b·N) + c, R²=0.64-1.0

### 2. Cross-architecture SSM comparison (NOVEL, STRONG)
- Mamba-1.4B (SSM, no attention) shows PI > RI (gap=+49-70%)
- This ELIMINATES causal attention, softmax, RoPE as sole causes
- Narrows to: autoregressive L→R + continuous gating + fixed-capacity state
- Jacobian at init: Mamba shows 295× primacy (extreme), Qwen shows 1.47× (mild)
- **No other paper has this cross-architecture Jacobian comparison**

### 3. Probing classifiers (NOVEL, STRONG)
- RI/PI condition discrimination: 97-100% across 3 models
- RI correctness encoded (60-87%); PI correctness at chance (50-61%)
- **Shows asymmetry in REPRESENTATION SPACE, not just output**
- Independent from logit lens, uses residual stream directly
- **No paper uses probing for PI/RI specifically**

### 4. Logit lens: value found then suppressed (STRONG)
- Consistent Pattern B across 4 models
- P(v_last) peaks at ~90% depth, gets outcompeted by penultimate
- P(v_first) rises cleanly and monotonically to 1.0
- Clean narrative: "the model finds the right answer but can't hold onto it"

### 5. Causal analysis: distributed mechanism (MODERATE)
- No bottleneck heads — ablating top heads HURTS retrieval
- Attribution-important heads at 60-80% depth (before value emergence)
- **This IS a finding:** PI > RI is architectural, not a circuit bug
- But the effects are small (+0.03-0.07 per head)

### 6. Error position characterization (NOVEL, MODERATE)
- Architecture-dependent failure modes:
  - Qwen: recency imprecision (off-by-one → diffuse with N)
  - Gemma: primacy fallback (4 heads → defaults to first value)
  - Pythia: off-by-one lock (base model → penultimate at high N)
- N-dependent transition is statistically significant (Qwen 3B: p=0.008)

### 7. Narrative transfer (NOVEL)
- PI > RI on Dota 2 narratives (gap=+18% on Qwen 1.5B)
- Not a KV-format artifact
- **No other paper tests PI/RI on naturalistic narrative text**

### 8. Jacobian at initialization (NOVEL)
- Primacy bias exists in UNTRAINED models
- Mamba: extreme primacy (295×) due to recurrent exponential decay
- Qwen: mild primacy (1.47×) due to causal attention compounding
- Training amplifies both primacy and recency
- **Novel cross-architecture comparison at initialization**

## What We're Missing (Critical Gaps)

### Gap 1: No Formal Theorem (CRITICAL for NeurIPS)

We have an informal "Three-Force Model" but no mathematical proof. The closest we have:
- PI(N) = a·exp(-b·N) + c (empirical fit, not derived from first principles)
- Verbal argument connecting softmax dispersion + cumulative reinforcement + positional confusion
- References to Veličković (softmax bound), Wu (position bias), Chowdhury (influence density)

**What would close this gap:**
- A simplified model (e.g., 1-layer linear attention) where PI(N) can be derived exactly
- Or: show that Pasten et al.'s continuity theorem directly predicts PI > RI
- Or: derive the PI floor as function of d_model and n_heads

**Honest risk:** Without this, reviewers will say "nice experiments but where's the theory?"

### Gap 2: Bidirectional Control (RUNNING)

The strongest test of the "autoregressive processing" claim is: does a bidirectional model show PI ≈ RI?

Currently running BERT MLM probe (bidirectional_probe_v2.py). Expected outcomes:
- If PI ≈ RI in BERT → confirms autoregressive processing is the cause
- If PI > RI in BERT → autoregressive is NOT the sole cause (deeper)

This experiment will complete within this session.

### Gap 3: Statistical Power on Some Models

| Model | Trials/cell | Wilson CI width at 50% | Sufficient? |
|---|---|---|---|
| Qwen 3B-Base | 200 | ±7% | YES |
| Qwen 1.5B | 200 | ±7% | YES |
| Qwen 3B-Inst | 200 | ±7% | YES |
| Qwen 0.5B | 50 | ±14% | BORDERLINE |
| Gemma 1B | 50 | ±14% | BORDERLINE |
| TinyLlama | 30 | ±18% | NO — need rerun |
| StableLM | 30 | ±18% | NO — need rerun |
| Mamba 1.4B | 30 | ±18% | NO — need rerun |
| Pythia | 50 | ±14% | BORDERLINE |

**Action needed:** Rerun TinyLlama, StableLM, Mamba at 200 trials. The gaps are large enough that the direction (PI > RI) is unambiguous, but ±18% CIs are embarrassing for NeurIPS.

### Gap 4: Scaling Law Needs More Points

Current: 3-9 model sizes for PI vs model_size correlation.
The correlation is significant at N=20 (r=0.886, p=0.019) but only 6 data points.
Would benefit from more sizes (API models at different N values could help).

## Revised Paper Strategy

### Option A: Strong Empirical + Weak Theory (Current trajectory)
- Lead with the cross-architecture universality finding (novel)
- Present the Three-Force Model as an organizing framework (not a proof)
- Emphasize novel MI techniques (probing, Jacobian at init)
- Risk: "nice experiments, where's the theorem?" from theory-minded reviewers
- Mitigation: position paper as "mechanistic investigation" not "theoretical contribution"

### Option B: Derive Formal Result (Higher ceiling, higher effort)
- Derive PI accuracy bound from a simplified attention model
- Show it matches empirical data
- The theory becomes the main contribution; experiments validate it
- Risk: derivation may not work or may require unrealistic simplifications

### Recommendation: Option A with a formal result in Section 3

Even an approximate analytical result would elevate the paper:

**Proposition:** For an L-layer causal attention model with softmax temperature θ and sequence length n, the retrieval probability satisfies:
- P(v_0 | "first") ≥ 1 - O(exp(-L·α_sink)) — approaches 1 with depth
- P(v_{N-1} | "last") ≤ O(1/N · exp(δ/θ)) — decays with N

Where α_sink is the attention sink coefficient (measurable) and δ is the key separation.

**Corollary:** PI accuracy decays as N increases; RI accuracy is robust.

This isn't a hard theorem — it's a bound argument. But it provides the formal hook NeurIPS expects.

## What Makes This Paper vs NOT a Paper

### Unique contributions no one else has:
1. Cross-architecture PI/RI comparison (Mamba + 6 transformer families)
2. Jacobian at initialization across architectures
3. Probing classifiers for PI/RI discrimination
4. Architecture-dependent error position patterns
5. Narrative transfer validation
6. Component elimination table (cross-architecture)
7. 200-trial statistical rigor

### What we're competing against:
- Chowdhury (2603.10123): formal theory, transformer-only
- Wang et al. (2506.15156): Mamba primacy/recency, but different task
- Wu et al. (ICML 2025): formal proof, attention-only
- Veličković et al. (ICML 2025): softmax bound, theoretical

**Our niche:** We're the only paper with BOTH:
(a) Cross-architecture behavioral evidence (11 models, 7 families)
(b) Cross-architecture mechanistic evidence (logit lens, probing, Jacobian, causal)

The theory papers prove things about specific architectures. We show the phenomenon is MORE general than any single theory predicts. That's the contribution.

## Priority Actions (Remaining)

1. **BERT MLM probe** — running, will determine if we can claim "autoregressive is the cause"
2. **200-trial reruns** — TinyLlama, StableLM, Mamba (after Phi finishes GPU)
3. **Formal bound derivation** — write the proposition + corollary in Section 3
4. **Paper main figure** — regenerate with all models, proper CIs
5. **LaTeX draft** — start writing with the outline we have
