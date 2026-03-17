# Bottlenecks vs. the MI Field: What We're Missing

**Date:** 2026-03-18
**Perspective:** How does this work compare to top MI papers at NeurIPS/ICML 2024-2026?

---

## Bottleneck 1: No Circuit Found — Top MI Papers Find Circuits

**The standard in MI (2023-2026):**
- Conmy et al. 2023 "Automated Circuit Discovery" → finds specific subgraphs
- Wang et al. 2023 "Interpretability in the Wild" → indirect object identification circuit
- Hanna et al. 2024 "How does GPT-2 compute greater-than" → named components, causal graph

**What we have:** "There's no circuit, it's distributed." Honest, but leaves a vacuum.

**The gap:** A reviewer will ask: "If you can't point to what's doing it, how do you
know it's architectural vs some distributed learned pattern?"

**What we could do:** Instead of searching for a suppression circuit (doesn't exist),
find the **retrieval circuit** — what IS the model doing when PI succeeds (40% of the
time)? Edge attribution patching between successful and failing PI cases would reveal
which components contribute to correct late-value retrieval.

**Effort:** 4-6 hours (edge attribution patching implementation)
**Impact:** MEDIUM — reframes the causal story from "we couldn't find anything" to
"here's what works, and here's why it's insufficient"

---

## Bottleneck 2: No SAE Analysis — The Field Has Moved Here

**The standard in MI (2025-2026):**
- Bricken et al. 2023 "Towards Monosemanticity" → SAE features in Claude
- Templeton et al. 2024 "Scaling Monosemanticity" → millions of interpretable features
- Cunningham et al. 2024 → SAE-based circuit analysis

**What we have:** Linear probing classifiers. These are the 2022-era approach. They
tell you THAT information is encoded, but not WHAT features encode it.

**The gap:** An SAE would let us say: "Feature #4721 ('recency marker') activates
strongly in RI trials but is absent in PI trials" — giving interpretable names to the
mechanism.

**What we could do:**
- Check if pre-trained SAEs exist for Pythia-410M (EleutherAI has released some)
- Or train a small SAE on Qwen 1.5B residual stream at answer position
- Compare feature activations: RI-correct vs PI-correct vs PI-incorrect

**Effort:** 4-6 hours (if pre-trained SAEs available), 1-2 days (if training from scratch)
**Impact:** HIGH — brings the work into 2025-era MI. Would be the strongest single
improvement to the paper.

---

## Bottleneck 3: No Training Dynamics — When Does PI > RI Emerge?

**What we have:** Untrained vs pretrained Jacobian for 2 models. Binary comparison.

**The gap:** The interesting question is the trajectory. Does PI > RI:
(a) Exist from step 0 and stay constant? → purely architectural
(b) Emerge gradually during training? → learned behavior
(c) Get WORSE during training? → training amplifies architectural bias

**What we could do:** Run Stage 1 on Pythia-410M training checkpoints (available at
steps 1K, 10K, 50K, 100K, 143K on HuggingFace). Plot PI accuracy vs training step.

**Effort:** 3-4 hours (5 checkpoints × 6 cells × 50 trials)
**Impact:** MEDIUM — novel, clean experiment. Would strengthen the "architectural vs
learned" narrative with temporal evidence.

---

## Bottleneck 4: All Mechanistic Evidence on Models ≤3B

**What we have:** Logit lens, probing, causal on models 0.5B-3B. API behavioral
data from Haiku (~25B) and GPT-4.1-mini.

**The gap:** A reviewer will say: "Maybe the mechanism is qualitatively different at
70B. Your logit lens shows suppression in small models — does it still happen in
large ones?"

**What we could do:** Limited options without large GPU access. Best argument:
- The Jacobian at init doesn't depend on training (proved for any architecture)
- The softmax bound (Veličković) applies at ALL scales
- Behavioral data from API models confirms the pattern scales
- Cite Barbero et al. 2024 who show over-squashing in Gemini 1.5 (large model)

**Effort:** 0 (argumentation only) or 8+ hours (if running logit lens on a 7B model)
**Impact:** LOW to address in paper, HIGH if we could actually run 7B experiments

---

## Bottleneck 5: Remedy on Different Model Than Mechanism

**What we have:**
- Mechanistic evidence (logit lens, probing, causal) on Qwen 1.5B/3B
- Remedy experiment on Claude Haiku (API)

**The gap:** The remedy and the mechanism are on DIFFERENT models. We can't show
HOW the remedy works mechanistically because Haiku is a black box.

**What we could do:** Run remedy interventions on Qwen 1.5B, then run logit lens
WITH landmarks to show the mechanism:
- Without landmarks: P(v_last) peaks at L26 then suppressed to 0.01
- With landmarks: P(v_last) stays high through final layers?

This would be the "mechanistic remedy" — showing exactly which force is addressed.

**Effort:** 2 hours (remedy on local model + logit lens comparison)
**Impact:** HIGH — directly connects our theory to the intervention

---

## Bottleneck 6: No Connection to Practical Downstream Tasks

**What we have:** KV retrieval task (synthetic) + Dota 2 narratives (semi-realistic)

**The gap:** "So what?" — when does PI > RI actually hurt users in practice?

**Potential connections:**
- **RAG:** When a document is updated, the model retrieves the old version (first
  indexed) instead of the new one (most recently added)
- **Multi-turn chat:** Model remembers user's first stated preference but forgets
  corrections ("Actually, I changed my mind...")
- **Agent tool use:** Model calls tool with outdated parameter values
- **Code generation:** Model uses the original variable value, not the reassigned one

**What we could do:** Design a 2-3 scenario experiment showing PI > RI in a practical
setting (e.g., code variable reassignment, preference update in dialogue).

**Effort:** 6-8 hours (design + run + analyze)
**Impact:** HIGH for paper narrative, but hard to control experimentally

---

## Bottleneck 7: Multi-Layer C1 Proof Gap

**What we have:** C1 proven for single-layer attention. Empirically verified for
multi-layer (4 models, 8/8 post-peak decreases). Conjectured for general case.

**The gap:** Theory-minded reviewers will push on this. "Your main proposition
requires C1. You've only proven it for a toy model."

**What we could do:** Prove C1 for a 2-layer attention model with residual
connections. The key challenge: layer 2 can pull v_i's information from OTHER
positions via indirect attention paths. Need to show the dilution effect still
dominates.

**Approach:** In a 2-layer model, h_query^{(2)} = h_query^{(1)} + Attn^{(2)}(...).
The attention at layer 2 reads from positions whose representations already contain
mixed information. Show that the SNR for v_i in the mixed representations is lower
than in the original, so the "fresh information" channel is weaker than single-layer.

**Effort:** 2-3 hours (careful math)
**Impact:** MEDIUM — closes the formal gap for the core proposition

---

## Priority Ranking (Impact / Effort)

| # | Action | Impact | Effort | Ratio |
|---|---|---|---|---|
| 1 | SAE features for RI vs PI | HIGH | 4-6h | ★★★ |
| 2 | Remedy on local model + logit lens | HIGH | 2h | ★★★★ |
| 3 | Training dynamics (Pythia checkpoints) | MEDIUM | 3-4h | ★★★ |
| 4 | Successful retrieval circuit | MEDIUM | 4-6h | ★★ |
| 5 | 2-layer C1 proof | MEDIUM | 2-3h | ★★★ |
| 6 | Downstream task demo | HIGH | 6-8h | ★★ |
| 7 | Scale argumentation (no experiment) | LOW | 0.5h | ★★★★ |

**Best bang for buck:** #2 (remedy + logit lens, 2h) and #7 (scale argument, 0.5h).
**Biggest upgrade to paper quality:** #1 (SAE analysis) if pre-trained SAEs exist.
**Most novel experiment:** #3 (training dynamics) — no one has plotted PI > RI vs
training step.
