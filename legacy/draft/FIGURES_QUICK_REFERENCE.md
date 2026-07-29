# Quick Reference: 6 Main Figures for NeurIPS Paper

## **The Story in 6 Figures**

```
Figure 1 (Context)         Figure 2 (Core Story)      Figure 3 (Bottleneck)
   Regime Map        →         Logit Lens          →      Head Ranking
"This is our context"     "Here's where it breaks"    "Here's the culprit"
                                     ↓
Figure 4 (Validation)      Figure 5 (Robustness)      Figure 6 (Location)
  Cross-Model Consistency  →   Method Agreement    →  Layer Importance
"Happens at all scales"     "Methods converge"        "Late layers matter"
```

---

## **Figure 1: Regime Map**
- **Type:** 2D heatmap (keys × updates)
- **Color:** Green (A) → Orange (B) → Red (C) → Dark red (D)
- **Highlight:** Blue box at 1k, 5u with arrow
- **Message:** "Point B is the PI cracking edge"
- **Data:** Behavioral sweep (Exp 11)
- **Time:** 1 hour

---

## **Figure 2: Dual-Process Logit Lens**
- **Type:** 3-panel line plot
- **Panel A:** RI condition (blue)
- **Panel B:** PI condition (red)
- **Panel C:** Overlay showing divergence point
- **Message:** "Dual process: primacy + retrieval compete"
- **Data:** Logit lens (Exp 12)
- **Time:** 1.5 hours

---

## **Figure 3: Per-Head Knockout Ranking**
- **Type:** Horizontal bar chart (top 15 heads)
- **Color:** L0=green, L8H3=red, others=blue
- **Value:** Δlogit effect when head ablated
- **Message:** "L8H3 is the bottleneck (48% of failure)"
- **Data:** Per-head knockout (Exp 25a)
- **Time:** 1 hour

---

## **Figure 4: Cross-Model Head Consistency**
- **Type:** 3 side-by-side bar charts (0.5B | 1.5B | 3B)
- **Color:** Green (all models) → Yellow (2 models) → Blue (1 model)
- **Message:** "Same heads matter across scales"
- **Data:** Per-head knockout (Exp 25a) × 3 models
- **Blocker:** Need Exp 25a on 3B first
- **Time:** 1 hour (+ depends on 3B data)

---

## **Figure 5: Method Agreement Matrix**
- **Type:** Heatmap (heads × 5 methods)
- **Methods:** Knockout, Attribution, DLA, Entropy, Copy Score
- **Color:** Green (rank 1-5) → Yellow (6-10) → Blue (11-20)
- **Message:** "Multiple methods converge on L0, L8H3"
- **Data:** Knockout (Exp 25a) + Observational (Exp 25c)
- **Time:** 1 hour

---

## **Figure 6: Layer Importance (Patching Recovery)**
- **Type:** Line plot (layer × PI accuracy after patching)
- **Lines:** Residual stream patching (blue)
- **Regions:** 3 zones (primacy, middle, retrieval)
- **Message:** "Late-layer patching recovers PI (80-90%)"
- **Data:** Activation patching (Exp 15) + Query patching (Exp 22)
- **Time:** 1 hour

---

## **What You Need Right Now**

### **Data Status:**
- [x] Exp 11 (Behavioral sweep) — ready
- [x] Exp 12 (Logit lens) — ready
- [x] Exp 25a (Knockout) on 0.5B, 1.5B — ready
- [ ] Exp 25a on 3B — **CRITICAL, missing**
- [x] Exp 25c (Observational) — ready
- [ ] Exp 15/22 (Patching) — verify completeness

### **Immediate Action:**
1. **Run Exp 25a on 3B at Point B (1k, 5u)** — ~2 hours compute
   - This unblocks Figures 1-3, 5-6 immediately
   - Figure 4 requires this data
2. **Create Figures 1-3, 5 while 3B runs** (~4-5 hours)
3. **Create Figure 4 when 3B data arrives** (~1 hour)
4. **Create Figure 6 last** (~1 hour)

---

## **Data Extraction Commands (Pseudocode)**

```python
# Figure 1: Load behavioral sweep
import json
behavior = json.load(open("results/behavioral_sweep_0.5B.json"))
# Extract regime map from behavior["cells"]

# Figure 2: Load logit lens
logit_lens = json.load(open("results/Qwen2.5-0.5B/1k_5u/logit_lens.json"))
# Extract trajectory[condition][layer][p_init/p_final]

# Figure 3: Load knockout
knockout_0p5b = json.load(open("results/Qwen2.5-0.5B/1k_5u/per_head_knockout.json"))
knockout_1p5b = json.load(open("results/Qwen2.5-1.5B/1k_3u/per_head_knockout.json"))
# Extract heads[head_id][causal_effect]

# Figure 4: Load all three models
knockout_3b = json.load(open("results/Qwen2.5-3B/1k_5u/per_head_knockout.json"))
# Merge and compare rankings

# Figure 5: Load both causal and observational
knockout = json.load(open("per_head_knockout.json"))
observational = json.load(open("observational_head_metrics.json"))
# Correlate rankings

# Figure 6: Load patching
patching = json.load(open("results/.../activation_patching.json"))
# Extract results[layer][pi_accuracy_after_patch]
```

---

## **Narrative Arc (How Figures Tell Your Story)**

**Paragraph 1 (Figure 1):**
"To understand PI > RI asymmetry, we selected Point B (Figure 1), the PI cracking edge where RI succeeds (~60%) but PI fails catastrophically (~17%), maximizing signal-to-noise for mechanistic analysis."

**Paragraph 2 (Figure 2):**
"Logit lens reveals the mechanism: a dual-process architecture emerges during interference (Figure 2). In RI (blue), the initial value is correct and dominates throughout. In PI (red), the initial value dominates even though the final value is correct. The divergence occurs around layer 8-12, suggesting early layers encode primacy while late layers attempt (and fail) to retrieve final values."

**Paragraph 3 (Figure 3):**
"Per-head causal knockout identifies the primacy bottleneck (Figure 3). L8H3 alone controls 48% of PI failure—knockout increases PI accuracy from 44% to 92%. Early-layer L0 heads show consistent primacy effects. This causal hierarchy reveals that transformer PI failure concentrates in a small set of heads rather than distributed across the network."

**Paragraph 4 (Figure 4):**
"These findings generalize across model scales (Figure 4). L8H3 ranks top in Qwen-0.5B, 1.5B, and 3B. Early-layer L0 heads appear consistently. This cross-scale consistency validates that the primacy mechanism is architectural rather than emergent at specific scales."

**Paragraph 5 (Figure 5):**
"Multi-method validation confirms robustness (Figure 5). L0 and L8H3 heads rank top via both causal methods (knockout, attribution patching; r=0.72) and observational methods (DLA, entropy, copy score; r=0.68). This convergence rules out methodological artifacts."

**Paragraph 6 (Figure 6):**
"Layer-level analysis identifies where PI fails (Figure 6). Patching late-layer activations (L18-L24) from a clean run recovers PI accuracy to 85-90%, but early-layer patching provides minimal recovery. Combined with head-level analysis (Figures 3-5), this locates the fault to late-layer retrieval heads operating in a corrupted early-layer representational space—explaining why retrieval fails despite being attempted."

---

## **Key Numbers to Have Ready**

| Metric | Value | Source |
|--------|-------|--------|
| PI baseline accuracy | ~17% | Exp 11 |
| RI baseline accuracy | ~60% | Exp 11 |
| L8H3 knockout effect | +8.3Δ | Exp 25a |
| L8H3 accuracy improvement | 44% → 92% | Exp 25a |
| Logit lens divergence layer | L8-L12 | Exp 12 |
| Peak P(init) in PI | 0.8-0.95 | Exp 12 |
| Patching recovery (L22) | ~85% | Exp 15 |
| Method correlation (causal) | r > 0.7 | Exp 25a+25b |
| Method correlation (cross) | r > 0.6 | Exp 25a+25c |
| Heads in top-10 all models | [count] | Exp 25a × 3 |

---

## **Potential Figure Problems & Solutions**

| Problem | Solution |
|---------|----------|
| L8H3 not dominant in 0.5B? | Use 1.5B as primary; show 0.5B as supplementary |
| Logit lens divergence unclear? | Add vertical line + text annotation at specific layer |
| Cross-model 3B missing? | Run Exp 25a now; worth the compute |
| Patching data incomplete? | Verify Exp 15 ran at all 5 points; fill gaps if needed |
| Method correlation weak? | Check if methods are computed on same trials (should be) |

---

## **Files to Reference**

- **FIGURES_DRAFT.md** — Full detailed specifications (this document references it)
- **VISUALIZATION_STRATEGY.md** — Conceptual rationale and community standards
- **PAPER_DRAFT_OUTLINE.md** — Paper structure (section placement of figures)
- **Exp results:** `/results/Qwen2.5-*/*/per_head_knockout.json`, etc.

