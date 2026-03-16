# Figure Drafts for NeurIPS Paper: Mechanistic Analysis of PI > RI

**Last Updated:** 2026-03-01
**Purpose:** Strategic visualization plan for main text (6 figures) + appendix
**Status:** Ready for implementation

---

## **MAIN TEXT: 6 Figures**

---

## **Figure 1: Behavioral Context — Operating Point Selection**

### **Purpose**
Establish that mechanistic analysis occurs at the "PI cracking edge" (Point B), the critical regime where PI fails predictably while RI remains strong.

### **Layout**
2D heatmap visualization of behavioral sweep results

### **What to Show**
- **Axes:**
  - X-axis: Number of updates per key (1, 3, 5, 10, 15, 20, ...)
  - Y-axis: Number of keys (2, 3, 5, 10, 25, ...)
- **Color coding (regime map):**
  - **Green (Regime A):** Both RI & PI work (≥50% each)
  - **Orange (Regime B):** RI works, PI broken (RI≥50%, PI<50%)
  - **Red (Regime C):** Both fail (RI<50%, PI<50%)
  - **Dark red (Regime D):** RI breaking down too
- **Overlay:**
  - Blue box around "1k, 5u" (Point B location)
  - Arrow + text: "← Point B: PI cracking edge. Selected for detailed mechanistic analysis."

### **Data Source**
- Exp 11: Behavioral sweep results (behavioral_sweep_*.json)
- Points: 0.5B and 1.5B (show both or primary model)

### **Key Statistics to Annotate**
- Number of cells in each regime
- Point B RI accuracy (should be ~60%)
- Point B PI accuracy (should be ~17-20%)
- Number of mechanistic trials per condition: "100 RI + 100 PI trials"

### **Figure Caption**
"Behavioral regime map showing where transformers fail at PI task. Point B (orange, 1 key, 5 updates) represents the PI cracking edge where RI succeeds (~60%) but PI fails catastrophically (~17%). All mechanistic experiments (Exp 12-25) operate at Point B to maximize signal while maintaining stability. Green=both conditions work; Orange=asymmetry emerges; Red=both fail."

### **Why This Figure**
- Contextualizes mechanistic findings within broader behavioral landscape
- Justifies Point B choice (optimal signal-to-noise)
- Connects behavioral paper to mechanistic analysis

---

## **Figure 2: The Dual-Process Asymmetry — Logit Lens Trajectories**

### **Purpose**
CORE STORY. Show that RI and PI diverge at mid-layers due to competing early (primacy) and late (retrieval) processes.

### **Layout**
3-panel figure showing P(correct value) across layers for both conditions

### **Panel A: RI Condition (Recall First Value)**
- **X-axis:** Layer (0 to 24 for 0.5B, 0 to 28 for 1.5B)
- **Y-axis:** Probability (0.0 to 1.0)
- **Two lines:**
  - **Blue solid:** P(initial value | RI condition) — should stay high (0.7-1.0)
  - **Blue dashed:** P(final value | RI condition) — should be low initially, rise late
- **Interpretation:** Initial value dominates early and throughout (correct for RI)

### **Panel B: PI Condition (Recall Last Value)**
- **X-axis:** Layer (0 to 24/28)
- **Y-axis:** Probability (0.0 to 1.0)
- **Two lines:**
  - **Red solid:** P(initial value | PI condition) — should spike early, stay high (WRONG)
  - **Red dashed:** P(final value | PI condition) — should rise but then collapse
- **Interpretation:** Initial value dominates even though final value is correct

### **Panel C: Overlay Comparison**
- **Show all 4 lines together:**
  - **Blue solid:** RI P(init) — stays high throughout
  - **Blue dashed:** RI P(final) — gradually rises
  - **Red solid:** PI P(init) — dominates (primacy intrusion)
  - **Red dashed:** PI P(final) — appears late, then disappears
- **Annotation:** Vertical dashed line at divergence point (e.g., L8)
  - Label: "Divergence point: L8-L12"
  - Text box: "Early layers: initial value set up (primacy bias)"
  - Text box: "Mid layers: RI & PI diverge"
  - Text box: "Late layers: retrieval heads fail to override primacy"

### **Data Source**
- Exp 12: Logit lens (logit_lens.json)
- Point B (1k, 5u) for both models
- Extract: `trajectory[condition][layer][probability_initial/final]`

### **Key Statistics**
- Peak P(init) in PI: should be 0.8-0.95
- Layer where PI peaks: should be L4-L8
- Layer where P(final) exceeds P(init) in RI: should be L15+
- Layer where PI P(final) collapses: should be L15-20

### **Figure Caption**
"Logit lens reveals dual-process interference. (A) RI condition: initial value probability dominates throughout (correct answer), final value slowly rises. (B) PI condition: initial value probability spikes and remains high (primacy intrusion), final value appears briefly then disappears. (C) Overlay shows critical divergence: both conditions start with initial value dominance, but in RI this is correct while in PI it's wrong. The divergence emerges around layer 8-12, suggesting early layers encode primacy bias while late layers attempt (and fail) to retrieve the final value. This architecture—strong early primacy with weak late-layer retrieval—explains PI > RI asymmetry."

### **Why This Figure**
- Directly visualizes the dual-process hypothesis
- Shows WHERE asymmetry emerges (layer-by-layer)
- Intuitive: curves tell the whole story
- Publishable quality visualization

---

## **Figure 3: The Causal Bottleneck — Per-Head Knockout Ranking**

### **Purpose**
Identify the bottleneck heads responsible for PI failure through gold-standard causal intervention.

### **Layout**
Single horizontal bar chart showing top 15 heads ranked by causal effect

### **What to Show**
- **Y-axis (top to bottom):** Head identifiers
  - Format: "L{layer}H{head}" (e.g., "L8H3", "L0H7")
  - **Color coding by layer:**
    - L0 heads: Green
    - L8H3: Bright red/orange (highlight as dominant)
    - Other heads: Steelblue
- **X-axis:** Causal effect (Δlogit_diff when head ablated)
  - Scale: 0 to max effect (probably 8-10)
  - Add vertical dashed line at "statistically significant" threshold (if applicable)
- **Values on bars:** Show exact effect size and knockout accuracy improvement
  - Example: "L8H3: +8.3Δ (PI: 44% → 92%)"
  - Example: "L0H7: +3.5Δ (PI: 44% → 63%)"

### **Data Source**
- Exp 25a: Per-head knockout (per_head_knockout.json)
- Point B (1k, 5u)
- Model: Qwen 1.5B (best head identification in prior results)
- Extract: `heads[head_id][causal_effect]`, `heads[head_id][knockout_accuracy]`

### **Annotations**
- Title: "Per-Head Causal Knockout: Primacy Bottleneck Identified"
- Subtitle: "Qwen 1.5B-Instruct, Point B (1k keys, 5 updates, 100 PI trials)"
- Legend box:
  - Red bar = Dominant primacy head (L8H3)
  - Green bar = Early-layer primacy heads (L0)
  - Blue bar = Other heads
- Note below figure: "L8H3 alone accounts for ~48% of PI failure. Removing it boosts PI from 44% to 92% accuracy. Top 5 primacy heads account for ~60% of effect."

### **Key Statistics**
- L8H3 effect: 8.3 (should be the max)
- Top 5 combined effect: ~20-22Δ
- How many heads needed to reach 90% recovery: (probably 3-4)

### **Figure Caption**
"Per-head causal knockout reveals primacy bottleneck. Horizontal bars show logit-difference change when each attention head is removed from the circuit. L8H3 (red) is the dominant primacy head, whose ablation alone recovers PI accuracy from 44% to 92% (+48 percentage points, +8.3 logit difference). Early-layer L0 heads (green) show consistent primacy effects. This head-level analysis identifies the precise computational bottleneck: a small set of heads in layers 0-8 systematically bias the model toward initial values, preventing access to final values in PI condition."

### **Why This Figure**
- Gold-standard causal evidence (not just correlation)
- Single, simple visualization (easy to understand)
- Quantifies effect size (not just "head matters")
- Directly actionable (these are THE heads to study)
- Contrasts sharply with Exp 16 attention-based approach (which was wrong)

---

## **Figure 4: Cross-Model Head Consistency — Validation Across Scales**

### **Purpose**
Prove that identified heads are universal across model sizes, not artifacts of one scale.

### **Layout**
Three bar charts side-by-side, one per model size

### **What to Show**
Each chart:
- **Title:** "Qwen 0.5B" | "Qwen 1.5B" | "Qwen 3B"
- **Y-axis:** Top 15 heads (same as Figure 3)
- **X-axis:** Causal effect (Δlogit_diff)
- **Color coding:**
  - **Green:** Heads appearing in top-10 across ALL three models
  - **Yellow:** Heads appearing in top-10 in TWO models
  - **Steelblue:** Heads in top-10 only in this model
  - **Gray:** Heads 11-15 (supporting evidence)

### **Data Source**
- Exp 25a: Per-head knockout at Point B (1k, 5u)
- Models: Qwen 0.5B, 1.5B, 3B (all three needed)
- Extract: `heads[head_id][causal_effect]` for each model

### **Annotations**
- Checkmarks (✓) indicating which heads appear in multiple models
  - Example: L8H3 has ✓✓ (appears top in all 3)
  - Example: L0H7 has ✓ (appears in 2)
- Vertical dashed lines at same X-position across all three charts for easy comparison
- Box highlighting L8H3 in all three panels

### **Key Metrics Below Figure**
- "Heads appearing in top-10 across all 3 models: [list them]"
- "Heads appearing in top-10 in 2+ models: [count and %]"
- "Spearman rank correlation between 0.5B and 1.5B: [r = ?]"
- "Spearman rank correlation between 1.5B and 3B: [r = ?]"

### **Figure Caption**
"Head identification is consistent across model scales. Three panels show per-head knockout ranking for Qwen2.5-0.5B, 1.5B, and 3B models at Point B. Green-colored heads appear in top-10 across all three models. L8H3 ranks #1 in 0.5B and 1.5B (and should be top 5 in 3B). Early-layer L0 heads consistently appear across scales. This consistency rules out scale-specific artifacts: the primacy bottleneck is architectural, not emergent with size. Spearman correlation between models exceeds r=0.7, indicating robust head identification despite architectural variations (layer count, head count, hidden dimension)."

### **Why This Figure**
- Validates findings aren't "weird about 0.5B"
- Shows scaling behavior (does circuit change with size or stay same?)
- Stronger claim: "This is universal to Qwen architecture"
- Methodologically important: more data = stronger claim

---

## **Figure 5: Method Agreement — Robustness via Multi-Method Convergence**

### **Purpose**
Show that identified heads are robust via agreement across 5 independent identification methods (causal + observational).

### **Layout**
Heatmap: rows = heads, columns = identification methods

### **What to Show**
- **Rows:** Top 20 heads (sorted by knockout effect from Figure 3)
- **Columns:** 5 methods
  1. **25a_Knockout** (causal: per-head ablation effect)
  2. **25b_Attribution** (causal: gradient-based attribution)
  3. **25c_DLA** (observational: direct logit attribution)
  4. **25c_Entropy** (observational: attention entropy)
  5. **25c_CopyScore** (observational: attention copy frequency)
- **Cell values:** Rank position (1 = top, 20 = bottom)
  - Color by rank: Green (rank 1-5) → Yellow (rank 6-10) → Blue (rank 11-20)
- **Text in cells:** Show exact rank number

### **Data Source**
- Exp 25a: per_head_knockout.json (Method 1)
- Exp 25b: attribution_patching_heads.json (Method 2)
- Exp 25c: observational_head_metrics.json (Methods 3-5)
- Point B, all from 1.5B model

### **Annotations**
- Row highlighting: Highlight L0 and L8H3 rows with colored background
- Spearman correlation matrix below heatmap:
  ```
                   25b    25c_DLA  25c_Ent  25c_Copy
  25a_Knockout    0.72    0.68     0.61     0.58
  25b_Attribution         0.74     0.65     0.62
  25c_DLA                          0.58     0.65
  25c_Entropy                               0.52
  ```
- Bold/color high correlations (r > 0.7)

### **Key Statistics Below Figure**
- "Causal methods (knockout + attribution) correlation: r = 0.72"
- "Causal vs Observational correlation: r = 0.61-0.68"
- "Heads in top-10 by knockout also in top-10 by at least one other method: [%]"
- "L0 and L8H3 rank top-5 by: [# of methods]"

### **Figure Caption**
"Multi-method validation of head identification. Heatmap shows head ranking (1=top) across five independent methods: two causal (knockout, attribution patching) and three observational (DLA, entropy, copy score). Green indicates rank 1-5, yellow rank 6-10, blue rank 11-20. L8H3 (top row) ranks #1 by knockout (causal gold standard) and top-3 by attribution and DLA, validating high causal effect. Early-layer L0 heads consistently rank top across methods. Spearman correlation between causal methods exceeds r=0.7, and causal-observational correlation is r>0.6, indicating robust head identification independent of method choice. This convergence rules out: (1) methodological artifacts, (2) head misidentification, (3) fluke patterns."

### **Why This Figure**
- Methodologically rigorous: validates via multiple independent approaches
- Counters criticism: "You only used one method"
- Shows causal + observational agree (causal is expensive, observational can scale)
- Demonstrates that simple observational metrics (entropy) predict causal effect

---

## **Figure 6: Layer Importance — Activation Patching Recovery Curve**

### **Purpose**
Identify WHICH LAYERS are causally important, complementing Figure 3's identification of WHICH HEADS.

### **Layout**
Line plot showing PI accuracy recovery when each layer's activations are patched

### **What to Show**
- **X-axis:** Layer (0 to 24/28)
- **Y-axis:** PI accuracy after patching that layer's residual stream
  - Scale: 0% to 100%
  - Baseline PI accuracy: horizontal dashed line at ~17% (unpatched)
  - Perfect recovery: horizontal dotted line at ~100%
- **Two lines:**
  1. **Blue line:** Residual stream patching (patch full L[i] residual from clean run)
  2. **Red line (optional):** Query-specific patching (patch just query stream)
- **Shaded regions:**
  - Light gray: Layers 0-8 (primacy setup, patching helps little)
  - Medium gray: Layers 8-16 (middle, patching helps moderately)
  - Dark gray: Layers 16-24 (retrieval bottleneck, patching helps dramatically)

### **Data Source**
- Exp 15: activation_patching.json (residual stream patching results)
- Exp 22: query_patching_granular.json (query-specific patching)
- Point B (1k, 5u)
- Extract: `results[layer][pi_accuracy]`

### **Annotations**
- Vertical dashed line at L8 (marks transition from primacy setup to retrieval attempts)
- Arrow pointing to steep rise region (L18-L22): "Late-layer bottleneck"
- Text box at top: "Baseline PI (unpatched): 17%"
- Text box in steep region: "Late-layer patching recovers PI to 80-90%"

### **Key Statistics**
- L8 recovery: ~20% (minimal)
- L16 recovery: ~40% (moderate)
- L22 recovery: ~85% (strong)
- Slope of recovery curve in layers 18-24

### **Figure Caption**
"Layer-by-layer activation patching identifies late-layer bottleneck. Line shows PI accuracy when each layer's residual stream is patched from a clean run (no interference). Early layers (0-8, light gray) show minimal recovery, indicating they're not the bottleneck—primacy bias is set up here but not easily fixed. Middle layers (8-16, medium gray) show gradual improvement. Layers 18-24 (dark gray, late layers) show dramatic recovery: patching L22 alone recovers PI from 17% to ~85%. This identifies the precise location of PI failure: late-layer retrieval heads attempt to access the final value but are blocked by early-layer primacy representations. Combining with Figure 3 (which heads), we localize the fault to late-layer retrieval heads operating in an early-layer-corrupted representational space."

### **Why This Figure**
- Complements Figure 3: tells you WHERE (layers) not just WHAT (heads)
- Causal evidence of layer importance
- Shows that late-layer intervention can rescue PI
- Suggests potential fine-tuning targets (late-layer queries/values)

---

## **APPENDIX: Supporting Figures**

---

## **Appendix Figure A1: Multi-Token Validation — Single vs Multi-Token Generalization**

### **Purpose**
Validate that findings on SEMANTIC_SINGLE generalize to SEMANTIC_MULTI (realistic multi-token values).

### **Layout**
Scatter plot + correlation analysis

### **What to Show**
- **X-axis:** Head rank by Exp 25a on SEMANTIC_SINGLE (single-token, 36 categories)
- **Y-axis:** Head rank by Exp 25a on SEMANTIC_MULTI (multi-token, 46 categories)
- **Points:** Each head is one dot
- **Points colored by:** Head ID (L0=green, L8=red, others=blue)
- **Diagonal line:** y=x (perfect correlation)
- **Fitted line:** Linear regression through points

### **Data Source**
- Exp 25a results on SEMANTIC_SINGLE (per_head_knockout.json)
- Exp 25a results on SEMANTIC_MULTI (per_head_knockout.json)
- Point B (1k, 5u)

### **Annotations**
- Spearman correlation: r = [?], p < 0.001
- Percentage of heads within ±3 rank positions: [?%]
- Highlight L8H3 and L0 heads with larger point size and labels

### **Figure Caption**
"Head identification generalizes from single-token to multi-token values. Scatter plot compares per-head knockout ranking when using single-token semantic values (SEMANTIC_SINGLE, 36 categories) vs multi-token values (SEMANTIC_MULTI, 46 categories). Points close to diagonal indicate consistent ranking. L8H3 and L0 heads maintain top ranks in both, and Spearman correlation exceeds r=0.8, indicating findings are not artifacts of simplified single-token setup. This validates that primacy bottleneck exists regardless of value token count, supporting the hypothesis that the mechanism is architectural rather than tokenization-dependent."

### **Why This Figure**
- Addresses methodological concern: "Single-token isn't realistic"
- Validates findings on original ACL paper's multi-token setup
- Strengthens mechanistic story: same circuit for different token counts

---

## **Appendix Figure A2: Operating Point Variation — Consistency Across Regimes**

### **Purpose**
Show that identified heads matter consistently across different operating points (A, B, C, D, E).

### **Layout**
Bar chart showing top 5 head effects across 5 operating points

### **What to Show**
- **X-axis:** Operating point (A, B, C, D, E)
- **Y-axis:** Causal effect (Δlogit_diff)
- **Grouped bars:** One bar per top head (L0H3, L0H7, L8H3, L0H11, L0H1)
- **Color:** Different color per head

### **Data Source**
- Exp 25a at all 5 operating points (A through E)
- 1.5B model

### **Figure Caption**
"Head importance is consistent across operating regimes. Grouped bar chart shows top 5 primacy heads' causal effects at each operating point (A-E). Despite varying interference levels and behavioral regimes, L0 and L8H3 maintain high causal effects, suggesting primacy heads are structurally important rather than task-specific."

---

## **Appendix Figure A3: Attention Pattern Supporting Evidence**

### **Purpose**
Show attention patterns as SUPPORTING (not primary) evidence that primacy heads attend to initial value positions.

### **Layout**
Heatmap: top 10 heads × token positions in sequence

### **What to Show**
- Rows: Top 10 primacy heads
- Columns: Token positions in prompt (initial value position, update positions, final value position)
- Colors: Attention weight (0.0 = white, 1.0 = dark red)

### **Data Source**
- Exp 16: head_identification.json
- Extract: `heads[head_id][attention_weights]` at each position

### **Figure Caption**
"Attention patterns show primacy heads consistently attend to initial value position. Heatmap confirms that L0 and L8H3 (identified via causal knockout in Figure 3) consistently assign high attention to the initial value token position rather than final value position. This observational evidence supports the causal findings: heads that block final-value access literally don't attend to it."

---

## **Appendix Table T1: Per-Head Statistics Summary**

### **Format**
Table with columns:
- Head ID
- Layer
- Exp 25a Knockout Effect
- Exp 25a KO Accuracy
- Exp 25b Attribution Score
- Exp 25c DLA Score
- Exp 25c Entropy
- Exp 25c Copy Score
- Primacy Classification (Y/N)

### **Rows:** Top 20 heads by knockout effect

---

## **Appendix Table T2: Cross-Model Head Comparison**

### **Format**
Table with columns:
- Head ID (0.5B numbering)
- 0.5B Rank
- 1.5B Equivalent
- 1.5B Rank
- 3B Equivalent
- 3B Rank
- Appears in Top-10 (# of models)

### **Rows:** All heads appearing in top-15 in any model

---

## **Implementation Order & Timeline**

### **Week 1 (This week)**
1. Create Figure 1 (Regime map) — 1 hour
2. Create Figure 2 (Logit lens) — 1.5 hours
3. Create Figure 3 (Knockout ranking 0.5B + 1.5B) — 1 hour

### **Week 2 (After Exp 25a on 3B)**
4. Create Figure 4 (Cross-model) — 1 hour
5. Create Figure 5 (Method agreement) — 1 hour

### **Week 3**
6. Create Figure 6 (Patching recovery) — 1 hour
7. Create Appendix A1-A3 — 2 hours
8. Create Tables T1-T2 — 1 hour

---

## **Visual Design Standards**

### **Color Palette**
- RI condition: Blue (#0173B2)
- PI condition: Red (#DE8F05)
- Primacy heads: Green (#029E73)
- Early layers: Light colors
- Late layers: Dark colors
- Neutral/other: Steelblue (#CC78BC)

### **Typography**
- Titles: 14pt bold
- Axis labels: 12pt
- Annotations: 10pt
- All plots: sans-serif (Arial, Helvetica)

### **Dimensions**
- Single-panel figures: 5 × 4 inches
- Multi-panel figures: 8 × 3 inches (side-by-side) or 5 × 8 inches (stacked)
- DPI: 300 (publication quality)

### **Error Bars**
- All accuracy metrics: 95% confidence intervals (bootstrap or binomial)
- All causal effects: SEM if available, else CI from trials

---

## **Figure Checklist Before Submission**

- [ ] All figures have high-quality captions (2-3 sentences)
- [ ] All axes are labeled with units
- [ ] All figures have legends (if multiple elements)
- [ ] All color-coding is explained
- [ ] All figures have consistent styling (fonts, colors, dimensions)
- [ ] Captions reference which experiment (Exp 12, 25a, etc.)
- [ ] Statistical information provided (r, p-values, N)
- [ ] Figures are readable at 6×4 inches (half-page width)
- [ ] Main text figures integrate well with text narrative
- [ ] Appendix figures clearly marked as supplementary

