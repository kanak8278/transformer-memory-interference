# Multi-Token Validation Plan: Single-Token → Multi-Token Generalization

**Purpose:** Validate that mechanistic findings (heads, layers, causal effects) from SEMANTIC_SINGLE (single-token values) generalize to SEMANTIC_MULTI (multi-token realistic values from ACL paper).

**Status:** Planned
**Priority:** High (appendix validation, strengthens paper narrative)
**Expected Runtime:** 3-4 weeks (parallelizable across GPUs)

---

## **Motivation**

### **Why This Matters**
Our mechanistic analysis (Exp 12-25) uses **SEMANTIC_SINGLE**: 36 categories with verified single-token values (ruby, jade, pearl). This is analytically clean but simplified.

The **original ACL paper** uses **SEMANTIC_MULTI**: 46 categories with realistic multi-token values (alexandrite, tourmaline, amethyst).

**Risk:** Are our mechanistic findings artifacts of single-token simplification?
- Do primacy heads still dominate for multi-token values?
- Do the same heads rank top?
- Is the causal hierarchy different?

**Upside:** If findings generalize → strong validation that the mechanism is fundamental, not tokenization-dependent.

---

## **Dataset Differences**

| Property | SEMANTIC_SINGLE | SEMANTIC_MULTI |
|----------|-----------------|-----------------|
| **Categories** | 36 | 46 |
| **Values per category** | 4-89 (mean 22) | 45-70 (mean 52) |
| **Token count per value** | 1 | 2-4 (mean 2.3) |
| **Total values** | 791 | 2403 |
| **Source** | Verified single-token | ACL paper dataset |
| **Semantic quality** | Real (ruby IS a gem) | Real (alexandrite IS a gem) |
| **Use case** | Mechanistic analysis (clean) | Original paper validation |

### **Key Difference**
- Single-token: Answer is ONE token → clean logit lens (track single token ID)
- Multi-token: Answer is 2-4 tokens → messy logit lens (track which token? combine how?)

---

## **Experiment Design**

### **Core Validation: Exp 26 — Per-Head Knockout on SEMANTIC_MULTI**

**What:** Replicate Exp 25a (per-head knockout) but using SEMANTIC_MULTI values

**Configuration:**
- **Dataset:** SEMANTIC_MULTI
- **Model:** Qwen 1.5B (primary), also 0.5B and 3B (if compute allows)
- **Operating points:** Point B only (1k keys, 3u updates for 1.5B, matched difficulty)
- **Trials:** 100 RI + 100 PI (same as Exp 25a for direct comparison)
- **Metric:** Causal effect (Δlogit_diff when head ablated) on PI task

**Procedure:**
1. Generate 100 PI trials using SEMANTIC_MULTI dataset
2. Get baseline PI accuracy on those trials
3. For each head (L0H0 through L27H11):
   - Zero out head output (hook on `attn.hook_z`)
   - Re-run 100 trials
   - Measure accuracy + logit_diff
   - Compute: `causal_effect[head] = mean(ld_knockout - ld_baseline)`
4. Rank heads by causal effect

**Expected Output:**
```json
{
  "experiment": "26_per_head_knockout_multitoken",
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "dataset_type": "SEMANTIC_MULTI",
  "num_keys": 1000,
  "num_updates": 3,
  "num_trials": 100,
  "condition": "PI",
  "baseline_pi_accuracy": 0.???,
  "heads": {
    "L0H0": {
      "layer": 0, "head": 0,
      "baseline_logit_diff": ???,
      "knockout_logit_diff": ???,
      "causal_effect": ???,
      "knockout_accuracy": ???,
      "rank": ???
    },
    ...
  },
  "top_15_heads": [...]
}
```

**Success Criteria:**
- L8H3 should still rank #1 (or top-3)
- L0 heads should still be in top-10
- Spearman correlation between SEMANTIC_SINGLE and SEMANTIC_MULTI rankings: r > 0.7

---

## **Extended Validation Suite (Optional but Valuable)**

### **Exp 26a — Multi-Token Logit Lens**

**Purpose:** Track how PI signal appears/disappears for multi-token values

**Challenge:** Multi-token values have multiple tokens. Strategy:
- **Token 1:** First token of value (e.g., "alex" from "alexandrite")
- **Token 2:** Second token (e.g., "andrite")
- **Combined:** Average probability across all tokens in value

**Output:**
```json
{
  "experiment": "26a_logit_lens_multitoken",
  "dataset_type": "SEMANTIC_MULTI",
  "layers": 28,
  "trajectories": {
    "PI": {
      "p_initial_token1": [0.8, 0.75, 0.70, ...],  // First token of initial value
      "p_initial_token2": [0.6, 0.55, 0.50, ...],  // Second token of initial value
      "p_initial_combined": [0.7, 0.65, 0.60, ...], // Mean
      "p_final_token1": [...],
      "p_final_token2": [...],
      "p_final_combined": [...]
    },
    "RI": { ... }
  },
  "divergence_point": "L8-L12"
}
```

**Success Criteria:**
- Divergence point should match SEMANTIC_SINGLE (L8-12)
- P(initial) should dominate similarly

---

### **Exp 26b — Attention Patterns on Multi-Token**

**Purpose:** Do primacy heads still attend to initial-value positions (even if multi-token)?

**Design:**
- Measure attention to first-token position of initial value
- Measure attention to first-token position of final value
- Do primacy heads prefer initial positions?

---

### **Exp 26c — Method Agreement on Multi-Token**

**Purpose:** Do observational metrics still predict causal effects?

**Design:**
- Run Exp 25c (observational metrics) on SEMANTIC_MULTI
- Measure agreement with Exp 26 (causal knockout)
- Compare to Exp 25c ↔ 25a agreement on SEMANTIC_SINGLE

---

## **Experimental Schedule**

### **Phase 1: Core Validation (Required)**
- **Exp 26 (Knockout on SEMANTIC_MULTI)** — Qwen 1.5B
  - ~1 hour compute (336 heads × 100 trials, batch processing)
  - ~2 weeks wall-clock (can parallelize by layer groups)
  - Output: Per-head ranking to compare with Fig 3

### **Phase 2: Extended Validation (Recommended)**
- **Exp 26a (Logit Lens on SEMANTIC_MULTI)** — Qwen 1.5B
  - ~30 min compute
  - Validates Fig 2 logit lens trajectory
- **Exp 26b (Attention on SEMANTIC_MULTI)** — Qwen 1.5B
  - ~20 min compute
  - Supporting evidence for primacy heads

### **Phase 3: Cross-Model Validation (Optional)**
- **Exp 26 on Qwen 0.5B, 3B**
  - Validates cross-model story (Fig 4) also holds for multi-token
  - Lower priority if compute-limited

---

## **Data & Implementation**

### **Script Location**
`mechanistic_probing_v2/experiments/26_per_head_knockout_multitoken.py`

### **Dataset Loading**
```python
from mechanistic_probing_v2.core.dataset_configs import (
    load_dataset_config,
    get_eligible_categories,
    generate_trial,
)

config = load_dataset_config("SEMANTIC_MULTI")
# Returns: 46 categories, 2403 total values
# Each category has 45-70 values

categories = get_eligible_categories("SEMANTIC_MULTI", min_values=3)
# For Point B (1k, 3u), need at least 3 updates per category
# SEMANTIC_MULTI has enough (all cats have 45+)
```

### **Trial Generation**
```python
rng = random.Random(seed)
trial = generate_trial(
    dataset_type="SEMANTIC_MULTI",
    num_keys=1000,
    num_updates=3,
    condition="PI",
    seed=seed,
    use_chat_format=True,
    tokenizer=tokenizer,
)
# trial["prompt"]: full interference prompt
# trial["expected"]: first token of final value (or should we handle multi-token?)
```

### **Multi-Token Answer Handling**
**Decision Point:** How to evaluate multi-token answers?

**Option A (Conservative): First Token Only**
- Expected answer: `final_value_tokens[0]` (e.g., "alexandrite" → expect "alex")
- Measure: Does model generate first token correctly?
- Pro: Clean, comparable to single-token
- Con: Doesn't capture full answer

**Option B (Realistic): Sequence Match**
- Expected answer: full value tokens (e.g., ["alex", "andrite"])
- Measure: Do first N generated tokens match value?
- Pro: Realistic
- Con: More complex, harder to fail (can get first token right by chance)

**Recommendation: Option A (First Token Only)**
- Maintains comparability with SEMANTIC_SINGLE
- Still measures causal effect (if first token is primacy-blocked, full answer fails)
- Cleaner mechanistic interpretation

---

## **Comparison & Analysis**

### **Analysis 1: Head Ranking Correlation**

```python
# Load both results
single = json.load(open("...SEMANTIC_SINGLE/per_head_knockout.json"))
multi = json.load(open("...SEMANTIC_MULTI/per_head_knockout.json"))

# Extract top-20 heads + rankings
single_heads = sorted(single["heads"].items(), key=lambda x: x[1]["causal_effect"], reverse=True)
multi_heads = sorted(multi["heads"].items(), key=lambda x: x[1]["causal_effect"], reverse=True)

# Compute Spearman correlation
from scipy.stats import spearmanr
single_ranks = {h: i for i, (h, _) in enumerate(single_heads[:30])}
multi_ranks = {h: i for i, (h, _) in enumerate(multi_heads[:30])}

common_heads = set(single_ranks.keys()) & set(multi_ranks.keys())
r, p = spearmanr([single_ranks[h] for h in common_heads],
                   [multi_ranks[h] for h in common_heads])
print(f"Spearman r: {r:.3f}, p-value: {p:.2e}")

# Success: r > 0.7
```

### **Analysis 2: Top-K Overlap**

```python
# How many heads in top-10 overlap?
single_top10 = set(h[0] for h in single_heads[:10])
multi_top10 = set(h[0] for h in multi_heads[:10])
overlap = len(single_top10 & multi_top10) / 10
print(f"Top-10 overlap: {overlap:.0%}")

# Success: > 70% overlap
```

### **Analysis 3: Causal Effect Magnitude**

```python
# Is the dominant head still dominant?
single_top = single_heads[0]
multi_top = multi_heads[0]

print(f"Single-token top head: {single_top[0]} ({single_top[1]['causal_effect']:.2f}Δ)")
print(f"Multi-token top head: {multi_top[0]} ({multi_top[1]['causal_effect']:.2f}Δ)")

# Success: Same head or consistent effect magnitude
```

---

## **Expected Results**

### **Best Case (Findings Validate)**
- L8H3 ranks #1 in multi-token too
- L0 heads in top-10 in both
- Spearman r > 0.8 between rankings
- → **Claim:** "Primacy bottleneck is universal, independent of token count"
- → **Paper:** Add Appendix Figure A1 (scatter: single-token rank vs multi-token rank)

### **Good Case (Mostly Consistent)**
- L8H3 in top-5 in multi-token
- L0 heads somewhat consistent
- Spearman r = 0.65-0.75
- → **Claim:** "Primacy bottleneck persists but mechanism adapts for multi-token"
- → **Paper:** Add Appendix note explaining differences

### **Concerning Case (Big Differences)**
- Different heads rank top
- Spearman r < 0.6
- → **Question:** Why does mechanism change?
- → **Action:** Debug: is multi-token handling wrong? Is dataset different?

---

## **Integration with Paper**

### **Main Text**
No change. Single-token findings remain primary story (cleaner, more direct).

### **Appendix Figure A1 (New)**
"Multi-token validation: Head rankings on SEMANTIC_MULTI vs SEMANTIC_SINGLE"
- Scatter plot: single-token rank (X) vs multi-token rank (Y)
- Spearman r reported
- Points on diagonal = consistent
- Points off diagonal = dataset-dependent

### **Appendix Section (New)**
"We validated findings on multi-token values from the original ACL paper's dataset (SEMANTIC_MULTI). Per-head knockout shows [L8H3, L0 heads] rank consistently, with Spearman r = 0.78, indicating the primacy bottleneck is not an artifact of single-token simplification."

---

## **File Outputs**

All results saved to:
```
results/Qwen2.5-{model}/{num_keys}k_{num_updates}u/
├── per_head_knockout_multitoken.json          (Exp 26)
├── logit_lens_multitoken.json                 (Exp 26a, if run)
├── attention_patterns_multitoken.json         (Exp 26b, if run)
├── observational_head_metrics_multitoken.json (Exp 26c, if run)
└── multitoken_validation_summary.txt          (comparison table)
```

---

## **Success Metrics**

- [x] Exp 26 complete (knockout on SEMANTIC_MULTI)
- [x] Head ranking Spearman r > 0.7 vs SEMANTIC_SINGLE
- [x] L8H3 in top-5 (ideally top-1)
- [x] Appendix Figure A1 created
- [x] Paper appendix section written

---

## **Dependencies & Resources**

- **Model:** Qwen2.5-1.5B-Instruct (loaded once, reused)
- **Compute:** ~1 hour per model (336 heads × 100 trials)
- **Memory:** ~24 GB VRAM (similar to Exp 25a)
- **Data:** SEMANTIC_MULTI loaded via dataset_configs.py
- **Code:** New script `26_per_head_knockout_multitoken.py` (~300 lines, copy-paste from Exp 25a)

---

## **Timeline**

- **Week 1:** Write & test Exp 26 script
- **Week 2:** Run Exp 26 (compute time)
- **Week 3:** Analyze results + create Figure A1
- **Week 4:** Optional — run Exp 26a/26b if needed

**Critical Path:** This doesn't block paper writing (main text uses single-token). Run in parallel with Exp 25a on 3B.

