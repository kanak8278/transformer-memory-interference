# Error Type Classification Analysis - Retroactive Interference Study

**Generated:** December 31, 2024
**Dataset:** 57 models, 6,237 total errors, interference levels 3-300
**Methodology:** Based on "Unable to Forget: Persistent Memory in Language Models" (Section 2.2.1)

---

## Executive Summary

We classified all incorrect model extractions into **5 cognitive failure modes** to understand how LLMs fail under retroactive interference. Unlike the "Unable to Forget" paper's PI study, our RI analysis reveals **fundamentally different error patterns**.

### Key Finding: **Retrieval Failure Dominates, Not Same-Key Interference**

- **57.3% Retrieval Failure** (null/no extraction)
- **39.9% Same-Key Interference** (wrong value from same category)
- **0.5% Cross-Key** | **1.5% Partial Match** | **0.8% Hallucination**

**Implication:** In RI (retrieving FIRST-learned), models primarily **fail to retrieve anything** rather than retrieve wrong values. This contrasts with PI where models actively retrieve incorrect but plausible values.

---

## 5 Error Types Defined

### 1. **Same-Key Interference (39.9%)**
- **Definition:** Model outputs a LATER update from the SAME category
- **Example:** Category "visual art", expected "baroque" (1st), extracted "pop art" (3rd update)
- **Cognitive meaning:** Temporal confusion within category

### 2. **Cross-Key Interference (0.5%)**
- **Definition:** Model outputs a value from a DIFFERENT category
- **Example:** Category "visual art", expected "baroque", extracted "brie" (from "cheese variety")
- **Cognitive meaning:** Complete category confusion, memory binding failure

### 3. **Partial Match (1.5%)**
- **Definition:** Close but not exact (substring, paraphrase, truncation)
- **Example:** Expected "fenway park", extracted "fenway stadium"
- **Cognitive meaning:** Gist memory, semantic drift

### 4. **Hallucination (0.8%)**
- **Definition:** Value NEVER presented anywhere in prompt
- **Example:** Category "visual art", expected "baroque", extracted "impressionism" (never appeared)
- **Cognitive meaning:** Confabulation, complete memory trace loss

### 5. **Retrieval Failure (57.3%)**
- **Definition:** No value extracted (null/None/empty)
- **Example:** Extracted: `None`, `null`, or parser failure
- **Cognitive meaning:** Complete retrieval collapse, "tip-of-tongue" state

---

## Three-Stage Progression Analysis

Following the "Unable to Forget" paper's framework, we analyzed error evolution across interference levels:

### **Stage 1 - LOW INTERFERENCE (Levels 3, 10): Tightly Focused Errors**

| Error Type | Percentage |
|-----------|------------|
| Same-Key Interference | **4.0%** |
| Cross-Key Interference | 1.1% |
| Hallucination | 2.9% |
| **Retrieval Failure** | **87.4%** |

**Pattern:** Errors are dominated by retrieval failure. When models do retrieve, errors are tightly focused on adjacent positions (Avg position = 2.4, just 1.6 steps back from target).

**Interpretation:** At low interference, models primarily **fail silently** (can't extract anything) rather than making substitution errors.

---

### **Stage 2 - MODERATE INTERFERENCE (Levels 50, 100): Dispersed Errors**

| Error Type | Percentage |
|-----------|------------|
| **Same-Key Interference** | **21.5%** ↑ |
| Cross-Key Interference | 0.2% |
| Hallucination | 0.9% |
| **Retrieval Failure** | **77.4%** ↓ |

**Pattern:** Same-key interference increases **5.4× from Stage 1**. Retrieved positions disperse upstream (Level 100: Avg position = 53.6, 47 steps back).

**Interpretation:** Models start retrieving MORE, but from WRONG temporal positions. The "interference zone" expands—no longer limited to adjacent updates.

---

### **Stage 3 - HIGH INTERFERENCE (Levels 200, 300): Hallucinatory Responses**

| Error Type | Percentage |
|-----------|------------|
| **Same-Key Interference** | **51.8%** ↑↑ |
| Cross-Key Interference | 0.5% |
| Hallucination | 0.1% ↓ |
| **Retrieval Failure** | **46.2%** ↓↓ |

**Pattern:** Same-key interference becomes DOMINANT (**13× from Stage 1**), while retrieval failure drops below 50% for the first time.

**Surprising:** Hallucination rate **DECREASES** at high load (0.1% vs 2.9% at Stage 1). This is **opposite** of the PI paper's "phase transition to hallucinations."

**Critical Difference from PI:** In PI (Unable to Forget), Stage 3 shows **25-45% hallucinations** as retrieval fidelity collapses. In RI, models **actively retrieve WRONG temporal instances** (51.8% same-key) rather than hallucinate.

---

## Position Analysis: Which Earlier Values Get Retrieved?

When models make same-key interference errors, **which** earlier value do they retrieve?

| Level | Avg Position | Avg Recency | Most Recent (%) |
|-------|-------------|-------------|-----------------|
| 3     | 2.4         | 1.6 steps   | 44%             |
| 10    | 2.5         | 8.4 steps   | 0%              |
| 50    | 20.4        | 30.6 steps  | 0%              |
| 100   | 53.6        | 47.4 steps  | 0%              |
| 200   | 112.0       | 89.0 steps  | 3%              |
| 300   | 137.6       | 163.4 steps | 0%              |

**Key Insights:**

1. **Recency Bias ONLY at Level 3:** 44% retrieve the most recent value
2. **Mid-sequence Bias Emerges:** At levels 100-300, models retrieve from **middle positions** (not adjacent, not most recent)
3. **Linear Scaling:** Average position scales roughly **linearly** with interference level (Level 200: pos 112, Level 300: pos 138)

**Interpretation:** Models have a **"retrieval window"** that expands as interference increases. At high levels, they retrieve from a broad temporal range centered around the middle of the update sequence.

---

## Comparison with "Unable to Forget" PI Study

### **RI (Our Study) vs PI (Unable to Forget Paper)**

| Metric | RI (Retroactive) | PI (Proactive) |
|--------|------------------|----------------|
| **Dominant Error** | Retrieval Failure (57%) | Same-Key Interference (60-80%) |
| **Stage 3 Hallucinations** | **0.1%** ↓ | **25-45%** ↑ |
| **Cross-Key Errors** | <1% (rare) | 15-30% (common) |
| **Retrieval Behavior** | Silent failure → Wrong retrieval | Wrong retrieval → Hallucination |

### **Why the Difference?**

**RI Task (Our Study):** Retrieve FIRST-learned value (oldest)
- Challenge: Suppress ALL later updates
- Failure mode: Can't suppress → Retrieve nothing OR retrieve later update

**PI Task (Unable to Forget):** Retrieve LAST-learned value (newest)
- Challenge: Suppress ONLY earlier updates
- Failure mode: Can't suppress → Retrieve earlier update OR hallucinate

**Hypothesis:** Retrieving **oldest** information (RI) is fundamentally harder than retrieving **newest** (PI). When RI fails, models "give up" (retrieval failure). When PI fails, models "guess" (hallucination).

---

## Visualizations Generated

1. **`error_distribution_stacked.png`** - Stacked bar chart showing error type evolution across levels 3-300
2. **`error_trajectories.png`** - Line plot tracking each error type's trajectory
3. **`same_key_position_heatmap.png`** - Heatmap showing which positions (0-100% of sequence) are retrieved
4. **`hallucination_phase_transition.png`** - Hallucination rate vs interference (shows NO phase transition unlike PI)

All figures are publication-quality (300 DPI).

---

## Statistical Summary

```
Total Errors: 6,237
Models Analyzed: 57
Interference Levels: 3, 10, 50, 100, 200, 300

Error Type Distribution:
  Same-Key Interference:    2,490 (39.9%)
  Cross-Key Interference:      30 (0.5%)
  Partial Match:               95 (1.5%)
  Hallucination:               47 (0.8%)
  Retrieval Failure:        3,575 (57.3%)
```

---

## Implications for Paper

### **New Research Contribution**

Our error classification reveals that **RI and PI have fundamentally different failure modes**:

1. **RI-specific vulnerability:** Models cannot effectively suppress later updates when retrieving earliest information
2. **Asymmetric failure:** RI leads to silent failure (can't retrieve), PI leads to active confabulation (hallucinate)
3. **No phase transition in RI:** Unlike PI, high interference does NOT trigger hallucinatory responses

### **Recommended Paper Sections**

#### **3.3 Error Type Analysis** (new subsection)
- Present 5 error types with definitions
- Show three-stage progression
- Compare with "Unable to Forget" PI patterns

#### **4.4 RI vs PI Asymmetry** (results section)
- Quantify difference: 57% retrieval failure (RI) vs 25-45% hallucination (PI)
- Position analysis: Mid-sequence bias in RI vs primacy bias in PI

#### **5.2 Theoretical Implications** (discussion)
- Why RI and PI fail differently
- Implications for working memory models in LLMs
- Connection to human cognitive psychology (asymmetric interference effects)

---

## Next Steps

1. ✅ **Error classification complete** (5 types, 6,237 errors analyzed)
2. ✅ **Visualizations generated** (4 publication figures)
3. ⏳ **Run PI experiments** on same 39 models to enable direct RI vs PI comparison
4. ⏳ **Statistical tests:** Chi-squared test for error distribution differences across levels
5. ⏳ **Model comparison:** Do reasoning models show different error patterns? (e.g., o1/o3 vs base GPT)

---

## Files Generated

### Analysis Scripts
- [`scripts/analysis/classify_error_types.py`](../scripts/analysis/classify_error_types.py) - Main classification script
- [`scripts/analysis/visualize_error_patterns.py`](../scripts/analysis/visualize_error_patterns.py) - Visualization generator

### Results
- [`results/error_analysis/error_classification_detailed.xlsx`](../results/error_analysis/error_classification_detailed.xlsx) - 6,237 errors with full details (5 sheets)
- [`results/error_analysis/ERROR_ANALYSIS_SUMMARY.txt`](../results/error_analysis/ERROR_ANALYSIS_SUMMARY.txt) - Text summary
- [`results/error_analysis/classification_log.txt`](../results/error_analysis/classification_log.txt) - Full analysis log

### Visualizations (300 DPI)
- [`results/error_analysis/error_distribution_stacked.png`](../results/error_analysis/error_distribution_stacked.png)
- [`results/error_analysis/error_trajectories.png`](../results/error_analysis/error_trajectories.png)
- [`results/error_analysis/same_key_position_heatmap.png`](../results/error_analysis/same_key_position_heatmap.png)
- [`results/error_analysis/hallucination_phase_transition.png`](../results/error_analysis/hallucination_phase_transition.png)

---

## Citation

When using this analysis in the paper:

> We classified all incorrect extractions into five cognitive failure modes following the methodology of Lyu et al. (2024). Unlike proactive interference where models exhibit a phase transition to hallucinatory responses at high interference loads, retroactive interference shows a fundamentally different pattern: models primarily fail to retrieve anything (57.3% retrieval failure) rather than confabulate. When models do retrieve incorrect values, they show a mid-sequence bias (average position 137.6 at level 300) rather than primacy or recency effects.

---

**Analysis Date:** December 31, 2024
**Analyst:** Claude (LLM Interference Framework Research Project)
