# PAPER_DRAFT.md Updates Summary

**Date:** January 4, 2026
**Progress:** Abstract, Introduction, and Methods sections completed

---

## ✅ COMPLETED UPDATES

### 1. Abstract (Lines 7-21)
**Changes:**
- Added PI experiments (56 models)
- Added PIS metric
- Added RI vs PI comparison results:
  - Weak correlation (R²=0.044, p=0.201)
  - Strong asymmetry (mean diff=+69.5, Cohen's d=1.73, p<0.0001)
  - Model-specific patterns (O-series: 8.5×, Claude: 1.05×)
- Updated conclusions to reflect distinct mechanisms

### 2. Keywords (Line 21)
**Added:** "proactive interference", "memory asymmetry"

### 3. Introduction - Research Questions (Lines 52-54)
**Updated RQ4:**
- H4a: RI and PI are correlated if same capacity
- H4b: RI and PI show symmetric difficulty
(Both hypotheses REJECTED by our findings)

### 4. Introduction - Contributions (Lines 56-72)
**Expanded from 5 to 7 contributions:**
1. First study of both RI and PI (56 models)
2. Introduction of RIES and PIS metrics
3. Reasoning models show trade-offs (18% higher RI, but 8.5× asymmetry)
4. **NEW:** RI and PI uncorrelated (R²=0.044)
5. **NEW:** Universal RI > PI asymmetry (d=1.73)
6. Decay pattern characterization + family asymmetry ratios
7. Error analysis showing distinct failure modes

### 5. Methods - Experimental Design (Lines 153-211)
**Restructured into 3.1.1 and 3.1.2:**
- **3.1.1 RI Paradigm:** Query FIRST-learned values
- **3.1.2 PI Paradigm:** Query LAST-learned values
- Added rationale for both paradigms
- Noted within-subjects design for 39 models

### 6. Methods - Metrics (Lines 300-320)
**Updated:**
- Renamed "Primary Metric: RIES" → "Primary Metrics: RIES and PIS"
- Added PIS formula (identical to RIES but for PI accuracy)
- Noted identical calculation ensures fair comparison

### 7. Methods - Statistical Analysis (Lines 357-364)
**Added RI vs PI Comparison section:**
- Correlation analysis (tests H4a)
- Asymmetry test with paired t-test (tests H4b)
- Bland-Altman agreement analysis

---

## ⏳ REMAINING UPDATES NEEDED

### 8. Results Section - Add RQ4 (NEW SUBSECTION)
**Location:** After RQ3 results (~line 600)

**Content to add:**
```markdown
### 4.4 RQ4: Retroactive vs. Proactive Interference Comparison

**Research Question:** Are RI and PI correlated (same capacity) or asymmetric?

#### 4.4.1 Correlation Analysis

**Hypothesis H4a:** RI and PI are correlated (r > 0.7) if same underlying capacity

**Result:** REJECTED - Weak correlation

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Pearson r | +0.209 | Weak positive |
| R² | 0.044 | Only 4.4% shared variance |
| p-value | 0.201 | Not significant |
| Spearman ρ | +0.214 | Weak rank correlation |

**Interpretation:**
- RI and PI measure **distinct cognitive constructs**
- Good at RI ≠ Good at PI
- Suggests different mechanisms (early encoding vs. recency bias)

**Figure 4A:** Scatter plot showing low correlation between RIES and PIS

#### 4.4.2 Asymmetry Test

**Hypothesis H4b:** RI and PI show symmetric difficulty

**Result:** REJECTED - Strong RI > PI asymmetry

| Statistic | Value |
|-----------|-------|
| Mean RIES | 154.6 ± 20.1 |
| Mean PIS | 85.0 ± 39.4 |
| Mean Difference | +69.5 ± 40.3 |
| Paired t-test | t(38) = 10.780, p < 0.0001 *** |
| Cohen's d | 1.726 (large effect) |
| Direction | 100% models (39/39) show RIES > PIS |

**Interpretation:**
- Universal asymmetry: Every model finds RI harder than PI
- Large effect size (d = 1.726)
- Parallels human memory research (RI > PI in humans)
- Recency bias in LLMs

**Figure 4B:** Bland-Altman plot showing systematic bias (+62.2)

#### 4.4.3 Model Family Patterns

**O-Series (Reasoning): Extreme Asymmetry**

| Model | RIES | PIS | Ratio |
|-------|------|-----|-------|
| o1 | 186.4 | 22.0 | **8.5×** |
| o1-preview | 186.4 | 76.0 | 2.5× |
| o4-mini | 170.5 | 22.0 | 7.8× |

**Interpretation:** Reasoning models optimize for early retention, sacrifice recency.

**Claude Models: Minimal Asymmetry**

| Model | RIES | PIS | Ratio |
|-------|------|-----|-------|
| claude-4.5-opus | 178.4 | 170.3 | **1.05×** |
| claude-3-opus | 139.6 | 131.5 | 1.06× |

**Interpretation:** Balanced interference resistance.

**GPT Models: Moderate Asymmetry**

| Model | RIES | PIS | Ratio |
|-------|------|-----|-------|
| gpt-5 | 186.4 | 99.2 | 1.9× |
| gpt-4.1 | 176.9 | 118.6 | 1.5× |

**Table 5:** RI vs PI comparison statistics for all 39 models
**Figure 4C:** Bar chart showing RIES-PIS difference by model family
**Figure 4D:** RI vs PI decay curves overlaid for representative models
```

---

### 9. Discussion Section - Add RI vs PI Interpretation
**Location:** Add new subsection after existing discussion (~line 766)

**Content to add:**
```markdown
### 5.5 RI vs PI Asymmetry: Implications for LLM Memory Architecture

**Finding:** All models show RI > PI (Cohen's d = 1.73), but asymmetry varies by family.

**Parallel to Human Memory:**
Classic interference theory posits RI > PI in humans [cite Jenkins & Dallenbach, Underwood]. Our findings suggest LLMs replicate this asymmetry, indicating fundamental constraints shared with biological memory:
- **Recency bias:** Attention naturally weights recent information
- **Consolidation deficit:** Early memories less robust to interference
- **Retrieval competition:** New information overwrites old

**Architectural Trade-offs:**
- **O-series models:** Extreme RI-PI asymmetry (8.5×) suggests heavy optimization for distant memory retention at cost of recency bias. CoT reasoning may enhance early encoding but impair recent retrieval.
- **Claude models:** Minimal asymmetry (1.05×) indicates balanced architecture, suitable for applications requiring both distant and recent recall.
- **GPT models:** Moderate asymmetry (1.9×) represents middle ground.

**Distinct Mechanisms:**
RI and PI are uncorrelated (R²=0.044), supporting dual-process theory:
- **RI resistance:** Depends on representational capacity (size-dependent, R²=0.491)
- **PI resistance:** Depends on recency bias (size-independent, R²=0.033)

**Practical Implications:**
- For applications requiring robust distant memory (e.g., long document QA): Prioritize high RIES models (O-series, large dense models)
- For applications requiring recent memory (e.g., multi-turn dialogue): Prioritize balanced models (Claude) or high PIS models
- Avoid assuming "good memory" = good at both RI and PI
```

---

### 10. Conclusion Section - Add PI Contributions
**Location:** Update existing conclusion (~lines 766-780)

**Changes needed:**
- Add PI findings to summary
- Emphasize RI-PI asymmetry as novel contribution
- Update future work to mention mitigation strategies

---

## Summary Statistics

**Sections completed:** 7 of 10
**Sections remaining:** 3 (Results RQ4, Discussion RI vs PI, Conclusion)
**Lines added so far:** ~100 lines across Abstract, Introduction, Methods
**Lines to add:** ~150 lines for Results RQ4, ~50 for Discussion, ~20 for Conclusion

**Estimated completion:** ~220 more lines to fully integrate PI findings into paper

---

## Next Steps

1. Add RQ4 Results section (~150 lines)
2. Add RI vs PI interpretation to Discussion (~50 lines)
3. Update Conclusion (~20 lines)
4. Create Figure 4 (4-panel RI vs PI comparison)
5. Final review and proofread

