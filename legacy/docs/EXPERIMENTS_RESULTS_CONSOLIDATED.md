# LLM Interference Research: Consolidated Experimental Results

**Date:** January 4, 2026
**Framework Version:** 2.0
**Dataset:** 46 categories × 8 interference levels (3, 10, 50, 100, 200, 300, 400, 500)
**Experiments:** Retroactive Interference (RI) + Proactive Interference (PI)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Dataset & Methodology](#dataset--methodology)
3. [Models Tested](#models-tested)
4. [Retroactive Interference (RI) Results](#retroactive-interference-ri-results)
5. [Proactive Interference (PI) Results](#proactive-interference-pi-results)
6. [RI vs PI Comparison Analysis](#ri-vs-pi-comparison-analysis)
7. [Statistical Analysis](#statistical-analysis)
8. [Comparison with Literature](#comparison-with-literature)
9. [Visualizations](#visualizations)

---

## Executive Summary

This document consolidates experimental results from testing **both retroactive (RI) and proactive interference (PI)** across **56 large language models** spanning 8 model families. We measured how interference affects memory recall across varying interference levels.

### Headline Results

**Retroactive Interference (RI) - Recalling FIRST-learned values (39 models):**
- **Model size predicts interference resistance**: R² = 0.491, p < 0.0001 ***
- **Context length does NOT predict resistance**: R² = 0.003, p = 0.746 (ns)
- **Reasoning models show 18% higher RIES**: 178.0 vs 150.3 (p = 0.001, Cohen's d = 1.76)
- **Decay patterns vary by architecture**: 35% exponential, 30% polynomial, 26% log-quadratic

**Proactive Interference (PI) - Recalling LAST-learned values (56 models):**
- **Mean PIS = 85.0 ± 39.4** (lower than mean RIES = 154.6 ± 20.1)
- **Model size weakly predicts PI resistance**: R² = 0.033 (vs RI's R² = 0.491)
- **Top performers**: Claude models (claude-4.5-opus: PIS=170.3, claude-4-sonnet: 137.8)
- **Bottom performers**: Mistral-7b, Titan models (PIS=0.0)

**RI vs PI Comparison (39 models with both measurements):**
- **Weak correlation between RI and PI**: R² = 0.044, Pearson r = +0.209, p = 0.201 (not significant)
  - **Interpretation**: RI and PI measure distinct cognitive constructs
- **Strong asymmetry - RI consistently harder than PI**:
  - Mean difference: RIES - PIS = +69.5 ± 40.3
  - Paired t-test: t(38) = 10.780, p < 0.0001 ***
  - Effect size: Cohen's d = 1.726 (large)
  - **100% of models (39/39) show RIES > PIS**
- **Model-specific patterns**:
  - O-series (reasoning): Extreme asymmetry (o1: 8.5× ratio)
  - Claude models: Minimal asymmetry (claude-4.5-opus: 1.05× ratio)
  - GPT models: Moderate asymmetry (gpt-5: 1.9× ratio)

**Error Analysis (54 models, 5,409 errors):**
- **Dominant failure mode:** 51% retrieval failure (null extraction), 46% same-key interference
- **RI vs PI asymmetry:** <1% hallucinations in RI vs 25-45% in PI at high interference
- **Retrieval pattern:** Mid-sequence bias (position ~46%) rather than primacy or recency

### Novel Contributions

1. **First study to demonstrate that chain-of-thought (CoT) reasoning models exhibit significantly stronger resistance to retroactive interference** (18% higher RIES, Cohen's d = 1.76)

2. **First comprehensive RI vs PI comparison in LLMs showing strong asymmetry**: Models are consistently better at resisting proactive interference (recalling recent information) than retroactive interference (recalling distant information), paralleling findings in human memory research

3. **First evidence that RI and PI are uncorrelated in LLMs** (R² = 0.044), suggesting they rely on distinct mechanisms rather than a unified "interference resistance" capacity

4. **First cognitive error analysis for RI showing fundamentally different failure modes than PI**: RI exhibits silent retrieval failure → same-key substitution, while PI exhibits wrong retrieval → hallucination

---

## Dataset & Methodology

### Experimental Design

**Task:** Interleaved fact learning with retroactive interference
- Present 46 initial facts (e.g., "visual art: abstract expressionism")
- Update N facts with new values (N = interference level)
- Query **FIRST-learned** values for all 46 categories
- Measure accuracy of recalled initial values

**Interference Levels:** 3, 10, 50, 100, 200, 300, 400, 500 updates

**Dataset Characteristics:**
- **46 semantic categories** spanning diverse domains
- Categories: visual art, tools, gemstone, building style, sport, cuisine, etc.
- Values: Meaningful, non-overlapping terms (e.g., "baroque", "screwdriver", "emerald")
- Pool size: 5-8 values per category (allowing multiple interference levels)

**Metrics:**
1. **Accuracy**: % of correctly recalled first-learned values (out of 46)
2. **RIES (Retroactive Interference Endurance Score)**: Area under accuracy curve
   - Calculated using trapezoidal integration on log-scaled interference
   - Formula: `∫ accuracy d(log₁₀(interference + 1))`
   - Range: 0-200 (higher = better resistance)

### Quality Control

**Data Filtering:**
- Excluded data points where accuracy = 0 AND error present
- Included data points where accuracy = 0 but NO error (real interference)
- Focused on levels 3-300 for maximum model coverage (39 models)

**Error Categories Excluded:**
- Context length exceeded
- Rate limit errors
- Validation errors
- Model format errors

---

## Models Tested

### Complete Dataset (39 Models, Levels 3-300)

| Family | Models | Size Range | Context Range |
|--------|--------|------------|---------------|
| **Claude** | 7 | 10B - 300B | 200K |
| **GPT** | 7 | 175B - 2500B | 16K - 128K |
| **O-series (Reasoning)** | 5 | 30B - 2000B | 128K - 200K |
| **Gemini** | 4 | 40B - 300B | 1M - 2M |
| **Llama** | 10 | 1B - 405B | 128K - 3.5M |
| **Nova** | 3 | 1B - 70B | 300K |
| **Qwen** | 2 | 30B - 235B | 256K |
| **DeepSeek** | 1 | 671B | 64K |
| **Total** | **39** | **1B - 2500B** | **16K - 3.5M** |

### Model Categorization

**By Size Tier:**
- **XS (1-10B):** 5 models - Mean RIES = 125.0 ± 6.6
- **S (10-100B):** 10 models - Mean RIES = 146.9 ± 13.2
- **M (100-500B):** 13 models - Mean RIES = 158.8 ± 13.7
- **L (500B+):** 11 models - Mean RIES = 170.0 ± 18.8

**By Architecture:**
- **Reasoning (CoT):** 6 models - Mean RIES = 178.0 ± 12.6
- **Non-Reasoning:** 33 models - Mean RIES = 150.3 ± 18.2
- **MoE:** 1 model (Llama 4 Maverick) - RIES = 178.3
- **Dense:** 38 models

### Models Excluded (15 Models)

**Context Limit Errors:**
- gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini (levels 100-300)
- mistral-7b, mistral-8x7b, mistral-large, mistral-small (levels 100-300)
- qwen-3-32b (levels 100-300)
- titan-express, titan-large (levels 50-300)

**Format/Validation Errors:**
- llama-3-70b, llama-3-8b (levels 50-300)
- ministral-3b, ministral-8b (level 300 only)

---

## Key Findings

### Finding 1: Model Size Strongly Predicts Interference Resistance

**Multiple Linear Regression:** RIES ~ Size Class + Context Length

| Predictor | β | t-statistic | p-value | Significance |
|-----------|---|-------------|---------|--------------|
| Size Class | 13.90 | 5.899 | < 0.0001 | *** |
| Context Length | 0.000001 | 0.353 | 0.726 | ns |
| **R² (combined)** | **0.493** | | | **49.3% variance explained** |

**Univariate Correlations:**
- RIES vs Size Class: r = +0.701, R² = 0.491, p < 0.0001 ***
- RIES vs Context Length: r = +0.054, R² = 0.003, p = 0.746 (ns)

**Spearman (non-parametric):**
- ρ(size) = +0.679, ρ² = 0.461, p < 0.0001 ***
- ρ(context) = +0.191, ρ² = 0.036, p = 0.245 (ns)

**Interpretation:**
> Model parameter count is the primary predictor of retroactive interference resistance, explaining nearly 50% of variance. Context window length has virtually no predictive power, suggesting that interference resistance is a function of representational capacity rather than memory buffer size.

### Finding 2: Reasoning Models Significantly Resist Interference

**Group Comparison:**

| Group | n | Mean RIES | Std Dev | Range |
|-------|---|-----------|---------|-------|
| **Reasoning (CoT)** | 6 | **178.0** | 12.6 | 155.4 - 186.4 |
| **Non-Reasoning** | 33 | **150.3** | 18.2 | 25.9 - 178.4 |

**Statistical Test:**
- **T-test:** t = 3.544, p = 0.001 **
- **Cohen's d:** 1.76 (large effect size)

**Reasoning Models:**
1. o1 (RIES = 186.4)
2. o1-preview (RIES = 186.4)
3. gpt-5 (RIES = 186.4)
4. o3 (RIES = 185.8)
5. o3-mini (RIES = 183.3)
6. o4-mini (RIES = 170.5)
7. deepseek-r1 (RIES = 155.4)

**Interpretation:**
> Chain-of-thought reasoning models demonstrate 18% higher interference resistance (Cohen's d = 1.76, large effect). This suggests that extended inference processes or architectural features supporting reasoning may enhance memory robustness and retrieval accuracy under interference.

### Finding 3: Size Tier Effect is Significant

**ANOVA Test:** F = 12.067, p < 0.0001 ***

**Mean RIES by Tier:**
- **XS (1-10B):** 125.0 ± 6.6 (n=5)
- **S (10-100B):** 146.9 ± 13.2 (n=10) → +17.5% vs XS
- **M (100-500B):** 158.8 ± 13.7 (n=13) → +8.1% vs S
- **L (500B+):** 170.0 ± 18.8 (n=11) → +7.1% vs M

**Trend:** Monotonic increase with size tier (p < 0.0001)

### Finding 4: Diverse Decay Patterns Across Architectures

**Best-Fit Decay Functions (n=46 models with ≥5 levels):**

| Pattern | Percentage | Families | R² (mean) |
|---------|-----------|----------|-----------|
| **Exponential** | 34.8% | Claude, Llama, Gemini | 0.698 |
| **Polynomial** | 30.4% | GPT, Nova, Mistral | 0.745 |
| **Log-Quadratic** | 26.1% | Mixed | 0.740 |
| **Power Law** | 8.7% | O-series (flat curves) | 0.479 |

**Key Observations:**
- **No universal log-linear decline** (unlike proactive interference in literature)
- Reasoning models show **power law** (near-flat curves, 97-100% accuracy)
- Dense models show **exponential** (fast early decay) or **polynomial** (gradual decline)

**Average R² by Pattern:**
- Polynomial: 0.745 (best fit)
- Log-Quadratic: 0.740
- Exponential: 0.698
- Log-Linear: 0.564
- Power Law: 0.479 (but appropriate for flat curves)

---

## Retroactive Interference (RI) Results

This section presents results from RI experiments where models were asked to recall **FIRST-learned** values after interference from new information.

### Top 10 Models by RIES (Retroactive Interference Endurance Score)

| Rank | Model | RIES | Size (B) | Context | Accuracy Range | Family |
|------|-------|------|----------|---------|----------------|--------|
| 1 | o1 | 186.4 | 1760 | 200K | 97.8% → 100.0% | O-series |
| 2 | o1-preview | 186.4 | 1760 | 128K | 97.8% → 100.0% | O-series |
| 3 | gpt-5 | 186.4 | 2500 | 128K | 97.8% → 100.0% | GPT |
| 4 | o3 | 185.8 | 2000 | 200K | 97.8% → 100.0% | O-series |
| 5 | o3-mini | 183.3 | 2000 | 200K | 95.7% → 100.0% | O-series |
| 6 | claude-4.5-opus | 178.4 | 300 | 200K | 8.7% → 100.0% | Claude |
| 7 | llama-4-maverick | 178.3 | 400 | 1M | 6.5% → 100.0% | Llama |
| 8 | qwen-3-vl-235b | 177.3 | 235 | 256K | 8.7% → 100.0% | Qwen |
| 9 | gpt-4.1 | 176.9 | 1760 | 128K | 2.2% → 100.0% | GPT |
| 10 | o4-mini | 170.5 | 30 | 200K | 32.6% → 100.0% | O-series |

### Bottom 10 Models by RIES

| Rank | Model | RIES | Size (B) | Context | Accuracy Range | Family |
|------|-------|------|----------|---------|----------------|--------|
| 39 | llama-3-8b | 25.9 | 8 | 8K | 0.0% → 52.2% | Llama |
| 38 | llama-3-70b | 71.0 | 70 | 8K | 0.0% → 93.5% | Llama |
| 37 | llama-3.2-3b | 113.6 | 3 | 128K | 0.0% → 97.8% | Llama |
| 36 | nova-micro | 126.1 | 1 | 300K | 2.2% → 97.8% | Nova |
| 35 | llama-3.2-1b | 126.1 | 1 | 128K | 0.0% → 97.8% | Llama |
| 34 | llama-3.1-8b | 128.3 | 8 | 128K | 21.7% → 97.8% | Llama |
| 33 | gpt-4.1-nano | 128.2 | 1760 | 128K | 6.5% → 89.1% | GPT |
| 32 | nova-lite | 130.7 | 7 | 300K | 2.2% → 100.0% | Nova |
| 31 | llama-3.1-70b | 134.6 | 70 | 128K | 0.0% → 100.0% | Llama |
| 30 | llama-3.2-90b | 134.6 | 90 | 128K | 0.0% → 100.0% | Llama |

**Observations:**
- Smallest models (1-8B) struggle most with interference
- llama-3-8b shows catastrophic forgetting (0% → 52% accuracy)
- Even large models (llama-3.1-70b, 70B params) can show poor resistance if context is limited

### Family-Level Analysis

**Mean RIES by Family:**

| Family | n | Mean RIES | Std Dev | Min RIES | Max RIES |
|--------|---|-----------|---------|----------|----------|
| **O-series** | 5 | **181.1** | 11.2 | 170.5 | 186.4 |
| **Qwen** | 2 | **171.8** | 7.7 | 166.4 | 177.3 |
| **GPT (non-CoT)** | 7 | **161.2** | 18.3 | 128.2 | 186.4 |
| **DeepSeek** | 1 | **155.4** | - | 155.4 | 155.4 |
| **Gemini** | 4 | **153.8** | 10.1 | 143.9 | 162.5 |
| **Claude** | 7 | **152.9** | 15.0 | 139.6 | 178.4 |
| **Nova** | 3 | **138.1** | 17.0 | 126.1 | 157.5 |
| **Llama** | 10 | **138.8** | 17.4 | 25.9 | 178.3 |

**Key Insights:**
- O-series dominates (5 of top 6 models)
- Qwen models punch above their weight (2nd best family mean)
- Large variance within families (especially GPT, Claude, Llama)

---

## Proactive Interference (PI) Results

This section presents results from PI experiments where models were asked to recall **LAST-learned** values (opposite of RI).

### PI Experimental Design

**Key Differences from RI:**
- **Query target**: LAST-learned values (most recent updates) instead of FIRST-learned values
- **Prompt template**: Modified to ask for "latest value" instead of "initial value"
- **Same dataset**: Identical interleaved sequence, same interference levels
- **Metric**: PIS (Proactive Interference Score) - calculated using same formula as RIES

**Hypothesis**: If recency bias exists, PI should be easier than RI (models better at recalling recent information).

### Top 10 Models by PIS (Proactive Interference Score)

| Rank | Model | PIS | Size (B) | Accuracy Range | RIES | RI-PI Diff |
|------|-------|-----|----------|----------------|------|------------|
| 1 | claude-4.5-opus | 170.3 | 300 | 71.7% → 100.0% | 178.4 | +8.1 |
| 2 | claude-4-sonnet | 137.8 | 200 | 8.7% → 100.0% | 162.5 | +24.6 |
| 3 | claude-3-opus | 131.5 | 175 | 2.2% → 100.0% | 139.6 | +8.1 |
| 4 | claude-4-opus | 131.0 | 200 | 4.3% → 100.0% | 162.5 | +31.5 |
| 5 | claude-4.5-haiku | 128.5 | 15 | 4.3% → 100.0% | 151.8 | +23.3 |
| 6 | gpt-4o | 126.5 | 1760 | 0.0% → 100.0% | - | - |
| 7 | claude-3.5-haiku | 126.4 | 10 | 4.3% → 95.7% | 141.4 | +15.0 |
| 8 | claude-4.5-sonnet | 124.2 | 250 | 0.0% → 100.0% | 162.5 | +38.3 |
| 9 | gemini-2.5-pro | 123.6 | 300 | 0.0% → 100.0% | 155.4 | +31.8 |
| 10 | qwen-3-vl-235b | 120.2 | 235 | 0.0% → 100.0% | 177.3 | +57.1 |

### Bottom 10 Models by PIS

| Rank | Model | PIS | Size (B) | Accuracy Range | RIES | RI-PI Diff |
|------|-------|-----|----------|----------------|------|------------|
| 56 | mistral-7b | 0.0 | 7 | 0.0% → 0.0% | - | - |
| 55 | nova-premier | 0.0 | 70 | 0.0% → 0.0% | - | - |
| 54 | titan-express | 0.0 | - | 0.0% → 0.0% | - | - |
| 53 | titan-large | 0.0 | - | 0.0% → 0.0% | - | - |
| 52 | titan-lite | 0.0 | - | 0.0% → 0.0% | - | - |
| 51 | llama-3.2-1b | 1.0 | 1 | 0.0% → 2.2% | 126.1 | +125.1 |
| 50 | llama-3-8b | 18.9 | 8 | 0.0% → 80.4% | 25.9 | +7.0 |
| 49 | mistral-small | 19.6 | 22 | 0.0% → 89.1% | - | - |
| 48 | gpt-5-nano | 21.0 | 1760 | 0.0% → 95.7% | 143.0 | +122.0 |
| 47 | o1 | 22.0 | 1760 | 0.0% → 100.0% | 186.4 | +164.4 |

### Key PI Findings

**1. Claude Models Dominate PI Performance**
- 7 of top 10 models are Claude family
- Minimal asymmetry between RI and PI (claude-4.5-opus: only 8.1 point difference)

**2. Extreme Asymmetry in Reasoning Models**
- O-series models show catastrophic PI performance despite stellar RI scores
- **o1**: RIES=186.4 (1st in RI) → PIS=22.0 (47th in PI) - **8.5× ratio**
- **o1-preview**: RIES=186.4 → PIS=76.0 - **2.5× ratio**
- Suggests these models optimize heavily for "first-learned" retention

**3. Model Size Weakly Predicts PI (vs Strong for RI)**
- **PIS vs Size**: R² = 0.033 (not significant)
- **RIES vs Size**: R² = 0.491 (p < 0.0001)
- Size matters more for remembering distant info (RI) than recent info (PI)

**4. Some Models Fail Completely at PI**
- 5 models score PIS=0.0 (Mistral-7b, Nova-premier, Titan models)
- Likely due to prompt format incompatibility or model limitations

### Family-Level PI Analysis

**Mean PIS by Family:**

| Family | n | Mean PIS | Std Dev | Min PIS | Max PIS | Mean RIES | RI-PI Diff |
|--------|---|----------|---------|---------|---------|-----------|------------|
| **Claude** | 7 | **137.1** | 15.2 | 124.2 | 170.3 | 152.9 | +15.8 |
| **Gemini** | 4 | **103.8** | 14.6 | 91.3 | 123.6 | 153.8 | +50.0 |
| **GPT (non-CoT)** | 9 | **102.5** | 35.4 | 21.0 | 126.5 | 161.2 | +58.7 |
| **Qwen** | 3 | **102.3** | 34.0 | 89.3 | 120.2 | 171.8 | +69.5 |
| **Llama** | 13 | **54.7** | 30.8 | 1.0 | 100.3 | 138.8 | +84.1 |
| **Nova** | 3 | **64.6** | 31.9 | 0.0 | 76.8 | 138.1 | +73.5 |
| **O-series** | 5 | **54.3** | 29.5 | 22.0 | 76.0 | **181.1** | **+126.8** |
| **Mistral** | 5 | **43.0** | 41.2 | 0.0 | 96.7 | - | - |

**Key Insights:**
- **Claude maintains consistency**: Smallest RI-PI gap (+15.8 points)
- **O-series shows extreme asymmetry**: Largest RI-PI gap (+126.8 points)
- **All families show RI > PI**: No exceptions

---

## RI vs PI Comparison Analysis

This section analyzes the relationship between retroactive and proactive interference across 39 models tested on both paradigms.

### Correlation Analysis

**Research Question**: Are models that resist RI also good at resisting PI?

**Hypothesis**: If RI and PI tap into the same underlying capacity (general "memory stability"), we should see high correlation (r > 0.7).

**Result**: **No significant correlation**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Pearson r** | +0.209 | Weak positive correlation |
| **R²** | 0.044 | Only 4.4% shared variance |
| **p-value** | 0.201 | Not statistically significant |
| **Spearman ρ** | +0.214 | Weak rank correlation |

**Interpretation**:
- RI and PI measure **distinct cognitive constructs** in LLMs
- A model good at RI is NOT necessarily good at PI (and vice versa)
- Suggests different mechanisms: RI may rely on early encoding strength, while PI may rely on recency bias/attention

### Asymmetry Test

**Research Question**: Is RI harder than PI (or vice versa)?

**Hypothesis (from human memory literature)**: RI typically > PI in humans (retroactive interference stronger).

**Result**: **Strong asymmetry - RI consistently harder than PI**

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| **Mean RIES** | 154.6 ± 20.1 | Retroactive interference resistance |
| **Mean PIS** | 85.0 ± 39.4 | Proactive interference resistance |
| **Mean Difference** | +69.5 ± 40.3 | RI significantly harder |
| **Paired t-test** | t(38) = 10.780 | Highly significant |
| **p-value** | < 0.0001 *** | Strong statistical evidence |
| **Cohen's d** | 1.726 | **Large effect size** |
| **Direction** | 39/39 models show RIES > PIS | **100% consistency** |

**Interpretation**:
- **Universal asymmetry**: Every single model finds RI harder than PI
- **Large effect**: Cohen's d = 1.726 indicates a massive difference
- **Parallels human memory**: Humans also show RI > PI (new info disrupts old more than vice versa)
- **Recency bias in LLMs**: Models naturally attend more to recent context

### Model-Specific Patterns

**1. O-Series (Reasoning Models): Extreme Asymmetry**

| Model | RIES | PIS | Difference | Ratio |
|-------|------|-----|------------|-------|
| o1 | 186.4 | 22.0 | +164.4 | **8.5×** |
| o1-preview | 186.4 | 76.0 | +110.4 | 2.5× |
| o4-mini | 170.5 | 22.0 | +148.5 | 7.8× |
| o3 | 185.8 | 76.0 | +109.8 | 2.4× |
| o3-mini | 183.3 | 75.6 | +107.8 | 2.4× |

**Interpretation**: Reasoning models heavily optimize for early memory retention but struggle with recency.

**2. Claude Models: Minimal Asymmetry**

| Model | RIES | PIS | Difference | Ratio |
|-------|------|-----|------------|-------|
| claude-4.5-opus | 178.4 | 170.3 | +8.1 | **1.05×** |
| claude-3-opus | 139.6 | 131.5 | +8.1 | 1.06× |
| claude-3.5-haiku | 141.4 | 126.4 | +15.0 | 1.12× |

**Interpretation**: Claude models exhibit balanced interference resistance across both RI and PI.

**3. GPT Models: Moderate Asymmetry**

| Model | RIES | PIS | Difference | Ratio |
|-------|------|-----|------------|-------|
| gpt-5 | 186.4 | 99.2 | +87.2 | 1.9× |
| gpt-4.1 | 176.9 | 118.6 | +58.3 | 1.5× |
| gpt-5-mini | 157.0 | 109.1 | +47.9 | 1.4�� |

**Interpretation**: GPT models show moderate RI advantage, between Claude and O-series extremes.

### Scatter Plot Analysis

**Visualization**: RIES vs PIS across 39 models
- **Regression line**: Slightly negative slope (r = -0.279 in uncorrected data)
- **Most points below diagonal**: Indicates RI > PI universally
- **Outliers**: O-series models (high RI, low PI)
- **Tight cluster**: Claude models (high on both)

### Bland-Altman Agreement Analysis

**Purpose**: Assess agreement between RI and PI measurements

**Results**:
- **Mean bias**: +62.2 (systematic tendency for RI > PI)
- **95% limits of agreement**: +154.0 to -29.5
- **No proportional bias**: Difference constant across magnitude

**Interpretation**: RI and PI are NOT interchangeable measures; they assess distinct capacities.

### Key Takeaways

1. **RI and PI are uncorrelated** (R² = 0.044) - distinct mechanisms
2. **RI universally harder** than PI (100% of models, d = 1.726)
3. **Model families differ in asymmetry**:
   - O-series: Extreme (8.5× ratio)
   - Claude: Minimal (1.05× ratio)
   - GPT: Moderate (1.9× ratio)
4. **Parallels human memory**: LLMs replicate classic RI > PI asymmetry
5. **Architectural implications**: Reasoning models trade PI resistance for RI strength

---

## Statistical Analysis

### Regression Analysis Summary

**Model:** RIES ~ Size Class + Context Length (n=39)

```
Coefficients:
  Intercept:      115.56
  Size Class:     +13.90  (p < 0.0001 ***)
  Context Length: +0.000001  (p = 0.726, ns)

Model Fit:
  R² = 0.493
  Adjusted R² = 0.465
  F-statistic = 17.5 (p < 0.0001 ***)
```

**Non-Reasoning Models Only (n=33):**

```
R² = 0.456
Size Class:      r = +0.669, R² = 0.448, p < 0.0001 ***
Context Length:  r = +0.151, R² = 0.023, p = 0.402 (ns)
```

**Conclusion:** Effect holds even when reasoning models excluded.

### Effect Sizes

| Comparison | Cohen's d | Interpretation |
|------------|-----------|----------------|
| Reasoning vs Non-Reasoning | **1.76** | **Very large** |
| XS vs L size tier | 2.91 | Very large |
| S vs M size tier | 0.87 | Large |
| M vs L size tier | 0.63 | Medium |

---

## Error Type Analysis

### Cognitive Failure Modes (54 Models, 5,409 Errors)

Following the methodology from "Unable to Forget: Persistent Memory in Language Models" (Section 2.2.1), we classified all incorrect extractions into 5 cognitive failure modes:

#### Error Type Distribution

| Error Type | Count | Percentage | Description |
|-----------|-------|------------|-------------|
| **Same-Key Interference** | 2,490 | **46.0%** | Later value from SAME category |
| **Retrieval Failure** | 2,747 | **50.8%** | No value extracted (null/None) |
| **Partial Match** | 95 | 1.8% | Close but not exact match |
| **Hallucination** | 47 | 0.9% | Value NEVER presented |
| **Cross-Key Interference** | 30 | 0.6% | Value from DIFFERENT category |

**Dataset:** 54 models (excludes 3 models with accuracy=0 for all levels: nova-premier, titan-lite, o1-mini)

#### Three-Stage Progression

| Stage | Levels | Same-Key | Retrieval Failure | Hallucination | Pattern |
|-------|--------|----------|-------------------|---------------|---------|
| **1 - Low** | 3, 10 | 6.4% | **80.1%** | 4.5% | Tightly focused errors |
| **2 - Moderate** | 50, 100 | 25.8% | **72.9%** | 1.1% | Dispersed errors |
| **3 - High** | 200, 300 | **55.8%** | 42.0% | 0.1% | Same-key dominates |

**Critical Finding:** Unlike Proactive Interference (PI) which shows 25-45% hallucinations at high load, Retroactive Interference (RI) shows **<1% hallucinations**. Instead, RI exhibits a shift from retrieval failure → same-key interference.

#### Position Analysis: Which Earlier Values Get Retrieved?

When models make same-key interference errors, **which** earlier value do they retrieve?

| Level | Avg Position | Avg Recency | Most Recent (%) | Pattern |
|-------|-------------|-------------|-----------------|---------|
| 3     | 2.4         | 1.6 steps   | 44%             | Adjacent values |
| 10    | 2.5         | 8.4 steps   | 0%              | Early updates |
| 50    | 20.4        | 30.6 steps  | 0%              | Mid-sequence |
| 100   | 53.6        | 47.4 steps  | 0%              | Mid-sequence |
| 200   | 112.0       | 89.0 steps  | 3%              | Mid-sequence |
| 300   | 137.6       | 163.4 steps | 0%              | **Mid-sequence bias** |

**Key Insight:** Models retrieve from a **mid-sequence window** (position ~46% of total), not primacy (first) or recency (last). This suggests a "retrieval window" that expands linearly with interference level.

#### RI vs PI Error Comparison

| Metric | RI (Our Study) | PI ("Unable to Forget") |
|--------|----------------|------------------------|
| **Dominant Error** | Retrieval Failure (51%) | Same-Key Interference (60-80%) |
| **Stage 3 Hallucinations** | **0.1%** ↓ | **25-45%** ↑ |
| **Cross-Key Errors** | <1% (rare) | 15-30% (common) |
| **Retrieval Behavior** | Silent failure → Wrong retrieval | Wrong retrieval → Hallucination |

**Hypothesis:** Retrieving **oldest** information (RI) is fundamentally harder than retrieving **newest** (PI). When RI fails, models "give up" (retrieval failure). When PI fails, models "guess" (hallucination).

### Analysis Scripts & Data

- **Script:** [`scripts/analysis/classify_error_types.py`](../scripts/analysis/classify_error_types.py)
- **Visualizations:** [`scripts/analysis/visualize_error_patterns.py`](../scripts/analysis/visualize_error_patterns.py)
- **Results:** [`results/error_analysis/error_classification_detailed.xlsx`](../results/error_analysis/error_classification_detailed.xlsx) - 5,409 errors with full classification
- **Report:** [`docs/ERROR_ANALYSIS_FINDINGS.md`](ERROR_ANALYSIS_FINDINGS.md) - Comprehensive 10-page analysis

---

## Comparison with Literature

### "Unable to Forget" Paper (Proactive Interference)

| Metric | Paper (IES, Proactive) | Our Study (RIES, Retroactive) |
|--------|------------------------|-------------------------------|
| **Dataset** | 30 open-source models | 39 models (incl. reasoning) |
| **Reasoning Models** | Excluded (latency) | **Included (novel)** |
| **Size Effect** | t = 3.03, p = 0.005 | t = 5.90, p < 0.0001 (stronger) |
| **Context Effect** | t = -0.14, p = 0.886 (ns) | t = 0.35, p = 0.726 (ns) ✓ Match |
| **R² (combined)** | 0.261 (26.1%) | **0.493 (49.3%)** (stronger) |
| **Spearman (size)** | ρ² = 0.673 (128K subset) | ρ² = 0.461 (all models) |
| **Decay Pattern** | Log-linear (universal) | **Diverse (35% exponential, 30% polynomial)** |

**Key Differences:**

1. **Stronger size effect in RI** (R² = 0.493 vs 0.261)
   - May indicate that retroactive interference is more sensitive to model capacity

2. **No universal decay pattern in RI**
   - Proactive interference: log-linear decline (universal)
   - Retroactive interference: architecture-dependent (exponential, polynomial, power law)

3. **Reasoning models resist RI strongly** (novel finding)
   - Paper excluded reasoning models due to latency
   - We show they have 18% higher resistance

4. **Context length irrelevant (confirmed)**
   - Both studies: context does NOT predict interference resistance
   - Supports "representational capacity" over "memory buffer" hypothesis

---

## Visualizations

### Figure 1: Decay Curves by Size Tier
**File:** `results/visualizations/decay_curves_size_tiers.png`

4-panel plot showing accuracy decay across interference levels (3-300):
- **XS models (1-10B):** Fast exponential decay (8.7% → 93.5%)
- **S models (10-100B):** Moderate decay (2.2% → 100%)
- **M models (100-500B):** Slower decay, higher baseline
- **L models (500B+):** Flattest curves, dominated by O-series (97.8% → 100%)

### Figure 2: Reasoning vs Non-Reasoning
**File:** `results/visualizations/decay_curves_reasoning.png`

2-panel comparison:
- **Reasoning (CoT):** Flat curves, 95-100% accuracy maintained
- **Non-Reasoning:** Variable decay patterns, 0-100% accuracy range

### Figure 3: Regression Analysis
**File:** `results/visualizations/regression_analysis_size_context.png`

4-panel statistical visualization:
- **Top-left:** RIES vs Size Class (R² = 0.491, strong linear trend)
- **Top-right:** RIES vs Context Length (R² = 0.003, no correlation)
- **Bottom-left:** Boxplot by size tier (ANOVA p < 0.0001)
- **Bottom-right:** Reasoning vs Non-Reasoning (T-test p = 0.001)

### Figure 4: MoE vs Dense
**File:** `results/visualizations/decay_curves_moe.png`

2-panel comparison (limited data: only 1 MoE model in clean dataset)

---

## Data Files

### Primary Data
- **Raw experiments:** `data/experiment_results_20251228_224744.xlsx`
  - 54 models × 8 levels = 433 data points
  - 39 models have complete data (levels 3-300)

### Analysis Results
- **RIES scores:** `results/key_results/ries_analysis.xlsx`
  - 39 models with RIES calculations
  - Includes: model_id, size, context, RIES, accuracy range, levels tested

- **Model categories:** `results/key_results/model_categories.xlsx`
  - Size tier (XS/S/M/L)
  - Architecture (Reasoning/Non-Reasoning, MoE/Dense)
  - Developer family

- **Regression results:** `results/key_results/regression_analysis_results.xlsx`
  - Statistical test outputs
  - R², correlation coefficients, p-values

---

## Conclusions

### Main Takeaways

1. **Model size is the primary factor** governing retroactive interference resistance (R² = 0.49)

2. **Context length is irrelevant** to interference resistance (R² = 0.003)

3. **Reasoning models show exceptional resistance** (+18%, p = 0.001, d = 1.76)

4. **Decay patterns are architecture-dependent** (no universal log-linear decline)

5. **RI and PI may have different mechanisms** (stronger size effect in RI, diverse decay patterns)

### Implications

**For LLM Development:**
- Scaling model parameters improves interference resistance more than expanding context windows
- Chain-of-thought architectures may provide inherent memory advantages
- MoE architectures need further study (limited data in this experiment)

**For LLM Applications:**
- Long-context tasks with frequent updates may benefit from reasoning models
- Small models (< 10B) show poor interference resistance, may need external memory
- Context window ≠ working memory capacity (confirmed)

---

## Next Steps

### Planned Experiments

1. **Proactive Interference (PI) Analysis**
   - Test same models with PI paradigm (query last-learned values)
   - Compare RI vs PI correlations
   - Test if models show asymmetric interference patterns

2. **Mitigation Strategies**
   - Prompt engineering: "Focus on first-learned values"
   - Explicit memory instructions: "Deprioritize recent updates"
   - Chain-of-thought prompting for non-CoT models

3. **Multiple Runs for Confidence Intervals**
   - Run each model 3-5 times
   - Calculate mean ± std for RIES
   - Generate error bars for figures

4. **Additional Analysis**
   - MoE vs Dense (with more MoE models)
   - Fine-grained size bins (instead of 4 tiers)
   - Training data effects (if information available)

---

**Document Version:** 1.0
**Last Updated:** December 30, 2024
**Contact:** [Your Email]
