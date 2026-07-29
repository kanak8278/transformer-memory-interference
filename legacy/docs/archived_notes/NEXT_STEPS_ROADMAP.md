# Next Steps Roadmap: Completing the Retroactive Interference Study

**Status:** RI experiments complete (39 models), paper draft ready
**Timeline:** Complete remaining experiments before paper submission

---

## ✅ Completed

1. **Repository cleanup and organization**
   - Clean directory structure
   - Updated shell scripts with multiple runs support
   - Documentation: README, PROJECT_STRUCTURE, EXPERIMENTS_RESULTS_CONSOLIDATED

2. **Retroactive Interference (RI) experiments**
   - 39 models with complete data (levels 3-300)
   - RIES scores calculated
   - Statistical analysis complete (regression, ANOVA, t-tests)
   - Visualizations generated (decay curves, regression plots)

3. **Key findings documented**
   - Size predicts RI (R² = 0.491, p < 0.0001)
   - Context irrelevant (R² = 0.003, p = 0.746)
   - Reasoning models 18% better (p = 0.001, d = 1.76)
   - Diverse decay patterns (35% exponential, 30% polynomial)

4. **Paper draft created**
   - Full structure with Abstract, Intro, Methods, Results, Discussion, Conclusion
   - Ready for LaTeX conversion
   - ~8,500 words

---

## 🔄 In Progress / Next Experiments

### 1. Proactive Interference (PI) Experiments

**Goal:** Compare RI vs PI to test if they have different mechanisms

**What:** Query LAST-learned values (vs. FIRST-learned in RI)
- Same 39 models
- Same interference levels (3, 10, 50, 100, 200, 300)
- Same dataset

**Script Ready:** `scripts/core/run_proactive_interference_experiment.py`

**How to Run:**

```bash
# Single model test
python scripts/core/run_proactive_interference_experiment.py \
    --model claude-4.5-sonnet \
    --levels 3 10 50 100 200 300 \
    --output-dir data/raw_experiments_pi

# All models (batch)
./scripts/shell/run_all_models_pi.sh 1
```

**Expected Time:** ~6-8 hours for all 39 models

**Analysis:**
- Calculate PIS (Proactive Interference Score) - same formula as RIES
- Compare RI vs PI:
  - Do models show asymmetric interference (RI > PI or vice versa)?
  - Correlation: r(RIES, PIS) - are resistant models resistant to both?
  - Regression: Does size/context predict PI similarly?

**Hypotheses:**
- H1: RIES and PIS correlate (r > 0.7) - same underlying capacity
- H2: RI > PI (asymmetric) - newer memories more fragile
- H3: Reasoning models resist both RI and PI equally

---

### 2. Mitigation Strategies

**Goal:** Test if prompting can reduce interference

#### Strategy 1: Explicit Memory Instructions

**Prompts to test:**
```
[Control - no instruction]
Standard RI prompt

[Treatment 1 - Priority instruction]
"IMPORTANT: Focus on recalling the INITIAL values you learned first.
Ignore any updates you saw later."

[Treatment 2 - Inhibition instruction]
"Recent updates are IRRELEVANT. Deprioritize new information.
Retrieve only the ORIGINAL values."

[Treatment 3 - Elaboration instruction]
"Before answering, think carefully about the FIRST time you learned each fact.
Visualize the initial learning phase."
```

**Models to test:** Select 10 diverse models
- 2 reasoning (o1, o3-mini)
- 2 large dense (claude-4.5-opus, gpt-4.1)
- 2 medium (llama-3.1-70b, gemini-2.5-pro)
- 2 small (llama-3.2-11b, nova-lite)
- 2 tiny (llama-3.2-1b, nova-micro)

**Interference levels:** 50, 100, 200 (high interference only)

**Expected result:** Modest improvements (10-20% accuracy gain) - based on human lit, explicit instructions help but don't eliminate interference

#### Strategy 2: Chain-of-Thought for Non-CoT Models

**Prompt addition:**
```
"Before answering, use step-by-step reasoning:
1. Recall when you FIRST learned about [category]
2. Identify if [category] was updated
3. Retrieve the INITIAL value, not the update
4. Provide your answer"
```

**Hypothesis:** Can we induce "reasoning-like" memory benefits in standard models?

**Models to test:** 5 non-reasoning models with varying RIES
- High RIES (claude-4.5-opus)
- Medium (llama-3.1-405b)
- Low (llama-3-8b)

**Expected result:** Small improvements (5-10%) but won't match true reasoning models

---

### 3. Multiple Runs for Confidence Intervals

**Goal:** Get error bars for all figures (publication quality)

**What:** Run each model 3-5 times at each interference level

**How:**
```bash
# RI experiments (3 runs each)
./scripts/shell/run_all_bedrock_models.sh 3
./scripts/shell/run_all_claude_gemini_models.sh 3
./scripts/shell/run_all_gpt_models.sh 3

# PI experiments (3 runs each)
./scripts/shell/run_all_models_pi.sh 3
```

**Expected Time:** ~18-24 hours total (3× longer than single runs)

**Analysis:**
- Calculate mean ± std for RIES at each level
- Generate error bars for decay curve plots
- Test if variance differs by model size (small models more variable?)

**Statistics:**
- Use mean RIES for correlations (more robust)
- Report 95% CI for all key results
- Check if findings hold with error bars

---

### 4. RI vs PI Comparison Analysis

**Script to create:** `scripts/analysis/compare_ri_vs_pi.py`

**Analyses:**

1. **Correlation Matrix**
   ```
   RIES vs PIS: r = ?
   RIES vs Size: r = 0.70
   PIS vs Size: r = ?
   RIES vs Context: r = 0.05
   PIS vs Context: r = ?
   ```

2. **Asymmetry Test**
   ```
   Paired t-test: RIES vs PIS (within-subject)
   Effect size: Cohen's d
   Do models show consistent RI > PI or vice versa?
   ```

3. **Decay Pattern Comparison**
   ```
   RI: 35% exponential, 30% polynomial, 26% log-quad, 9% power
   PI: ? (to be determined)
   Chi-square test: Are distributions different?
   ```

4. **Reasoning Model Effect**
   ```
   RI: +18% (p = 0.001)
   PI: +?% (p = ?)
   Interaction: Does reasoning help more for RI or PI?
   ```

**Visualization:**
- Scatter plot: RIES vs PIS (each model = point)
- Bland-Altman plot: Agreement between RI and PI
- Decay curves: RI vs PI overlaid (per model family)

---

## 📊 Final Analyses & Figures

### Figures for Paper

**Main Text (4 figures):**

1. **Figure 1: Decay curves by size tier** ✅ DONE
   - 4 panels (XS, S, M, L)
   - Shows larger models degrade slower

2. **Figure 2: Regression analysis** ✅ DONE
   - 4 panels: RIES vs Size, RIES vs Context, Boxplots by tier, CoT vs non-CoT

3. **Figure 3: Reasoning vs Non-Reasoning** ✅ DONE
   - 2 panels: CoT models (flat curves) vs standard (varied decay)

4. **Figure 4: RI vs PI Comparison** 🔄 TO CREATE
   - Panel A: Scatter (RIES vs PIS)
   - Panel B: Bland-Altman (agreement)
   - Panel C: Asymmetry (RI - PI) by model
   - Panel D: Decay patterns overlaid

**Supplementary (6-8 figures):**
- S1: All individual model decay curves
- S2: Family-level analysis (Claude, GPT, Llama, etc.)
- S3: MoE vs Dense (when more data available)
- S4: Mitigation strategy results
- S5: Error analysis (types of errors by interference level)
- S6: Residual plots (regression diagnostics)

### Tables for Paper

**Main Text (6-8 tables):**

1. **Table 1: Model specifications** ✅ DONE
   - 39 models: family, size, context, RIES

2. **Table 2: Regression results** ✅ DONE
   - Coefficients, t-stats, p-values, R²

3. **Table 3: Group comparisons** ✅ DONE
   - Reasoning vs Non-Reasoning, Size tiers

4. **Table 4: Decay pattern distribution** ✅ DONE
   - Exponential, polynomial, log-quad, power

5. **Table 5: RI vs PI comparison** 🔄 TO CREATE
   - Correlations, asymmetry tests

6. **Table 6: Mitigation strategies** 🔄 TO CREATE
   - Effect sizes for each strategy

7. **Table 7: Top/Bottom performers** ✅ DONE
   - Top 10 and Bottom 10 models

8. **Table 8: Comparison with "Unable to Forget" paper** ✅ DONE
   - Side-by-side comparison of findings

---

## 🎯 Timeline (Suggested)

### Week 1: PI Experiments
- **Days 1-2:** Run PI experiments on all 39 models (single run)
- **Days 3-4:** Export PI results to Excel, calculate PIS
- **Days 5-7:** Analyze RI vs PI, create comparison figures

### Week 2: Mitigation Strategies
- **Days 1-3:** Test explicit memory instructions (10 models × 3 strategies)
- **Days 4-5:** Test CoT prompting for non-CoT models
- **Days 6-7:** Analyze mitigation results, create figures

### Week 3: Multiple Runs
- **Days 1-5:** Run 3-5 trials per model (RI and PI)
- **Days 6-7:** Calculate confidence intervals, update all figures

### Week 4: Paper Finalization
- **Days 1-2:** Generate all final figures and tables
- **Days 3-4:** Convert paper draft to LaTeX
- **Days 5-6:** Write discussion section incorporating new findings
- **Day 7:** Proofread, format references, submit!

---

## 📝 Scripts to Create

### 1. `compare_ri_vs_pi.py`
```python
# Load RI and PI results
# Calculate correlations, asymmetry tests
# Generate comparison visualizations
```

### 2. `test_mitigation_strategies.py`
```python
# Run experiments with modified prompts
# Compare control vs treatments
# Calculate effect sizes
```

### 3. `calculate_confidence_intervals.py`
```python
# Aggregate multiple runs
# Calculate mean ± std for each model
# Generate error bars for plots
```

### 4. `generate_paper_figures.py`
```python
# Create all 10-12 figures at publication quality
# High DPI (300-600), journal formatting
# Save as PDF and PNG
```

---

## 🎓 Key Questions to Answer

### RQ4: RI vs PI Comparison
- Do models show asymmetric interference? (RI > PI or PI > RI?)
- Are RIES and PIS correlated? (r > 0.7 = same capacity)
- Do reasoning models resist both equally?

### RQ5: Mitigation Effectiveness
- Can explicit instructions reduce RI? (expected: +10-20%)
- Does CoT prompting help non-CoT models? (expected: +5-10%)
- Which strategy is most effective?

### RQ6: Variability Across Runs
- How much do RIES scores vary across runs?
- Do small models show more variability?
- Are top performers consistently top?

---

## 📦 Deliverables

### For Paper Submission

1. **Main manuscript** (LaTeX, 8-10K words)
   - Abstract, Intro, Methods, Results, Discussion, Conclusion
   - 4 main figures
   - 6-8 main tables

2. **Supplementary materials**
   - 6-8 supplementary figures
   - Full model specifications table
   - Complete dataset description
   - Analysis scripts (GitHub repo link)

3. **Data availability**
   - Raw experiment results (JSON files)
   - Processed data (Excel files)
   - Analysis code (Python scripts)
   - All available on GitHub or OSF

### For Repository

1. **Documentation**
   - README with clear instructions
   - Paper preprint (arXiv)
   - Experiment replication guide

2. **Code**
   - All experiment scripts
   - All analysis scripts
   - Requirements.txt
   - Docker container (optional, for reproducibility)

---

## 🚀 How to Execute

### Priority Order

**High Priority (Required for paper):**
1. ✅ PI experiments (all 39 models, single run)
2. ✅ RI vs PI analysis script
3. ✅ Generate RI vs PI comparison figures

**Medium Priority (Strengthen paper):**
4. ⬜ Mitigation strategies (10 models, 2-3 strategies)
5. ⬜ Multiple runs for error bars (3 runs × 39 models)

**Low Priority (Nice to have):**
6. ⬜ Additional MoE models (if APIs accessible)
7. ⬜ Cross-lingual tests (English → Spanish, etc.)
8. ⬜ Mechanistic probing (attention patterns)

### Commands to Run

```bash
# 1. PI Experiments
cd /Users/schatta/Downloads/personal/areas_of_work/llm_interference_framework
./scripts/shell/run_all_models_pi.sh 1

# 2. Export PI results
python scripts/analysis/export_results_to_excel.py --input data/raw_experiments_pi --output data/pi_results.xlsx

# 3. Calculate PIS scores
python scripts/analysis/calculate_ries.py --input data/pi_results.xlsx --max-level 300 --output results/key_results/pis_analysis.xlsx

# 4. Compare RI vs PI
python scripts/analysis/compare_ri_vs_pi.py --ri results/key_results/ries_analysis.xlsx --pi results/key_results/pis_analysis.xlsx

# 5. Generate final figures
python scripts/analysis/generate_paper_figures.py

# 6. Convert to LaTeX
# Manual: Copy docs/PAPER_DRAFT.md → paper.tex
```

---

## 💡 Tips for Success

1. **Start with PI experiments ASAP**
   - Longest-running task (6-8 hours)
   - Can run overnight

2. **Parallelize where possible**
   - PI experiments (run in background)
   - Paper writing (work on intro/discussion while experiments run)

3. **Track progress**
   - Use todo list to check off completed tasks
   - Document any unexpected findings

4. **Budget time for failures**
   - Some models will hit rate limits
   - Some API keys may expire
   - Plan for 10-20% retry rate

5. **Focus on core findings**
   - Don't get distracted by minor details
   - Paper deadline > perfect analysis

---

## ✅ Success Criteria

**Paper is ready to submit when:**
- [ ] All 4 main RQs answered (RI predictors, CoT effect, decay patterns, RI vs PI)
- [ ] All 4 main figures generated
- [ ] All 6-8 main tables complete
- [ ] Abstract written (200-250 words)
- [ ] Discussion section addresses limitations
- [ ] References formatted (40-60 papers)
- [ ] Supplementary materials prepared
- [ ] Code repository public and documented
- [ ] LaTeX compiles without errors

**Target venues:**
- NeurIPS 2025 (June deadline)
- ICLR 2026 (October deadline)
- EMNLP 2025 (May deadline)
- Transactions on Machine Learning Research (rolling)

---

**Last Updated:** December 30, 2024
**Next Review:** After PI experiments complete
