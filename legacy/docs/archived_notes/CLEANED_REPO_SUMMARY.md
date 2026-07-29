# ✅ Repository Cleanup Complete

## What Was Done

### 1. **Reorganized Structure**
```
llm_interference_framework/
├── data/                           # All experimental data
│   ├── experiment_results_20251228_224744.xlsx  (Main results - 54 models)
│   ├── raw_experiments/            (New experiment outputs go here)
│   └── bedrock_models_with_limits.csv
│
├── scripts/
│   ├── core/                       # Core experiment scripts
│   ├── analysis/                   # Analysis & RIES calculation
│   └── shell/                      # Batch execution (UPDATED)
│
├── results/
│   ├── key_results/                # For paper (3 Excel files)
│   ├── visualizations/             # For paper (4 main PNG files)
│   └── archived/                   # Old experiments
│
├── models/                         # Model implementations (unchanged)
├── datasets/                       # Dataset utilities (unchanged)
└── docs/                          # Documentation
```

### 2. **Updated Shell Scripts** ✨

All three batch scripts now support:
- **Multiple runs per model** for confidence intervals
- **Correct paths** (runs from project root automatically)
- **New output location**: `data/raw_experiments/`

**Usage:**
```bash
# Single run (default)
./scripts/shell/run_all_bedrock_models.sh

# Multiple runs for confidence intervals
./scripts/shell/run_all_bedrock_models.sh 3      # 3 runs per model
./scripts/shell/run_all_claude_gemini_models.sh 5   # 5 runs per model
./scripts/shell/run_all_gpt_models.sh 3
```

### 3. **Cleaned Up Files**

**Removed:**
- All `test_*.py` debugging scripts (8 files)
- All `debug_*.py` scripts (2 files)
- All `verify_*.py` scripts (3 files)
- `compare_*.py`, `diagnose_*.py` (3 files)
- Old markdown: `SAGEMAKER_BEDROCK_GUIDE.md`, `REPOSITORY_SUMMARY.md`, `EXPERIMENTS_ROADMAP.md`
- Duplicate scripts: `generate_meaningful_dataset_old.py`, `calculate_ies.py`
- Temp files: `~$*.xlsx`, `.DS_Store`

**Archived:**
- `results_interleaved/` → `results/archived/`
- `results_old/` → `results/archived/`
- `bedrock_responses_deepseek/` → `results/archived/`
- `ries_analysis_levels_3_400.xlsx` → `results/archived/`

### 4. **Key Files for Paper**

**Data:**
- `data/experiment_results_20251228_224744.xlsx` - Main experimental data

**Results:**
- `results/key_results/ries_analysis.xlsx` - 39 models, RIES scores
- `results/key_results/model_categories.xlsx` - Model categorization
- `results/key_results/regression_analysis_results.xlsx` - Statistical results

**Figures:**
- `results/visualizations/decay_curves_size_tiers.png` - Main 4-panel figure
- `results/visualizations/decay_curves_reasoning.png` - CoT comparison
- `results/visualizations/decay_curves_moe.png` - MoE vs Dense
- `results/visualizations/regression_analysis_size_context.png` - Regression plots

---

## How to Run Experiments

### Option 1: Run All Models (Batch)

```bash
# AWS Bedrock models (Llama, Claude, Nova, Mistral, etc.)
./scripts/shell/run_all_bedrock_models.sh 3

# Anthropic Claude + Google Gemini
./scripts/shell/run_all_claude_gemini_models.sh 3

# OpenAI GPT + O-series
./scripts/shell/run_all_gpt_models.sh 3
```

### Option 2: Run Single Model

```bash
python scripts/core/run_interleaved_experiment.py \
    --model claude-4.5-sonnet \
    --levels 3 10 50 100 200 300 \
    --output-dir data/raw_experiments
```

### Option 3: Analyze Existing Data

```bash
# Calculate RIES from existing data
python scripts/analysis/calculate_ries.py \
    --input data/experiment_results_20251228_224744.xlsx \
    --max-level 300

# Generate visualizations
python scripts/analysis/visualize_decay_patterns.py

# Run regression analysis
python scripts/analysis/regression_analysis_size_classes.py
```

---

## Multiple Runs for Confidence Intervals

To get error bars for your paper:

1. **Run each model 3-5 times:**
   ```bash
   ./scripts/shell/run_all_bedrock_models.sh 5
   ```

2. **Export all runs to Excel:**
   ```bash
   python scripts/analysis/export_results_to_excel.py
   ```

3. **Calculate mean ± std for RIES:**
   The calculate_ries.py script will automatically compute:
   - Mean RIES across runs
   - Standard deviation
   - 95% confidence intervals (for plotting error bars)

---

## File Counts

**Before cleanup:**
- 28+ Python scripts in root
- 5+ markdown docs
- Multiple result directories scattered

**After cleanup:**
- 0 Python scripts in root
- 7 core scripts in `scripts/core/` and `scripts/analysis/`
- 3 shell scripts in `scripts/shell/`
- 1 main README
- Clean organized structure

---

## Next Steps for Paper

1. **Run multiple experiments** (if you haven't):
   ```bash
   ./scripts/shell/run_all_bedrock_models.sh 3
   ./scripts/shell/run_all_claude_gemini_models.sh 3
   ./scripts/shell/run_all_gpt_models.sh 3
   ```

2. **Export and calculate RIES with confidence intervals:**
   ```bash
   python scripts/analysis/export_results_to_excel.py
   python scripts/analysis/calculate_ries.py --input data/experiment_results.xlsx --max-level 300
   ```

3. **Generate final figures:**
   ```bash
   python scripts/analysis/visualize_decay_patterns.py
   python scripts/analysis/regression_analysis_size_classes.py
   ```

4. **Use these files for paper:**
   - Figures: `results/visualizations/*.png`
   - Tables: `results/key_results/*.xlsx`
   - Stats: `results/key_results/regression_analysis_results.xlsx`

---

## Main Findings (Ready for Paper)

- **39 models** with complete data (levels 3-300)
- **RIES vs Model Size**: R² = 0.491, p < 0.0001 *** (SIGNIFICANT)
- **RIES vs Context Length**: R² = 0.003, p = 0.746 (NOT significant)
- **Reasoning models** (CoT): Mean RIES = 178.0 ± 12.6
- **Non-reasoning models**: Mean RIES = 150.3 ± 18.2
- **T-test**: p = 0.001 ** (Cohen's d = 1.76, large effect)

**Novel Finding:**
> Chain-of-thought reasoning models show significantly higher resistance to retroactive interference (p = 0.001), suggesting that extended inference processes may enhance memory robustness in large language models.

---

## Repository Status

✅ Cleaned and organized
✅ Shell scripts updated with correct paths
✅ Multiple runs support added
✅ Output directory standardized (`data/raw_experiments/`)
✅ Key results isolated for paper (`results/key_results/`, `results/visualizations/`)
✅ Old experiments archived
✅ Documentation updated

**Repository is now ready for:**
- Multiple experimental runs
- Confidence interval calculation
- Paper writing
- Reproducibility
