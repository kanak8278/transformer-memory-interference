# LLM Interference Framework - Project Organization

## Directory Structure

### 1. Scripts (`scripts/`)

#### Shell Scripts (`scripts/shell/`)
**Retroactive Interference (RI) Experiments:**
- `run_all_claude_gemini_models.sh` - Run RI on Claude & Gemini models
- `run_all_gpt_models.sh` - Run RI on GPT models
- `run_all_bedrock_models.sh` - Run RI on Bedrock models

**Proactive Interference (PI) Experiments:**
- `run_all_claude_gemini_models_pi.sh` - Run PI on Claude & Gemini models
- `run_all_gpt_models_pi.sh` - Run PI on GPT models
- `run_all_bedrock_models_pi.sh` - Run PI on Bedrock models

#### Core Scripts (`scripts/core/`)
- `run_interleaved_experiment.py` - Core RI experiment runner (queries FIRST values)
- `run_proactive_interference_experiment.py` - Core PI experiment runner (queries LAST values)

#### Analysis Scripts (`scripts/analysis/`)
- `calculate_ries.py` - Calculate RIES (Retroactive Interference Endurance Score)
- `calculate_pis.py` - Calculate PIS (Proactive Interference Score)
- `compare_ri_vs_pi.py` - Compare RI vs PI (correlation, asymmetry testing, plots)
- `export_results_to_excel.py` - Export raw results to Excel

### 2. Data (`data/`)

#### Raw Experiment Results
- `raw_ri_claude_gpt_gemini/` - RI results for Claude, GPT, Gemini (27 models)
- `raw_ri_bedrock/` - RI results for Bedrock models (29 models)
- `raw_pi_claude_gpt_gemini/` - PI results for Claude, GPT, Gemini (27 models)
- `raw_pi_bedrock/` - PI results for Bedrock models (29 models)

#### Processed Results
- `pi_experiment_results.xlsx` - Aggregated PI experiment results (56 models)
- `interleaved_dataset.json` - Base dataset for experiments
- `interleaved_dataset_meaningful.json` - Dataset with meaningful categories
- `bedrock_models_with_limits.csv` - Bedrock model configuration

### 3. Results (`results/key_results/`)

#### Analysis Results
- `ries_analysis.xlsx` - RIES scores for 39 models (levels 3-300)
- `pis_analysis.xlsx` - PIS scores for 56 models (levels 3-300)
- `model_categories.xlsx` - Model metadata and categorization
- `regression_analysis_results.xlsx` - Model size vs interference analysis

#### RI vs PI Comparison (`ri_vs_pi_comparison/`)
- `ries_vs_pis_scatter.png` - Scatter plot showing R²=0.044
- `bland_altman_plot.png` - Agreement plot showing systematic bias (RIES > PIS)
- `ri_vs_pi_comparison_table.xlsx` - 39 models with RIES, PIS, difference, ratio
- `merged_ri_pi_data.xlsx` - Merged dataset for analysis

### 4. Prompts (`prompts/`)
- `batch_interleaved_prompt.jinja` - Jinja2 template for RI experiments (asks for FIRST values)
- `batch_proactive_prompt.jinja` - Jinja2 template for PI experiments (asks for LAST values)

### 5. Datasets (`datasets/`)
- `interleaved_dataset_loader.py` - Dataset loader for experiments
- `category_response_parser.py` - Parser for model responses (handles both RI and PI formats)

## Experiment Pipeline

### Retroactive Interference (RI)
1. **Run:** `bash scripts/shell/run_all_claude_gemini_models.sh`
2. **Raw Data:** → `data/raw_ri_claude_gpt_gemini/`
3. **Export:** `python scripts/analysis/export_results_to_excel.py`
4. **Calculate:** `python scripts/analysis/calculate_ries.py`
5. **Results:** → `results/key_results/ries_analysis.xlsx`

### Proactive Interference (PI)
1. **Run:** `bash scripts/shell/run_all_claude_gemini_models_pi.sh`
2. **Raw Data:** → `data/raw_pi_claude_gpt_gemini/`
3. **Export:** `python scripts/analysis/export_results_to_excel.py` (needs PI mode)
4. **Calculate:** `python scripts/analysis/calculate_pis.py`
5. **Results:** → `results/key_results/pis_analysis.xlsx`

### RI vs PI Comparison
1. **Compare:** `python scripts/analysis/compare_ri_vs_pi.py`
2. **Results:** → `results/key_results/ri_vs_pi_comparison/`

## Key Findings (39 Models)

### Correlation Analysis
- **Pearson r:** +0.209 (p = 0.201, not significant)
- **R² = 0.044** (weak correlation - RI and PI measure different constructs)

### Asymmetry Test
- **Mean RIES:** 154.6 ± 20.1
- **Mean PIS:** 85.0 ± 39.4
- **Difference:** +69.5 ± 40.3 (RI > PI)
- **Effect size:** Cohen's d = 1.726 (large)
- **Significance:** t(38) = 10.780, p < 0.0001 ***
- **Direction:** 100% of models (39/39) show RIES > PIS

### Interpretation
Models are **significantly better at resisting proactive interference** than retroactive interference. This means:
- **PI (easier):** Recalling LAST-learned values
- **RI (harder):** Recalling FIRST-learned values when new information interferes

## Cleaned Files (2026-01-04)

### Removed
- ❌ `data/experiment_results_20251228_224744.xlsx` (old RI export)
- ❌ `data/experiment_results_20251231_132627.xlsx` (old RI export)
- ❌ `data/experiment_results_20260102_201645.xlsx` (old RI export)
- ❌ `data/pi_experiment_results.xlsx` (contained wrong RI data)
- ❌ `results/key_results/pis_analysis.xlsx` (calculated with wrong data)
- ❌ `data/bedrock_models.zip` (old archive)
- ❌ `data/raw_experiments/` (empty folder)
- ❌ All `.DS_Store` files

### Renamed
- ✅ `ri_vs_pi_comparison.xlsx/` → `ri_vs_pi_comparison/` (folder, not Excel file)
- ✅ `pis_analysis_corrected.xlsx` → `pis_analysis.xlsx` (now the standard)
- ✅ `pi_experiment_results_corrected.xlsx` → `pi_experiment_results.xlsx`
- ✅ `raw_experiments_ri_run/` → `raw_ri_claude_gpt_gemini/`
- ✅ `raw_experiments_ri_run_bedrock/` → `raw_ri_bedrock/`
- ✅ `raw_experiments_pi/` → `raw_pi_claude_gpt_gemini/`
- ✅ `raw_experiments_pi_bedrock/` → `raw_pi_bedrock/`

## Notes

### Model ID Consistency
- All model IDs normalized (removed "bedrock-" prefix)
- Ensures proper matching between RI and PI datasets
- 39 models with complete data (6 interference levels: 3, 10, 50, 100, 200, 300)

### Data Quality
- PI data now correctly reflects proactive interference (not RI data)
- All analysis files use correct normalized model IDs
- Comparison includes all 39 models with complete RI and PI data
