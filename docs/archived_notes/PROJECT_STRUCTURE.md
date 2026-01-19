# LLM Retroactive Interference Framework - Project Structure

## Directory Overview

```
llm_interference_framework/
│
├── 📂 data/                          # Experimental Data
│   ├── experiment_results_20251228_224744.xlsx  (Main experiment results)
│   └── bedrock_models_with_limits.csv           (Model configurations)
│
├── 📂 scripts/                       # Executable Scripts
│   │
│   ├── 📂 core/                      # Core Experiment Scripts
│   │   ├── run_interleaved_experiment.py        (Main experiment runner)
│   │   └── generate_meaningful_dataset.py       (Dataset generator)
│   │
│   ├── 📂 analysis/                  # Analysis Scripts
│   │   ├── calculate_ries.py                    (Calculate RIES scores)
│   │   ├── test_decay_patterns.py               (Fit decay patterns)
│   │   ├── visualize_decay_patterns.py          (Generate decay curves)
│   │   ├── regression_analysis_size_classes.py  (Regression analysis)
│   │   └── export_results_to_excel.py           (Export utility)
│   │
│   └── 📂 shell/                     # Batch Scripts
│       ├── run_all_bedrock_models.sh
│       ├── run_all_claude_gemini_models.sh
│       └── run_all_gpt_models.sh
│
├── 📂 results/                       # Analysis Results
│   │
│   ├── 📂 key_results/               # Main Findings (For Paper)
│   │   ├── ries_analysis.xlsx                   (39 models RIES scores)
│   │   ├── model_categories.xlsx                (Model categorization)
│   │   └── regression_analysis_results.xlsx     (Statistical results)
│   │
│   ├── 📂 visualizations/            # Generated Plots (For Paper)
│   │   ├── decay_curves_size_tiers.png          (Figure: 4-panel XS/S/M/L)
│   │   ├── decay_curves_reasoning.png           (Figure: CoT vs non-CoT)
│   │   ├── decay_curves_moe.png                 (Figure: MoE vs Dense)
│   │   ├── regression_analysis_size_context.png (Figure: Regression)
│   │   ├── decay_curves/                        (Individual model curves)
│   │   ├── family_analysis/                     (Family comparisons)
│   │   ├── scatter_plots/                       (RIES correlations)
│   │   └── statistical_tests/                   (Statistical plots)
│   │
│   └── 📂 archived/                  # Old Experiments
│       ├── ries_analysis_levels_3_400.xlsx
│       ├── results_interleaved/
│       ├── results_old/
│       └── bedrock_responses_deepseek/
│
├── 📂 models/                        # Model Implementations
│   ├── config.py                               (Model configurations)
│   ├── bedrock_model.py                        (AWS Bedrock interface)
│   ├── gemini_model.py                         (Google Gemini interface)
│   └── openai_model.py                         (OpenAI interface)
│
├── 📂 datasets/                      # Dataset Utilities
│   ├── category_response_parser.py             (Response parser)
│   └── [other dataset files]
│
├── 📂 prompts/                       # Prompt Templates
│   └── [prompt files]
│
├── 📂 notebooks/                     # Jupyter Notebooks (Optional)
│   └── [analysis notebooks]
│
├── 📂 docs/                          # Documentation
│   ├── SETUP.md
│   └── USAGE.md
│
├── README.md                         # Main documentation
├── requirements.txt                  # Python dependencies
└── .gitignore                       # Git ignore rules
```

## Key Files for Paper

### Data
- `data/experiment_results_20251228_224744.xlsx` - Raw experimental data (54 models, 8 levels)

### Results (Main Findings)
- `results/key_results/ries_analysis.xlsx` - 39 models with RIES scores
- `results/key_results/model_categories.xlsx` - Model categorization (XS/S/M/L, CoT, MoE)
- `results/key_results/regression_analysis_results.xlsx` - Statistical analysis

### Figures
- `results/visualizations/decay_curves_size_tiers.png` - Main figure (4-panel)
- `results/visualizations/decay_curves_reasoning.png` - CoT comparison
- `results/visualizations/regression_analysis_size_context.png` - Regression plots

## Running Analyses

### 1. Calculate RIES (from data)
```bash
cd scripts/analysis
python calculate_ries.py --input ../../data/experiment_results_20251228_224744.xlsx --max-level 300
```

### 2. Generate Visualizations
```bash
python visualize_decay_patterns.py
```

### 3. Run Regression Analysis
```bash
python regression_analysis_size_classes.py
```

## Main Findings Summary

- **39 models** with complete data (levels 3-300)
- **RIES vs Size**: R² = 0.491, p < 0.0001 *** (size matters!)
- **RIES vs Context**: R² = 0.003, p = 0.746 (context doesn't matter)
- **Reasoning models**: RIES = 178.0 ± 12.6 (significantly better than non-reasoning)
- **Decay patterns**: Exponential (35%), Polynomial (30%), Log-Quadratic (26%)

## Cleaned Up

### Removed
- All `test_*.py` debugging scripts
- All `debug_*.py` scripts
- All `verify_*.py` scripts
- Old markdown docs (SAGEMAKER_BEDROCK_GUIDE.md, etc.)
- Temporary Excel files (~$*.xlsx)
- Old experiment scripts (generate_meaningful_dataset_old.py, calculate_ies.py)

### Archived
- Old experiment results moved to `results/archived/`
- Previous RIES calculations (with level 400)
