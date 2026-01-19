# LLM Interference Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A comprehensive framework for measuring **retroactive interference (RI)** and **proactive interference (PI)** in Large Language Models. This repository contains the code, data, and analysis scripts for our ICML 2026 paper:

> **"Unable to Remember: Retroactive Interference Reveals Memory Consolidation Limits in Large Language Models"**
>
> Sourav Chattaraj, Arjit Jain

## Key Findings

Our study of **54 LLMs** (39 with complete data for quantitative analysis) reveals:

| Finding | Details |
|---------|---------|
| **Opposite to Humans** | All 39 models show PI > RI effect (d = 1.73), opposite to human memory where RI dominates |
| **Substantial Degradation** | RI: 94% → 27% accuracy; PI: 85% → 2% accuracy at high interference |
| **Capacity > Context** | Model size predicts RI resistance (R² = 0.49), context length does not (R² ≈ 0) |
| **Distinct Mechanisms** | RI and PI are uncorrelated (R² = 0.04) with different failure modes |
| **Reasoning Trade-off** | O-series models show +18% RI advantage but 8.5× PI vulnerability |

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/llm-interference-framework.git
cd llm-interference-framework
pip install -r requirements.txt
```

### Set API Keys

```bash
export ANTHROPIC_API_KEY="your_key"      # For Claude
export OPENAI_API_KEY="your_key"         # For GPT/O-series models
export GOOGLE_API_KEY="your_key"         # For Gemini
aws configure                            # For AWS Bedrock
```

### Run an Experiment

```bash
# Run Retroactive Interference (RI) experiment
python scripts/core/run_interleaved_experiment.py \
    --model claude-3.5-haiku \
    --levels 3 10 50 100 200 300

# Run Proactive Interference (PI) experiment
python scripts/core/run_proactive_interference_experiment.py \
    --model claude-3.5-haiku \
    --levels 3 10 50 100 200 300
```

### Analyze Results

```bash
# Calculate RIES (Retroactive Interference Endurance Score)
python scripts/analysis/calculate_ries.py \
    --input results/experiment_results.xlsx

# Calculate PIES (Proactive Interference Endurance Score)
python scripts/analysis/calculate_pis.py \
    --input results/pi_experiment_results.xlsx

# Compare RI vs PI
python scripts/analysis/compare_ri_vs_pi.py

# Generate visualizations
python scripts/analysis/visualize_decay_patterns.py
```

## Project Structure

```
llm_interference_framework/
├── data/                        # Input datasets
│   ├── interleaved_dataset.json           # Main experiment dataset
│   ├── interleaved_dataset_meaningful.json # Human-readable version
│   └── bedrock_models_with_limits.csv     # Model configurations
│
├── results/                     # Experiment results
│   ├── key_results/             # Final processed results & figures
│   │   ├── ries_analysis.xlsx   # RIES scores for all models
│   │   ├── pies_analysis.xlsx   # PIES scores for all models
│   │   ├── *.png                # Publication figures
│   │   └── ri_vs_pi_comparison/ # Comparison tables
│   ├── raw_ri_bedrock/          # Raw RI responses (Bedrock models)
│   ├── raw_ri_claude_gpt_gemini/# Raw RI responses (Claude/GPT/Gemini)
│   ├── raw_pi_bedrock/          # Raw PI responses (Bedrock models)
│   ├── raw_pi_claude_gpt_gemini/# Raw PI responses (Claude/GPT/Gemini)
│   └── error_analysis/          # Error classification results
│
├── scripts/
│   ├── core/                    # Experiment runners
│   │   ├── run_interleaved_experiment.py           # RI experiments
│   │   ├── run_proactive_interference_experiment.py # PI experiments
│   │   └── generate_meaningful_dataset.py          # Dataset generation
│   ├── analysis/                # Analysis & visualization
│   │   ├── calculate_ries.py              # RIES calculation
│   │   ├── calculate_pis.py               # PIES calculation
│   │   ├── compare_ri_vs_pi.py            # RI vs PI comparison
│   │   ├── regression_analysis_size_classes.py # Size effect analysis
│   │   ├── visualize_decay_patterns.py    # Decay curve visualization
│   │   └── classify_error_types.py        # Error analysis
│   └── shell/                   # Batch execution scripts
│
├── models/                      # LLM interfaces
│   ├── bedrock_model.py         # AWS Bedrock (Llama, Mistral, Nova, etc.)
│   ├── claude_model.py          # Anthropic Claude
│   ├── openai_model.py          # OpenAI GPT/O-series
│   ├── gemini_model.py          # Google Gemini
│   └── config.py                # Model configurations
│
├── prompts/                     # Experiment prompt templates
│   ├── batch_interleaved_prompt.jinja     # RI prompt template
│   └── batch_proactive_prompt.jinja       # PI prompt template
│
├── docs/                        # Documentation & paper
│   ├── paper.tex                # ICML 2026 submission
│   └── references.bib           # Bibliography
│
└── datasets/                    # Dataset loaders
    └── interleaved_dataset_loader.py
```

## Experimental Paradigm

### Retroactive Interference (RI)
Tests **consolidation**: Can models retain access to initial encodings despite subsequent interference?
- Present 46 category-value pairs (initial learning)
- Present N interfering updates per category
- Query for **INITIAL** values
- Measures resistance to new information disrupting old

### Proactive Interference (PI)
Tests **recency access**: Can models retrieve the most recent information despite earlier competing content?
- Same sequence as RI
- Query for **LAST** values instead
- Measures resistance to old information disrupting new

## Metrics

### RIES (Retroactive Interference Endurance Score)
Area under the accuracy-vs-interference curve using log-scaled integration:
- **Higher RIES** = better resistance to retroactive interference
- Range: 0-200 (theoretical maximum at 100% accuracy across all levels)

### PIES (Proactive Interference Endurance Score)
Same calculation as RIES but for PI task:
- **Higher PIES** = better resistance to proactive interference
- **RIES > PIES** indicates PI is harder (true for all 39 tested models)

## Supported Models (54 Total)

### Reasoning Models (n=6)
`o1`, `o1-preview`, `o3`, `o3-mini`, `o4-mini`, `deepseek-r1`

### GPT Family (n=11)
`gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo`, `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano`

### Claude Family (n=7)
`claude-3-opus`, `claude-3.5-haiku`, `claude-4-sonnet`, `claude-4-opus`, `claude-4.5-haiku`, `claude-4.5-sonnet`, `claude-4.5-opus`

### Gemini Family (n=4)
`gemini-2.0-flash`, `gemini-2.0-flash-lite`, `gemini-2.5-pro`, `gemini-2.5-flash-lite`

### Llama Family (n=12)
`llama-3-8b`, `llama-3-70b`, `llama-3.1-8b`, `llama-3.1-70b`, `llama-3.1-405b`, `llama-3.2-1b`, `llama-3.2-3b`, `llama-3.2-11b`, `llama-3.2-90b`, `llama-3.3-70b`, `llama-4-scout-17b`, `llama-4-maverick-17b`

### Other Models (n=14)
Mistral, Nova, Titan, Qwen variants

See [models/config.py](models/config.py) for the complete list with specifications.

## Key Results Summary

| Metric | Value |
|--------|-------|
| Models tested | 54 total, 39 with complete data |
| Interference levels | 3, 10, 50, 100, 200, 300 |
| Mean RIES | 154.6 ± 20.1 |
| Mean PIES | 85.0 ± 39.4 |
| Top RI performer | o1, o1-preview, gpt-5 (RIES = 186.4) |
| Bottom RI performer | llama-3.2-3b (RIES = 113.6) |
| Size → RI correlation | R² = 0.49 (p < 0.0001) |
| Size → PI correlation | R² = 0.06 (n.s.) |
| RI-PI correlation | R² = 0.04 (uncorrelated) |
| Universal asymmetry | 39/39 models show RIES > PIES |

## Error Analysis

### Error Type Distribution (Our Data)

| Error Type | RI (N=5,409) | PI (N=9,753) | Interpretation |
|------------|--------------|--------------|----------------|
| Same-Key Interference | 46.0% | 56.1% | Returns value from same category but wrong position |
| Retrieval Failure | 50.8% | 42.5% | No value extracted (null/empty response) |
| Hallucination | 0.9% | 0.2% | Value never presented in prompt |
| Cross-Key Interference | 0.6% | 0.7% | Value from different category |
| Partial Match | 1.8% | 0.7% | Fuzzy match to presented value |

### Position Analysis (Primacy vs Recency Bias)

| Metric | RI | PI |
|--------|----|----|
| Same-Key Errors | 2,490 | 5,467 |
| Mean Error Position | 0.505 (late) | 0.488 |
| Errors from late half (>0.5) | 55.6% | 51.5% |
| Errors from early half (<0.5) | 44.4% | 48.5% |
| Most recent position errors | 1.4% | - |
| First position errors | - | 13.6% |

At Level 3: RI mean pos = 0.722 (late), PI mean pos = 0.121 (early) - confirms opposite biases!

### Key Insights

1. **Same-Key Interference Dominates PI** (56.1% vs 46.0%): PI shows stronger within-category interference
2. **Retrieval Failure Dominates RI** (50.8% vs 42.5%): RI more likely to produce no response
3. **Hallucination is Rare for Both** (<1%): Unlike "Unable to Forget" paper's higher rates
4. **Opposite Position Biases**: RI shows recency bias, PI shows primacy bias

### Comparison with "Unable to Forget" Paper

| Their Category | Our Equivalent | Notes |
|----------------|----------------|-------|
| PI-response | same_key_interference | Earlier value from same key |
| Off-keys | retrieval_failure | No response |
| Off-values | hallucination + partial_match + cross_key | Their broader "hallucination" |

**Hallucination Rate Discrepancy**: They report 25-45% "off-values" at extreme interference, we see <1%. This is because:
- Their "off-values" includes any value not in that key's stream (including cross-key)
- Our 5-category system is more granular
- Different experimental design (interleaved vs sequential updates)

## Citation

If you use this framework, please cite:

```bibtex
@inproceedings{chattaraj2026unable,
  title={Unable to Remember: Retroactive Interference Reveals Memory Consolidation Limits in Large Language Models},
  author={Chattaraj, Sourav and Jain, Arjit},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

This work builds upon the methodology introduced in:
- Liu et al. (2024). "Unable to Forget: Proactive Interference Reveals Working Memory Limits in LLMs"
