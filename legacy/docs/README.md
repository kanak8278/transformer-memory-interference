# LLM Retroactive Interference Evaluation Framework

A unified framework for evaluating retroactive interference (RI) effects across multiple Large Language Models (LLMs).

## Overview

This framework tests how well LLMs can recall initial information when presented with interfering updates. It supports:

- **5 LLM Providers**: Claude (Anthropic), Gemini (Google), GPT (OpenAI), Bedrock (AWS), Llama (Meta)
- **54+ Models**: Across all major providers
- **Flexible Testing**: Interference levels from 3 to 500 updates
- **Unified Interface**: Same code works across all providers

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API Keys

```bash
# Claude (Anthropic)
export ANTHROPIC_API_KEY="your_key_here"

# Gemini (Google AI Studio)
export GOOGLE_API_KEY="your_key_here"

# OpenAI/GPT
export OPENAI_API_KEY="your_key_here"

# AWS Bedrock (for Llama, Mistral, etc.)
aws configure  # or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
```

### 3. Run Experiment

```bash
# Test Claude Haiku at levels 3, 10, 50, 100
python scripts/core/run_interleaved_experiment.py \
    --model claude-3.5-haiku \
    --levels 3 10 50 100 \
    --dataset data/interleaved_dataset.json \
    --output-dir results/

# Test via AWS Bedrock
python scripts/core/run_interleaved_experiment.py \
    --model bedrock-llama-3.1-8b \
    --levels 3 10 50 100
```

### 4. View Results

Results are saved as JSON in the `results/` directory:

```json
{
  "model": "claude-3.5-haiku",
  "levels": [3, 10, 50, 100],
  "results": [
    {
      "interference_level": 3,
      "accuracy": 100.0,
      "correct_count": 46,
      "total_count": 46
    }
  ]
}
```

## Supported Models

### Claude (Anthropic)
- `claude-3.5-haiku` - Claude 3.5 Haiku (fast, cost-effective)
- `claude-3.5-sonnet` - Claude 3.5 Sonnet (balanced)
- `claude-3-opus` - Claude 3 Opus (powerful)

### Gemini (Google)
- `gemini-2.0-flash` - Gemini 2.0 Flash
- `gemini-2.5-pro` - Gemini 2.5 Pro
- `gemini-2.0-flash-lite` - Gemini 2.0 Flash Lite

### GPT (OpenAI)
- `gpt-4o` - GPT-4o
- `gpt-4o-mini` - GPT-4o Mini
- `o1` - OpenAI o1 (reasoning)
- `o3-mini` - OpenAI o3-mini

### AWS Bedrock
- `bedrock-llama-3.1-8b` - Llama 3.1 8B
- `bedrock-llama-3.1-70b` - Llama 3.1 70B
- `bedrock-nova-pro` - Amazon Nova Pro
- `bedrock-mistral-large` - Mistral Large
- `bedrock-claude-3.5-sonnet` - Claude via Bedrock

See `models/config.py` for the full list of supported models.

## Architecture

```
llm_interference_framework/
├── models/                    # Model interfaces
│   ├── base_model.py         # Abstract base class
│   ├── claude_model.py       # Anthropic Claude
│   ├── gemini_model.py       # Google Gemini
│   ├── openai_model.py       # OpenAI/GPT
│   ├── bedrock_model.py      # AWS Bedrock
│   ├── llama_model.py        # Meta Llama (local)
│   ├── config.py             # Model configurations
│   └── model_factory.py      # Auto-detection
│
├── datasets/                  # Data handling
│   ├── interleaved_dataset_loader.py
│   └── category_response_parser.py
│
├── scripts/
│   ├── core/                 # Main experiment scripts
│   │   ├── run_interleaved_experiment.py
│   │   └── run_proactive_interference_experiment.py
│   ├── analysis/             # Analysis & visualization
│   │   ├── calculate_ries.py
│   │   └── visualize_decay_patterns.py
│   └── shell/                # Batch scripts
│
├── data/                      # Dataset files
│   └── interleaved_dataset.json
│
├── results/                   # Output directory
└── docs/                      # Documentation
```

## How It Works

### 1. Dataset

The framework uses a category-value dataset with:
- **46 categories** (e.g., "visual art", "tools", "gemstone")
- **500+ values** per category (e.g., "Artist123", "Tool456")
- **8 interference levels**: 3, 10, 50, 100, 200, 300, 400, 500

### 2. Experiment Flow

```python
# 1. Create interleaved sequence
sequence = [
    "visual art: Artist123",  # BASELINE (remember this!)
    "tools: Tool456",
    "visual art: Artist999",  # Interference
    "tools: Tool888",
    # ... randomly shuffled
]

# 2. Ask model
"What was the INITIAL value for 'visual art'?"

# 3. Expected: "Artist123"
# 4. Evaluate: Correct / Incorrect / Missing
```

### 3. TRUE Random Interleaving

Unlike simple sequential updates, this framework uses **TRUE random interleaving**:

- All category-value pairs are shuffled randomly
- Baseline positions vary (not always first)
- More realistic cognitive load

## Advanced Usage

### Custom Configuration

```python
from models import create_model

# Create model with custom config
config = {
    'max_tokens': 8000,
    'temperature': 0.0,
}

model = create_model('bedrock-llama-3.1-8b', config)
```

### Running Multiple Models

```bash
# Run all Bedrock models
./scripts/shell/run_all_bedrock_models.sh

# Run all Claude/Gemini models
./scripts/shell/run_all_claude_gemini_models.sh
```

### Analysis Scripts

```bash
# Calculate RIES scores
python scripts/analysis/calculate_ries.py --input results/experiment_results.xlsx

# Visualize decay patterns
python scripts/analysis/visualize_decay_patterns.py

# Compare RI vs PI
python scripts/analysis/compare_ri_vs_pi.py
```

## Results Analysis

### Key Findings

1. **Retroactive Interference is Real**: All models show degraded performance as interference increases
2. **Model Size Matters**: Larger models handle interference better (R² = 0.49)
3. **Context Length Doesn't Help**: Context window size doesn't predict RI resistance

### RIES Score

The Retrospective Interference Endurance Score (RIES) measures a model's ability to resist retroactive interference:

- Higher RIES = better resistance to interference
- Calculated as area under the accuracy vs log(interference) curve
- Comparable to IES from "Unable to Forget" paper

## Troubleshooting

### "Model not available"
- Check API keys are set correctly
- Verify AWS credentials for Bedrock models
- Try different model name variations

### "Context length exceeded"
- Use models with larger context windows
- Test lower interference levels (3, 10, 50)
- Use batch mode to split large prompts

### "Rate limit exceeded"
- Add delays between API calls
- Reduce number of categories (`--sample-size 20`)
- Use exponential backoff (built-in)

## Contributing

### Adding a New Model

1. Create `models/your_model.py` inheriting from `BaseModelInterface`
2. Implement `_get_model_id()` and `generate()` methods
3. Add to `model_factory.py`
4. Test with `run_interleaved_experiment.py`

### Adding New Datasets

1. Create dataset in same format as `data/interleaved_dataset.json`
2. Update `datasets/interleaved_dataset_loader.py` if needed
3. Test with existing models

## License

MIT License - See LICENSE file

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{chattaraj2026retroactive,
  title={Retroactive Interference in Large Language Models: A Systematic Investigation},
  author={Chattaraj, Sourav and Jain, Arjit},
  journal={ICML},
  year={2026}
}
```

## Contact

For questions or issues, please open a GitHub issue.
