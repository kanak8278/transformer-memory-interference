# Usage Guide

## Basic Usage

### Running a Single Experiment

```bash
python run_interleaved_experiment.py \
    --model claude-haiku \
    --levels 3 10 50 100 \
    --dataset data/interleaved_dataset.json \
    --output-dir results/
```

**Parameters**:
- `--model`: Model name (see [Supported Models](#supported-models))
- `--levels`: Interference levels to test (space-separated)
- `--sample-size`: Number of categories (default: 46)
- `--dataset`: Path to dataset JSON file
- `--output-dir`: Directory to save results

### Quick Test

Test with 4 categories at Level 3 (takes ~30 seconds):

```bash
./scripts/quick_test.sh claude-haiku
```

## Supported Models

### Claude (Anthropic)
```bash
# Fast and cost-effective
python run_interleaved_experiment.py --model claude-haiku --levels 3 10 50 100

# Balanced performance
python run_interleaved_experiment.py --model claude-sonnet --levels 3 10 50 100

# Most powerful
python run_interleaved_experiment.py --model claude-opus --levels 3 10 50
```

### Gemini (Google)
```bash
# Gemini 1.5 Pro
python run_interleaved_experiment.py --model gemini-pro --levels 3 10 50 100

# Gemini 1.5 Flash (faster)
python run_interleaved_experiment.py --model gemini-flash --levels 3 10 50 100

# Gemini 2.5 Pro (latest)
python run_interleaved_experiment.py --model gemini-2.5-pro --levels 3 10 50 100
```

### GPT (OpenAI)
```bash
# GPT-3.5 Turbo
python run_interleaved_experiment.py --model gpt-35-turbo --levels 3 10 50 100

# GPT-4
python run_interleaved_experiment.py --model gpt-4 --levels 3 10 50

# GPT-4 Turbo
python run_interleaved_experiment.py --model gpt-4-turbo --levels 3 10 50 100
```

### Llama (Meta)
```bash
# Llama 3.2 1B (fast, 128K context)
export HF_TOKEN="your_token"
python run_interleaved_experiment.py --model llama-3.2-1b --levels 3 10 50 100

# Llama 3.1 8B (128K context)
python run_interleaved_experiment.py --model llama-3.1-8b --levels 3 10 50 100

# Llama 3 70B (8K context)
python run_interleaved_experiment.py --model llama-3-70b --levels 3 10 50
```

## Advanced Usage

### Custom Sample Size

Test with fewer categories (faster):

```bash
# Test with 10 categories
python run_interleaved_experiment.py \
    --model claude-haiku \
    --levels 3 10 50 \
    --sample-size 10
```

### Multiple Levels

Test multiple interference levels:

```bash
# Test all standard levels
python run_interleaved_experiment.py \
    --model claude-haiku \
    --levels 3 10 50 100 200 300 400
```

**Note**: Levels 200+ may exceed context limits for some models/APIs.

### Custom Configuration

Pass custom config to model:

```python
from models import create_model

config = {
    'max_tokens': 8000,       # Override default (2000)
    'temperature': 0.0,       # Deterministic output
    'access_method': 'hf_router',  # For Llama: 'hf_router', 'bedrock', 'local'
}

model = create_model('llama-3.2-1b', config)
response = model.generate("Your prompt here")
```

## Programmatic Usage

### Basic Example

```python
from models import create_model
from datasets.interleaved_dataset_loader import InterleavedDatasetLoader
from datasets.category_response_parser import CategoryResponseParser

# 1. Load dataset
loader = InterleavedDatasetLoader('data/interleaved_dataset.json')

# 2. Get categories
categories = loader.get_categories()[:4]  # First 4 categories

# 3. Create prompt for Level 10
prompt_data = loader.create_interleaved_prompt(
    categories=categories,
    interference_level=10
)

# 4. Initialize model
model = create_model('claude-haiku')

# 5. Generate response
response = model.generate(prompt_data['prompt'])

# 6. Parse response
extracted = CategoryResponseParser.parse_batch_response(response, categories)

# 7. Evaluate
for cat in categories:
    expected = prompt_data['expected_answers'][cat]
    extracted_val = extracted.get(cat)
    correct = extracted_val and extracted_val.lower() == expected.lower()
    print(f"{cat}: {'✅' if correct else '❌'} (Expected: {expected}, Got: {extracted_val})")
```

### Batch Processing Multiple Models

```python
from models import create_model

models_to_test = [
    'claude-haiku',
    'claude-sonnet',
    'gemini-flash',
    'gpt-35-turbo',
    'llama-3.2-1b',
]

results = {}

for model_name in models_to_test:
    print(f"\nTesting {model_name}...")

    model = create_model(model_name)
    if not model.is_available():
        print(f"⚠️ {model_name} not available")
        continue

    response = model.generate("What is 2+2?")
    results[model_name] = response
    print(f"Response: {response}")
```

## Output Format

Results are saved as JSON:

```json
{
  "experiment": "interleaved_retroactive_interference",
  "model": "claude-haiku",
  "levels": [3, 10, 50, 100],
  "sample_size": 46,
  "timestamp": "20241212_150000",
  "results": [
    {
      "model": "claude-haiku",
      "interference_level": 3,
      "accuracy": 100.0,
      "correct_count": 46,
      "total_count": 46,
      "missing_count": 0,
      "sequence_length": 138,
      "results": [
        {
          "category": "visual art",
          "expected": "Artist213",
          "extracted": "Artist213",
          "correct": true,
          "missing": false,
          "baseline_position": 14
        }
      ]
    }
  ]
}
```

## Thomson Reuters Workspace

### Running on TR Network

```bash
# Connect to TR VPN if remote
# Clone repository
git clone <repo-url>
cd llm_interference_framework

# Install dependencies
pip install -r requirements.txt

# Set API keys (Claude and Llama only - OpenAI/Gemini use internal)
export ANTHROPIC_API_KEY="sk-ant-..."
export HF_TOKEN="hf_..."

# Run experiments
python run_interleaved_experiment.py --model claude-haiku --levels 3 10 50 100
```

### Available Models in TR Workspace

| Model | Access | Notes |
|-------|--------|-------|
| Claude | External API | Requires ANTHROPIC_API_KEY |
| Gemini | Internal (Vertex AI) | Requires GCP credentials |
| GPT | Internal (Azure) | No key needed (uses TR token service) |
| Llama | External (HF Router) | Requires HF_TOKEN |

## Google Colab (Level 200+)

For testing Level 200+ (requires full 128K context):

1. Open `notebooks/Llama_Interference_Experiments.ipynb` in Colab
2. Enable T4 GPU
3. Upload dataset and parser files
4. Run all cells

See notebook for detailed instructions.

## Troubleshooting

### Model Not Available

```bash
# Check API keys
python -c "import os; print('ANTHROPIC_API_KEY:', bool(os.getenv('ANTHROPIC_API_KEY')))"

# Test model
python -c "
from models import create_model
model = create_model('claude-haiku')
print(f'Available: {model.is_available()}')
print(f'Info: {model.get_model_info()}')
"
```

### Context Length Exceeded

For Level 200+:
- Use Google Colab with local inference (full 128K context)
- Or reduce sample size: `--sample-size 20`
- Or test lower levels only: `--levels 3 10 50 100`

### Parsing Errors

Check actual response format:

```python
# Results include 'batch_response' field
import json
with open('results/your_results.json') as f:
    data = json.load(f)

# Check what the model actually returned
print(data['results'][0]['batch_response'])
```

## Examples

### Example 1: Compare Two Models

```bash
# Test Claude Haiku
python run_interleaved_experiment.py --model claude-haiku --levels 3 10 50 100

# Test GPT-3.5
python run_interleaved_experiment.py --model gpt-35-turbo --levels 3 10 50 100

# Compare results
ls -lh results/
```

### Example 2: Quick Validation

```bash
# Test 4 categories, Level 3, multiple models
for model in claude-haiku gemini-flash gpt-35-turbo llama-3.2-1b; do
    echo "Testing $model..."
    python run_interleaved_experiment.py \
        --model $model \
        --levels 3 \
        --sample-size 4
done
```

### Example 3: Full Evaluation

```bash
# Test all levels with one model
python run_interleaved_experiment.py \
    --model claude-sonnet \
    --levels 3 10 50 100 200 300 400 \
    --sample-size 46
```

## Tips

1. **Start Small**: Test with `--sample-size 4 --levels 3` first
2. **Check Context Limits**: Different models/APIs have different limits
3. **Monitor Costs**: Some APIs charge per token
4. **Save Results**: Results are automatically saved to `results/` directory
5. **Use Batch Mode**: Already enabled by default (1 API call per level)

## Next Steps

- Analyze results: See `results/` directory
- Compare models: Use same levels across models
- Test higher levels: Use Google Colab for Level 200+
- Visualize: Create plots from JSON results
