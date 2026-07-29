# Setup Guide

## Prerequisites

- Python 3.8+
- pip or conda
- Access to at least one LLM provider

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/llm-interference-framework.git
cd llm-interference-framework
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n llm-interference python=3.10
conda activate llm-interference
```

### 3. Install Dependencies

```bash
# Core dependencies (required)
pip install -r requirements.txt
```

### 4. Optional Dependencies

#### For AWS Bedrock
```bash
pip install boto3 botocore
```

#### For Local Llama Inference (GPU required)
```bash
pip install torch transformers bitsandbytes accelerate
```

#### For Jupyter Notebooks
```bash
pip install jupyter ipykernel
```

## API Key Setup

### Claude (Anthropic)

1. Get API key from: https://console.anthropic.com/
2. Set environment variable:
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```

### Gemini (Google)

**Option A: Google AI Studio (Recommended for quick start)**
1. Get API key from: https://aistudio.google.com/app/apikey
2. Set environment variable:
   ```bash
   export GOOGLE_API_KEY="AIza..."
   ```

**Option B: Vertex AI (For production)**
1. Create GCP project
2. Enable Vertex AI API
3. Create service account and download JSON key
4. Set environment variable:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
   export GOOGLE_CLOUD_PROJECT="your-project-id"
   ```

### OpenAI/GPT

**Option A: Standard OpenAI API**
1. Get API key from: https://platform.openai.com/api-keys
2. Set environment variable:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

**Option B: Azure OpenAI**
1. Create Azure OpenAI resource
2. Get endpoint and key from Azure portal
3. Set environment variables:
   ```bash
   export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
   export AZURE_OPENAI_API_KEY="your-key"
   ```

### AWS Bedrock (Llama, Claude, Mistral, etc.)

1. Configure AWS credentials:
   ```bash
   aws configure
   # OR set environment variables
   export AWS_ACCESS_KEY_ID="..."
   export AWS_SECRET_ACCESS_KEY="..."
   export AWS_DEFAULT_REGION="us-west-2"
   ```

2. Ensure you have Bedrock access enabled in your AWS account

### Llama (Hugging Face - for local inference)

1. Create account: https://huggingface.co/
2. Get token: https://huggingface.co/settings/tokens
3. Accept Llama model licenses (for gated models)
4. Set environment variable:
   ```bash
   export HF_TOKEN="hf_..."
   ```

## Verify Installation

```bash
# Test imports
python -c "from models import create_model; print('✅ Installation successful!')"

# Test model loading (requires API key)
python -c "
from models import create_model
model = create_model('claude-3.5-haiku')  # Or your preferred model
print(f'Model available: {model.is_available()}')
"
```

## Directory Structure

After setup, your directory should look like:

```
llm_interference_framework/
├── venv/                      # Virtual environment
├── models/
├── datasets/
├── data/
│   └── interleaved_dataset.json
├── results/                   # Will store experiment results
├── notebooks/
├── scripts/
│   ├── core/
│   └── analysis/
├── requirements.txt
├── README.md
└── SETUP.md
```

## Configuration

### Environment Variables

Create a `.env` file in the root directory:

```bash
# .env file
ANTHROPIC_API_KEY=sk-ant-your-key-here
GOOGLE_API_KEY=AIza-your-key-here
OPENAI_API_KEY=sk-your-key-here
HF_TOKEN=hf_your_token_here

# Optional: AWS Bedrock
AWS_REGION=us-west-2
AWS_PROFILE=your-profile

# Optional: Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key

# Optional: Google Cloud / Vertex AI
GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcp-credentials.json
GOOGLE_CLOUD_PROJECT=your-project-id
```

Load in your script:
```python
from dotenv import load_dotenv
load_dotenv()
```

### Model Configuration

Default configurations are in `models/base_model.py`:
- `max_tokens`: 2000 (8000 for Llama)
- `temperature`: 0.0 (deterministic)

Override when creating models:
```python
config = {
    'max_tokens': 8000,
    'temperature': 0.0,
}
model = create_model('llama-3.2-1b', config)
```

## Troubleshooting

### Import Errors

```bash
# Error: ModuleNotFoundError: No module named 'anthropic'
pip install anthropic

# Error: No module named 'google.generativeai'
pip install google-generativeai

# Error: No module named 'openai'
pip install openai
```

### API Key Issues

```bash
# Test if keys are set
python -c "import os; print('ANTHROPIC_API_KEY:', bool(os.getenv('ANTHROPIC_API_KEY')))"
python -c "import os; print('GOOGLE_API_KEY:', bool(os.getenv('GOOGLE_API_KEY')))"
python -c "import os; print('OPENAI_API_KEY:', bool(os.getenv('OPENAI_API_KEY')))"
```

### Permission Errors

```bash
# Hugging Face gated models
# Solution: Accept model license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
```

## Quick Test

Run a quick test to verify everything works:

```bash
# Test with 4 categories at Level 3
python scripts/core/run_interleaved_experiment.py \
    --model claude-3.5-haiku \
    --levels 3 \
    --sample-size 4
```

Expected output:
```
================================================================================
EXPERIMENT: claude-3.5-haiku, Level 3
================================================================================
...
Results:
  Correct: 4/4 (100.00%)
  Missing: 0/4 (0.00%)
```

## Next Steps

1. Read [README.md](../README.md) for usage instructions
2. Run experiments: `python scripts/core/run_interleaved_experiment.py --help`
3. For Level 200+ testing, see `notebooks/Llama_Interference_Experiments.ipynb`

## Support

For issues:
1. Check [Troubleshooting](#troubleshooting) section
2. Review logs in console output
3. Check API key permissions
4. Open GitHub issue with error details
