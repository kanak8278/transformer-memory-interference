# Contributing to LLM Retroactive Interference Framework

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/llm-interference-framework.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Install dependencies: `pip install -r requirements.txt`

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8
```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and small

## Adding a New Model

1. Create a new file in `models/` (e.g., `models/your_model.py`)
2. Inherit from `BaseModelInterface`:

```python
from .base_model import BaseModelInterface

class YourModelInterface(BaseModelInterface):
    def __init__(self, model_name: str, config=None):
        super().__init__(model_name, config)
        # Your initialization code

    def generate(self, prompt: str) -> str:
        # Your generation code
        pass

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update({
            "provider": "Your Provider",
            "model_id": self.model_id,
        })
        return info
```

3. Add model mappings to `models/config.py`
4. Register in `models/model_factory.py`
5. Test with the experiment script

## Adding New Analysis Scripts

Place new analysis scripts in `scripts/analysis/`. Follow the existing patterns:

```python
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Your analysis')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='output.xlsx')
    args = parser.parse_args()

    # Your analysis code

if __name__ == '__main__':
    main()
```

## Testing

```bash
# Run a quick test
python scripts/core/run_interleaved_experiment.py \
    --model claude-3.5-haiku \
    --levels 3 \
    --sample-size 4

# Test analysis scripts
python scripts/analysis/calculate_ries.py \
    --input data/experiment_results.xlsx \
    --output test_output.xlsx
```

## Submitting Changes

1. Ensure your code follows the style guidelines
2. Test your changes thoroughly
3. Update documentation if needed
4. Commit with clear messages:
   ```
   git commit -m "Add: new model support for XYZ"
   git commit -m "Fix: parsing issue in response handler"
   ```
5. Push to your fork: `git push origin feature/your-feature-name`
6. Open a Pull Request

## Pull Request Guidelines

- Provide a clear description of the changes
- Reference any related issues
- Include test results if applicable
- Keep PRs focused on a single feature/fix

## Reporting Issues

When reporting issues, please include:

1. Python version and OS
2. Steps to reproduce
3. Expected vs actual behavior
4. Error messages (if any)
5. Model being tested

## Questions?

Open an issue with the "question" label or reach out to the maintainers.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
