"""
Standardized output paths for experiment results.

Folder structure:
    results/
        {model_short}/
            behavioral_sweep.json          # Phase 1 (no operating point)
            {keys}k_{updates}u/
                logit_lens.json
                head_identification.json
                forced_attention.json
                ...

All experiment scripts should use get_output_path() to construct paths.
"""

import json
from pathlib import Path


RESULTS_ROOT = Path(__file__).parent.parent / "results"


def model_short_name(model: str) -> str:
    """Qwen/Qwen2.5-0.5B-Instruct -> Qwen2.5-0.5B-Instruct"""
    return model.split("/")[-1]


def get_output_dir(model: str, keys: int, updates: int) -> Path:
    """Get the output directory for a specific model + operating point.

    Returns: results/{model_short}/{keys}k_{updates}u/
    """
    d = RESULTS_ROOT / model_short_name(model) / f"{keys}k_{updates}u"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_output_path(model: str, keys: int, updates: int, experiment: str) -> Path:
    """Get the full output path for a specific experiment result.

    Args:
        model: Full model name (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
        keys: Number of categories
        updates: Number of values per category
        experiment: Experiment name (e.g., "logit_lens", "head_identification")

    Returns: results/{model_short}/{keys}k_{updates}u/{experiment}.json
    """
    return get_output_dir(model, keys, updates) / f"{experiment}.json"


def load_results(model: str, keys: int, updates: int, experiment: str) -> dict:
    """Load results JSON from the standard path. Raises FileNotFoundError if missing."""
    path = get_output_path(model, keys, updates, experiment)
    with open(path) as f:
        return json.load(f)


def load_head_identification(model: str, keys: int, updates: int) -> list[tuple[int, int]]:
    """Load primacy-biased heads from head_identification results.

    Returns list of (layer, head) tuples.
    """
    data = load_results(model, keys, updates, "head_identification")
    return [(h["layer"], h["head"]) for h in data["primacy_biased_heads"]]


def save_results(data: dict, model: str, keys: int, updates: int, experiment: str) -> Path:
    """Save results JSON and return the path."""
    path = get_output_path(model, keys, updates, experiment)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved to {path}")
    return path
