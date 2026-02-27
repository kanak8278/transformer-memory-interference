"""
Standardized output paths for experiment results.

Folder structure:
    results/
        {model_short}/
            behavioral_sweep.json          # Phase 1 (no operating point)
            {keys}k_{updates}u/
                logit_lens.json            # symlink → latest timestamped version
                logit_lens_20260227_031500.json
                head_identification.json
                head_identification_20260227_041200.json
                ...

All experiment scripts should use get_output_path() to construct paths.
Timestamped copies are saved alongside so re-runs never overwrite prior results.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path


RESULTS_ROOT = Path(__file__).parent.parent / "results"

# Default data setup for all mechanistic experiments.
# Behavioral sweep (Colab) uses "synthetic" with different pool.
DATA_SETUP = {
    "data_setup": "single_token_english",
    "value_pool_size": 2300,
    "category_source": "ORIGINAL_CATEGORIES_46",
}


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


def _timestamp_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def save_results(data: dict, model: str, keys: int, updates: int, experiment: str) -> Path:
    """Save results JSON with standard metadata and return the path.

    Saves TWO files:
      1. {experiment}_{timestamp}.json  — timestamped, never overwritten
      2. {experiment}.json              — always points to the latest run

    Automatically injects data_setup, value_pool_size, category_source,
    and timestamp into the saved data if not already present.
    """
    ts = _timestamp_str()

    # Inject standard metadata if missing
    if "data_setup" not in data and "config" not in data:
        data.update(DATA_SETUP)
        data["timestamp"] = datetime.now(timezone.utc).isoformat()
    elif "config" in data:
        for k, v in DATA_SETUP.items():
            if k not in data["config"]:
                data["config"][k] = v
        if "timestamp" not in data["config"]:
            data["config"]["timestamp"] = datetime.now(timezone.utc).isoformat()

    out_dir = get_output_dir(model, keys, updates)

    # 1. Timestamped file (archival — never overwritten)
    ts_path = out_dir / f"{experiment}_{ts}.json"
    with open(ts_path, "w") as f:
        json.dump(data, f, indent=2)

    # 2. Latest file (stable name for downstream code to read)
    latest_path = out_dir / f"{experiment}.json"
    with open(latest_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved to {ts_path}")
    print(f"  (latest: {latest_path})")
    return latest_path
