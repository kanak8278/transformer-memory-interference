"""
Semantic dataset generation for interference experiments.

Like dataset.py but uses semantically meaningful single-token values
(e.g., 'fruit variety: apple' instead of 'fruit variety: Fruit37').

Values are verified single-token in both Qwen2.5 and Gemma-3 tokenizers.
Categories with <4 values are excluded.

Usage:
    from core.semantic_dataset import generate_semantic_trial, SEMANTIC_CATEGORIES
    trial = generate_semantic_trial(num_keys=5, num_updates=3, condition="RI", seed=42)
"""

import json
import random
from pathlib import Path

from core.dataset import (
    InterferenceTrial,
    build_interleaved_sequence,
    build_prompt,
    format_for_chat,
)

# ── Load verified semantic values ────────────────────────────────────────────

_VALUES_FILE = Path(__file__).parent / "semantic_values_qwen_gemma.json"

with open(_VALUES_FILE) as _f:
    _DATA = json.load(_f)

# category -> list of single-token meaningful values
SEMANTIC_VALUES: dict[str, list[str]] = {
    cat: info["values"] for cat, info in _DATA["categories"].items()
}

# All available categories (sorted by pool size descending for stable ordering)
SEMANTIC_CATEGORIES: list[str] = sorted(
    SEMANTIC_VALUES.keys(),
    key=lambda c: len(SEMANTIC_VALUES[c]),
    reverse=True,
)

# Quick lookup: category -> pool size
SEMANTIC_POOL_SIZES: dict[str, int] = {
    cat: len(vals) for cat, vals in SEMANTIC_VALUES.items()
}


def get_eligible_categories(min_values: int) -> list[str]:
    """Return categories with at least `min_values` values.

    Sorted by pool size descending (largest pools first).
    """
    return [c for c in SEMANTIC_CATEGORIES if SEMANTIC_POOL_SIZES[c] >= min_values]


def generate_semantic_trial(
    num_keys: int,
    num_updates: int,
    condition: str,
    seed: int,
    test_category_idx: int = 0,
    categories: list[str] = None,
) -> InterferenceTrial:
    """Generate a single interference trial with semantic values.

    Same interface as dataset.generate_trial() but draws values from
    the semantic pool instead of synthetic prefixes.

    Args:
        num_keys: number of categories to use
        num_updates: number of value updates per category
        condition: "RI" or "PI"
        seed: random seed for reproducibility
        test_category_idx: which category (by index) to query
        categories: optional fixed category list. If None, randomly sampled
                    from categories that have enough values.

    Returns:
        InterferenceTrial with prompt and metadata

    Raises:
        ValueError: if num_updates exceeds available values for any category
    """
    rng = random.Random(seed)

    if categories is None:
        eligible = get_eligible_categories(num_updates)
        if len(eligible) < num_keys:
            raise ValueError(
                f"Need {num_keys} categories with >= {num_updates} values, "
                f"but only {len(eligible)} categories qualify. "
                f"Max updates for {num_keys}+ categories: "
                f"{_max_updates_for_n_keys(num_keys)}"
            )
        categories = rng.sample(eligible, num_keys)
    else:
        categories = categories[:num_keys]

    # Sample values from semantic pool for each category
    values_per_cat = {}
    for cat in categories:
        pool = SEMANTIC_VALUES[cat]
        if num_updates > len(pool):
            raise ValueError(
                f"Category '{cat}' has {len(pool)} values but "
                f"need {num_updates}. Use a category with more values."
            )
        values_per_cat[cat] = rng.sample(pool, num_updates)

    test_category = categories[test_category_idx % len(categories)]

    # Build interleaved sequence (reuse from dataset.py)
    sequence = build_interleaved_sequence(categories, values_per_cat, rng)

    # Build prompt (reuse from dataset.py)
    prompt, expected = build_prompt(sequence, condition, test_category)

    cat_values = [item["value"] for item in sequence if item["category"] == test_category]

    return InterferenceTrial(
        categories=categories,
        test_category=test_category,
        condition=condition,
        num_keys=num_keys,
        num_updates=num_updates,
        prompt=prompt,
        expected_answer=expected,
        initial_value=cat_values[0],
        final_value=cat_values[-1],
        all_values=cat_values,
        seed=seed,
    )


def _max_updates_for_n_keys(n: int) -> int:
    """Return the max num_updates possible if we need n categories."""
    sizes = sorted(SEMANTIC_POOL_SIZES.values(), reverse=True)
    if n > len(sizes):
        return 0
    # The bottleneck is the n-th largest pool
    return sizes[n - 1]


def preflight_context_check(
    num_keys: int,
    num_updates: int,
    tokenizer,
    context_limit: int,
    safety_margin: float = 0.90,
) -> tuple[bool, int]:
    """Check if a (num_keys, num_updates) cell fits within context."""
    trial = generate_semantic_trial(num_keys, num_updates, "RI", seed=0)
    formatted = format_for_chat(trial.prompt, tokenizer)
    tokens = tokenizer.encode(formatted)
    n_tokens = len(tokens)
    limit = int(context_limit * safety_margin)
    return n_tokens <= limit, n_tokens


def compute_feasible_grid(
    key_levels: list[int],
    update_levels: list[int],
    tokenizer,
    context_limit: int,
) -> dict[tuple[int, int], int]:
    """Pre-compute which grid cells are feasible for semantic dataset."""
    feasible = {}
    for nk in key_levels:
        for nu in update_levels:
            # Check pool size first (cheaper than tokenizing)
            eligible = get_eligible_categories(nu)
            if len(eligible) < nk:
                break
            ok, n_tokens = preflight_context_check(nk, nu, tokenizer, context_limit)
            if ok:
                feasible[(nk, nu)] = n_tokens
            else:
                break
    return feasible


# ── Grid definitions (constrained by semantic pool sizes) ────────────────────

# Max updates limited by pool sizes:
#   36 cats with >=4, 31 with >=9, 27 with >=10, 17 with >=15, 12 with >=20
KEY_LEVELS = [2, 3, 5, 7, 10, 15, 20, 25, 30, 36]

UPDATE_LEVELS = [
    3, 4, 5, 7, 10, 15, 20, 30,
]


# ── Summary helper ───────────────────────────────────────────────────────────

def print_pool_summary():
    """Print a summary of available categories and pool sizes."""
    print(f"Semantic value pool: {len(SEMANTIC_CATEGORIES)} categories, "
          f"{sum(SEMANTIC_POOL_SIZES.values())} total values\n")
    for cat in SEMANTIC_CATEGORIES:
        n = SEMANTIC_POOL_SIZES[cat]
        print(f"  {n:3d}  {cat}")
    print()
    for threshold in [4, 10, 15, 20, 30]:
        eligible = len(get_eligible_categories(threshold))
        print(f"  Categories with >= {threshold:2d} values: {eligible}")
    print(f"\n  Max grid capacity:")
    for nk in [2, 5, 10, 15, 20, 30, 36]:
        mu = _max_updates_for_n_keys(nk)
        print(f"    {nk:2d} keys → max {mu} updates/key")


if __name__ == "__main__":
    print_pool_summary()
    print("\n--- Sample trial ---")
    trial = generate_semantic_trial(num_keys=5, num_updates=5, condition="RI", seed=42)
    print(f"Categories: {trial.categories}")
    print(f"Test: {trial.test_category}")
    print(f"Expected: {trial.expected_answer}")
    print(f"Initial: {trial.initial_value}, Final: {trial.final_value}")
    print(f"All values: {trial.all_values}")
    print(f"\nPrompt:\n{trial.prompt}")
