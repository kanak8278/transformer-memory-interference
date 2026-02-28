"""
Unified dataset configuration for interference experiments.

Single source of truth for all 4 dataset types. All notebooks and sweep scripts
should use this module for value generation. Replaces inline generation in
notebooks and the separate semantic_dataset.py.

Four dataset types organized by two features:
  - Token count: single-token vs multi-token
  - Semantic relation: values ARE category members vs random assignment

  ┌───────────────────────┬──────────────────────────┬──────────────────────────┐
  │                       │ Single-token             │ Multi-token              │
  ├───────────────────────┼──────────────────────────┼──────────────────────────┤
  │ No semantic relation  │ ARBITRARY_SINGLE         │ ARBITRARY_MULTI          │
  │                       │ gemstone: wire           │ gemstone: Gem42          │
  ├───────────────────────┼──────────────────────────┼──────────────────────────┤
  │ Real semantic relation│ SEMANTIC_SINGLE          │ SEMANTIC_MULTI           │
  │                       │ gemstone: ruby           │ gemstone: alexandrite    │
  └───────────────────────┴──────────────────────────┴──────────────────────────┘

Usage:
    from mechanistic_probing_v2.core.dataset_configs import (
        load_dataset_config,
        get_value_pool,
        get_eligible_categories,
        get_max_updates,
        generate_values_for_trial,
    )

    # Load config
    config = load_dataset_config("SEMANTIC_SINGLE")

    # Get categories that have enough values for your experiment
    cats = get_eligible_categories("SEMANTIC_SINGLE", min_values=20)
    # → 14 categories, each with 20+ single-token semantic values

    # Generate values for a trial (deterministic with seed)
    rng = random.Random(42)
    categories = rng.sample(cats, 5)
    values = generate_values_for_trial("SEMANTIC_SINGLE", categories, num_updates=15, rng=rng)

    # Or get the raw pool for a category
    pool = get_value_pool("SEMANTIC_SINGLE", category="gemstone")
    # → ["amber", "coral", "crystal", "diamond", "jade", "jet", "pearl", "quartz", "ruby", "turquoise"]

All data files live in core/:
  - single_token_values.py                  → ARBITRARY_SINGLE (2300 words)
  - semantic_values_qwen_gemma_deduped.json → SEMANTIC_SINGLE (791 values, 36 cats)
  - interleaved_dataset_meaningful.json     → SEMANTIC_MULTI (2403 values, 46 cats)
  - ARBITRARY_MULTI generated at runtime    → no file needed
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass




# ═══════════════════════════════════════════════════════════════════════════
# DATASET CONFIGS
# ═══════════════════════════════════════════════════════════════════════════

DATA_DIR = Path(__file__).parent / "data"

DATASET_CONFIGS = {
    "ARBITRARY_SINGLE": {
        "name": "Arbitrary Single-Token",
        "description": "Random English words assigned to any category. No key-value semantic link.",
        "token_type": "single",
        "semantic": False,
        "source": "data/arbitrary_single.json",
        "pool_type": "shared",
    },
    "ARBITRARY_MULTI": {
        "name": "Arbitrary Multi-Token (Synthetic)",
        "description": "Prefix+number values (Art375, Gem42). No real semantic content.",
        "token_type": "multi",
        "semantic": False,
        "source": "generated",
        "pool_type": "per_category",
    },
    "SEMANTIC_SINGLE": {
        "name": "Semantic Single-Token",
        "description": (
            "Real category members, verified single-token on Qwen2.5 and Gemma-3. "
            "ruby IS a gemstone. 36 categories, 791 values total."
        ),
        "token_type": "single",
        "semantic": True,
        "source": "data/semantic_single.json",
        "pool_type": "per_category",
    },
    "SEMANTIC_MULTI": {
        "name": "Semantic Multi-Token",
        "description": (
            "Real category members, multi-token. alexandrite IS a gemstone. "
            "46 categories, 2403 values total. From ACL paper dataset."
        ),
        "token_type": "multi",
        "semantic": True,
        "source": "data/semantic_multi.json",
        "pool_type": "per_category",
    },
}

VALID_DATASET_TYPES = list(DATASET_CONFIGS.keys())


# ═══════════════════════════════════════════════════════════════════════════
# VALUE POOL LOADERS (cached)
# ═══════════════════════════════════════════════════════════════════════════

_cached_pools = {}


def _load_arbitrary_single_pool():
    """Load the 2300 single-token English words from JSON."""
    cache_key = "arbitrary_single"
    if cache_key in _cached_pools:
        return _cached_pools[cache_key]

    path = DATA_DIR / "arbitrary_single.json"
    with open(path) as f:
        data = json.load(f)

    values = data["values"]
    _cached_pools[cache_key] = values
    return values


def _load_arbitrary_multi_prefix_map():
    """Load the prefix map from JSON."""
    cache_key = "arbitrary_multi_prefixes"
    if cache_key in _cached_pools:
        return _cached_pools[cache_key]

    path = DATA_DIR / "arbitrary_multi.json"
    with open(path) as f:
        data = json.load(f)

    _cached_pools[cache_key] = data["prefix_map"]
    return data["prefix_map"]


def _load_arbitrary_multi_pool(category, pool_size=500):
    """Generate synthetic Prefix+Number values for a category."""
    prefix_map = _load_arbitrary_multi_prefix_map()
    prefix = prefix_map.get(category, category.split()[0].capitalize())
    return [f"{prefix}{i}" for i in range(1, pool_size + 1)]


def _load_semantic_pool(name):
    """Load per-category semantic values from a clean JSON file.

    Works for both semantic_single.json and semantic_multi.json —
    same structure: {"categories": {"cat": {"values": [...], "count": N}, ...}}

    Returns dict: {category: [value1, value2, ...], ...}
    """
    if name in _cached_pools:
        return _cached_pools[name]

    path = DATA_DIR / f"{name}.json"
    with open(path) as f:
        data = json.load(f)

    pools = {}
    for cat, info in data["categories"].items():
        pools[cat] = info["values"]

    _cached_pools[name] = pools
    return pools


def _load_semantic_single_pool():
    """Load per-category single-token semantic values.

    36 categories, 791 total values (4-89 per category).
    Verified single-token on both Qwen2.5 and Gemma-3 tokenizers.
    """
    return _load_semantic_pool("semantic_single")


def _load_semantic_multi_pool():
    """Load per-category multi-token semantic values.

    46 categories, 2403 total values (45-70 per category).
    Extracted from ACL paper dataset.
    """
    return _load_semantic_pool("semantic_multi")


def _get_pools(dataset_type):
    """Internal helper to get the pool dict for a dataset type."""
    if dataset_type == "SEMANTIC_SINGLE":
        return _load_semantic_single_pool()
    elif dataset_type == "SEMANTIC_MULTI":
        return _load_semantic_multi_pool()
    else:
        raise ValueError(f"No per-category pools for {dataset_type}")


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def _validate_dataset_type(dataset_type):
    if dataset_type not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset type '{dataset_type}'. "
            f"Valid: {VALID_DATASET_TYPES}"
        )


def load_dataset_config(dataset_type):
    """Load a dataset configuration by type name.

    Args:
        dataset_type: One of "ARBITRARY_SINGLE", "ARBITRARY_MULTI",
                      "SEMANTIC_SINGLE", "SEMANTIC_MULTI"

    Returns:
        Config dict with name, description, token_type, semantic, source,
        pool_type, categories, pool_sizes, total_values.
    """
    _validate_dataset_type(dataset_type)
    config = DATASET_CONFIGS[dataset_type].copy()

    if dataset_type == "ARBITRARY_SINGLE":
        path = DATA_DIR / "arbitrary_single.json"
        with open(path) as f:
            data = json.load(f)
        pool = data["values"]
        categories = data["categories"]
        config["categories"] = categories
        config["pool_sizes"] = {cat: len(pool) for cat in categories}
        config["total_values"] = len(pool)

    elif dataset_type == "ARBITRARY_MULTI":
        prefix_map = _load_arbitrary_multi_prefix_map()
        categories = list(prefix_map.keys())
        config["categories"] = categories
        config["pool_sizes"] = {cat: 500 for cat in categories}
        config["total_values"] = 500 * len(categories)

    elif dataset_type in ("SEMANTIC_SINGLE", "SEMANTIC_MULTI"):
        pools = _get_pools(dataset_type)
        config["categories"] = sorted(
            pools.keys(),
            key=lambda c: len(pools[c]),
            reverse=True,
        )
        config["pool_sizes"] = {cat: len(vals) for cat, vals in pools.items()}
        config["total_values"] = sum(len(v) for v in pools.values())

    return config


def get_value_pool(dataset_type, category=None):
    """Get the value pool for a dataset type.

    Args:
        dataset_type: One of the four dataset types.
        category: Required for ARBITRARY_MULTI, SEMANTIC_SINGLE, SEMANTIC_MULTI.
                  Ignored for ARBITRARY_SINGLE (shared pool).

    Returns:
        List of value strings.
    """
    _validate_dataset_type(dataset_type)

    if dataset_type == "ARBITRARY_SINGLE":
        return _load_arbitrary_single_pool()

    elif dataset_type == "ARBITRARY_MULTI":
        if category is None:
            raise ValueError("ARBITRARY_MULTI requires a category argument")
        return _load_arbitrary_multi_pool(category)

    elif dataset_type in ("SEMANTIC_SINGLE", "SEMANTIC_MULTI"):
        pools = _get_pools(dataset_type)
        if category is None:
            return [v for vals in pools.values() for v in vals]
        if category not in pools:
            raise ValueError(
                f"Category '{category}' not in {dataset_type} "
                f"(has {len(pools)} categories)"
            )
        return pools[category]

    raise ValueError(f"Unknown dataset type: {dataset_type}")


def get_eligible_categories(dataset_type, min_values=1):
    """Get categories that have at least `min_values` values in their pool.

    For ARBITRARY datasets, all categories qualify (pools are large/unlimited).
    For SEMANTIC datasets, filters to categories with sufficient pool size.

    Args:
        dataset_type: One of the four dataset types.
        min_values: Minimum number of values required per category.

    Returns:
        List of category names, sorted by pool size descending (largest first).
    """
    _validate_dataset_type(dataset_type)

    if dataset_type == "ARBITRARY_SINGLE":
        config = load_dataset_config(dataset_type)
        pool_size = config["total_values"]
        if min_values > pool_size:
            return []
        return list(config["categories"])

    elif dataset_type == "ARBITRARY_MULTI":
        if min_values > 500:
            return []
        config = load_dataset_config(dataset_type)
        return list(config["categories"])

    elif dataset_type in ("SEMANTIC_SINGLE", "SEMANTIC_MULTI"):
        pools = _get_pools(dataset_type)
        eligible = [
            cat for cat, vals in pools.items()
            if len(vals) >= min_values
        ]
        return sorted(eligible, key=lambda c: len(pools[c]), reverse=True)

    return []


def get_pool_sizes(dataset_type):
    """Get per-category pool sizes.

    Returns:
        Dict mapping category → number of available values, sorted descending.
    """
    _validate_dataset_type(dataset_type)
    config = load_dataset_config(dataset_type)
    return config["pool_sizes"]


def get_max_updates(dataset_type, num_keys):
    """Get the maximum number of updates possible for a given key count.

    Smart calculation: assumes you pick the BEST categories (largest pools),
    not the worst. This matches how generate_values_for_trial works when
    categories are auto-selected via get_eligible_categories.

    Args:
        dataset_type: One of the four dataset types.
        num_keys: Number of categories needed.

    Returns:
        Maximum updates per category. 0 if not enough categories exist.
    """
    _validate_dataset_type(dataset_type)

    if dataset_type == "ARBITRARY_SINGLE":
        pool = _load_arbitrary_single_pool()
        return len(pool) // num_keys

    elif dataset_type == "ARBITRARY_MULTI":
        return 500

    elif dataset_type in ("SEMANTIC_SINGLE", "SEMANTIC_MULTI"):
        pools = _get_pools(dataset_type)
        sizes = sorted([len(v) for v in pools.values()], reverse=True)
        if num_keys > len(sizes):
            return 0
        # The bottleneck is the num_keys-th largest pool (the smallest
        # among the best categories you'd pick)
        return sizes[num_keys - 1]

    return 0


def generate_values_for_trial(dataset_type, categories, num_updates, rng):
    """Generate values for each category in a trial.

    Deterministic given the same rng state. Use with random.Random(seed).

    Args:
        dataset_type: One of the four types.
        categories: List of category names to generate values for.
                    For SEMANTIC datasets, use get_eligible_categories() first
                    to ensure all categories have enough values.
        num_updates: Number of unique values per category.
        rng: random.Random instance for reproducibility.

    Returns:
        Dict mapping category → list of values (length = num_updates).

    Raises:
        ValueError: if any category doesn't have enough values.
    """
    _validate_dataset_type(dataset_type)
    values_per_cat = {}

    if dataset_type == "ARBITRARY_SINGLE":
        pool = _load_arbitrary_single_pool()
        total_needed = len(categories) * num_updates
        if total_needed > len(pool):
            raise ValueError(
                f"Need {total_needed} values but ARBITRARY_SINGLE pool has "
                f"{len(pool)}. Reduce keys×updates or use ARBITRARY_MULTI."
            )
        selected = rng.sample(pool, total_needed)
        idx = 0
        for cat in categories:
            values_per_cat[cat] = selected[idx:idx + num_updates]
            idx += num_updates

    elif dataset_type == "ARBITRARY_MULTI":
        for cat in categories:
            pool = _load_arbitrary_multi_pool(cat)
            values_per_cat[cat] = rng.sample(pool, num_updates)

    elif dataset_type in ("SEMANTIC_SINGLE", "SEMANTIC_MULTI"):
        pools = _get_pools(dataset_type)
        for cat in categories:
            pool = pools.get(cat, [])
            if len(pool) < num_updates:
                eligible = get_eligible_categories(dataset_type, min_values=num_updates)
                raise ValueError(
                    f"{dataset_type}: category '{cat}' has {len(pool)} values, "
                    f"need {num_updates}. Use get_eligible_categories("
                    f"'{dataset_type}', min_values={num_updates}) to get "
                    f"valid categories ({len(eligible)} available)."
                )
            values_per_cat[cat] = rng.sample(pool, num_updates)

    return values_per_cat


# ═══════════════════════════════════════════════════════════════════════════
# TRIAL GENERATION (unified interface)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class InterferenceTrial:
    """A single interference trial with all metadata."""
    categories: list
    test_category: str
    condition: str          # "RI" or "PI"
    num_keys: int
    num_updates: int
    prompt: str
    expected_answer: str
    initial_value: str
    final_value: str
    all_values: list        # all values for test_category in sequence order
    seed: int = 0
    dataset_type: str = ""


SYSTEM_PROMPT = "Answer with ONLY the exact value. No explanation."


def _shuffle_no_consecutive(items, rng, max_attempts=100):
    """Shuffle items so no two consecutive items share the same category."""
    for _ in range(max_attempts):
        candidate = items.copy()
        rng.shuffle(candidate)
        ok = all(
            candidate[i]["category"] != candidate[i - 1]["category"]
            for i in range(1, len(candidate))
        )
        if ok:
            return candidate

    # Fallback: greedy placement
    remaining = items.copy()
    rng.shuffle(remaining)
    result = []
    last_cat = None
    while remaining:
        valid = [i for i, item in enumerate(remaining) if item["category"] != last_cat]
        if not valid:
            result.extend(remaining)
            break
        idx = rng.choice(valid)
        item = remaining.pop(idx)
        result.append(item)
        last_cat = item["category"]
    return result


def generate_trial(
    dataset_type,
    num_keys,
    num_updates,
    condition,
    seed,
    test_category_idx=0,
    categories=None,
    use_chat_format=True,
    tokenizer=None,
):
    """Generate a single interference trial.

    This is the unified entry point. It:
    1. Selects categories (filtering by pool size for semantic datasets)
    2. Generates values via generate_values_for_trial
    3. Builds an interleaved sequence (no consecutive same-category)
    4. Constructs the prompt (chat or completion format)

    Args:
        dataset_type: One of the four dataset types.
        num_keys: Number of categories.
        num_updates: Number of value updates per category.
        condition: "RI" (recall first) or "PI" (recall last).
        seed: Random seed for full reproducibility.
        test_category_idx: Which category index to query (default 0).
        categories: Optional fixed category list. If None, auto-selected.
        use_chat_format: If True, uses chat template. If False, raw prompt.
        tokenizer: Required if use_chat_format=True and tokenizer has
                   apply_chat_template.

    Returns:
        InterferenceTrial with prompt and all metadata.
    """
    _validate_dataset_type(dataset_type)
    rng = random.Random(seed)

    # Select categories
    if categories is None:
        eligible = get_eligible_categories(dataset_type, min_values=num_updates)
        if len(eligible) < num_keys:
            raise ValueError(
                f"Need {num_keys} categories with >= {num_updates} values in "
                f"{dataset_type}, but only {len(eligible)} qualify. "
                f"Max updates for {num_keys} keys: "
                f"{get_max_updates(dataset_type, num_keys)}"
            )
        categories = rng.sample(eligible, num_keys)
    else:
        categories = list(categories[:num_keys])

    # Generate values
    values_per_cat = generate_values_for_trial(
        dataset_type, categories, num_updates, rng
    )

    # Build interleaved sequence
    items = []
    for cat in categories:
        for idx, val in enumerate(values_per_cat[cat]):
            items.append({"category": cat, "value": val, "update_idx": idx})
    sequence = _shuffle_no_consecutive(items, rng)

    # Select test category
    test_category = categories[test_category_idx % len(categories)]

    # Build prompt
    stream_lines = [f"{item['category']}: {item['value']}" for item in sequence]
    stream_text = "\n".join(stream_lines)
    query_word = "first" if condition == "RI" else "last"

    cat_values = [
        item["value"] for item in sequence
        if item["category"] == test_category
    ]
    expected = cat_values[0] if condition == "RI" else cat_values[-1]

    raw_prompt = (
        f"Read the following key-value stream. "
        f"Each key gets updated multiple times.\n\n"
        f"{stream_text}\n\n"
        f"What was the {query_word} value of {test_category}?"
    )

    # Apply chat template if requested
    if use_chat_format and tokenizer and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": raw_prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = raw_prompt

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
        dataset_type=dataset_type,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY / DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════

def print_dataset_summary(dataset_type=None):
    """Print summary of one or all dataset configurations."""
    types = [dataset_type] if dataset_type else VALID_DATASET_TYPES

    for dt in types:
        config = load_dataset_config(dt)
        print(f"\n{'=' * 70}")
        print(f"  {dt}: {config['name']}")
        print(f"{'=' * 70}")
        print(f"  Description:  {config['description']}")
        print(f"  Token type:   {config['token_type']}")
        print(f"  Semantic:     {config['semantic']}")
        print(f"  Source:       {config['source']}")
        print(f"  Pool type:    {config['pool_type']}")
        print(f"  Categories:   {len(config['categories'])}")
        print(f"  Total values: {config['total_values']}")

        # Pool size distribution
        sizes = sorted(config["pool_sizes"].values())
        print(f"  Pool range:   {sizes[0]}-{sizes[-1]} per category "
              f"(mean={sum(sizes)/len(sizes):.0f})")

        # Sample
        if dt in ("SEMANTIC_SINGLE", "SEMANTIC_MULTI"):
            pools = _get_pools(dt)
            sample_cat = "gemstone" if "gemstone" in pools else config["categories"][0]
            print(f"  Sample ({sample_cat}): {pools[sample_cat][:5]}")
        elif dt == "ARBITRARY_SINGLE":
            pool = _load_arbitrary_single_pool()
            print(f"  Sample:       {pool[:5]}")
        elif dt == "ARBITRARY_MULTI":
            print(f"  Sample (gem): {_load_arbitrary_multi_pool('gemstone')[:5]}")

        # Eligible categories at various thresholds
        print(f"  Eligible categories:")
        for threshold in [5, 10, 15, 20, 30, 50]:
            n = len(get_eligible_categories(dt, min_values=threshold))
            if n == 0:
                break
            print(f"    >= {threshold:3d} values: {n} categories")

        # Max updates for common key counts
        print(f"  Max updates (smart, picks best categories):")
        for nk in [2, 5, 10, 20]:
            mx = get_max_updates(dt, nk)
            if mx == 0:
                break
            print(f"    {nk:2d} keys → {mx} updates")

        print()


if __name__ == "__main__":
    print_dataset_summary()
