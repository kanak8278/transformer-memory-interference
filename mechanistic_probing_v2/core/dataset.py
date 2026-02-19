"""
Dataset generation for interference experiments.

Generates key-value streams with controlled interference levels.
Uses the same 46 categories from the original ACL paper.
Prompt format matches the original paper (no few-shot).

Interleaving: TRUE random interleaving (no consecutive same-category repeats),
matching the original paper methodology.
"""

import json
import random
import math
from pathlib import Path
from dataclasses import dataclass, field


# ── Categories from original ACL paper ──────────────────────────────────────

ORIGINAL_CATEGORIES = [
    "visual art", "tools", "landform", "musical instrument", "gemstone",
    "fabric", "tree species", "cheese variety", "architectural style",
    "cloud formation", "bird species", "culinary herb", "flower species",
    "wine variety", "dance style", "pasta shape", "literary genre",
    "cooking method", "mathematical concept", "weather phenomenon",
    "ocean current", "mineral type", "coffee variety", "telescope type",
    "martial art", "sea creature", "psychology term", "chemical element",
    "dinosaur genus", "programming language", "ancient civilization",
    "bridge type", "photography technique", "boat type", "cartoon character",
    "stadium name", "surgical procedure", "constellation", "spice blend",
    "guitar type", "hat style", "painting medium", "volcano name",
    "fruit variety", "sword type", "board game",
]


def _make_short_prefix(category: str) -> str:
    """Generate a short synthetic value prefix from category name.

    E.g., 'visual art' -> 'Art', 'tree species' -> 'Tree'
    """
    # Map categories to short prefixes for synthetic values
    PREFIX_MAP = {
        "visual art": "Art", "tools": "Tool", "landform": "Land",
        "musical instrument": "Inst", "gemstone": "Gem", "fabric": "Fab",
        "tree species": "Tree", "cheese variety": "Cheese",
        "architectural style": "Style", "cloud formation": "Cloud",
        "bird species": "Bird", "culinary herb": "Herb",
        "flower species": "Flower", "wine variety": "Wine",
        "dance style": "Dance", "pasta shape": "Pasta",
        "literary genre": "Genre", "cooking method": "Cook",
        "mathematical concept": "Math", "weather phenomenon": "Weather",
        "ocean current": "Current", "mineral type": "Mineral",
        "coffee variety": "Coffee", "telescope type": "Scope",
        "martial art": "Martial", "sea creature": "Sea",
        "psychology term": "Psych", "chemical element": "Elem",
        "dinosaur genus": "Dino", "programming language": "Lang",
        "ancient civilization": "Civ", "bridge type": "Bridge",
        "photography technique": "Photo", "boat type": "Boat",
        "cartoon character": "Toon", "stadium name": "Stadium",
        "surgical procedure": "Surg", "constellation": "Star",
        "spice blend": "Spice", "guitar type": "Guitar",
        "hat style": "Hat", "painting medium": "Medium",
        "volcano name": "Volcano", "fruit variety": "Fruit",
        "sword type": "Sword", "board game": "Game",
    }
    return PREFIX_MAP.get(category, category.split()[0].capitalize())


def generate_value_pool(category: str, pool_size: int = 500) -> list[str]:
    """Generate a pool of synthetic values for a category.

    Values are like 'Art1', 'Art2', ..., 'Art500'.
    Uses synthetic values (matching original paper) to allow arbitrary pool sizes.
    """
    prefix = _make_short_prefix(category)
    return [f"{prefix}{i}" for i in range(1, pool_size + 1)]


@dataclass
class InterferenceTrial:
    """A single interference trial with all metadata."""
    # Core
    categories: list[str]
    test_category: str
    condition: str  # "RI" or "PI"
    num_keys: int
    num_updates: int
    prompt: str
    expected_answer: str

    # All values for the test category (ordered by appearance)
    initial_value: str
    final_value: str
    all_values: list[str]  # all values assigned to test_category

    # Metadata for controls
    seed: int = 0
    prompt_tokens: int = 0  # filled after tokenization
    initial_value_position: int = 0  # token position of initial value in prompt
    final_value_position: int = 0  # token position of final value in prompt


def build_interleaved_sequence(
    categories: list[str],
    values_per_cat: dict[str, list[str]],
    rng: random.Random,
) -> list[dict]:
    """Build a truly random interleaved sequence (no consecutive same-category).

    Returns list of dicts: [{"category": ..., "value": ..., "update_idx": ...}, ...]
    """
    num_updates = len(next(iter(values_per_cat.values())))

    # Create all (category, value, update_idx) triples
    items = []
    for cat in categories:
        for idx, val in enumerate(values_per_cat[cat]):
            items.append({"category": cat, "value": val, "update_idx": idx})

    # Shuffle with constraint: no consecutive same-category
    # Use rejection sampling with a fallback to greedy placement
    shuffled = _shuffle_no_consecutive(items, rng, max_attempts=100)
    return shuffled


def _shuffle_no_consecutive(
    items: list[dict], rng: random.Random, max_attempts: int = 100
) -> list[dict]:
    """Shuffle items ensuring no two consecutive items share the same category.

    Falls back to greedy placement if pure random shuffle fails.
    """
    for _ in range(max_attempts):
        candidate = items.copy()
        rng.shuffle(candidate)
        if _no_consecutive_repeats(candidate):
            return candidate

    # Fallback: greedy placement
    remaining = items.copy()
    rng.shuffle(remaining)
    result = []
    last_cat = None

    while remaining:
        # Find items that don't match the last category
        valid = [i for i, item in enumerate(remaining) if item["category"] != last_cat]
        if not valid:
            # Can't avoid consecutive — just append remaining
            result.extend(remaining)
            break
        idx = rng.choice(valid)
        item = remaining.pop(idx)
        result.append(item)
        last_cat = item["category"]

    return result


def _no_consecutive_repeats(sequence: list[dict]) -> bool:
    for i in range(1, len(sequence)):
        if sequence[i]["category"] == sequence[i - 1]["category"]:
            return False
    return True


def build_prompt(
    sequence: list[dict],
    condition: str,
    test_category: str,
) -> tuple[str, str]:
    """Build a prompt from an interleaved sequence.

    Args:
        sequence: interleaved key-value pairs
        condition: "RI" (recall first) or "PI" (recall last)
        test_category: which category to query

    Returns:
        (prompt_text, expected_answer)
    """
    # Build stream text
    stream_lines = [f"{item['category']}: {item['value']}" for item in sequence]
    stream_text = "\n".join(stream_lines)

    query_word = "first" if condition == "RI" else "last"

    # Find expected answer
    cat_values = [item["value"] for item in sequence if item["category"] == test_category]
    expected = cat_values[0] if condition == "RI" else cat_values[-1]

    prompt = (
        f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{stream_text}\n\n"
        f"What was the {query_word} value of {test_category}?"
    )

    return prompt, expected


def generate_trial(
    num_keys: int,
    num_updates: int,
    condition: str,
    seed: int,
    test_category_idx: int = 0,
    categories: list[str] = None,
) -> InterferenceTrial:
    """Generate a single interference trial.

    Args:
        num_keys: number of categories to use
        num_updates: number of value updates per category
        condition: "RI" or "PI"
        seed: random seed for reproducibility
        test_category_idx: which category (by index in selected set) to query
        categories: optional fixed category list. If None, randomly sampled.

    Returns:
        InterferenceTrial with prompt and metadata
    """
    rng = random.Random(seed)

    if categories is None:
        categories = rng.sample(ORIGINAL_CATEGORIES, min(num_keys, len(ORIGINAL_CATEGORIES)))
    else:
        categories = categories[:num_keys]

    # Generate values for each category
    values_per_cat = {}
    for cat in categories:
        pool = generate_value_pool(cat, pool_size=max(num_updates + 50, 500))
        selected = rng.sample(pool, num_updates)
        values_per_cat[cat] = selected

    # Select test category
    test_category = categories[test_category_idx % len(categories)]

    # Build interleaved sequence
    sequence = build_interleaved_sequence(categories, values_per_cat, rng)

    # Build prompt
    prompt, expected = build_prompt(sequence, condition, test_category)

    # Find initial and final values of test category
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


SYSTEM_PROMPT = "Answer with ONLY the exact value. No explanation."


def format_for_chat(prompt: str, tokenizer) -> str:
    """Wrap a raw prompt in chat template if tokenizer supports it."""
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt


def preflight_context_check(
    num_keys: int,
    num_updates: int,
    tokenizer,
    context_limit: int,
    safety_margin: float = 0.90,
) -> tuple[bool, int]:
    """Check if a (num_keys, num_updates) cell fits within context.

    Generates ONE sample prompt, tokenizes it (with chat template), checks against limit.

    Returns:
        (feasible: bool, estimated_tokens: int)
    """
    trial = generate_trial(num_keys, num_updates, "RI", seed=0)
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
    """Pre-compute which grid cells are feasible.

    Returns:
        dict mapping (num_keys, num_updates) -> estimated_tokens
        Only includes feasible cells.
    """
    feasible = {}
    for nk in key_levels:
        for nu in update_levels:
            ok, n_tokens = preflight_context_check(nk, nu, tokenizer, context_limit)
            if ok:
                feasible[(nk, nu)] = n_tokens
            else:
                # Once infeasible for this key count, all higher updates will also fail
                break
    return feasible


# ── Grid definitions from research plan ─────────────────────────────────────

KEY_LEVELS = [2, 3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 46]

UPDATE_LEVELS = [
    1, 3, 5, 10, 15, 20,           # diff=5 from 10-20
    30, 40, 50, 60,                 # diff=10 from 20-60
    80, 100, 120, 140, 160, 180,    # diff=20 from 60-300
    200, 220, 240, 260, 280, 300,
]
