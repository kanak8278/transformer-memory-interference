"""
Single-token value pool for mechanistic probing (Phase 2).

For Phase 2 analysis (logit lens, DLA, attention), we need values where
each value is a SINGLE token. This is critical because:
  - Logit lens looks at P(token) at each layer for ONE token ID
  - DLA computes per-component contribution to ONE token's logit
  - Multi-token values like "Art42" share prefix tokens across values

These 98 values are verified single-token in Qwen2.5-0.5B-Instruct's tokenizer.
They're common English words — the model knows them well.
"""

# All verified single-token in Qwen2.5-0.5B-Instruct (151,936 vocab)
SINGLE_TOKEN_VALUES = [
    "ale", "amber", "ant", "ash", "axe", "bat", "bay", "bear", "bee", "bell",
    "black", "blue", "bow", "brown", "cake", "cat", "chain", "coal", "coin", "cow",
    "cream", "dart", "deer", "dog", "eel", "elm", "fern", "fig", "flag", "folk",
    "fox", "gin", "gold", "gray", "green", "ham", "hawk", "hen", "hill", "horn",
    "ice", "iris", "iron", "jam", "key", "king", "lace", "lake", "lamp", "lead",
    "lion", "lock", "lord", "mask", "metal", "mint", "mist", "night", "noon", "nut",
    "opal", "orange", "owl", "pie", "pig", "pine", "pink", "pipe", "pond", "pop",
    "punk", "purple", "rain", "ram", "rat", "red", "rice", "ring", "rock", "rose",
    "ruby", "rum", "salt", "sand", "silver", "snow", "soup", "steel", "storm", "sword",
    "tan", "tea", "tin", "vine", "wand", "white", "wind", "wolf",
]


def get_single_token_pool(n: int = None) -> list[str]:
    """Get a pool of single-token values.

    Args:
        n: number of values to return. None = all 98.

    Returns:
        List of single-token value strings.
    """
    if n is None or n >= len(SINGLE_TOKEN_VALUES):
        return list(SINGLE_TOKEN_VALUES)
    return list(SINGLE_TOKEN_VALUES[:n])


def verify_single_token(tokenizer, values: list[str] = None) -> dict:
    """Verify that values are single-token WITH space prefix.

    In prompts, values appear after ": " so the tokenizer sees " red" not "red".
    The space-prefixed form is what the model predicts and what appears in token IDs.

    Returns dict mapping value -> space-prefixed token_id.
    """
    if values is None:
        values = SINGLE_TOKEN_VALUES

    verified = {}
    for v in values:
        # Check space-prefixed form (as it appears in "category: value")
        toks = tokenizer.encode(f" {v}", add_special_tokens=False)
        if len(toks) == 1:
            verified[v] = toks[0]
    return verified
