"""Fixed vocabulary for the synthetic key-value interference task.

51 single-token symbols, no BPE:
  KEY   <Ka>..<Kz>   ids 0-25    (26-symbol pool; a given example uses only K of them)
  VALUE <V0>..<V9>   ids 26-35   (10 symbols, i.i.d. with replacement)
  STEP  <S1>..<S12>  ids 36-47   (12 symbols, local per-key update index, sized to max N)
  special           ids 48-50   <BOS>=48 <QUERY>=49 <EOS>=50

See ../setup.md for the full design rationale.
"""

NUM_KEYS = 26
NUM_VALUES = 10
NUM_STEPS = 12

KEY_BASE = 0
VALUE_BASE = KEY_BASE + NUM_KEYS       # 26
STEP_BASE = VALUE_BASE + NUM_VALUES    # 36
BOS = STEP_BASE + NUM_STEPS            # 48
QUERY = BOS + 1                        # 49
EOS = QUERY + 1                        # 50
VOCAB_SIZE = EOS + 1                   # 51


def key_id(key_idx: int) -> int:
    """key_idx in [0, 26)."""
    return KEY_BASE + key_idx


def value_id(value: int) -> int:
    """value in [0, 10)."""
    return VALUE_BASE + value


def step_id(step: int) -> int:
    """step in [1, 12] (1-indexed local update count)."""
    return STEP_BASE + (step - 1)


def value_of(token_id: int) -> int:
    assert VALUE_BASE <= token_id < STEP_BASE
    return token_id - VALUE_BASE


def key_letter(key_idx: int) -> str:
    return chr(ord("a") + key_idx)


def token_repr(token_id: int) -> str:
    """Human-readable token for debugging / val-sample dumps."""
    if KEY_BASE <= token_id < VALUE_BASE:
        return f"<K{key_letter(token_id - KEY_BASE)}>"
    if VALUE_BASE <= token_id < STEP_BASE:
        return f"<V{token_id - VALUE_BASE}>"
    if STEP_BASE <= token_id < BOS:
        return f"<S{token_id - STEP_BASE + 1}>"
    return {BOS: "<BOS>", QUERY: "<QUERY>", EOS: "<EOS>"}[token_id]


def decode(token_ids) -> str:
    return " ".join(token_repr(t) for t in token_ids)
