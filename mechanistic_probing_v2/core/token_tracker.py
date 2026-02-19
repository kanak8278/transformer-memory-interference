"""
Token position tracker for mechanistic probing.

Maps token positions in a formatted prompt to semantic roles:
  - initial_value: first occurrence of initial value token
  - final_value: last occurrence of final value token
  - intermediate_value: other value tokens for the test category
  - query: the final question tokens
  - instruction: system/instruction prefix tokens
  - other: everything else

Designed for single-token values (Phase 2).
Works with TransformerLens HookedTransformer.
"""

from dataclasses import dataclass, field


@dataclass
class TokenMap:
    """Maps every token position to a semantic role."""
    roles: list[str]
    initial_value_positions: list[int] = field(default_factory=list)
    final_value_positions: list[int] = field(default_factory=list)
    intermediate_value_positions: list[int] = field(default_factory=list)
    query_positions: list[int] = field(default_factory=list)
    instruction_positions: list[int] = field(default_factory=list)
    answer_position: int = -1
    seq_len: int = 0
    test_category: str = ""
    condition: str = ""


def build_token_map(
    str_tokens: list[str],
    token_ids: list[int],
    initial_value_tid: int,
    final_value_tid: int,
    intermediate_value_tids: list[int],
) -> TokenMap:
    """Build a TokenMap by scanning token IDs directly.

    For single-token values, this is exact — no substring heuristics needed.

    Args:
        str_tokens: list of string tokens from model.to_str_tokens()
        token_ids: list of token IDs from model.to_tokens()[0]
        initial_value_tid: token ID of the initial value
        final_value_tid: token ID of the final value
        intermediate_value_tids: token IDs of intermediate values

    Returns:
        TokenMap with per-position labels and key indices.
    """
    seq_len = len(token_ids)
    roles = ["other"] * seq_len
    tmap = TokenMap(roles=roles, seq_len=seq_len, answer_position=seq_len - 1)

    inter_set = set(intermediate_value_tids)

    # Scan for value tokens by ID (exact match)
    initial_found = []
    final_found = []

    for i, tid in enumerate(token_ids):
        if tid == initial_value_tid:
            initial_found.append(i)
        if tid == final_value_tid and final_value_tid != initial_value_tid:
            final_found.append(i)
        if tid in inter_set:
            tmap.intermediate_value_positions.append(i)
            roles[i] = "intermediate_value"

    # For initial: take the FIRST occurrence
    if initial_found:
        tmap.initial_value_positions = [initial_found[0]]
        roles[initial_found[0]] = "initial_value"

    # For final: take the LAST occurrence
    if final_found:
        tmap.final_value_positions = [final_found[-1]]
        roles[final_found[-1]] = "final_value"

    # Find query region: look for "What" or "first"/"last" near the end
    joined = "".join(str_tokens)
    _mark_query_region(str_tokens, roles, tmap)
    _mark_instruction_region(str_tokens, roles, tmap)

    tmap.instruction_positions = [i for i, r in enumerate(roles) if r == "instruction"]

    return tmap


def _mark_query_region(str_tokens: list[str], roles: list[str], tmap: TokenMap):
    """Mark query tokens (the final question)."""
    # Search from the end for "What"
    for i in range(len(str_tokens) - 1, -1, -1):
        if "What" in str_tokens[i] or "what" in str_tokens[i]:
            tmap.query_positions = list(range(i, len(str_tokens)))
            for j in range(i, len(str_tokens)):
                if roles[j] == "other":
                    roles[j] = "query"
            return


def _mark_instruction_region(str_tokens: list[str], roles: list[str], tmap: TokenMap):
    """Mark instruction/system tokens at the start."""
    # Everything before the first value token is instruction
    first_value_pos = None
    for i, r in enumerate(roles):
        if r in ("initial_value", "final_value", "intermediate_value"):
            first_value_pos = i
            break

    if first_value_pos is not None:
        for i in range(first_value_pos):
            if roles[i] == "other":
                roles[i] = "instruction"


def summarize_token_map(tmap: TokenMap) -> dict:
    """Summarize a TokenMap for logging."""
    role_counts = {}
    for r in tmap.roles:
        role_counts[r] = role_counts.get(r, 0) + 1

    return {
        "seq_len": tmap.seq_len,
        "answer_position": tmap.answer_position,
        "role_distribution": role_counts,
        "initial_value_at": tmap.initial_value_positions,
        "final_value_at": tmap.final_value_positions,
        "n_intermediate": len(tmap.intermediate_value_positions),
        "query_start": tmap.query_positions[0] if tmap.query_positions else -1,
    }
