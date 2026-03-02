"""
Analysis utilities for Phase 2 mechanistic probing.

Provides reusable functions for:
  - Logit lens: project residual stream through unembedding at each layer
  - Attention extraction: per-head attention to value positions
  - Direct Logit Attribution (DLA): per-head contribution to logit of specific tokens
  - Activation patching helpers

All functions work with TransformerLens HookedTransformer + ActivationCache.
"""

import torch
import numpy as np
from dataclasses import dataclass, field


# ── Logit Lens ──────────────────────────────────────────────────────────────

@dataclass
class LogitLensResult:
    """Logit lens output for one trial at one layer."""
    layer: int
    prob_initial: float  # P(initial_value_token)
    prob_final: float    # P(final_value_token)
    rank_initial: int    # rank of initial_value_token in vocab
    rank_final: int      # rank of final_value_token in vocab
    top5_tokens: list[str] = field(default_factory=list)
    top5_probs: list[float] = field(default_factory=list)


def logit_lens_all_layers(
    model,
    cache,
    initial_token_ids: list[int],
    final_token_ids: list[int],
    answer_position: int = -1,
) -> list[LogitLensResult]:
    """Run logit lens at every layer.

    Projects the residual stream at `answer_position` through the unembedding
    matrix to get per-layer token probabilities.

    Args:
        model: HookedTransformer
        cache: ActivationCache from model.run_with_cache()
        initial_token_ids: token ids for the initial value (space-prefixed + bare)
        final_token_ids: token ids for the final value (space-prefixed + bare)
        answer_position: which position to analyze (-1 = last)

    Returns:
        List of LogitLensResult, one per layer.
    """
    n_layers = model.cfg.n_layers
    results = []

    for layer in range(n_layers):
        resid = cache["resid_post", layer]  # [batch, seq, d_model]
        # Get residual at answer position
        h = resid[0, answer_position, :]  # [d_model]

        # Project through unembedding: logits = h @ W_U + b_U
        # W_U shape: [d_model, vocab_size]
        logits = h @ model.W_U + model.b_U  # [vocab_size]
        probs = torch.softmax(logits, dim=-1)

        # Get max probability across all token variants (space-prefixed + bare)
        prob_initial = max(probs[tid].item() for tid in initial_token_ids)
        prob_final = max(probs[tid].item() for tid in final_token_ids)
        # Use the best-ranking variant for rank
        best_init_tid = max(initial_token_ids, key=lambda t: probs[t].item())
        best_final_tid = max(final_token_ids, key=lambda t: probs[t].item())

        # Get ranks (use best variant)
        sorted_indices = torch.argsort(probs, descending=True)
        rank_initial = (sorted_indices == best_init_tid).nonzero(as_tuple=True)[0].item()
        rank_final = (sorted_indices == best_final_tid).nonzero(as_tuple=True)[0].item()

        # Top 5
        top5_idx = sorted_indices[:5]
        top5_probs = probs[top5_idx].tolist()
        top5_tokens = [model.to_string(idx.unsqueeze(0)) for idx in top5_idx]

        results.append(LogitLensResult(
            layer=layer,
            prob_initial=prob_initial,
            prob_final=prob_final,
            rank_initial=rank_initial,
            rank_final=rank_final,
            top5_tokens=top5_tokens,
            top5_probs=top5_probs,
        ))

    return results


def logit_lens_to_dict(results: list[LogitLensResult]) -> dict:
    """Convert logit lens results to a serializable dict."""
    return {
        "layers": [r.layer for r in results],
        "prob_initial": [r.prob_initial for r in results],
        "prob_final": [r.prob_final for r in results],
        "rank_initial": [r.rank_initial for r in results],
        "rank_final": [r.rank_final for r in results],
        "top5_per_layer": [
            {"tokens": r.top5_tokens, "probs": r.top5_probs}
            for r in results
        ],
    }


# ── Attention Pattern Analysis ──────────────────────────────────────────────

@dataclass
class HeadAttentionStats:
    """Attention statistics for one head at the answer position."""
    layer: int
    head: int
    attn_to_initial: float     # attention weight to initial value position(s)
    attn_to_final: float       # attention weight to final value position(s)
    attn_to_intermediate: float # attention weight to intermediate value positions
    attn_to_query: float       # attention to query tokens
    attn_to_instruction: float # attention to instruction tokens
    retrieval_score: float     # total attention to any value position
    primacy_score: float       # attn_initial / (attn_initial + attn_final)
                               # OBSERVATIONAL ONLY — high score means head looks at initial
                               # positions more than final, but does NOT mean the head causes
                               # PI failures. Use exp 25a causal knockout to establish causation.


def extract_attention_stats(
    cache,
    n_layers: int,
    n_heads: int,
    answer_position: int,
    initial_positions: list[int],
    final_positions: list[int],
    intermediate_positions: list[int],
    query_positions: list[int],
    instruction_positions: list[int],
) -> list[HeadAttentionStats]:
    """Extract attention statistics for all heads at the answer position.

    Args:
        cache: ActivationCache
        n_layers, n_heads: model dimensions
        answer_position: token position to analyze
        *_positions: lists of token indices for each role

    Returns:
        List of HeadAttentionStats, one per (layer, head).
    """
    results = []

    for layer in range(n_layers):
        pattern = cache["pattern", layer]  # [batch, n_heads, seq, seq]
        # Attention FROM answer_position TO all positions
        attn = pattern[0, :, answer_position, :]  # [n_heads, seq]

        for head in range(n_heads):
            head_attn = attn[head]  # [seq]

            a_init = head_attn[initial_positions].sum().item() if initial_positions else 0.0
            a_final = head_attn[final_positions].sum().item() if final_positions else 0.0
            a_inter = head_attn[intermediate_positions].sum().item() if intermediate_positions else 0.0
            a_query = head_attn[query_positions].sum().item() if query_positions else 0.0
            a_instr = head_attn[instruction_positions].sum().item() if instruction_positions else 0.0

            retrieval = a_init + a_final + a_inter
            denom = a_init + a_final
            primacy = a_init / denom if denom > 1e-8 else 0.5

            results.append(HeadAttentionStats(
                layer=layer,
                head=head,
                attn_to_initial=a_init,
                attn_to_final=a_final,
                attn_to_intermediate=a_inter,
                attn_to_query=a_query,
                attn_to_instruction=a_instr,
                retrieval_score=retrieval,
                primacy_score=primacy,
            ))

    return results


def attention_stats_to_arrays(
    stats: list[HeadAttentionStats], n_layers: int, n_heads: int,
) -> dict:
    """Convert attention stats to numpy arrays for easy plotting.

    Returns dict with keys like 'primacy_score' -> shape [n_layers, n_heads].
    """
    fields = [
        "attn_to_initial", "attn_to_final", "attn_to_intermediate",
        "retrieval_score", "primacy_score",
    ]
    arrays = {f: np.zeros((n_layers, n_heads)) for f in fields}

    for s in stats:
        for f in fields:
            arrays[f][s.layer, s.head] = getattr(s, f)

    return arrays


# ── Direct Logit Attribution (DLA) ──────────────────────────────────────────
#
# FRAMING NOTE (updated 2026-03):
# Original framing: DLA measured logit(init) - logit(final) per component,
# treating the two tokens as competitors in the same trial. This assumed
# "primacy promotion" as the mechanism — i.e. heads push init over final.
#
# Corrected framing: PI failure = model fails to retrieve the FINAL value.
# Whether this is because (a) final is suppressed or (b) initial is promoted
# is an empirical question answered by exp 14. DLA now computes BOTH projections
# separately:
#   - head_logit_final: how much does this component support final-value retrieval?
#     (negative = component suppresses the final value — primary PI story)
#   - head_logit_init: how much does this component support initial-value retrieval?
#     (positive = component promotes initial value — secondary/corroborating story)
#   - head_logit_diff: head_logit_init - head_logit_final (kept for backward compat
#     and for exp 20 minority override analysis which uses the signed difference)
#
# Interpretation: compare head_logit_final on PI trials vs RI trials.
# A head with high head_logit_init on RI but LOW head_logit_final on PI is the
# asymmetry we care about — helps retrieval when answer is initial, hurts when final.

@dataclass
class DLAResult:
    """Per-component contribution to logit difference and individual token logits."""
    # Per attention head: shape [n_layers, n_heads]
    head_logit_diff: np.ndarray   # logit(initial) - logit(final) [backward compat]
    head_logit_init: np.ndarray   # component contribution to logit(initial)
    head_logit_final: np.ndarray  # component contribution to logit(final)
    # Per MLP layer: shape [n_layers]
    mlp_logit_diff: np.ndarray
    mlp_logit_init: np.ndarray
    mlp_logit_final: np.ndarray
    # Embed contribution
    embed_logit_diff: float = 0.0
    embed_logit_init: float = 0.0
    embed_logit_final: float = 0.0


def compute_dla(
    model,
    cache,
    initial_token_id: int,
    final_token_id: int,
    answer_position: int = -1,
) -> DLAResult:
    """Compute Direct Logit Attribution for each component.

    For each attention head and MLP layer, computes both:
        component_output @ W_U[:, initial_id]   → head_logit_init
        component_output @ W_U[:, final_id]     → head_logit_final
        difference of the two                   → head_logit_diff

    Positive head_logit_final = component supports retrieval of the final value.
    Negative head_logit_final = component suppresses the final value (PI failure mode).

    The difference (head_logit_diff) is kept for backward compatibility and for
    experiments that need the signed comparison (e.g. exp 20 minority override).

    Args:
        model: HookedTransformer
        cache: ActivationCache
        initial_token_id: token id for the initial value
        final_token_id: token id for the final value
        answer_position: position to analyze (-1 = last)

    Returns:
        DLAResult with per-component contributions.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Separate directions in logit space
    dir_init  = model.W_U[:, initial_token_id]                               # [d_model]
    dir_final = model.W_U[:, final_token_id]                                  # [d_model]
    dir_diff  = dir_init - dir_final                                           # [d_model]

    head_diff  = np.zeros((n_layers, n_heads))
    head_init  = np.zeros((n_layers, n_heads))
    head_final = np.zeros((n_layers, n_heads))
    mlp_diff   = np.zeros(n_layers)
    mlp_init   = np.zeros(n_layers)
    mlp_final  = np.zeros(n_layers)

    for layer in range(n_layers):
        z   = cache["z", layer][0, answer_position, :, :]  # [n_heads, d_head]
        W_O = model.W_O[layer]                              # [n_heads, d_head, d_model]

        for head in range(n_heads):
            head_out = z[head] @ W_O[head]               # [d_model]
            head_init[layer, head]  = (head_out @ dir_init).item()
            head_final[layer, head] = (head_out @ dir_final).item()
            head_diff[layer, head]  = head_init[layer, head] - head_final[layer, head]

        mlp_out = cache["mlp_out", layer][0, answer_position, :]  # [d_model]
        mlp_init[layer]  = (mlp_out @ dir_init).item()
        mlp_final[layer] = (mlp_out @ dir_final).item()
        mlp_diff[layer]  = mlp_init[layer] - mlp_final[layer]

    embed_out = cache["embed"][0, answer_position, :]
    e_init  = (embed_out @ dir_init).item()
    e_final = (embed_out @ dir_final).item()

    return DLAResult(
        head_logit_diff=head_diff,
        head_logit_init=head_init,
        head_logit_final=head_final,
        mlp_logit_diff=mlp_diff,
        mlp_logit_init=mlp_init,
        mlp_logit_final=mlp_final,
        embed_logit_diff=e_init - e_final,
        embed_logit_init=e_init,
        embed_logit_final=e_final,
    )


def dla_to_dict(result: DLAResult) -> dict:
    """Serialize DLA result."""
    return {
        "head_logit_diff":  result.head_logit_diff.tolist(),
        "head_logit_init":  result.head_logit_init.tolist(),
        "head_logit_final": result.head_logit_final.tolist(),
        "mlp_logit_diff":   result.mlp_logit_diff.tolist(),
        "mlp_logit_init":   result.mlp_logit_init.tolist(),
        "mlp_logit_final":  result.mlp_logit_final.tolist(),
        "embed_logit_diff":  result.embed_logit_diff,
        "embed_logit_init":  result.embed_logit_init,
        "embed_logit_final": result.embed_logit_final,
    }


# ── Activation Patching ────────────────────────────────────────────────────

def patch_residual_stream(
    model,
    clean_cache,
    corrupted_tokens,
    layer: int,
    position: int,
    component: str = "resid_post",
) -> torch.Tensor:
    """Patch clean activations into a corrupted run.

    Runs the corrupted input through the model, but at (layer, position),
    replaces the activation with the clean version.

    Args:
        model: HookedTransformer
        clean_cache: cache from clean run
        corrupted_tokens: tokenized corrupted input
        layer: which layer to patch
        position: which position to patch
        component: "resid_post", "attn_out", or "mlp_out"

    Returns:
        Logits from the patched run.
    """
    clean_activation = clean_cache[component, layer][0, position, :].clone()

    def patch_hook(activation, hook):
        activation[0, position, :] = clean_activation
        return activation

    hook_name = f"blocks.{layer}.hook_{component.replace('_post', '_post').replace('attn_out', 'attn.hook_result').replace('mlp_out', 'mlp.hook_result')}"

    # Simpler: use the cache key format
    if component == "resid_post":
        hook_name = f"blocks.{layer}.hook_resid_post"
    elif component == "attn_out":
        hook_name = f"blocks.{layer}.hook_attn_out"
    elif component == "mlp_out":
        hook_name = f"blocks.{layer}.hook_mlp_out"

    logits = model.run_with_hooks(
        corrupted_tokens,
        fwd_hooks=[(hook_name, patch_hook)],
    )
    return logits


# ── Patching Recovery Metric ───────────────────────────────────────────────
#
# Following Wang et al. (2022, "Interpretability in the Wild: IOI Circuit")
# and Heimersheim & Nanda (2024, "How to use and interpret activation patching"):
#
# Use LOGIT DIFFERENCE as the base metric for recovery, not P(correct).
#
# Why: P(correct)-based recovery explodes when clean_p ≈ corrupted_p (small
# denominator). This happens in RI where the model is robust to interference.
# Logit difference is linear in the model's internal representations and produces
# stable denominators in the ±100 logit range.
#
# Formula:
#   logit_diff = logit(correct_token) - logit(incorrect_token)
#   recovery = (patched_ld - corrupted_ld) / (clean_ld - corrupted_ld)
#
# Where:
#   correct_token = expected answer (initial for RI, final for PI)
#   incorrect_token = wrong answer (final for RI, initial for PI)
#
# References:
#   - Wang et al. 2022 (IOI): logit_diff = logit(IO) - logit(S)
#   - Heimersheim & Nanda 2024: "prefer logit difference over probability"
#   - Conmy et al. 2023 (ACDC): uses KL divergence as alternative


def compute_logit_diff(logits, correct_tids, incorrect_tids):
    """Compute logit difference: logit(correct) - logit(incorrect).

    Takes max over token variants (space-prefixed + bare) for each side.
    Returns a single float.
    """
    correct_logit = max(logits[t].item() for t in correct_tids)
    incorrect_logit = max(logits[t].item() for t in incorrect_tids)
    return correct_logit - incorrect_logit


def compute_recovery(patched_ld, corrupted_ld, clean_ld, min_denom=0.01):
    """Compute patching recovery using logit difference.

    Args:
        patched_ld: logit_diff after patching
        corrupted_ld: logit_diff in corrupted (baseline) run
        clean_ld: logit_diff in clean run
        min_denom: minimum absolute denominator to prevent division by zero.
                   Default 0.01 — only guards against true numerical zero.
                   Trials with small denominators should be filtered at
                   aggregation time, not silently clamped here.

    Returns:
        recovery fraction (0 = no help, 1 = fully restored, can be >1 or <0).
        Returns None if denominator is below min_denom (caller should filter).

    Note on >100% recovery:
        Values above 1.0 (100%) mean the patched run's logit_diff exceeds
        the clean run's. This is a known, expected phenomenon in activation
        patching (reported in Wang et al. 2022, IOI circuit). It happens
        because:
          1. The corrupted prompt has richer context (N values vs 1) than
             the clean prompt. Non-patched layers retain this richer
             representation while the patched layer fixes the retrieval error.
          2. The clean prompt is artificially minimal (1 value per key with
             instructions saying "each key gets updated multiple times"),
             which may slightly confuse the model, making its clean logit_diff
             a soft rather than hard upper bound.
        Overshoots of 110-160% are typical and do not indicate a bug.
        The core claim (late-layer patching recovers PI) holds regardless
        of whether recovery is 95% or 150%.
    """
    denom = clean_ld - corrupted_ld
    if abs(denom) < min_denom:
        return None
    return (patched_ld - corrupted_ld) / denom


def aggregate_recovery(recovery_values, min_healthy_denom=1.0):
    """Aggregate recovery values, filtering out degenerate trials.

    Trials where compute_recovery returned None (denominator < 0.01) are
    always excluded. Additionally, this function is typically called after
    collecting raw recovery values — callers should also track and report
    the raw logit_diffs so reviewers can assess denominator health.

    Args:
        recovery_values: list of recovery floats (may contain None)
        min_healthy_denom: not used here (filtering happens at compute time),
                           but callers should pick operating points where
                           |clean_ld - corrupted_ld| > 1.0 for stable results.

    Returns:
        dict with mean, std, n_total, n_valid, n_filtered
    """
    valid = [r for r in recovery_values if r is not None]
    n_filtered = len(recovery_values) - len(valid)
    if not valid:
        return {
            "mean": 0.0, "std": 0.0,
            "n_total": len(recovery_values), "n_valid": 0, "n_filtered": n_filtered,
        }
    import numpy as np
    return {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "n_total": len(recovery_values),
        "n_valid": len(valid),
        "n_filtered": n_filtered,
    }


# ── Helpers ─────────────────────────────────────────────────────────────────

def get_first_token_id(model, value: str) -> int:
    """Get the first token ID for a value string.

    For synthetic values like 'Art42', this is typically a single token
    or we take the first subtoken.
    """
    tokens = model.to_tokens(value, prepend_bos=False)
    return tokens[0, 0].item()


def get_positions_by_role(roles: list[str], role: str) -> list[int]:
    """Get all token positions with a given role."""
    return [i for i, r in enumerate(roles) if r == role]
