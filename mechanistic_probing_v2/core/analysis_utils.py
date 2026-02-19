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

@dataclass
class DLAResult:
    """Per-component contribution to logit difference."""
    # Per attention head: shape [n_layers, n_heads]
    head_logit_diff: np.ndarray  # contribution to logit(initial) - logit(final)
    # Per MLP layer: shape [n_layers]
    mlp_logit_diff: np.ndarray
    # Embed contribution
    embed_logit_diff: float = 0.0


def compute_dla(
    model,
    cache,
    initial_token_id: int,
    final_token_id: int,
    answer_position: int = -1,
) -> DLAResult:
    """Compute Direct Logit Attribution for each component.

    For each attention head and MLP layer, computes:
        component_output @ (W_U[:, initial_id] - W_U[:, final_id])

    This gives each component's contribution to the logit difference
    between initial and final value tokens.

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

    # Direction in logit space: initial - final
    logit_dir = model.W_U[:, initial_token_id] - model.W_U[:, final_token_id]  # [d_model]

    head_diff = np.zeros((n_layers, n_heads))
    mlp_diff = np.zeros(n_layers)

    for layer in range(n_layers):
        # Attention head outputs
        # cache["z", layer] has shape [batch, seq, n_heads, d_head]
        # We need to project through W_O to get per-head output in residual stream space
        z = cache["z", layer][0, answer_position, :, :]  # [n_heads, d_head]
        # W_O shape: [n_heads, d_head, d_model]
        W_O = model.W_O[layer]  # [n_heads, d_head, d_model]

        for head in range(n_heads):
            head_out = z[head] @ W_O[head]  # [d_model]
            head_diff[layer, head] = (head_out @ logit_dir).item()

        # MLP output
        mlp_out = cache["mlp_out", layer][0, answer_position, :]  # [d_model]
        mlp_diff[layer] = (mlp_out @ logit_dir).item()

    # Embedding contribution
    embed_out = cache["embed"][0, answer_position, :]  # [d_model]
    embed_diff = (embed_out @ logit_dir).item()

    return DLAResult(
        head_logit_diff=head_diff,
        mlp_logit_diff=mlp_diff,
        embed_logit_diff=embed_diff,
    )


def dla_to_dict(result: DLAResult) -> dict:
    """Serialize DLA result."""
    return {
        "head_logit_diff": result.head_logit_diff.tolist(),
        "mlp_logit_diff": result.mlp_logit_diff.tolist(),
        "embed_logit_diff": result.embed_logit_diff,
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
