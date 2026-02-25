"""
Model loading utilities for mechanistic probing.

Two loading modes:
- TransformerLens: For mechanistic analysis (Phase 2). n_ctx=2048 default.
- HuggingFace: For behavioral sweep (Phase 1). Full context window.

All experiments should use load_model() or load_model_hf() from this module.
"""

import torch
from dataclasses import dataclass


@dataclass
class ModelInfo:
    name: str
    n_layers: int
    n_heads: int
    d_model: int
    n_ctx: int
    device: str
    backend: str  # "transformer_lens" or "huggingface"


# Known context limits (actual max_position_embeddings).
CONTEXT_LIMITS = {
    "Qwen/Qwen2.5-0.5B-Instruct": 32_768,
    "Qwen/Qwen2.5-1.5B-Instruct": 32_768,
    "google/gemma-2-2b-it": 8_192,
    "HuggingFaceTB/SmolLM2-135M-Instruct": 8_192,
    "HuggingFaceTB/SmolLM2-360M-Instruct": 8_192,
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": 8_192,
    "EleutherAI/pythia-160m": 2_048,
    "EleutherAI/pythia-160m-deduped": 2_048,
    "EleutherAI/pythia-410m": 2_048,
}

# Models that are base (completion-only, no chat template)
BASE_MODELS = {
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-160m-deduped",
    "EleutherAI/pythia-410m",
    "openai-community/gpt2",
    "openai-community/gpt2-medium",
}


def is_base_model(model_name: str) -> bool:
    """Check if model is a base (completion-only) model."""
    return model_name in BASE_MODELS


def get_device():
    """Get best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_name: str, device: str = None, n_ctx: int = 8192) -> tuple:
    """
    Load model via TransformerLens (for mechanistic analysis).

    Returns:
        (model, tokenizer, model_info)

    model is a HookedTransformer instance with cache support.
    n_ctx is just a buffer size — doesn't change model behavior.
    RoPE values are computed on-the-fly from actual token positions.
    """
    if device is None:
        device = get_device()

    from transformer_lens import HookedTransformer

    print(f"Loading {model_name} via TransformerLens on {device} (n_ctx={n_ctx})...")
    model = HookedTransformer.from_pretrained(model_name, device=device, n_ctx=n_ctx)
    model.eval()

    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    info = ModelInfo(
        name=model_name,
        n_layers=model.cfg.n_layers,
        n_heads=model.cfg.n_heads,
        d_model=model.cfg.d_model,
        n_ctx=model.cfg.n_ctx,
        device=device,
        backend="transformer_lens",
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded: {n_params:,} params, {info.n_layers} layers, "
          f"{info.n_heads} heads, d_model={info.d_model}, ctx={info.n_ctx}")

    return model, tokenizer, info


def load_model_hf(model_name: str, device: str = None) -> tuple:
    """
    Load model via HuggingFace transformers (for behavioral sweep).

    Returns:
        (model, tokenizer, model_info)

    model is a standard HF CausalLM. Full context window available.
    """
    if device is None:
        device = get_device()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name} via HuggingFace on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, device_map=device,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = model.config
    n_ctx = getattr(config, "max_position_embeddings", 4096)
    n_layers = getattr(config, "num_hidden_layers", 0)
    n_heads = getattr(config, "num_attention_heads", 0)
    d_model = getattr(config, "hidden_size", 0)

    info = ModelInfo(
        name=model_name,
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        n_ctx=n_ctx,
        device=device,
        backend="huggingface",
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded: {n_params:,} params, {info.n_layers} layers, "
          f"{info.n_heads} heads, d_model={info.d_model}, ctx={info.n_ctx}")

    return model, tokenizer, info


def get_context_limit(model_name: str, model_info: ModelInfo = None) -> int:
    """Get context token limit for a model. Uses known limits or model config."""
    if model_name in CONTEXT_LIMITS:
        return CONTEXT_LIMITS[model_name]
    if model_info is not None:
        return model_info.n_ctx
    return 4096  # conservative fallback
