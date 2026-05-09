"""
Model loading utilities for all experiments.

Handles:
  - Device auto-detection (cuda with GPU selection → mps → cpu)
  - TransformerLens loading (for mechanistic analysis)
  - HuggingFace loading (for behavioral sweeps)
  - dtype selection (float16 for CUDA, float32 for MPS/CPU)
  - Single-token value verification

All experiments should use this module for model loading:
  from core.model_loader import load_model, load_model_hf, detect_device

Usage:
  # TransformerLens (mechanistic probing)
  model, tokenizer, info = load_model("Qwen/Qwen2.5-0.5B-Instruct", n_ctx=8192)

  # HuggingFace (behavioral sweep)
  model, tokenizer, info = load_model_hf("Qwen/Qwen2.5-3B-Instruct")

  # With explicit GPU
  model, tokenizer, info = load_model_hf("Qwen/Qwen2.5-3B-Instruct", gpu_idx=5)

  # Verify single-token values
  value_to_tid = verify_single_token(tokenizer)
"""

import os
import torch
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelInfo:
    name: str
    n_layers: int
    n_heads: int
    d_model: int
    n_ctx: int
    device: str
    device_name: str
    backend: str  # "transformer_lens" or "huggingface"
    dtype: str = "float32"


# ═══════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

# Known context limits (actual max_position_embeddings).
CONTEXT_LIMITS = {
    # Qwen 2.5
    "Qwen/Qwen2.5-0.5B-Instruct": 32_768,
    "Qwen/Qwen2.5-1.5B-Instruct": 32_768,
    "Qwen/Qwen2.5-3B":             32_768,
    "Qwen/Qwen2.5-3B-Instruct":   32_768,
    # Qwen3.5 series (qwen3_5 architecture, May 2025)
    "Qwen/Qwen3.5-0.8B":               32_768,
    "Qwen/Qwen3.5-4B":                 32_768,
    "Qwen/Qwen3.5-9B":                 32_768,
    "Qwen/Qwen3.5-0.8B-Base":          32_768,
    "Qwen/Qwen3.5-4B-Base":            32_768,
    "Qwen/Qwen3.5-9B-Base":            32_768,

    "Qwen/Qwen2.5-7B-Instruct":   32_768,
    # Gemma 3 (all sizes)
    "google/gemma-3-270m-it": 32_768,
    "google/gemma-3-1b-it":   8_192,
    "google/gemma-3-4b-it":   131_072,
    "google/gemma-3-12b-it":  131_072,
    "google/gemma-3-27b-it":  131_072,
    # Gemma 2
    "google/gemma-2-2b-it": 8_192,
    # SmolLM2
    "HuggingFaceTB/SmolLM2-135M-Instruct": 8_192,
    "HuggingFaceTB/SmolLM2-360M-Instruct": 8_192,
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": 8_192,
    # Pythia
    "EleutherAI/pythia-160m":         2_048,
    "EleutherAI/pythia-160m-deduped": 2_048,
    "EleutherAI/pythia-410m":         2_048,
    # TinyLlama
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 2_048,
    # StableLM
    "stabilityai/stablelm-2-1_6b-chat": 4_096,
    # Mamba
    "state-spaces/mamba-370m-hf":  2_048,
    "state-spaces/mamba2-370m-hf": 2_048,
    "state-spaces/mamba-1.4b-hf":  2_048,
}

# Instruction-tuned models (use chat template).
INSTRUCT_MODELS = {
    # Qwen 2.5
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    # Gemma 3 (all -it variants)
    "google/gemma-3-270m-it",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
    # Gemma 2
    "google/gemma-2-2b-it",
    # SmolLM2
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}

# Base models (completion-only, no chat template).
BASE_MODELS = {
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-160m-deduped",
    "EleutherAI/pythia-410m",
    "openai-community/gpt2",
    "openai-community/gpt2-medium",
    "state-spaces/mamba-370m-hf",
    "state-spaces/mamba2-370m-hf",
}


def is_instruct_model(model_name):
    """Check if model uses chat template (instruct) or completion format (base).
    Falls back to name heuristic for unknown models."""
    if model_name in INSTRUCT_MODELS:
        return True
    # Heuristic: -it, -Instruct, -chat, -Chat suffixes indicate instruct models
    lower = model_name.lower()
    return any(lower.endswith(s) for s in ["-it", "-instruct", "-chat"])


def is_base_model(model_name):
    """Check if model is a base (completion-only) model."""
    return model_name in BASE_MODELS


def model_short_name(model_name):
    """state-spaces/mamba-370m-hf → mamba-370m-hf"""
    return model_name.split("/")[-1]


# ═══════════════════════════════════════════════════════════════════════════
# DEVICE DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def detect_device(gpu_idx=None):
    """Auto-detect best available device: cuda → mps → cpu.

    Args:
        gpu_idx: If set and CUDA available, sets CUDA_VISIBLE_DEVICES
                 to this index so the model lands on the right physical GPU.
                 If None, uses whatever CUDA device is default.

    Returns:
        (device_str, device_name) — e.g. ("cuda:0", "NVIDIA V100, 16.0 GB")
    """
    if torch.cuda.is_available():
        if gpu_idx is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        device = "cuda:0"
        props = torch.cuda.get_device_properties(0)
        name = f"{props.name}, {props.total_memory / 1e9:.1f} GB"
        return device, name

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", "Apple Silicon MPS"

    return "cpu", "CPU"


def get_device(gpu_idx=None):
    """Convenience wrapper — returns just the device string.

    Backward-compatible with old code that called get_device().
    """
    device, _ = detect_device(gpu_idx=gpu_idx)
    return device


def clear_accelerator_cache(device=None):
    """Clear accelerator memory cache.

    Safe to call on any device — does nothing on CPU.
    """
    if device is None:
        device = get_device()
    device_str = str(device)

    if "cuda" in device_str and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device_str == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def _select_dtype(device, model_name=None, force_dtype=None):
    """Select appropriate dtype for device and model.

    Args:
        device: Device string ("cuda:0", "mps", "cpu")
        model_name: Optional model name for heuristics
        force_dtype: If set, use this dtype regardless

    Returns:
        torch.dtype
    """
    if force_dtype is not None:
        return force_dtype

    # MPS has spotty float16 support
    if str(device) == "mps":
        return torch.float32

    # CUDA: float16 for all models (safe for inference, halves memory)
    if "cuda" in str(device):
        return torch.float16

    # CPU: always float32
    return torch.float32


# ═══════════════════════════════════════════════════════════════════════════
# TRANSFORMERLENS LOADING (mechanistic analysis)
# ═══════════════════════════════════════════════════════════════════════════

def load_model(model_name, device=None, n_ctx=8192, gpu_idx=None, dtype=None, **kwargs):
    """Load model via TransformerLens (for mechanistic analysis).

    Args:
        model_name: HuggingFace model name
        device: Device string. If None, auto-detected.
        n_ctx: Context window size for TransformerLens buffer.
        gpu_idx: GPU index for CUDA_VISIBLE_DEVICES. Only used if device is None.
        dtype: torch dtype (e.g. torch.float16). If None, auto-selected:
               CUDA → float16, MPS → float32, CPU → float32.
        **kwargs: Extra args passed to HookedTransformer.from_pretrained

    Returns:
        (model, tokenizer, model_info)
    """
    if device is None:
        device, device_name = detect_device(gpu_idx=gpu_idx)
    else:
        device_name = str(device)

    if dtype is None:
        dtype = _select_dtype(device, model_name)

    from transformer_lens import HookedTransformer

    print(f"Loading {model_name} via TransformerLens on {device} (n_ctx={n_ctx}, dtype={dtype})...")
    try:
        model = HookedTransformer.from_pretrained(
            model_name, device=device, n_ctx=n_ctx, dtype=dtype, **kwargs
        )
    except TypeError as e:
        if "n_ctx" in str(e):
            # Some TransformerLens versions leak n_ctx to HuggingFace kwargs
            print(f"  Retrying without n_ctx (TL version compat)...")
            model = HookedTransformer.from_pretrained(
                model_name, device=device, dtype=dtype, **kwargs
            )
        else:
            raise
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
        device=str(device),
        device_name=device_name,
        backend="transformer_lens",
        dtype=str(model.cfg.dtype),
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded: {n_params:,} params, {info.n_layers} layers, "
          f"{info.n_heads} heads, d_model={info.d_model}, ctx={info.n_ctx}")
    print(f"  Device: {device} ({device_name})")

    return model, tokenizer, info


# ═══════════════════════════════════════════════════════════════════════════
# HUGGINGFACE LOADING (behavioral sweeps)
# ═══════════════════════════════════════════════════════════════════════════

def load_model_hf(model_name, device=None, gpu_idx=None, dtype=None, **kwargs):
    """Load model via HuggingFace transformers (for behavioral sweeps).

    Handles device detection, dtype selection, and MPS/CPU compatibility.

    Args:
        model_name: HuggingFace model name
        device: Device string. If None, auto-detected.
        gpu_idx: GPU index for CUDA_VISIBLE_DEVICES. Only used if device is None.
        dtype: Force a specific torch.dtype. If None, auto-selected.
        **kwargs: Extra args passed to AutoModelForCausalLM.from_pretrained

    Returns:
        (model, tokenizer, model_info)
    """
    if device is None:
        device, device_name = detect_device(gpu_idx=gpu_idx)
    else:
        device_name = str(device)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    selected_dtype = _select_dtype(device, model_name, force_dtype=dtype)

    print(f"Loading {model_name} via HuggingFace on {device} (dtype={selected_dtype})...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if str(device) in ("cpu", "mps"):
        # MPS and CPU don't support device_map, load then move
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=selected_dtype, trust_remote_code=True, **kwargs
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=selected_dtype, device_map=device,
            trust_remote_code=True, **kwargs
        )
    model.eval()

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
        device=str(device),
        device_name=device_name,
        backend="huggingface",
        dtype=str(selected_dtype),
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded: {n_params:,} params, {info.n_layers} layers, "
          f"{info.n_heads} heads, d_model={info.d_model}, ctx={info.n_ctx}")
    print(f"  Device: {device} ({device_name})")

    return model, tokenizer, info


# ═══════════════════════════════════════════════════════════════════════════
# CONTEXT LIMITS
# ═══════════════════════════════════════════════════════════════════════════

def get_context_limit(model_name, model_info=None):
    """Get context token limit for a model. Uses known limits or model config."""
    if model_name in CONTEXT_LIMITS:
        return CONTEXT_LIMITS[model_name]
    if model_info is not None:
        return model_info.n_ctx
    return 4096


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE-TOKEN VALUE VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def verify_single_token(tokenizer, values=None):
    """Verify that values are single-token WITH space prefix.

    In prompts, values appear after ": " so the tokenizer sees " ruby" not "ruby".
    The space-prefixed form is what the model predicts and what logit lens tracks.

    Args:
        tokenizer: HuggingFace tokenizer or TransformerLens tokenizer
        values: List of words to verify. If None, loads ARBITRARY_SINGLE pool.

    Returns:
        Dict mapping value → space-prefixed token_id.
        Only includes values that are exactly 1 token.
    """
    if values is None:
        from .dataset_configs import get_value_pool
        values = get_value_pool("ARBITRARY_SINGLE")

    verified = {}
    for v in values:
        toks = tokenizer.encode(f" {v}", add_special_tokens=False)
        if len(toks) == 1:
            verified[v] = toks[0]
    return verified


def get_single_token_pool(n=None):
    """Get a pool of single-token values.

    Backward-compatible wrapper. Loads from dataset_configs.

    Args:
        n: Number of values to return. None = all.

    Returns:
        List of single-token value strings.
    """
    from .dataset_configs import get_value_pool
    pool = get_value_pool("ARBITRARY_SINGLE")
    if n is None or n >= len(pool):
        return list(pool)
    return list(pool[:n])
