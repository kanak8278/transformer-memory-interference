"""
Model factory for creating model instances based on provider.
"""

from typing import Dict, Optional
from .base_model import BaseModelInterface
from .claude_model import ClaudeModelInterface
from .gemini_model import GeminiModelInterface
from .openai_model import OpenAIModelInterface
from .llama_model import LlamaModelInterface
from .bedrock_model import BedrockModelInterface


def create_model(model_name: str, config: Optional[Dict] = None) -> BaseModelInterface:
    """
    Create a model instance based on the model name.

    Args:
        model_name: Model identifier (e.g., 'claude-haiku', 'gemini-pro', 'gpt-4', 'llama-3.1-8b', 'bedrock-llama-3-70b')
        config: Optional configuration dictionary

    Returns:
        Model interface instance

    Raises:
        ValueError: If model provider is not recognized
    """
    # Determine provider from model name
    if model_name.startswith('bedrock-'):
        return BedrockModelInterface(model_name, config)
    elif model_name.startswith('claude'):
        return ClaudeModelInterface(model_name, config)
    elif model_name.startswith('gemini'):
        return GeminiModelInterface(model_name, config)
    elif model_name.startswith('gpt') or model_name.startswith('o1') or model_name.startswith('o3') or model_name.startswith('o4'):
        return OpenAIModelInterface(model_name, config)
    elif model_name.startswith('llama'):
        return LlamaModelInterface(model_name, config)
    else:
        raise ValueError(f"Unknown model provider for model: {model_name}")


def list_available_models() -> Dict[str, list]:
    """
    List all available models by provider.

    Returns:
        Dictionary mapping provider names to list of model names
    """
    return {
        "claude": [
            "claude-haiku (claude-3-5-haiku-20241022)",
            "claude-sonnet (claude-sonnet-4-20250514)",
            "claude-opus (claude-3-opus-20240229)",
        ],
        "gemini": [
            "gemini-pro (gemini-1.5-pro)",
            "gemini-flash (gemini-1.5-flash)",
            "gemini-2.5-pro",
        ],
        "openai": [
            # Legacy GPT models
            "gpt-4",
            "gpt-4-turbo",
            "gpt-35-turbo (gpt-3.5-turbo)",
            # GPT-4o models
            "gpt-4o",
            "gpt-4o-mini",
            # GPT-5 models
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            # GPT-4.1 models
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            # o-series reasoning models
            "o1",
            "o1-mini",
            "o1-preview",
            "o3",
            "o3-mini",
            "o4-mini",
        ],
        "llama": [
            "llama-3.1-8b (meta-llama/Llama-3.1-8B) ✓ HF Router",
            "llama-3.2-1b (meta-llama/Llama-3.2-1B) ✓ HF Router",
            "llama-3.2-3b (meta-llama/Llama-3.2-3B) ✓ HF Router",
            "llama-3-8b (meta-llama/Meta-Llama-3-8B) ✓ HF Router",
            "llama-3-70b (meta-llama/Meta-Llama-3-70B) ✓ HF Router",
            "llama-3.1-70b (meta-llama/Llama-3.1-70B) ⚠️ Bedrock only",
            "llama-3.1-405b (meta-llama/Llama-3.1-405B) ⚠️ Bedrock only",
        ],
        "bedrock": [
            # Llama via Bedrock
            "bedrock-llama-3-8b (meta.llama3-8b-instruct-v1:0)",
            "bedrock-llama-3-70b (meta.llama3-70b-instruct-v1:0)",
            "bedrock-llama-3.1-8b (meta.llama3-1-8b-instruct-v1:0)",
            "bedrock-llama-3.1-70b (meta.llama3-1-70b-instruct-v1:0)",
            "bedrock-llama-3.2-1b (meta.llama3-2-1b-instruct-v1:0)",
            "bedrock-llama-3.2-3b (meta.llama3-2-3b-instruct-v1:0)",
            "bedrock-llama-3.2-90b (meta.llama3-2-90b-instruct-v1:0)",
            "bedrock-llama-3.3-70b (meta.llama3-3-70b-instruct-v1:0)",
            # Amazon Titan
            "bedrock-titan-large (amazon.titan-tg1-large)",
            "bedrock-titan-express (amazon.titan-text-express-v1)",
            # Amazon Nova
            "bedrock-nova-micro (amazon.nova-micro-v1:0)",
            "bedrock-nova-lite (amazon.nova-lite-v1:0)",
            "bedrock-nova-pro (amazon.nova-pro-v1:0)",
            # Anthropic Claude via Bedrock
            "bedrock-claude-3-haiku (anthropic.claude-3-haiku-20240307-v1:0)",
            "bedrock-claude-3.5-sonnet (anthropic.claude-3-5-sonnet-20241022-v2:0)",
            "bedrock-claude-4-sonnet (anthropic.claude-sonnet-4-20250514-v1:0)",
            # Mistral
            "bedrock-mistral-7b (mistral.mistral-7b-instruct-v0:2)",
            "bedrock-mistral-8x7b (mistral.mixtral-8x7b-instruct-v0:1)",
            "bedrock-mistral-large (mistral.mistral-large-2402-v1:0)",
            # Others
            "bedrock-deepseek-r1 (deepseek.r1-v1:0)",
            "bedrock-cohere-command-r (cohere.command-r-v1:0)",
        ]
    }
