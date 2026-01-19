"""
Base model interface for LLM experiments.
All model implementations should inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseModelInterface(ABC):
    """Abstract base class for LLM model interfaces"""

    def __init__(self, model_name: str, config: Optional[Dict] = None):
        """
        Initialize model interface.

        Args:
            model_name: Name/identifier of the model
            config: Optional configuration dictionary
        """
        self.model_name = model_name
        self.config = config if config is not None else {}
        self.available = False

        # Common configuration parameters
        self.max_tokens = self.config.get('max_tokens', 2000)
        self.temperature = self.config.get('temperature', 0.0)

        # Token usage tracking (populated by generate() method)
        self.last_input_tokens = None
        self.last_output_tokens = None
        self.last_total_tokens = None
        self.last_was_truncated = False
        self.last_error = None

    @abstractmethod
    def _get_model_id(self, model_name: str) -> str:
        """
        Map friendly model names to API model IDs.

        Args:
            model_name: Friendly name (e.g., 'claude-haiku', 'gpt-4')

        Returns:
            API model identifier
        """
        pass

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate response from the model.

        Args:
            prompt: Input prompt string

        Returns:
            Generated response string
        """
        pass

    def is_available(self) -> bool:
        """Check if model is available"""
        return self.available

    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "provider": self.__class__.__name__,
            "available": self.available,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
