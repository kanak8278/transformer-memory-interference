"""
Claude model interface for Anthropic API.
Supports Claude 3 family: Haiku, Sonnet, Opus
"""

from typing import Dict, Optional
from .base_model import BaseModelInterface


class ClaudeModelInterface(BaseModelInterface):
    """Claude model interface for Anthropic API"""

    def __init__(self, model_name: str, config: Optional[Dict] = None):
        """
        Initialize Claude model interface.

        Args:
            model_name: Model name (claude-haiku, claude-sonnet, claude-opus)
            config: Optional configuration dictionary
        """
        super().__init__(model_name, config)

        # Import anthropic
        try:
            import anthropic
            self.client = anthropic.Anthropic()
            self.available = True
        except ImportError as e:
            error_msg = (
                "Anthropic library not installed. Install with: pip install anthropic\n"
                "Or install all dependencies: pip install -r requirements.txt"
            )
            print(f"\n⚠️  ERROR: {error_msg}\n")
            raise ImportError(error_msg)
        except Exception as e:
            error_msg = str(e)
            print(f"\n⚠️  ERROR: Claude initialization failed")
            print(f"    {error_msg}")
            print(f"\n    Make sure ANTHROPIC_API_KEY is set:")
            print(f"    export ANTHROPIC_API_KEY='your-api-key'\n")
            raise RuntimeError(f"Claude initialization failed: {error_msg}")

        # Get model ID from friendly name
        self.model_id = self._get_model_id(model_name)

    def _get_model_id(self, model_name: str) -> str:
        """
        Map friendly model names to Anthropic API model IDs.

        Args:
            model_name: Friendly name (claude-haiku, claude-sonnet, claude-opus, etc.)

        Returns:
            Anthropic API model identifier
        """
        model_mapping = {
            # Latest generation (default aliases point to 4.5)
            'claude-haiku': 'claude-haiku-4-5-20251001',
            'claude-sonnet': 'claude-sonnet-4-5-20250929',
            'claude-opus': 'claude-opus-4-5-20251101',

            # Claude 4.5 family (current as of Dec 2025)
            'claude-4.5-haiku': 'claude-haiku-4-5-20251001',
            'claude-4.5-sonnet': 'claude-sonnet-4-5-20250929',
            'claude-4.5-opus': 'claude-opus-4-5-20251101',

            # Claude 4 family
            'claude-4-sonnet': 'claude-sonnet-4-20250514',
            'claude-4-opus': 'claude-opus-4-20250514',

            # Claude 3.5 family (only Haiku is available)
            'claude-3.5-haiku': 'claude-3-5-haiku-20241022',

            # Claude 3 family (only Opus is available)
            'claude-3-opus': 'claude-3-opus-20240229',
        }
        return model_mapping.get(model_name, model_name)

    def generate(self, prompt: str) -> str:
        """
        Generate response from Claude.

        Args:
            prompt: Input prompt string

        Returns:
            Generated response string
        """
        if not self.available:
            return f"Mock response for {self.model_name}"

        try:
            message = self.client.messages.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text
        except Exception as e:
            print(f"Error calling {self.model_name}: {e}")
            return f"Error: {str(e)}"

    def get_model_info(self) -> Dict:
        """Get Claude model information"""
        info = super().get_model_info()
        info.update({
            "provider": "Anthropic",
            "model_id": self.model_id,
            "family": "Claude 3"
        })
        return info
