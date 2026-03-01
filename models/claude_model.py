"""
Claude model interface for Anthropic API.
Supports Claude 3, 4, 4.5 families including extended thinking (reasoning) mode.

Extended thinking is auto-enabled for opus models (claude-4.5-opus, claude-4-opus).
Override via config: {'thinking': True/False, 'thinking_budget': N}
"""

import time
from typing import Dict, Optional
from .base_model import BaseModelInterface
from .retry_utils import call_with_retry

# Models that use extended thinking by default
CLAUDE_REASONING_MODELS = {'claude-4.5-opus', 'claude-4-opus'}

# Extended thinking token budget (tokens reserved for internal reasoning)
THINKING_BUDGET_TOKENS = 8000
# max_tokens must exceed budget — output tokens on top of thinking budget
THINKING_MAX_TOKENS = 10000


class ClaudeModelInterface(BaseModelInterface):
    """Claude model interface for Anthropic API"""

    def __init__(self, model_name: str, config: Optional[Dict] = None):
        """
        Initialize Claude model interface.

        Args:
            model_name: Model name (claude-haiku, claude-sonnet, claude-opus, etc.)
            config: Optional configuration dictionary
                Optional keys:
                - thinking: bool — force enable/disable extended thinking
                            (default: auto-detect from model name)
                - thinking_budget: int — token budget for thinking
                            (default: THINKING_BUDGET_TOKENS = 8000)
        """
        super().__init__(model_name, config)

        # Resolve whether to use extended thinking
        self.use_thinking = self.config.get(
            'thinking',
            model_name in CLAUDE_REASONING_MODELS
        )
        self.thinking_budget = self.config.get('thinking_budget', THINKING_BUDGET_TOKENS)

        try:
            import anthropic
            self.client = anthropic.Anthropic()
            self.available = True
        except ImportError:
            error_msg = (
                "Anthropic library not installed. Install with: pip install anthropic"
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

        self.model_id = self._get_model_id(model_name)

        mode = "extended thinking" if self.use_thinking else "standard"
        if self.config.get('verbose', True):
            print(f"✓ Claude initialized: {self.model_id} [{mode}]")

    def _get_model_id(self, model_name: str) -> str:
        model_mapping = {
            # Latest generation aliases
            'claude-haiku':  'claude-haiku-4-5-20251001',
            'claude-sonnet': 'claude-sonnet-4-5-20250929',
            'claude-opus':   'claude-opus-4-5-20251101',

            # Claude 4.5 family
            'claude-4.5-haiku':  'claude-haiku-4-5-20251001',
            'claude-4.5-sonnet': 'claude-sonnet-4-5-20250929',
            'claude-4.5-opus':   'claude-opus-4-5-20251101',

            # Claude 4 family
            'claude-4-sonnet': 'claude-sonnet-4-20250514',
            'claude-4-opus':   'claude-opus-4-20250514',

            # Claude 3.5 family
            'claude-3.5-haiku': 'claude-3-5-haiku-20241022',

            # Claude 3 family
            'claude-3-opus': 'claude-3-opus-20240229',
        }
        return model_mapping.get(model_name, model_name)

    def generate(self, prompt: str) -> str:
        """
        Generate response from Claude.

        For reasoning models (claude-4.5-opus, claude-4-opus):
          - Uses extended thinking API with thinking budget
          - temperature is forced to 1 (Anthropic requirement)
          - Extracts only the text block, discarding the thinking block

        Args:
            prompt: Input prompt string

        Returns:
            Generated response string (thinking content stripped)
        """
        if not self.available:
            return f"Mock response for {self.model_name}"

        def _call():
            if self.use_thinking:
                msg = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=THINKING_MAX_TOKENS,
                    temperature=1,
                    thinking={"type": "enabled", "budget_tokens": self.thinking_budget},
                    messages=[{"role": "user", "content": prompt}]
                )
                text = next((b.text for b in msg.content if b.type == "text"), "")
            else:
                msg = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                text = msg.content[0].text

            self.last_input_tokens  = msg.usage.input_tokens
            self.last_output_tokens = msg.usage.output_tokens
            self.last_total_tokens  = msg.usage.input_tokens + msg.usage.output_tokens
            self.last_error         = None
            return text

        try:
            return call_with_retry(
                _call,
                max_retries=self.config.get('max_retries', 5),
                base_delay=self.config.get('retry_base_delay', 2.0),
                label=self.model_name,
            )
        except Exception as e:
            self.last_error = str(e)
            print(f"  ✗ {self.model_name} failed permanently: {str(e)[:80]}")
            return f"Error: {str(e)}"

    def get_model_info(self) -> Dict:
        info = super().get_model_info()
        info.update({
            "provider":      "Anthropic",
            "model_id":      self.model_id,
            "family":        "Claude",
            "use_thinking":  self.use_thinking,
            "thinking_budget": self.thinking_budget if self.use_thinking else None,
        })
        return info
