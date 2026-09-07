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


def _sdk_takes_temperature() -> bool:
    """Whether the installed SDK accepts `temperature` as a named argument.

    Probed once at import: the check needs a client instance, and building one
    per request would be wasteful in a sweep making tens of thousands of calls.
    """
    try:
        import inspect

        import anthropic as _a
        return "temperature" in inspect.signature(_a.Anthropic().messages.create).parameters
    except Exception:  # noqa: BLE001 - assume 1.x behaviour and use extra_body
        return False


_SDK_TAKES_TEMPERATURE = _sdk_takes_temperature()


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

        # Optional assistant-turn prefill. The model continues from this string
        # instead of starting fresh, which is the only reliable way to stop it
        # reasoning out loud — instructions alone do not ("No explanation" is
        # routinely ignored on counting tasks). Available on the Claude 4.5 line;
        # 4.6+ rejects prefills, so this is tied to the pinned snapshots.
        self.prefill = self.config.get('prefill')

        # Optional stop sequences. Paired with a prefill, '</answer>' makes the
        # no-reasoning arm airtight: generation halts at the closing tag, so the
        # model cannot append a self-correction after committing. Without it,
        # models do continue past the tag and reason their way to a different
        # answer — which would quietly make the control arm a CoT arm.
        self.stop_sequences = self.config.get('stop_sequences')
        if self.prefill and self.use_thinking:
            raise ValueError(
                "prefill and extended thinking are mutually exclusive: "
                "an assistant prefill cannot precede a thinking block."
            )

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

        # Side-channel outputs of the most recent generate() call, following the
        # same `last_*` convention as base_model.py so callers need no interface
        # change to read them.
        self.last_thinking_blocks = []
        self.last_text_block_present = None
        self.last_stop_reason = None

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

    def _temperature_kwargs(self, temperature: float) -> Dict:
        """Pass `temperature` in whichever way the installed SDK accepts.

        anthropic 1.x removed `temperature` / `top_p` / `top_k` from the
        Messages.create() signature — passing one is a TypeError, not a 400. The
        parameter is gone from the SDK, NOT from the API, and the Claude 4.5 line
        (Haiku 4.5, Sonnet 4.5, Opus 4.5) still honours it, so on 1.x it travels
        in extra_body and is merged into the request JSON as-is. Opus 4.7+ would
        400 on it either way.

        Probed once at import (see _SDK_TAKES_TEMPERATURE) rather than per call.
        """
        if _SDK_TAKES_TEMPERATURE:
            return {"temperature": temperature}
        return {"extra_body": {"temperature": temperature}}

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
                # Extended thinking requires temperature=1 on this model family,
                # and max_tokens must exceed the thinking budget (thinking tokens
                # are drawn from the same allowance as visible output).
                max_tok = max(self.max_tokens, self.thinking_budget + 2048)
                # Extended thinking requires temperature=1, which is the API
                # default — omitted rather than passed, so no SDK-version
                # handling is needed on this path.
                msg = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=max_tok,
                    thinking={"type": "enabled", "budget_tokens": self.thinking_budget},
                    messages=[{"role": "user", "content": prompt}]
                )
            else:
                messages = [{"role": "user", "content": prompt}]
                if self.prefill:
                    messages.append({"role": "assistant", "content": self.prefill})
                extra = {}
                if self.stop_sequences:
                    extra["stop_sequences"] = list(self.stop_sequences)
                msg = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=self.max_tokens,
                    messages=messages,
                    **extra,
                    **self._temperature_kwargs(self.temperature),
                )

            # Type-filtered on BOTH paths. Positional indexing (content[0]) breaks
            # as soon as a non-text block is emitted first. `None` (no text block
            # at all — e.g. thinking exhausted the allowance) is a distinct outcome
            # from "" and must not be scored as a wrong answer.
            text_block = next((b.text for b in msg.content if b.type == "text"), None)

            # Reasoning is the object of study here, so keep it rather than
            # discarding it. `thinking` blocks carry a summary of the chain (not
            # the verbatim trace); `redacted_thinking` blocks carry encrypted data.
            self.last_thinking_blocks = [
                {
                    "type": b.type,
                    "text": getattr(b, "thinking", None),
                    "data": getattr(b, "data", None),
                }
                for b in msg.content
                if b.type in ("thinking", "redacted_thinking")
            ]

            # The response continues the prefill, so the prefill itself is not
            # echoed back. Stitch it on, or downstream parsing sees a value with
            # no opening tag.
            if self.prefill and text_block is not None:
                text_block = self.prefill + text_block

            # A triggered stop sequence is excluded from the returned text, so
            # put it back — otherwise the closing tag is missing and the answer
            # parses as truncated.
            if (text_block is not None
                    and msg.stop_reason == "stop_sequence"
                    and getattr(msg, "stop_sequence", None)):
                text_block = text_block + msg.stop_sequence

            self.last_text_block_present = text_block is not None
            self.last_stop_reason   = msg.stop_reason
            self.last_was_truncated = msg.stop_reason == "max_tokens"
            self.last_input_tokens  = msg.usage.input_tokens
            self.last_output_tokens = msg.usage.output_tokens
            self.last_total_tokens  = msg.usage.input_tokens + msg.usage.output_tokens
            self.last_error         = None
            return text_block if text_block is not None else ""

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
