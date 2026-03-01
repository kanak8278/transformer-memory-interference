"""
OpenAI model interface for GPT models.
Supports both standard OpenAI API and Azure OpenAI endpoints.
"""

import os
import time
from typing import Dict, Optional
from .base_model import BaseModelInterface
from . import config as model_config
from .retry_utils import call_with_retry


class OpenAIModelInterface(BaseModelInterface):
    """OpenAI model interface supporting both OpenAI API and Azure OpenAI"""

    def __init__(self, model_name: str, config: Optional[Dict] = None):
        """
        Initialize OpenAI model interface.

        Args:
            model_name: Model name (gpt-4, gpt-4o, o1, etc.)
            config: Optional configuration dictionary
                Optional keys:
                - api_key: OpenAI API key (or set OPENAI_API_KEY env var)
                - azure_endpoint: Azure OpenAI endpoint (for Azure deployments)
                - azure_api_key: Azure OpenAI API key (or set AZURE_OPENAI_API_KEY)
                - api_version: API version for Azure (default: "2023-03-15-preview")
                - use_azure: Force Azure OpenAI (default: auto-detect from env vars)
        """
        super().__init__(model_name, config)

        # Override max_tokens using model_config helper
        if 'max_tokens' not in self.config:
            self.max_tokens = model_config.get_openai_max_tokens(model_name)

        # Configuration from environment or config dict
        defaults = model_config.OPENAI_DEFAULTS
        self.api_version = self.config.get('api_version', defaults['api_version'])
        self.timeout = self.config.get('timeout', defaults['timeout'])
        self.max_retries = self.config.get('max_retries', defaults['max_retries'])
        self.retry_delay = self.config.get('retry_delay', defaults['retry_delay'])

        # Initialize OpenAI
        try:
            from openai import OpenAI, AzureOpenAI

            self.OpenAI = OpenAI
            self.AzureOpenAI = AzureOpenAI

            # Get API model ID using model_config
            self.model_id = model_config.get_openai_model_id(model_name)

            # Determine if using Azure or standard OpenAI
            use_azure = self.config.get('use_azure', False)
            azure_endpoint = self.config.get('azure_endpoint', os.getenv('AZURE_OPENAI_ENDPOINT'))
            azure_api_key = self.config.get('azure_api_key', os.getenv('AZURE_OPENAI_API_KEY'))
            openai_api_key = self.config.get('api_key', os.getenv('OPENAI_API_KEY'))

            if use_azure or (azure_endpoint and azure_api_key):
                # Use Azure OpenAI
                if not azure_endpoint or not azure_api_key:
                    print(f"⚠️  Azure OpenAI requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY")
                    self.available = False
                    return

                self.client = self.AzureOpenAI(
                    azure_endpoint=azure_endpoint,
                    api_key=azure_api_key,
                    api_version=self.api_version,
                    timeout=self.timeout,
                )
                self.use_azure = True
                print(f"✓ Azure OpenAI initialized for {self.model_id}")
                self.available = True

            elif openai_api_key:
                # Use standard OpenAI API
                self.client = self.OpenAI(
                    api_key=openai_api_key,
                    timeout=self.timeout,
                )
                self.use_azure = False
                print(f"✓ OpenAI initialized for {self.model_id}")
                self.available = True

            else:
                print(f"⚠️  No OpenAI API key found. Set OPENAI_API_KEY or AZURE_OPENAI_API_KEY")
                self.available = False

        except ImportError as e:
            print(f"⚠️  OpenAI library not available: {e}")
            print("Install with: pip install openai")
            self.available = False

    def _get_model_id(self, model_name: str) -> str:
        """Map friendly model names to OpenAI model IDs (delegates to config)"""
        return model_config.get_openai_model_id(model_name)

    def generate(self, prompt: str) -> str:
        """
        Generate response from OpenAI with retry logic for rate limits.

        Args:
            prompt: Input prompt string

        Returns:
            Generated response string
        """
        if not self.available:
            return f"Mock response for {self.model_name}"

        def _call():
            if model_config.should_use_max_completion_tokens(self.model_id):
                call_params = {
                    "model": self.model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": self.max_tokens,
                }
                if model_config.supports_reasoning_effort(self.model_id):
                    reasoning_effort = self.config.get('reasoning_effort')
                    if reasoning_effort:
                        call_params["reasoning_effort"] = reasoning_effort
                if model_config.supports_temperature(self.model_id):
                    call_params["temperature"] = self.temperature
                response = self.client.chat.completions.create(**call_params)
            else:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )

            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                self.last_input_tokens  = usage.prompt_tokens
                self.last_output_tokens = usage.completion_tokens
                self.last_total_tokens  = usage.total_tokens
                self.last_was_truncated = usage.completion_tokens >= self.max_tokens

            content = response.choices[0].message.content
            return content or ""

        try:
            result = call_with_retry(
                _call,
                max_retries=self.max_retries,
                base_delay=self.retry_delay,
                label=self.model_name,
            )
            self.last_error = None
            return result
        except Exception as e:
            self.last_error = str(e)
            print(f"  ✗ {self.model_name} failed permanently: {str(e)[:80]}")
            return f"Error: {str(e)}"

    def get_model_info(self) -> Dict:
        """Get OpenAI model information"""
        info = super().get_model_info()
        provider = "Azure OpenAI" if getattr(self, 'use_azure', False) else "OpenAI"
        info.update({
            "provider": provider,
            "model_id": self.model_id,
            "family": "GPT",
            "api_version": self.api_version
        })
        return info
