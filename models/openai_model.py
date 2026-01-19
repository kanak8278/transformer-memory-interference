"""
OpenAI model interface for GPT models.
Supports both standard OpenAI API and Azure OpenAI endpoints.
"""

import os
import time
from typing import Dict, Optional
from .base_model import BaseModelInterface
from . import config as model_config


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

        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Create chat completion with model-specific parameters using model_config helpers
                if model_config.should_use_max_completion_tokens(self.model_id):
                    # New models use max_completion_tokens instead of max_tokens
                    call_params = {
                        "model": self.model_id,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "max_completion_tokens": self.max_tokens
                    }

                    # Only o-series models support reasoning_effort
                    if model_config.supports_reasoning_effort(self.model_id):
                        reasoning_effort = self.config.get('reasoning_effort')
                        if reasoning_effort:
                            call_params["reasoning_effort"] = reasoning_effort  # "low", "medium", or "high"

                    # Add temperature if model supports it
                    if model_config.supports_temperature(self.model_id):
                        call_params["temperature"] = self.temperature

                    response = self.client.chat.completions.create(**call_params)
                else:
                    # Standard GPT models
                    response = self.client.chat.completions.create(
                        model=self.model_id,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )

                # Log token usage and store in instance variables
                if hasattr(response, 'usage') and response.usage:
                    usage = response.usage
                    input_tokens = usage.prompt_tokens
                    output_tokens = usage.completion_tokens
                    total_tokens = usage.total_tokens

                    # Store in instance variables for experiment tracking
                    self.last_input_tokens = input_tokens
                    self.last_output_tokens = output_tokens
                    self.last_total_tokens = total_tokens
                    self.last_was_truncated = output_tokens >= self.max_tokens

                    print(f"  Tokens: Input={input_tokens:,}, Output={output_tokens:,}, Total={total_tokens:,}")

                    # Warn if output was truncated due to max_tokens
                    if output_tokens >= self.max_tokens:
                        print(f"  ⚠️  Output may be truncated (reached max_tokens={self.max_tokens})")

                # Extract response content
                content = response.choices[0].message.content

                # Check if content is None or empty (can happen with truncation)
                if content is None:
                    print(f"  ⚠️  Response content is None!")
                    print(f"  📋 Full response object: {response}")
                    print(f"  📋 Choice finish_reason: {response.choices[0].finish_reason}")
                    return ""
                elif len(content.strip()) == 0:
                    print(f"  ⚠️  Response content is empty!")
                    print(f"  📋 Full response object: {response}")
                    print(f"  📋 Choice finish_reason: {response.choices[0].finish_reason}")

                return content

            except Exception as e:
                last_error = e
                error_str = str(e)

                # Store error in instance variable
                self.last_error = error_str

                # Check if it's a rate limit error (429)
                if "429" in error_str or "Rate limit" in error_str:
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (attempt + 1)  # Exponential backoff
                        print(f"⚠️  Rate limit hit. Waiting {wait_time}s before retry {attempt + 2}/{self.max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"❌ Rate limit exceeded after {self.max_retries} attempts")
                        return f"Error: Rate limit exceeded after {self.max_retries} retries"

                # Check if it's a timeout
                elif "timed out" in error_str.lower() or "timeout" in error_str.lower():
                    if attempt < self.max_retries - 1:
                        wait_time = 30
                        print(f"⚠️  Request timed out. Retrying {attempt + 2}/{self.max_retries} after {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"❌ Request timed out after {self.max_retries} attempts")
                        return f"Error: Request timed out after {self.max_retries} retries"

                # For other errors, don't retry
                else:
                    print(f"❌ Error calling {self.model_name}: {e}")
                    return f"Error: {str(e)}"

        # If we exhausted retries
        print(f"❌ Failed after {self.max_retries} attempts: {last_error}")
        return f"Error: {str(last_error)}"

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
