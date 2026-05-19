"""
OpenAI model interface for GPT and o-series models.
Supports standard OpenAI API, Azure OpenAI, and TR AI Platform Workspace.

TR AI Platform Workspace auth (set TR_WORKSPACE_ID env var):
  - Fetches short-lived credentials from TR token endpoint
  - Auto-refreshes when < 5 minutes from expiry (recreates AzureOpenAI client)
  - No other credentials needed — workspace_id is sufficient
"""

import os
import random
from datetime import datetime, timezone
from typing import Dict, Optional
from .base_model import BaseModelInterface
from . import config as model_config
from .retry_utils import call_with_retry

TR_OPENAI_TOKEN_URL = "https://aiplatform.gcs.int.thomsonreuters.com/v1/openai/token"
TR_OPENAI_BASE_URL  = "https://eais2-use.int.thomsonreuters.com"


class OpenAIModelInterface(BaseModelInterface):
    """OpenAI model interface supporting OpenAI API, Azure OpenAI, and TR Workspace"""

    def __init__(self, model_name: str, config: Optional[Dict] = None):
        """
        Initialize OpenAI model interface.

        Auth priority (first match wins):
          1. TR_WORKSPACE_ID env var / config['tr_workspace_id'] → TR AI Platform (AzureOpenAI)
          2. AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY        → Standard Azure OpenAI
          3. OPENAI_API_KEY                                       → Standard OpenAI

        Args:
            model_name: Model name (gpt-4.1, gpt-5, o3-mini, etc.)
            config: Optional configuration dictionary
        """
        super().__init__(model_name, config)

        if 'max_tokens' not in self.config:
            self.max_tokens = model_config.get_openai_max_tokens(model_name)

        defaults = model_config.OPENAI_DEFAULTS
        self.api_version  = self.config.get('api_version',  defaults['api_version'])
        self.timeout      = self.config.get('timeout',      defaults['timeout'])
        self.max_retries  = self.config.get('max_retries',  defaults['max_retries'])
        self.retry_delay  = self.config.get('retry_delay',  defaults['retry_delay'])

        try:
            from openai import OpenAI, AzureOpenAI
            self._OpenAI      = OpenAI
            self._AzureOpenAI = AzureOpenAI
        except ImportError:
            print("⚠️  OpenAI library not available. Install with: pip install openai")
            self.available = False
            return

        self.model_id = model_config.get_openai_model_id(model_name)

        tr_workspace_id  = self.config.get('tr_workspace_id', os.getenv('TR_WORKSPACE_ID'))
        azure_endpoint   = self.config.get('azure_endpoint',  os.getenv('AZURE_OPENAI_ENDPOINT'))
        azure_api_key    = self.config.get('azure_api_key',   os.getenv('AZURE_OPENAI_API_KEY'))
        openai_api_key   = self.config.get('api_key',         os.getenv('OPENAI_API_KEY'))

        self._use_tr_workspace = False

        if tr_workspace_id:
            self._init_tr_workspace(tr_workspace_id)
        elif azure_endpoint and azure_api_key:
            self._init_azure(azure_endpoint, azure_api_key)
        elif openai_api_key:
            self._init_openai(openai_api_key)
        else:
            print("⚠️  No OpenAI credentials found. Set TR_WORKSPACE_ID, OPENAI_API_KEY, or AZURE_OPENAI_* vars")
            self.available = False

    # ------------------------------------------------------------------
    # TR AI Platform Workspace auth
    # ------------------------------------------------------------------

    def _init_tr_workspace(self, workspace_id: str):
        """Initialize using TR AI Platform Workspace (token-brokered Azure OpenAI)."""
        self._tr_workspace_id = workspace_id
        self._tr_expires      = None
        self._use_tr_workspace = True
        self.use_azure = True

        import time as _time
        for _attempt in range(5):
            try:
                self._fetch_tr_token()
                if self.config.get('verbose', True):
                    expiry = self._tr_expires.strftime('%H:%M UTC') if self._tr_expires else "no expiry"
                    print(f"✓ TR OpenAI initialized: {self.model_id} (token: {expiry})")
                self.available = True
                break
            except Exception as e:
                if _attempt < 4:
                    wait = random.uniform(1, 4 * (2 ** _attempt))
                    print(f"⚠️  TR workspace init failed: {e} — retrying in {wait:.1f}s")
                    _time.sleep(wait)
                else:
                    print(f"⚠️  TR workspace init failed permanently: {e}")
                    self.available = False

    def _fetch_tr_token(self):
        """Fetch fresh TR credentials and recreate the AzureOpenAI client."""
        import requests

        payload = {"workspace_id": self._tr_workspace_id, "model_name": self.model_id}
        resp = requests.post(TR_OPENAI_TOKEN_URL, json=payload, timeout=15)
        resp.raise_for_status()
        creds = resp.json()

        if "openai_key" not in creds or "openai_endpoint" not in creds:
            raise RuntimeError(f"TR OpenAI token fetch failed — response: {creds}")

        # OpenAI endpoint tokens don't carry expires_on — treat as non-expiring
        self._tr_expires = None

        deployment_id    = creds["azure_deployment"]
        llm_profile_key  = deployment_id.split("/")[0]

        headers = {
            "Authorization":           f"Bearer {creds['token']}",
            "api-key":                 creds["openai_key"],
            "Content-Type":            "application/json",
            "x-tr-chat-profile-name":  "ai-platforms-chatprofile-prod",
            "x-tr-userid":             self._tr_workspace_id,
            "x-tr-llm-profile-key":    llm_profile_key,
            "x-tr-user-sensitivity":   "true",
            "x-tr-sessionid":          deployment_id,
            "x-tr-asset-id":           self._tr_workspace_id,
            "x-tr-authorization":      TR_OPENAI_BASE_URL,
        }

        # Recreate client with fresh credentials (headers are baked in at init)
        self.client = self._AzureOpenAI(
            azure_endpoint=creds["openai_endpoint"],
            api_key=creds["openai_key"],
            api_version=creds["openai_api_version"],
            azure_deployment=deployment_id,
            default_headers=headers,
            timeout=self.timeout,
        )

    def _refresh_token_if_needed(self):
        """Re-fetch TR token if expiring within 5 minutes (no-op if no expiry)."""
        if self._tr_expires is None:
            return  # OpenAI endpoint tokens don't expire
        now = datetime.now(timezone.utc)
        if (self._tr_expires - now).total_seconds() < 300:
            self._fetch_tr_token()

    # ------------------------------------------------------------------
    # Standard auth paths
    # ------------------------------------------------------------------

    def _init_azure(self, endpoint: str, api_key: str):
        """Initialize using standard Azure OpenAI."""
        self.client = self._AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=self.api_version,
            timeout=self.timeout,
        )
        self.use_azure = True
        print(f"✓ Azure OpenAI initialized for {self.model_id}")
        self.available = True

    def _init_openai(self, api_key: str):
        """Initialize using standard OpenAI API."""
        self.client = self._OpenAI(
            api_key=api_key,
            timeout=self.timeout,
        )
        self.use_azure = False
        print(f"✓ OpenAI initialized for {self.model_id}")
        self.available = True

    def _get_model_id(self, model_name: str) -> str:
        return model_config.get_openai_model_id(model_name)

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """
        Generate response from OpenAI / Azure OpenAI.

        For TR Workspace: auto-refreshes credentials if expiring within 5 minutes.

        Args:
            prompt: Input prompt string

        Returns:
            Generated response string
        """
        if not self.available:
            return f"Mock response for {self.model_name}"

        def _call():
            if self._use_tr_workspace:
                self._refresh_token_if_needed()

            if model_config.should_use_max_completion_tokens(self.model_id):
                call_params = {
                    "model":                  self.model_id,
                    "messages":               [{"role": "user", "content": prompt}],
                    "max_completion_tokens":  self.max_tokens,
                }
                if model_config.supports_reasoning_effort(self.model_id):
                    reasoning_effort = self.config.get('reasoning_effort')
                    if reasoning_effort:
                        call_params["reasoning_effort"] = reasoning_effort
                if model_config.supports_temperature(self.model_id):
                    call_params["temperature"] = self.temperature
            else:
                call_params = {
                    "model":       self.model_id,
                    "messages":    [{"role": "user", "content": prompt}],
                    "max_tokens":  self.max_tokens,
                    "temperature": self.temperature,
                }

            response = self.client.chat.completions.create(**call_params)

            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                self.last_input_tokens  = usage.prompt_tokens
                self.last_output_tokens = usage.completion_tokens
                self.last_total_tokens  = usage.total_tokens
                self.last_was_truncated = usage.completion_tokens >= self.max_tokens

            return response.choices[0].message.content or ""

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

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def get_model_info(self) -> Dict:
        info = super().get_model_info()
        if self._use_tr_workspace:
            provider = "TR AI Platform (Azure OpenAI)"
        elif getattr(self, 'use_azure', False):
            provider = "Azure OpenAI"
        else:
            provider = "OpenAI"
        info.update({
            "provider":    provider,
            "model_id":    self.model_id,
            "family":      "GPT",
            "api_version": self.api_version,
        })
        return info
