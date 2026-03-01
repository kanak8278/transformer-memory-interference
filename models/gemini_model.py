"""
Gemini model interface for Google's Generative AI.
Supports Google AI Studio, Vertex AI, and TR AI Platform Workspace.

Thinking mode is auto-enabled for gemini-2.5-pro.
Override via config: {'thinking': True/False, 'thinking_budget': N}
  thinking_budget = -1  → dynamic (model decides, default)
  thinking_budget =  0  → disabled
  thinking_budget =  N  → fixed N tokens for thinking

TR AI Platform Workspace auth (set TR_WORKSPACE_ID env var):
  - Fetches short-lived OAuth2 token from TR token endpoint
  - Auto-refreshes token when < 5 minutes from expiry
  - No other credentials needed
"""

import os
import warnings
from datetime import datetime, timezone
from typing import Dict, Optional
from .base_model import BaseModelInterface
from . import config as model_config
from .retry_utils import call_with_retry

# Models that use thinking mode by default
GEMINI_REASONING_MODELS = {'gemini-2.5-pro'}

# -1 = dynamic budget (model decides how much thinking to use)
GEMINI_THINKING_BUDGET = -1

# TR AI Platform token endpoint
TR_TOKEN_URL = "https://aiplatform.gcs.int.thomsonreuters.com/v1/gemini/token"


class GeminiModelInterface(BaseModelInterface):
    """Gemini model interface supporting Google AI Studio, Vertex AI, and TR Workspace"""

    def __init__(self, model_name: str, config: Optional[Dict] = None):
        """
        Initialize Gemini model interface.

        Auth priority (first match wins):
          1. TR_WORKSPACE_ID env var / config['tr_workspace_id'] → TR AI Platform
          2. GOOGLE_CLOUD_PROJECT env var / config['project_id']  → Vertex AI
          3. GOOGLE_API_KEY env var / config['api_key']           → Google AI Studio

        Args:
            model_name: Model name (gemini-2.0-flash, gemini-2.5-pro, etc.)
            config: Optional configuration dictionary
        """
        super().__init__(model_name, config)

        # Override max_tokens using model_config
        if 'max_tokens' not in self.config:
            self.max_tokens = model_config.GEMINI_MAX_TOKENS

        # Timeout and retry configuration
        self.timeout = self.config.get('timeout', model_config.GEMINI_TIMEOUT)
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 30)

        # Resolve whether to use thinking mode
        self.use_thinking = self.config.get(
            'thinking',
            model_name in GEMINI_REASONING_MODELS
        )
        self.thinking_budget = self.config.get('thinking_budget', GEMINI_THINKING_BUDGET)

        # Get API model ID
        self.model_id = self._get_model_id(model_name)

        # Auth detection — priority order
        tr_workspace_id  = self.config.get('tr_workspace_id', os.getenv('TR_WORKSPACE_ID'))
        use_vertex       = self.config.get('use_vertex', False)
        project_id       = self.config.get('project_id', os.getenv('GOOGLE_CLOUD_PROJECT'))
        google_api_key   = self.config.get('api_key', os.getenv('GOOGLE_API_KEY'))

        self._use_tr_workspace = False

        if tr_workspace_id:
            self._init_tr_workspace(tr_workspace_id)
        elif use_vertex or project_id:
            self._init_vertex_ai(project_id)
        elif google_api_key:
            self._init_google_ai(google_api_key)
        else:
            print(f"⚠️  No Gemini credentials found. Set TR_WORKSPACE_ID, GOOGLE_API_KEY, or GOOGLE_CLOUD_PROJECT")
            self.available = False

    # ------------------------------------------------------------------
    # TR AI Platform Workspace auth
    # ------------------------------------------------------------------

    def _init_tr_workspace(self, workspace_id: str):
        """Initialize using TR AI Platform Workspace (token-brokered Vertex AI)."""
        self._tr_workspace_id = workspace_id
        self._tr_expires = None   # will be set on first fetch
        self._use_tr_workspace = True
        self.use_vertex = True

        # Eagerly fetch the first token to validate credentials at init time
        try:
            self._fetch_tr_token()
            mode = "thinking" if self.use_thinking else "standard"
            if self.config.get('verbose', True):
                print(f"✓ TR Gemini initialized: {self.model_id} [{mode}] (token expires: {self._tr_expires.strftime('%H:%M UTC')})")
            self.available = True
        except Exception as e:
            print(f"⚠️  TR workspace init failed: {e}")
            self.available = False

    def _fetch_tr_token(self):
        """Fetch a fresh token from TR token endpoint and re-initialize vertexai."""
        import requests
        from google.oauth2.credentials import Credentials as OAuth2Credentials

        # Suppress deprecation warning from vertexai.generative_models
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            import vertexai
            from vertexai.generative_models import GenerativeModel

        payload = {"workspace_id": self._tr_workspace_id, "model_name": self.model_id}
        resp = requests.post(TR_TOKEN_URL, json=payload, timeout=15)
        resp.raise_for_status()
        creds = resp.json()

        if "token" not in creds:
            raise RuntimeError(f"TR token fetch failed — response: {creds}")

        # Parse expiry: "2026-03-01 14:01 UTC+0000"
        expires_clean = creds["expires_on"].replace(" UTC+0000", "").strip()
        self._tr_expires = datetime.strptime(expires_clean, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        self.project_id = creds["project_id"]
        self.location   = creds["region"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vertexai.init(
                project=self.project_id,
                location=self.location,
                credentials=OAuth2Credentials(creds["token"]),
            )
            self.model = GenerativeModel(self.model_id)

    def _refresh_token_if_needed(self):
        """Re-fetch token if expired or expiring within 5 minutes."""
        now = datetime.now(timezone.utc)
        if self._tr_expires is None or (self._tr_expires - now).total_seconds() < 300:
            self._fetch_tr_token()

    # ------------------------------------------------------------------
    # Standard auth paths
    # ------------------------------------------------------------------

    def _init_google_ai(self, api_key: str):
        """Initialize using Google AI Studio (generative AI SDK)."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_id)
            self.use_vertex = False
            mode = "thinking" if self.use_thinking else "standard"
            print(f"✓ Google AI initialized: {self.model_id} [{mode}]")
            self.available = True

        except ImportError as e:
            print(f"⚠️  Google Generative AI library not available: {e}")
            print("Install with: pip install google-generativeai")
            self.available = False

    def _init_vertex_ai(self, project_id: str):
        """Initialize using standard Vertex AI (ADC credentials)."""
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel

            location = self.config.get('location', os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1'))

            if not project_id:
                print(f"⚠️  Vertex AI requires GOOGLE_CLOUD_PROJECT environment variable")
                self.available = False
                return

            vertexai.init(project=project_id, location=location)
            self.model = GenerativeModel(self.model_id)
            self.use_vertex = True
            self.project_id = project_id
            self.location = location
            print(f"✓ Vertex AI initialized for {self.model_id} (project: {project_id})")
            self.available = True

        except ImportError as e:
            print(f"⚠️  Vertex AI libraries not available: {e}")
            print("Install with: pip install google-cloud-aiplatform")
            self.available = False

    # ------------------------------------------------------------------
    # Model ID mapping
    # ------------------------------------------------------------------

    def _get_model_id(self, model_name: str) -> str:
        if model_name in model_config.GEMINI_MODEL_IDS:
            return model_config.GEMINI_MODEL_IDS[model_name]
        model_mapping = {
            'gemini-pro':   'gemini-1.5-pro',
            'gemini-flash': 'gemini-1.5-flash',
        }
        return model_mapping.get(model_name, model_name)

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """
        Generate response from Gemini.

        For TR Workspace: auto-refreshes token if expiring within 5 minutes.

        Args:
            prompt: Input prompt string

        Returns:
            Generated response string
        """
        if not self.available:
            return f"Mock response for {self.model_name}"

        def _call():
            # Refresh TR token if needed (fast ~100ms, only when near expiry)
            if self._use_tr_workspace:
                self._refresh_token_if_needed()

            gen_config = {
                "max_output_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
            if self.use_thinking:
                gen_config["thinking_config"] = {
                    "thinking_budget": self.thinking_budget
                }

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                response = self.model.generate_content(prompt, generation_config=gen_config)

            if hasattr(response, 'text'):
                text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                text = response.candidates[0].content.parts[0].text
            else:
                text = ""

            if hasattr(response, 'usage_metadata'):
                um = response.usage_metadata
                self.last_input_tokens  = getattr(um, 'prompt_token_count', None)
                self.last_output_tokens = getattr(um, 'candidates_token_count', None)
                self.last_total_tokens  = getattr(um, 'total_token_count', None)

            return text

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
            provider = "TR AI Platform (Vertex AI)"
        elif getattr(self, 'use_vertex', False):
            provider = "Google Vertex AI"
        else:
            provider = "Google AI Studio"
        info.update({
            "provider":        provider,
            "model_id":        self.model_id,
            "family":          "Gemini",
            "use_thinking":    self.use_thinking,
            "thinking_budget": self.thinking_budget if self.use_thinking else None,
        })
        if getattr(self, 'use_vertex', False):
            info["project_id"] = getattr(self, 'project_id', 'unknown')
            info["location"]   = getattr(self, 'location', 'unknown')
        return info
