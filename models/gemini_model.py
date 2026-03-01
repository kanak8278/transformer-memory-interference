"""
Gemini model interface for Google's Generative AI.
Supports both Google AI Studio API and Vertex AI.

Thinking mode is auto-enabled for gemini-2.5-pro.
Override via config: {'thinking': True/False, 'thinking_budget': N}
  thinking_budget = -1  → dynamic (model decides, default)
  thinking_budget =  0  → disabled
  thinking_budget =  N  → fixed N tokens for thinking
"""

import os
import time
from typing import Dict, Optional
from .base_model import BaseModelInterface
from . import config as model_config
from .retry_utils import call_with_retry

# Models that use thinking mode by default
GEMINI_REASONING_MODELS = {'gemini-2.5-pro'}

# -1 = dynamic budget (model decides how much thinking to use)
GEMINI_THINKING_BUDGET = -1


class GeminiModelInterface(BaseModelInterface):
    """Gemini model interface supporting Google AI Studio and Vertex AI"""

    def __init__(self, model_name: str, config: Optional[Dict] = None):
        """
        Initialize Gemini model interface.

        Args:
            model_name: Model name (gemini-2.0-flash, gemini-2.5-pro, etc.)
            config: Optional configuration dictionary
                Optional keys:
                - api_key: Google AI API key (or set GOOGLE_API_KEY env var)
                - use_vertex: Use Vertex AI instead of Google AI Studio
                - project_id: GCP project ID (for Vertex AI)
                - location: GCP location (for Vertex AI, default: us-central1)
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

        # Determine if using Vertex AI or Google AI Studio
        use_vertex = self.config.get('use_vertex', False)
        project_id = self.config.get('project_id', os.getenv('GOOGLE_CLOUD_PROJECT'))
        google_api_key = self.config.get('api_key', os.getenv('GOOGLE_API_KEY'))

        if use_vertex or project_id:
            # Use Vertex AI
            self._init_vertex_ai(project_id)
        elif google_api_key:
            # Use Google AI Studio (generative AI SDK)
            self._init_google_ai(google_api_key)
        else:
            print(f"⚠️  No Google API key found. Set GOOGLE_API_KEY or configure Vertex AI")
            self.available = False

    def _init_google_ai(self, api_key: str):
        """Initialize using Google AI Studio (generative AI SDK)"""
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
        """Initialize using Vertex AI"""
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
            print("Install with: pip install google-cloud-aiplatform vertexai")
            self.available = False

    def _get_model_id(self, model_name: str) -> str:
        """
        Map friendly model names to Gemini model IDs.

        Args:
            model_name: Friendly name (gemini-2.0-flash, gemini-2.5-pro, etc.)

        Returns:
            Gemini model identifier
        """
        # Check config mapping first
        if model_name in model_config.GEMINI_MODEL_IDS:
            return model_config.GEMINI_MODEL_IDS[model_name]

        # Legacy mappings
        model_mapping = {
            'gemini-pro': 'gemini-1.5-pro',
            'gemini-flash': 'gemini-1.5-flash',
        }
        return model_mapping.get(model_name, model_name)

    def generate(self, prompt: str) -> str:
        """
        Generate response from Gemini.

        Args:
            prompt: Input prompt string

        Returns:
            Generated response string
        """
        if not self.available:
            return f"Mock response for {self.model_name}"

        def _call():
            gen_config = {
                "max_output_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
            if self.use_thinking:
                gen_config["thinking_config"] = {
                    "thinking_budget": self.thinking_budget
                }

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

    def get_model_info(self) -> Dict:
        """Get Gemini model information"""
        info = super().get_model_info()
        provider = "Google Vertex AI" if getattr(self, 'use_vertex', False) else "Google AI Studio"
        info.update({
            "provider":       provider,
            "model_id":       self.model_id,
            "family":         "Gemini",
            "use_thinking":   self.use_thinking,
            "thinking_budget": self.thinking_budget if self.use_thinking else None,
        })
        if getattr(self, 'use_vertex', False):
            info["project_id"] = getattr(self, 'project_id', 'unknown')
            info["location"] = getattr(self, 'location', 'unknown')
        return info
