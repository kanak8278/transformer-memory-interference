"""
Llama model interface supporting multiple access methods:
1. Hugging Face Router API (Primary - no downloads, fast)
2. AWS Bedrock (if AWS credentials available)
3. Hugging Face Inference API (fallback)
4. Local Transformers (last resort, requires 16GB+ download)
"""

import os
import time
import json
from typing import Dict, Optional
from .base_model import BaseModelInterface


class LlamaModelInterface(BaseModelInterface):
    """Llama model interface with multiple backend support"""

    def __init__(self, model_name: str, config: Optional[Dict] = None):
        """
        Initialize Llama model interface.

        Args:
            model_name: Model name (llama-3.1-8b, llama-3.1-70b, llama-3.2-1b, etc.)
            config: Optional configuration dictionary
                Common keys:
                - access_method: "hf_router", "bedrock", "hf_api", or "local" (default: auto-detect)
                - hf_token: Hugging Face API token (for HF Router/API)
                - aws_region: AWS region for Bedrock (default: "us-east-1")
                - max_retries: Number of retries (default: 3)
                - retry_delay: Delay between retries (default: 30)
        """
        super().__init__(model_name, config)

        # Override max_tokens default for Llama models
        # Llama uses verbose list format, needs more tokens for batch responses
        if 'max_tokens' not in self.config:
            self.max_tokens = 8000  # Enough for 46 categories in list format

        # Get HF token from config or environment
        self.hf_token = self.config.get('hf_token') or os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')

        # Retry configuration
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 30)

        # AWS configuration
        self.aws_region = self.config.get('aws_region', 'us-east-1')

        # Get model ID
        self.model_id = self._get_model_id(model_name)

        # Auto-detect best access method if not specified
        access_method = self.config.get('access_method', 'auto')

        if access_method == 'auto':
            self._auto_detect_access_method()
        else:
            self.access_method = access_method
            self._initialize_backend()

    def _auto_detect_access_method(self):
        """Auto-detect the best available access method"""

        # Try HF Router API first (best: no downloads, fast)
        if self.hf_token:
            print(f"✓ HF_TOKEN found, using Hugging Face Router API")
            self.access_method = 'hf_router'
            self._init_hf_router()
            return

        # Try AWS Bedrock second
        try:
            import boto3
            boto3.client('sts', region_name=self.aws_region).get_caller_identity()
            print(f"✓ AWS credentials valid, using Bedrock")
            self.access_method = 'bedrock'
            self._init_bedrock()
            return
        except:
            pass

        # Fallback to HF Inference API
        print(f"⚠️  No HF_TOKEN or AWS credentials found")
        print(f"⚠️  Set HF_TOKEN for best performance: export HF_TOKEN=your_token")
        self.access_method = None
        self.available = False

    def _initialize_backend(self):
        """Initialize the specified backend"""
        if self.access_method == 'hf_router':
            self._init_hf_router()
        elif self.access_method == 'bedrock':
            self._init_bedrock()
        elif self.access_method == 'hf_api':
            self._init_hf_api()
        elif self.access_method == 'local':
            self._init_local()
        else:
            print(f"⚠️  Unknown access method: {self.access_method}")
            self.available = False

    def _get_model_id(self, model_name: str) -> str:
        """Map friendly model names to actual model IDs"""
        # HF Router mapping (with provider suffix)
        hf_router_mapping = {
            'llama-3.1-8b': 'meta-llama/Llama-3.1-8B-Instruct:novita',
            'llama-3.1-70b': 'meta-llama/Llama-3.1-70B-Instruct:novita',
            'llama-3.1-405b': 'meta-llama/Llama-3.1-405B-Instruct:novita',
            'llama-3.2-1b': 'meta-llama/Llama-3.2-1B-Instruct:novita',
            'llama-3.2-3b': 'meta-llama/Llama-3.2-3B-Instruct:novita',
            'llama-3-8b': 'meta-llama/Meta-Llama-3-8B-Instruct:novita',
            'llama-3-70b': 'meta-llama/Meta-Llama-3-70B-Instruct:novita',
        }

        # Store both mappings
        self.hf_router_id = hf_router_mapping.get(model_name, model_name)

        # Bedrock mapping
        bedrock_mapping = {
            'llama-3.1-8b': 'meta.llama3-1-8b-instruct-v1:0',
            'llama-3.1-70b': 'meta.llama3-1-70b-instruct-v1:0',
            'llama-3.1-405b': 'meta.llama3-1-405b-instruct-v1:0',
            'llama-3-8b': 'meta.llama3-8b-instruct-v1:0',
            'llama-3-70b': 'meta.llama3-70b-instruct-v1:0',
        }
        self.bedrock_id = bedrock_mapping.get(model_name, model_name)

        # Default HF model ID
        hf_mapping = {
            'llama-3.1-8b': 'meta-llama/Llama-3.1-8B-Instruct',
            'llama-3.1-70b': 'meta-llama/Llama-3.1-70B-Instruct',
            'llama-3.1-405b': 'meta-llama/Llama-3.1-405B-Instruct',
            'llama-3.2-1b': 'meta-llama/Llama-3.2-1B-Instruct',
            'llama-3.2-3b': 'meta-llama/Llama-3.2-3B-Instruct',
            'llama-3-8b': 'meta-llama/Meta-Llama-3-8B-Instruct',
            'llama-3-70b': 'meta-llama/Meta-Llama-3-70B-Instruct',
        }
        return hf_mapping.get(model_name, model_name)

    def _init_hf_router(self):
        """Initialize Hugging Face Router API (BEST - no downloads!)"""
        try:
            import requests
            self.requests = requests

            if not self.hf_token:
                print(f"⚠️  HF_TOKEN not found. Set: export HF_TOKEN=your_token")
                self.available = False
                return

            self.api_url = "https://router.huggingface.co/v1/chat/completions"
            self.available = True
            print(f"✓ HF Router API initialized for {self.model_name}")

        except ImportError:
            print(f"⚠️  requests library not available")
            self.available = False

    def _init_bedrock(self):
        """Initialize AWS Bedrock"""
        try:
            import boto3
            self.bedrock = boto3.client('bedrock-runtime', region_name=self.aws_region)

            # Test credentials
            boto3.client('sts', region_name=self.aws_region).get_caller_identity()
            self.available = True
            print(f"✓ AWS Bedrock initialized for {self.bedrock_id}")

        except Exception as e:
            print(f"⚠️  AWS Bedrock not available: {e}")
            self.available = False

    def _init_hf_api(self):
        """Initialize Hugging Face Inference API"""
        try:
            import requests
            self.requests = requests

            if not self.hf_token:
                print(f"⚠️  HF_TOKEN not found")
                self.available = False
                return

            self.hf_api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
            self.available = True
            print(f"✓ HF Inference API initialized for {self.model_id}")

        except ImportError:
            print(f"⚠️  requests library not available")
            self.available = False

    def _init_local(self):
        """Initialize local transformers (WARNING: 16GB+ download!)"""
        print(f"⚠️  WARNING: Local mode will download 16GB+ model files!")
        print(f"⚠️  Consider using HF Router API instead (no downloads needed)")
        print(f"⚠️  Set: export HF_TOKEN=your_token")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"🔄 Loading {self.model_id} locally (downloading 16GB+)...")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                token=self.hf_token
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="cpu",  # Force CPU on Mac
                token=self.hf_token
            )

            self.available = True
            print(f"✓ Local model loaded")

        except Exception as e:
            print(f"⚠️  Error loading local model: {e}")
            self.available = False

    def generate(self, prompt: str) -> str:
        """Generate response from Llama"""
        if not self.available:
            return f"Mock response for {self.model_name}"

        # Route to appropriate backend
        if self.access_method == 'hf_router':
            return self._generate_hf_router(prompt)
        elif self.access_method == 'bedrock':
            return self._generate_bedrock(prompt)
        elif self.access_method == 'hf_api':
            return self._generate_hf_api(prompt)
        elif self.access_method == 'local':
            return self._generate_local(prompt)
        else:
            return f"Error: Unknown access method {self.access_method}"

    def _generate_hf_router(self, prompt: str) -> str:
        """Generate using HF Router API (BEST method)"""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                headers = {"Authorization": f"Bearer {self.hf_token}"}

                response = self.requests.post(
                    self.api_url,
                    headers=headers,
                    json={
                        "messages": [{"role": "user", "content": prompt}],
                        "model": self.hf_router_id,
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature
                    },
                    timeout=300
                )

                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    last_error = f"HTTP {response.status_code}: {response.text}"

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    print(f"⚠️  HF Router error, retrying {attempt + 2}/{self.max_retries}...")
                    time.sleep(self.retry_delay)
                    continue

        return f"Error: {str(last_error)}"

    def _generate_bedrock(self, prompt: str) -> str:
        """Generate using AWS Bedrock"""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                payload = json.dumps({
                    'prompt': prompt,
                    'max_gen_len': self.max_tokens,
                    'temperature': self.temperature,
                    'top_p': 0.9
                })

                response = self.bedrock.invoke_model(
                    body=payload,
                    modelId=self.bedrock_id,
                    accept='application/json',
                    contentType='application/json'
                )

                result = json.loads(response['body'].read())
                return result['generation'].strip()

            except Exception as e:
                last_error = e
                error_str = str(e)

                if 'ExpiredToken' in error_str:
                    print(f"⚠️  AWS token expired. Run: aws sso login")
                    return f"Error: AWS credentials expired"

                if attempt < self.max_retries - 1:
                    print(f"⚠️  Bedrock error, retrying {attempt + 2}/{self.max_retries}...")
                    time.sleep(self.retry_delay)
                    continue

        return f"Error: {str(last_error)}"

    def _generate_hf_api(self, prompt: str) -> str:
        """Generate using HF Inference API"""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                headers = {"Authorization": f"Bearer {self.hf_token}"}

                response = self.requests.post(
                    self.hf_api_url,
                    headers=headers,
                    json={
                        "inputs": prompt,
                        "parameters": {
                            "max_new_tokens": self.max_tokens,
                            "temperature": self.temperature,
                            "return_full_text": False
                        }
                    },
                    timeout=300
                )

                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get('generated_text', str(result))
                    return str(result)
                elif response.status_code == 503:
                    if attempt < self.max_retries - 1:
                        print(f"⚠️  Model loading, waiting {self.retry_delay}s...")
                        time.sleep(self.retry_delay)
                        continue
                else:
                    last_error = f"HTTP {response.status_code}: {response.text}"

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    print(f"⚠️  HF API error, retrying {attempt + 2}/{self.max_retries}...")
                    time.sleep(self.retry_delay)
                    continue

        return f"Error: {str(last_error)}"

    def _generate_local(self, prompt: str) -> str:
        """Generate using local transformers"""
        try:
            import torch

            inputs = self.tokenizer(prompt, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            if response.startswith(prompt):
                response = response[len(prompt):].strip()

            return response

        except Exception as e:
            print(f"❌ Error generating with local model: {e}")
            return f"Error: {str(e)}"

    def get_model_info(self) -> Dict:
        """Get Llama model information"""
        info = super().get_model_info()
        info.update({
            "provider": f"Llama ({self.access_method or 'unavailable'})",
            "model_id": self.model_id,
            "access_method": self.access_method,
            "family": "Llama"
        })

        if self.access_method == 'hf_router':
            info["router_model"] = self.hf_router_id
            info["api_url"] = self.api_url
        elif self.access_method == 'bedrock':
            info["bedrock_id"] = self.bedrock_id
            info["aws_region"] = self.aws_region

        return info
