"""
SageMaker JumpStart endpoint model interface.

Calls a pre-deployed JumpStart endpoint. Does NOT manage endpoint lifecycle —
deploy/delete separately using experiments_cloud/deploy_jumpstart.py.

Usage:
    # 1. Deploy once (takes ~10 min):
    #    python experiments_cloud/deploy_jumpstart.py --model qwen3-5-9b
    #
    # 2. Run experiments (pass endpoint name via config or env):
    #    python ucurve_sweep.py --model jumpstart-qwen3-5-9b \
    #        --endpoint-name <name-printed-by-deploy-script>
    #
    # 3. Delete when done:
    #    python experiments_cloud/deploy_jumpstart.py --delete <endpoint-name>
"""

import json
import os
from typing import Dict, Optional

from .base_model import BaseModelInterface


JUMPSTART_MODEL_IDS = {
    "jumpstart-qwen3-5-9b":  "huggingface-llm-qwen3-5-9b-instruct",
    "jumpstart-qwen3-5-27b": "huggingface-llm-qwen3-5-27b-instruct",
    "jumpstart-qwen2-5-7b":  "huggingface-llm-qwen2-5-7b-instruct",
    "jumpstart-qwen2-5-72b": "huggingface-llm-qwen2-5-72b-instruct",
}

JUMPSTART_DEFAULT_INSTANCES = {
    "jumpstart-qwen3-5-9b":  "ml.g5.2xlarge",
    "jumpstart-qwen3-5-27b": "ml.g5.12xlarge",
    "jumpstart-qwen2-5-7b":  "ml.g5.2xlarge",
    "jumpstart-qwen2-5-72b": "ml.g5.48xlarge",
}


class JumpStartModelInterface(BaseModelInterface):
    """Calls a pre-deployed SageMaker JumpStart endpoint."""

    def __init__(self, model_name: str, config: Optional[Dict] = None):
        super().__init__(model_name, config)

        self.endpoint_name = (
            self.config.get("endpoint_name")
            or os.getenv("JUMPSTART_ENDPOINT_NAME")
        )
        if not self.endpoint_name:
            raise ValueError(
                f"endpoint_name required for {model_name}. "
                "Pass via config['endpoint_name'] or JUMPSTART_ENDPOINT_NAME env var."
            )

        self.aws_region = self.config.get("aws_region", "us-east-1")
        self.aws_profile = self.config.get("aws_profile", os.getenv("AWS_PROFILE"))

        self._init_client()

    def _get_model_id(self, model_name: str) -> str:
        return JUMPSTART_MODEL_IDS.get(model_name, model_name)

    def _init_client(self):
        import boto3
        if self.aws_profile:
            session = boto3.Session(profile_name=self.aws_profile)
        else:
            session = boto3.Session()

        self.runtime = session.client(
            "sagemaker-runtime",
            region_name=self.aws_region,
        )
        self.available = True
        print(f"✓ JumpStart endpoint: {self.endpoint_name} ({self.aws_region})")

    def generate(self, prompt: str) -> str:
        # TGI/vLLM endpoints on JumpStart use the OpenAI-compatible messages format
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        try:
            response = self.runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Body=json.dumps(payload),
            )
            body = json.loads(response["Body"].read().decode("utf-8"))

            # OpenAI-compatible response shape
            if "choices" in body:
                return body["choices"][0]["message"]["content"]

            # HF TGI legacy shape
            if isinstance(body, list) and "generated_text" in body[0]:
                return body[0]["generated_text"]

            # Fallback
            return str(body)

        except Exception as e:
            err = str(e)
            if "ThrottlingException" in err:
                return f"Error: Rate limited — {err}"
            if "ModelError" in err:
                return f"Error: Endpoint model error — {err}"
            return f"Error: {err}"
