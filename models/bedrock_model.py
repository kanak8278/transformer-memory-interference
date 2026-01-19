"""
AWS Bedrock model interface supporting multiple model families:
- Meta Llama (3, 3.1, 3.2, 4)
- Amazon Titan
- Amazon Nova
- Anthropic Claude
- Mistral / Ministral
- DeepSeek
- Qwen
- AI21 (infrastructure support)
- Cohere (infrastructure support)
"""

import os
import json
from typing import Dict, Optional
from .base_model import BaseModelInterface
from . import config as model_config


class BedrockModelInterface(BaseModelInterface):
    """AWS Bedrock model interface with multi-model support"""

    def __init__(self, model_name: str, config: Optional[Dict] = None):
        """
        Initialize Bedrock model interface.

        Args:
            model_name: Model name (e.g., 'bedrock-llama-3-70b', 'bedrock-titan-large')
            config: Optional configuration dictionary
                - aws_region: AWS region (default: 'us-east-1')
                - aws_profile: AWS profile to use
                - max_retries: Number of retries (default: 3)
        """
        super().__init__(model_name, config)

        # Get model ID and family first (needed for region selection)
        self.model_id = model_config.get_bedrock_model_id(model_name)
        self.model_family = model_config.detect_model_family(self.model_id)

        # AWS configuration - use model_config helper
        self.aws_region = self.config.get('aws_region', model_config.get_aws_region(self.model_family))
        self.aws_profile = self.config.get('aws_profile')

        # Retry configuration - use model_config default
        self.max_retries = self.config.get('max_retries', model_config.BEDROCK_MAX_RETRIES)

        # Override max_tokens using model_config helper
        if 'max_tokens' not in self.config:
            self.max_tokens = model_config.get_bedrock_max_tokens(self.model_family)

        # Initialize Bedrock client
        self._init_bedrock()

    def _get_model_id(self, model_name: str) -> str:
        """Map friendly model names to Bedrock model IDs (delegates to config)"""
        return model_config.get_bedrock_model_id(model_name)

    def _init_bedrock(self):
        """Initialize AWS Bedrock client"""
        try:
            import boto3
            from botocore.config import Config
            from botocore.exceptions import ClientError

            # Configure timeout based on model family
            # Strategy: Use shorter timeout for invoke_model, then fallback to streaming
            read_timeout = 120  # Default 2 minutes
            if self.model_family == 'titan':
                read_timeout = 120  # 2 minutes for Titan invoke_model, then try streaming
            elif self.model_family == 'deepseek':
                read_timeout = 300  # 5 minutes for DeepSeek (reasoning can be slow)
            elif self.model_family in ['claude', 'llama', 'mistral']:
                read_timeout = 180  # 3 minutes for other models

            boto_config = Config(
                read_timeout=read_timeout,
                connect_timeout=60,
                retries={'max_attempts': 0}  # We handle retries ourselves
            )

            # Create session with profile if specified
            if self.aws_profile:
                session = boto3.Session(profile_name=self.aws_profile)
                self.bedrock_runtime = session.client(
                    service_name='bedrock-runtime',
                    region_name=self.aws_region,
                    config=boto_config
                )
            else:
                self.bedrock_runtime = boto3.client(
                    service_name='bedrock-runtime',
                    region_name=self.aws_region,
                    config=boto_config
                )

            # Test credentials
            sts = boto3.client('sts', region_name=self.aws_region)
            sts.get_caller_identity()

            self.available = True
            print(f"✓ AWS Bedrock initialized for {self.model_id}")
            print(f"  Region: {self.aws_region}")
            print(f"  Model family: {self.model_family}")

        except ImportError as e:
            error_msg = (
                "boto3 not installed. Install with: pip install boto3\n"
                "Or install all dependencies: pip install -r requirements.txt"
            )
            print(f"\n⚠️  ERROR: {error_msg}\n")
            raise ImportError(error_msg)
        except Exception as e:
            error_msg = str(e)
            print(f"\n⚠️  ERROR: Bedrock initialization failed")
            print(f"    {error_msg}")
            print(f"\n    Make sure AWS credentials are configured:")
            print(f"    - Run: aws configure")
            print(f"    - OR: aws sso login")
            print(f"    - For SageMaker: Ensure execution role has Bedrock permissions\n")
            raise RuntimeError(f"Bedrock initialization failed: {error_msg}")

    def _format_request_body(self, prompt: str) -> dict:
        """Format request body based on model family"""

        if self.model_family == 'llama':
            # Meta Llama on Bedrock requires instruction format with special tokens
            # Reference: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html
            formatted_prompt = model_config.format_llama_prompt(prompt)
            return {
                "prompt": formatted_prompt,
                "max_gen_len": self.max_tokens,
                "temperature": self.temperature,
            }

        elif self.model_family == 'titan':
            # Amazon Titan format with textGenerationConfig
            # Reference: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-text.html
            return {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": self.max_tokens,
                    "stopSequences": [],
                    "temperature": self.temperature,
                    "topP": 0.9
                }
            }

        elif self.model_family == 'nova':
            # Amazon Nova format
            return {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": self.max_tokens,
                    "temperature": self.temperature,
                    "topP": 0.9,
                }
            }

        elif self.model_family == 'claude':
            # Anthropic Claude format (messages API)
            return {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
                    }
                ]
            }

        elif self.model_family == 'mistral':
            # Mistral format with instruction tags
            # Reference: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral.html
            formatted_prompt = model_config.format_mistral_prompt(prompt)
            return {
                "prompt": formatted_prompt,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }

        elif self.model_family == 'ai21':
            # AI21 Jurassic/Jamba format (simple)
            return {
                "prompt": prompt
            }

        elif self.model_family == 'deepseek':
            # DeepSeek R1 format with instruction tags
            # Reference: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-deepseek.html
            formatted_prompt = model_config.format_deepseek_prompt(prompt)
            return {
                "prompt": formatted_prompt,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": 0.9,
            }

        elif self.model_family == 'cohere':
            # Cohere format
            return {
                "message": prompt,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }

        else:
            # Generic format
            return {
                "prompt": prompt,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }

    def _log_token_usage(self, response_body: dict, prompt: str) -> None:
        """Extract and log token usage from response"""
        try:
            # Extract token counts from response metadata
            input_tokens = None
            output_tokens = None

            # Different models have different response structures for token counts
            if 'prompt_token_count' in response_body and 'generation_token_count' in response_body:
                # Llama format (directly in response body)
                input_tokens = response_body.get('prompt_token_count')
                output_tokens = response_body.get('generation_token_count')
            elif 'usage' in response_body:
                # Common format (Claude, Mistral, DeepSeek)
                usage = response_body['usage']
                input_tokens = usage.get('input_tokens', usage.get('prompt_tokens'))
                output_tokens = usage.get('output_tokens', usage.get('completion_tokens'))
            elif 'amazon-bedrock-invocationMetrics' in response_body:
                # Bedrock metrics format
                metrics = response_body['amazon-bedrock-invocationMetrics']
                input_tokens = metrics.get('inputTokenCount')
                output_tokens = metrics.get('outputTokenCount')
            elif 'inputTextTokenCount' in response_body:
                # Titan format
                input_tokens = response_body.get('inputTextTokenCount')
                results = response_body.get('results', [])
                if results:
                    output_tokens = results[0].get('tokenCount')

            # Store token usage in instance variables
            if input_tokens is not None and output_tokens is not None:
                total = input_tokens + output_tokens
                self.last_input_tokens = input_tokens
                self.last_output_tokens = output_tokens
                self.last_total_tokens = total
                self.last_was_truncated = output_tokens >= self.max_tokens
                print(f"  Tokens: Input={input_tokens:,}, Output={output_tokens:,}, Total={total:,}")
            elif output_tokens is not None:
                self.last_output_tokens = output_tokens
                print(f"  Tokens: Output={output_tokens:,}")
        except Exception:
            # Silently fail if token extraction fails
            pass

    def _extract_response_text(self, response_body: dict) -> str:
        """Extract text from response based on model family"""

        if self.model_family == 'llama':
            # Llama response: {"generation": "text"}
            return response_body.get('generation', '')

        elif self.model_family == 'titan':
            # Titan response: {"results": [{"outputText": "text", "tokenCount": N, "completionReason": "FINISH/LENGTH/..."}]}
            results = response_body.get('results', [])
            if results:
                result = results[0]
                # Log completion reason to understand why model stopped
                completion_reason = result.get('completionReason', 'UNKNOWN')
                if completion_reason != 'FINISH':
                    print(f"  ⚠️  Titan stopped with reason: {completion_reason}")
                return result.get('outputText', '')
            return ''

        elif self.model_family == 'nova':
            # Nova response: {"output": {"message": "text"}}
            output = response_body.get('output', {})
            return output.get('message', '')

        elif self.model_family == 'claude':
            # Claude response: {"content": [{"text": "text"}]}
            content = response_body.get('content', [])
            if content:
                return content[0].get('text', '')
            return ''

        elif self.model_family == 'mistral':
            # Mistral response: {"outputs": [{"text": "text"}]}
            outputs = response_body.get('outputs', [])
            if outputs:
                return outputs[0].get('text', '')
            return ''

        elif self.model_family == 'ai21':
            # AI21 response: {"completions": [{"data": {"text": "text"}}]}
            completions = response_body.get('completions', [])
            if completions:
                data = completions[0].get('data', {})
                return data.get('text', '')
            return ''

        elif self.model_family == 'deepseek':
            # DeepSeek response: {"choices": [{"text": "text"}]}
            choices = response_body.get('choices', [])
            if choices:
                return choices[0].get('text', '')
            return ''

        elif self.model_family == 'cohere':
            # Cohere response: {"text": "text"}
            return response_body.get('text', '')

        else:
            # Try common patterns
            if 'generation' in response_body:
                return response_body['generation']
            elif 'text' in response_body:
                return response_body['text']
            elif 'output' in response_body:
                return response_body['output']
            else:
                return str(response_body)

    def generate(self, prompt: str) -> str:
        """Generate response from Bedrock model with streaming fallback"""
        if not self.available:
            return f"Error: Bedrock not available for {self.model_name}"

        try:
            # Nova, Qwen, Mistral, and Ministral models use Converse API instead of InvokeModel
            # AWS migrated Mistral models to Converse API (InvokeModel returns ValidationException)
            if self.model_family in ['nova', 'qwen', 'mistral', 'ministral']:
                return self._converse(prompt)

            # Try regular InvokeModel first for other models
            return self._invoke_model(prompt)

        except Exception as e:
            error_msg = str(e)

            # Try streaming fallback for models that support it
            # Titan: 2min timeout on invoke_model → try streaming
            # Llama: timeout or connection errors → try streaming
            if model_config.supports_streaming(self.model_family) and self._should_try_streaming(error_msg):
                print(f"⚠️  InvokeModel failed, trying streaming fallback...")
                try:
                    return self._invoke_model_streaming(prompt)
                except Exception as stream_error:
                    error_msg = f"Both InvokeModel and streaming failed. Last error: {str(stream_error)}"

            print(f"⚠️  Bedrock error: {error_msg}")

            # Provide helpful error messages
            if 'AccessDeniedException' in error_msg:
                return f"Error: Access denied. Check IAM permissions for Bedrock."
            elif 'ValidationException' in error_msg:
                return f"Error: Invalid request. Model might not support this format."
            elif 'ThrottlingException' in error_msg:
                return f"Error: Rate limit exceeded. Wait and retry."
            elif 'ResourceNotFoundException' in error_msg:
                return f"Error: Model {self.model_id} not found in {self.aws_region}."
            else:
                return f"Error: {error_msg}"

    def _converse(self, prompt: str) -> str:
        """
        Invoke Nova and Qwen models using Converse API.
        Reference: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
        """
        # Format conversation for Converse API
        conversation = [
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ]

        # Prepare API call parameters
        api_params = {
            "modelId": self.model_id,
            "messages": conversation,
            "inferenceConfig": {
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
                "topP": 0.9
            }
        }

        # Add system prompt if configured (helps with instruction following)
        system_prompt = self.config.get('system_prompt')
        if system_prompt:
            api_params["system"] = [{"text": system_prompt}]

        try:
            # Call Converse API
            response = self.bedrock_runtime.converse(**api_params)

            # Extract response text
            response_text = response["output"]["message"]["content"][0]["text"]

            # Check stopReason for truncation (more accurate than token count)
            stop_reason = response.get("stopReason", "")
            was_truncated = (stop_reason == "max_tokens")

            # Warn if truncated
            if was_truncated:
                print(f"  ⚠️  Response truncated: stopReason={stop_reason}")

            # Extract and store token usage
            if "usage" in response:
                usage = response["usage"]
                input_tokens = usage.get("inputTokens")
                output_tokens = usage.get("outputTokens")
                if input_tokens and output_tokens:
                    total = input_tokens + output_tokens
                    self.last_input_tokens = input_tokens
                    self.last_output_tokens = output_tokens
                    self.last_total_tokens = total
                    self.last_was_truncated = was_truncated
                    print(f"  Tokens: Input={input_tokens:,}, Output={output_tokens:,}, Total={total:,}")

            return response_text

        except Exception as e:
            # Store error
            self.last_error = str(e)
            # Try streaming fallback
            error_msg = str(e)
            if self._should_try_streaming(error_msg):
                print(f"⚠️  Converse failed, trying converse_stream fallback...")
                return self._converse_stream(prompt)
            else:
                raise

    def _converse_stream(self, prompt: str) -> str:
        """
        Invoke Nova models using Converse Stream API.
        Reference: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
        """
        # Format conversation for Converse API
        conversation = [
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ]

        print(f"  Using streaming mode (converse_stream)...")

        try:
            # Call Converse Stream API
            streaming_response = self.bedrock_runtime.converse_stream(
                modelId=self.model_id,
                messages=conversation,
                inferenceConfig={
                    "maxTokens": self.max_tokens,
                    "temperature": self.temperature,
                    "topP": 0.9
                },
            )

            # Assemble response from chunks
            full_response = ""
            for chunk in streaming_response["stream"]:
                if "contentBlockDelta" in chunk:
                    text = chunk["contentBlockDelta"]["delta"]["text"]
                    full_response += text

            print(f"  Streaming completed ({len(full_response)} chars)")
            return full_response

        except Exception as e:
            raise Exception(f"Converse streaming failed: {str(e)}")

    def _should_try_streaming(self, error_msg: str) -> bool:
        """Determine if streaming should be attempted as fallback"""
        return model_config.should_try_streaming(error_msg)

    def _invoke_model(self, prompt: str) -> str:
        """Invoke model using standard InvokeModel API"""
        # Check input length limits for models with constraints (e.g., Titan 42K char limit)
        if self.model_family in model_config.BEDROCK_INPUT_LIMITS:
            limit = model_config.BEDROCK_INPUT_LIMITS[self.model_family]
            if len(prompt) > limit:
                raise ValueError(
                    f"{self.model_family.title()} input limit exceeded: "
                    f"{len(prompt):,} chars > {limit:,} char limit. "
                    f"Consider reducing interference level or using a model with larger context."
                )

        # Format request body
        request_body = self._format_request_body(prompt)

        # Invoke model
        response = self.bedrock_runtime.invoke_model(
            modelId=self.model_id,
            body=json.dumps(request_body),
            accept='application/json',
            contentType='application/json'
        )

        # Parse response
        response_body = json.loads(response.get('body').read().decode('utf-8'))

        # Extract and log token usage
        self._log_token_usage(response_body, prompt)

        # Extract text
        text = self._extract_response_text(response_body)

        return text

    def _invoke_model_streaming(self, prompt: str) -> str:
        """
        Invoke model using InvokeModelWithResponseStream API.
        This is used as a fallback when InvokeModel times out or fails.

        Supported models:
        - Llama 3, 3.1, 3.2, 3.3, 4 ✅ (all support streaming)
        - Claude ✅
        - Mistral ✅
        - Cohere ✅
        - Titan ✅ (supports streaming)

        References:
        - Llama: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-runtime_example_bedrock-runtime_InvokeModelWithResponseStream_MetaLlama3_section.html
        - Titan: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-runtime_example_bedrock-runtime_InvokeModelWithResponseStream_TitanText_section.html
        """
        # Check if model supports streaming
        if not model_config.supports_streaming(self.model_family):
            raise Exception(f"Streaming not supported for model family: {self.model_family}")

        # Format request body (same format as non-streaming)
        request_body = self._format_request_body(prompt)

        print(f"  Using streaming mode (InvokeModelWithResponseStream)...")

        try:
            # Invoke model with streaming (per AWS documentation)
            streaming_response = self.bedrock_runtime.invoke_model_with_response_stream(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )

            # Assemble response from chunks
            full_response = ""
            prompt_token_count = None
            generation_token_count = 0

            # Process streaming response (per AWS example)
            for event in streaming_response["body"]:
                chunk = json.loads(event["chunk"]["bytes"])

                # Extract text chunk based on model family
                if "generation" in chunk:
                    # Llama format: {"generation": "text"}
                    text_chunk = chunk["generation"]
                    full_response += text_chunk
                elif "generations" in chunk:
                    # Cohere format: {"generations": [{"text": "text"}]}
                    text_chunk = chunk["generations"][0]["text"]
                    full_response += text_chunk
                elif "outputs" in chunk:
                    # Mistral format: {"outputs": [{"text": "text"}]}
                    outputs = chunk.get("outputs", [])
                    if outputs:
                        text_chunk = outputs[0].get("text", "")
                        full_response += text_chunk
                elif "outputText" in chunk:
                    # Titan format: {"outputText": "text", "index": 0, "completionReason": "FINISHED"}
                    text_chunk = chunk["outputText"]
                    full_response += text_chunk
                    # Check completion reason for Titan (only warn on errors, not None or success)
                    completion_reason = chunk.get("completionReason")
                    if completion_reason and completion_reason not in ["FINISHED", "FINISH", None]:
                        print(f"  ⚠️  Titan streaming stopped: {completion_reason}")

                # Extract token counts (usually in final chunk)
                if "prompt_token_count" in chunk:
                    prompt_token_count = chunk["prompt_token_count"]
                if "generation_token_count" in chunk:
                    generation_token_count = chunk["generation_token_count"]
                # Titan token counts
                if "inputTextTokenCount" in chunk:
                    prompt_token_count = chunk["inputTextTokenCount"]
                if "totalOutputTextTokenCount" in chunk:
                    generation_token_count = chunk["totalOutputTextTokenCount"]

            # Log token usage
            if prompt_token_count is not None and generation_token_count > 0:
                total = prompt_token_count + generation_token_count
                print(f"  Tokens: Input={prompt_token_count:,}, Output={generation_token_count:,}, Total={total:,}")
            else:
                print(f"  Streaming completed ({len(full_response)} chars)")

            return full_response

        except Exception as e:
            raise Exception(f"Streaming invocation failed: {str(e)}")

    def get_model_info(self) -> Dict:
        """Get model information"""
        info = super().get_model_info()
        info.update({
            "model_id": self.model_id,
            "model_family": self.model_family,
            "aws_region": self.aws_region,
        })
        return info
