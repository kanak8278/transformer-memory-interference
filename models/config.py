"""
Configuration file for LLM Interference Framework.
Contains all model-specific settings, API configurations, and tunable parameters.
"""

# =============================================================================
# AWS BEDROCK CONFIGURATION
# =============================================================================

# AWS Region preferences by model family
# Each model family has an optimal region where direct model IDs work
AWS_REGIONS = {
    'nova': 'us-east-1',      # Nova models work with direct IDs in us-east-1
    'titan': 'us-east-1',     # Titan models available in us-east-1 (EOL in us-west-2)
    'llama': 'us-west-2',     # Llama 3.1 works with direct IDs in us-west-2
    'claude': 'us-west-2',    # Claude models generally available
    'mistral': 'us-east-1',   # Mistral models available in us-east-1 (7B, 8x7B, Large, Small)
    'ministral': 'us-east-1', # Ministral 3-3b and 3-8b require us-east-1 with Converse API
    'ai21': 'us-west-2',      # AI21 models generally available
    'deepseek': 'us-west-2',  # DeepSeek uses cross-region inference profiles
    'cohere': 'us-west-2',    # Cohere models generally available
    'qwen': 'us-west-2',      # Qwen models from Bedrock Marketplace
    'default': 'us-west-2'    # Default fallback region
}

# Max input tokens by model family (context window size)
# Based on bedrock_models_with_limits.csv - maximum supported input tokens
BEDROCK_MAX_INPUT_TOKENS = {
    'llama': 128000,    # Llama 3.1/3.2/3.3: 128K input (Llama 3: 8K, Llama 4: 1M)
    'deepseek': 64000,  # DeepSeek R1: 64K input context
    'claude': 200000,   # Claude 3/3.5/3.7/4: 200K input context
    'titan': 8000,      # Titan: 8K input tokens (combined input+output budget)
    'nova': 300000,     # Nova Gen1: 300K input (Nova Gen2: 1M but varies)
    'mistral': 32000,   # Mistral 7B/Large/Small: 32K input (Large 3: 128K)
    'ministral': 128000,# Ministral 3-3b/8b/14b: 128K input context
    'ai21': 256000,     # AI21 Jamba 1.5: 256K input context
    'cohere': 128000,   # Cohere Command R/R+: 128K input context
    'qwen': 256000,     # Qwen3: 256K input context (32B: 32K)
    'default': 128000   # Default fallback (most modern models support 128K)
}

# Max output tokens by model family
# Based on bedrock_models_with_limits.csv - maximum supported output tokens
BEDROCK_MAX_OUTPUT_TOKENS = {
    'llama': 2048,      # Llama 3/3.1/3.2/3.3: 2048 max_gen_len (Llama 4: 8192)
    'deepseek': 8192,   # DeepSeek R1: 8192 max output
    'claude': 8192,     # Claude 3.5: 8192 (Claude 3: 4096, Claude 3.7/4: 65K-131K)
    'titan': 8000,      # Titan: 8000 max output (combined input+output budget)
    'nova': 5000,       # Nova Gen1: 5000 max output (Nova Gen2: 8192)
    'mistral': 4096,    # Mistral 7B/Large/Small: 4096 max output (Converse API limit)
    'ministral': 8192,  # Ministral 3-3b/8b/14b: 8192 max output
    'ai21': 8192,       # AI21 Jamba 1.5: 8192 max output
    'cohere': 4096,     # Cohere Command R/R+: 4096 max output
    'qwen': 8192,       # Qwen models: 8192 max output
    'default': 8192     # Default fallback (most models support 8192)
}

# Legacy BEDROCK_MAX_TOKENS for backward compatibility (maps to output tokens)
BEDROCK_MAX_TOKENS = BEDROCK_MAX_OUTPUT_TOKENS

# Input length limits (in characters) for models with strict constraints
# Reference: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-text.html
BEDROCK_INPUT_LIMITS = {
    'titan': 42000,     # Titan has 42K character input limit (not tokens!)
}

# Bedrock Model ID mappings
# Maps friendly names (bedrock-llama-3-8b) to actual AWS model IDs
BEDROCK_MODEL_IDS = {
    # Llama 3 (original) - direct model IDs, 8K context
    'bedrock-llama-3-8b': 'meta.llama3-8b-instruct-v1:0',
    'bedrock-llama-3-70b': 'meta.llama3-70b-instruct-v1:0',

    # Llama 3.1 (128K context) - direct model IDs
    'bedrock-llama-3.1-8b': 'meta.llama3-1-8b-instruct-v1:0',
    'bedrock-llama-3.1-70b': 'meta.llama3-1-70b-instruct-v1:0',
    'bedrock-llama-3.1-405b': 'meta.llama3-1-405b-instruct-v1:0',

    # Llama 3.2 (128K context) - require cross-region inference profiles (us. prefix)
    'bedrock-llama-3.2-1b': 'us.meta.llama3-2-1b-instruct-v1:0',
    'bedrock-llama-3.2-3b': 'us.meta.llama3-2-3b-instruct-v1:0',
    'bedrock-llama-3.2-11b': 'us.meta.llama3-2-11b-instruct-v1:0',
    'bedrock-llama-3.2-90b': 'us.meta.llama3-2-90b-instruct-v1:0',

    # Llama 3.3 (128K context) - require cross-region inference profiles (us. prefix)
    'bedrock-llama-3.3-70b': 'us.meta.llama3-3-70b-instruct-v1:0',

    # Llama 4 - require cross-region inference profiles (us. prefix)
    'bedrock-llama-4-scout-17b': 'us.meta.llama4-scout-17b-instruct-v1:0',
    'bedrock-llama-4-maverick-17b': 'us.meta.llama4-maverick-17b-instruct-v1:0',

    # Amazon Titan
    'bedrock-titan-large': 'amazon.titan-tg1-large',
    'bedrock-titan-lite': 'amazon.titan-text-lite-v1',
    'bedrock-titan-express': 'amazon.titan-text-express-v1',

    # Amazon Nova
    'bedrock-nova-micro': 'amazon.nova-micro-v1:0',
    'bedrock-nova-lite': 'amazon.nova-lite-v1:0',
    'bedrock-nova-pro': 'amazon.nova-pro-v1:0',
    'bedrock-nova-premier': 'amazon.nova-premier-v1:0',

    # Anthropic Claude (via Bedrock)
    'bedrock-claude-3-haiku': 'anthropic.claude-3-haiku-20240307-v1:0',
    'bedrock-claude-3-sonnet': 'anthropic.claude-3-sonnet-20240229-v1:0',
    'bedrock-claude-3-opus': 'anthropic.claude-3-opus-20240229-v1:0',
    'bedrock-claude-3.5-sonnet': 'anthropic.claude-3-5-sonnet-20241022-v2:0',
    'bedrock-claude-3.5-haiku': 'anthropic.claude-3-5-haiku-20241022-v1:0',
    'bedrock-claude-3.7-sonnet': 'anthropic.claude-3-7-sonnet-20250219-v1:0',
    'bedrock-claude-4-sonnet': 'anthropic.claude-sonnet-4-20250514-v1:0',
    'bedrock-claude-4-opus': 'anthropic.claude-opus-4-20250514-v1:0',

    # Mistral
    'bedrock-mistral-7b': 'mistral.mistral-7b-instruct-v0:2',
    'bedrock-mistral-8x7b': 'mistral.mixtral-8x7b-instruct-v0:1',
    'bedrock-mistral-large': 'mistral.mistral-large-2402-v1:0',
    'bedrock-mistral-small': 'mistral.mistral-small-2402-v1:0',
    'bedrock-ministral-3b': 'mistral.ministral-3-3b-instruct',
    'bedrock-ministral-8b': 'mistral.ministral-3-8b-instruct',

    # DeepSeek (requires cross-region inference profile)
    'bedrock-deepseek-r1': 'us.deepseek.r1-v1:0',

    # Qwen (Bedrock Marketplace)
    'bedrock-qwen-3-32b': 'qwen.qwen3-32b-v1:0',
    'bedrock-qwen-3-vl-235b': 'qwen.qwen3-vl-235b-a22b',
    'bedrock-qwen-3-coder-30b': 'qwen.qwen3-coder-30b-a3b-v1:0',
}

# Model families that support streaming via InvokeModelWithResponseStream
# Reference: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-runtime_example_bedrock-runtime_InvokeModelWithResponseStream_TitanText_section.html
BEDROCK_STREAMING_SUPPORTED = ['llama', 'claude', 'mistral', 'titan']

# Timeout indicators for streaming fallback
BEDROCK_TIMEOUT_INDICATORS = [
    'timeout', 'timed out', 'ReadTimeout',
    'Connection', 'too long', 'ResponseStream'
]

# =============================================================================
# OPENAI CONFIGURATION
# =============================================================================

# OpenAI API Configuration
# Users should set OPENAI_API_KEY environment variable
# For Azure OpenAI, set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY

OPENAI_DEFAULTS = {
    'api_version': '2023-03-15-preview',
    'timeout': 300,  # 5 minutes
    'max_retries': 3,
    'retry_delay': 60,  # 60 seconds for rate limits
}

# Max output tokens by OpenAI model family
OPENAI_MAX_TOKENS = {
    # o-series reasoning models (highest output limits)
    'o1': 32768,
    'o3': 32768,
    'o4': 32768,
    # GPT-5 models (reasoning models - need high limits for reasoning + response)
    'gpt-5': 32768,  # Reasoning model: needs tokens for thinking + response
    'gpt-5-mini': 32768,
    'gpt-5-nano': 16384,
    # GPT-4.1 models (improved GPT-4)
    'gpt-4.1': 16384,
    'gpt-4.1-mini': 16384,
    'gpt-4.1-nano': 16384,
    # GPT-4o models (optimized GPT-4)
    'gpt-4o': 16384,
    'gpt-4o-mini': 16384,
    # GPT-4 family
    'gpt-4': 8192,
    'gpt-4-turbo': 8192,
    # GPT-3.5-turbo
    'gpt-35-turbo': 4096,
    'gpt-3.5-turbo': 4096,
    # Default
    'default': 2000
}

# OpenAI Model ID mappings
OPENAI_MODEL_IDS = {
    # Legacy GPT models
    'gpt-4': 'gpt-4',
    'gpt-4-turbo': 'gpt-4-turbo',
    'gpt-35-turbo': 'gpt-35-turbo',
    'gpt-3.5-turbo': 'gpt-35-turbo',
    # GPT-4o models
    'gpt-4o': 'gpt-4o',
    'gpt-4o-mini': 'gpt-4o-mini',
    # GPT-5 models
    'gpt-5': 'gpt-5',
    'gpt-5-mini': 'gpt-5-mini',
    'gpt-5-nano': 'gpt-5-nano',
    # GPT-4.1 models
    'gpt-4.1': 'gpt-4.1',
    'gpt-4.1-mini': 'gpt-4.1-mini',
    'gpt-4.1-nano': 'gpt-4.1-nano',
    # o-series reasoning models
    'o1': 'o1',
    'o1-mini': 'o1-mini',
    'o1-preview': 'o1-preview',
    'o3': 'o3',
    'o3-mini': 'o3-mini',
    'o4-mini': 'o4-mini',
}

# Models that use max_completion_tokens instead of max_tokens
OPENAI_USE_MAX_COMPLETION_TOKENS = [
    'o1', 'o1-mini', 'o1-preview', 'o3', 'o3-mini', 'o4-mini',
    'gpt-5', 'gpt-5-mini', 'gpt-5-nano',
    'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano',
    'gpt-4o', 'gpt-4o-mini'
]

# Models that support reasoning_effort parameter (o-series only)
OPENAI_SUPPORTS_REASONING_EFFORT = ['o1', 'o1-mini', 'o1-preview', 'o3', 'o3-mini', 'o4-mini']

# Models that do NOT support temperature parameter
# o-series reasoning models and gpt-5 series don't support temperature
OPENAI_NO_TEMPERATURE = ['o1', 'o1-mini', 'o1-preview', 'o3', 'o3-mini', 'o4-mini', 'gpt-5', 'gpt-5-mini', 'gpt-5-nano']

# =============================================================================
# CLAUDE DIRECT API CONFIGURATION
# =============================================================================

CLAUDE_API_URL = 'https://api.anthropic.com/v1/messages'
CLAUDE_API_VERSION = '2023-06-01'
CLAUDE_MAX_TOKENS = 4096
CLAUDE_TIMEOUT = 300  # 5 minutes

CLAUDE_MODEL_IDS = {
    'claude-3-opus': 'claude-3-opus-20240229',
    'claude-3-sonnet': 'claude-3-sonnet-20240229',
    'claude-3-haiku': 'claude-3-haiku-20240307',
    'claude-3.5-sonnet': 'claude-3-5-sonnet-20241022',
    'claude-3.5-haiku': 'claude-3-5-haiku-20241022',
}

# =============================================================================
# GEMINI API CONFIGURATION
# =============================================================================

GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models'
GEMINI_MAX_TOKENS = 8192
GEMINI_TIMEOUT = 300  # 5 minutes

GEMINI_MODEL_IDS = {
    'gemini-2.0-flash': 'gemini-2.0-flash',
    'gemini-2.0-flash-lite': 'gemini-2.0-flash-lite',
    'gemini-2.5-pro': 'gemini-2.5-pro',
    'gemini-2.5-flash-lite': 'gemini-2.5-flash-lite',
}

# =============================================================================
# PROMPT FORMATTING TEMPLATES
# =============================================================================

# Llama prompt template (Bedrock)
LLAMA_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{prompt}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

# DeepSeek R1 prompt template (Bedrock)
DEEPSEEK_PROMPT_TEMPLATE = """<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜><think>
"""

# Mistral prompt template (Bedrock)
# Reference: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral.html
MISTRAL_PROMPT_TEMPLATE = """<s>[INST] {prompt} [/INST]"""

# =============================================================================
# RETRY AND TIMEOUT CONFIGURATION
# =============================================================================

# Default retry settings (can be overridden per model)
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 300  # 5 minutes in seconds

# Bedrock-specific retry settings
BEDROCK_MAX_RETRIES = 3

# OpenAI-specific retry settings
OPENAI_MAX_RETRIES = 3
OPENAI_RETRY_DELAY = 60  # seconds between retries for rate limits

# =============================================================================
# TEMPERATURE AND GENERATION PARAMETERS
# =============================================================================

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_aws_region(model_family: str) -> str:
    """Get preferred AWS region for a model family."""
    return AWS_REGIONS.get(model_family, AWS_REGIONS['default'])

def get_bedrock_max_tokens(model_family: str) -> int:
    """Get max output tokens for a Bedrock model family (legacy - use get_bedrock_max_output_tokens)."""
    return BEDROCK_MAX_OUTPUT_TOKENS.get(model_family, BEDROCK_MAX_OUTPUT_TOKENS['default'])

def get_bedrock_max_input_tokens(model_family: str) -> int:
    """Get max input tokens (context window) for a Bedrock model family."""
    return BEDROCK_MAX_INPUT_TOKENS.get(model_family, BEDROCK_MAX_INPUT_TOKENS['default'])

def get_bedrock_max_output_tokens(model_family: str) -> int:
    """Get max output tokens for a Bedrock model family."""
    return BEDROCK_MAX_OUTPUT_TOKENS.get(model_family, BEDROCK_MAX_OUTPUT_TOKENS['default'])

def get_openai_max_tokens(model_name: str) -> int:
    """Get max output tokens for an OpenAI model."""
    model_lower = model_name.lower()

    # Check specific models first
    if model_lower in OPENAI_MAX_TOKENS:
        return OPENAI_MAX_TOKENS[model_lower]

    # Check by prefix/pattern
    if model_lower.startswith('o1') or model_lower.startswith('o3') or model_lower.startswith('o4'):
        return OPENAI_MAX_TOKENS['o1']
    elif 'gpt-5' in model_lower:
        return OPENAI_MAX_TOKENS['gpt-5']
    elif 'gpt-4.1' in model_lower:
        return OPENAI_MAX_TOKENS['gpt-4.1']
    elif 'gpt-4o' in model_lower:
        return OPENAI_MAX_TOKENS['gpt-4o']
    elif 'gpt-4' in model_lower:
        return OPENAI_MAX_TOKENS['gpt-4']
    elif 'gpt-35-turbo' in model_lower or 'gpt-3.5-turbo' in model_lower:
        return OPENAI_MAX_TOKENS['gpt-35-turbo']

    return OPENAI_MAX_TOKENS['default']

def should_use_max_completion_tokens(model_name: str) -> bool:
    """Check if model should use max_completion_tokens instead of max_tokens."""
    model_lower = model_name.lower()
    return any(model in model_lower for model in OPENAI_USE_MAX_COMPLETION_TOKENS)

def supports_reasoning_effort(model_name: str) -> bool:
    """Check if model supports reasoning_effort parameter (o-series only)."""
    model_lower = model_name.lower()
    return any(model in model_lower for model in OPENAI_SUPPORTS_REASONING_EFFORT)

def supports_temperature(model_name: str) -> bool:
    """Check if model supports temperature parameter."""
    model_lower = model_name.lower()
    return not any(model in model_lower for model in OPENAI_NO_TEMPERATURE)

def supports_streaming(model_family: str) -> bool:
    """Check if Bedrock model family supports streaming."""
    return model_family in BEDROCK_STREAMING_SUPPORTED

def should_try_streaming(error_msg: str) -> bool:
    """Determine if streaming should be attempted based on error message."""
    return any(indicator.lower() in error_msg.lower()
               for indicator in BEDROCK_TIMEOUT_INDICATORS)

def get_bedrock_model_id(model_name: str) -> str:
    """Get Bedrock model ID from friendly name."""
    return BEDROCK_MODEL_IDS.get(model_name, model_name)

def get_openai_model_id(model_name: str) -> str:
    """Get OpenAI model ID from friendly name."""
    return OPENAI_MODEL_IDS.get(model_name, model_name)

def detect_model_family(model_id: str) -> str:
    """Detect model family from Bedrock model ID."""
    # Handle cross-region inference profile IDs (us. prefix)
    if model_id.startswith('meta.llama') or model_id.startswith('us.meta.llama'):
        return 'llama'
    elif model_id.startswith('amazon.titan'):
        return 'titan'
    elif model_id.startswith('amazon.nova'):
        return 'nova'
    elif model_id.startswith('anthropic.claude'):
        return 'claude'
    elif model_id.startswith('mistral.ministral'):
        return 'ministral'  # Ministral is separate from Mistral (uses Converse API)
    elif model_id.startswith('mistral.'):
        return 'mistral'
    elif model_id.startswith('ai21.'):
        return 'ai21'
    elif model_id.startswith('deepseek.') or model_id.startswith('us.deepseek.'):
        return 'deepseek'
    elif model_id.startswith('cohere.'):
        return 'cohere'
    elif model_id.startswith('qwen.'):
        return 'qwen'
    else:
        return 'unknown'

def format_llama_prompt(prompt: str) -> str:
    """Format prompt for Llama models."""
    return LLAMA_PROMPT_TEMPLATE.format(prompt=prompt)

def format_deepseek_prompt(prompt: str) -> str:
    """Format prompt for DeepSeek R1 models."""
    return DEEPSEEK_PROMPT_TEMPLATE.format(prompt=prompt)

def format_mistral_prompt(prompt: str) -> str:
    """Format prompt for Mistral models."""
    return MISTRAL_PROMPT_TEMPLATE.format(prompt=prompt)
