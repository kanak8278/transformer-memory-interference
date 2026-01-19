"""
Export all experiment results to Excel.

This script reads all JSON result files from both locations:
- results/ (Claude, Gemini, OpenAI models)
- bedrock_responses_deepseek/results/ (Bedrock models)

Creates an Excel file with columns: model_id, model_name, input_max_tokens, interference_level, accuracy
Only includes models that have results for at least 1 interference level.
"""

import json
import os
from pathlib import Path
import pandas as pd
from typing import List, Dict

# Model input context limits (in tokens unless specified)
MODEL_CONTEXT_LIMITS = {
    # Llama models
    'llama-3-8b': '8K', 'llama-3-70b': '8K',
    'llama-3.1-8b': '128K', 'llama-3.1-70b': '128K', 'llama-3.1-405b': '128K',
    'llama-3.2-1b': '128K', 'llama-3.2-3b': '128K', 'llama-3.2-11b': '128K', 'llama-3.2-90b': '128K',
    'llama-3.3-70b': '128K',
    'llama-4-maverick-17b': '1M',      # 1 million token context (128 experts, 400B total params)
    'llama-4-scout-17b': '3.5M',       # 3.5 million token context (16 experts, 109B total params)

    # Amazon Titan (note: Lite/Express are 8K, but using chars for compatibility)
    'titan-lite': '8K',           # 8,000 tokens
    'titan-express': '8K',        # 8,000 tokens
    'titan-large': '32K',         # 32,000 tokens (Titan G1 Premier)
    'titan-premier': '32K',       # 32,000 tokens

    # Amazon Nova
    'nova-micro': '300K', 'nova-lite': '300K', 'nova-pro': '300K', 'nova-premier': '300K',

    # Mistral models
    'mistral-7b': '32K', 'mistral-8x7b': '32K', 'mistral-large': '32K', 'mistral-small': '32K',
    'ministral-3b': '128K', 'ministral-8b': '128K',

    # DeepSeek
    'deepseek-r1': '64K',

    # Qwen models
    'qwen-turbo': '32K',          # Qwen2.5 Turbo: 32K tokens
    'qwen-plus': '32K',           # Qwen2.5 Plus: 32K tokens
    'qwen-3-32b': '128K',         # Qwen3-32B: 128K tokens (native 32K, extendable to 128K)
    'qwen-3-coder-30b': '256K',   # Qwen3-Coder: 256K tokens
    'qwen-3-vl-235b': '256K',     # Qwen3-VL: 256K tokens

    # Claude models
    'claude-3-opus': '200K', 'claude-3-sonnet': '200K',
    'claude-3.5-haiku': '200K', 'claude-3.5-sonnet': '200K',
    'claude-3.7-sonnet': '200K',
    'claude-4-sonnet': '200K', 'claude-4-opus': '200K',
    'claude-4.5-haiku': '200K', 'claude-4.5-sonnet': '200K', 'claude-4.5-opus': '200K',

    # Gemini models
    'gemini-1.5-pro': '2M', 'gemini-1.5-flash': '1M',
    'gemini-2.0-flash': '1M', 'gemini-2.0-flash-exp': '1M', 'gemini-2.0-flash-lite': '1M',
    'gemini-2.5-pro': '2M', 'gemini-2.5-flash-lite': '1M',

    # OpenAI GPT models
    'gpt-3.5-turbo': '16K',
    'gpt-4': '8K', 'gpt-4-turbo': '128K',
    'gpt-4o': '128K', 'gpt-4o-mini': '128K',
    'gpt-4.1': '128K', 'gpt-4.1-mini': '128K', 'gpt-4.1-nano': '128K',
    'gpt-5': '128K', 'gpt-5-mini': '128K', 'gpt-5-nano': '128K',

    # OpenAI O-series (reasoning models)
    'o1': '200K', 'o1-mini': '128K', 'o1-preview': '128K', 'o1-pro': '128K',
    'o3': '200K', 'o3-mini': '200K',
    'o4-mini': '200K',
}


def find_result_files() -> List[Path]:
    """Find all JSON result files from both directories."""
    result_files = []

    # Main results directory (Claude, GPT, Gemini)
    results_dir = Path("results/run1_claude_gpt_gemini_results")
    if results_dir.exists():
        result_files.extend(results_dir.glob("*.json"))

    # Fallback to main results if run1 doesn't exist
    results_dir_main = Path("results")
    if results_dir_main.exists() and not results_dir.exists():
        result_files.extend(results_dir_main.glob("*.json"))

    # Bedrock results directory
    bedrock_dir = Path("results/archived/bedrock_responses_deepseek/bedrock_results")
    if bedrock_dir.exists():
        result_files.extend(bedrock_dir.glob("*.json"))

    return result_files


def extract_model_info(filename: str) -> tuple:
    """
    Extract model ID and name from filename.

    Example filenames:
    - claude-4.5-sonnet_levels_3_10_50_100_200_300_400_500_20251226_115158.json
    - bedrock-llama-3-8b_levels_3_10_50_100_200_300_400_500_20251225_140625.json

    Returns:
        (model_id, model_name) tuple
    """
    # Remove .json extension
    name = filename.replace(".json", "")

    # Split by _levels_ to get model part
    parts = name.split("_levels_")
    if len(parts) < 2:
        return None, None

    model_id = parts[0]

    # For Bedrock models, remove "bedrock-" prefix from model_id
    # bedrock-titan-large -> titan-large
    if model_id.startswith("bedrock-"):
        model_id = model_id.replace("bedrock-", "", 1)

    # Create a cleaner model name
    model_name = model_id.replace("-", " ").title()

    return model_id, model_name


def load_results_from_file(file_path: Path) -> List[Dict]:
    """
    Load results from a JSON file.

    Returns:
        List of dicts with keys: model_id, model_name, input_max_tokens, max_output_tokens,
        interference_level, accuracy, correct_count, total_count, missing_count, missing_rate,
        sequence_length, input_tokens, output_tokens, total_tokens, was_truncated, error, is_valid
        Returns empty list if model has accuracy = 0 for ALL levels
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract model info from filename
        model_id, model_name = extract_model_info(file_path.name)
        if not model_id:
            print(f"⚠️  Skipping {file_path.name}: cannot parse model name")
            return []

        # Get context limit for this model
        input_max_tokens = MODEL_CONTEXT_LIMITS.get(model_id, 'Unknown')

        # Determine max output tokens based on model family
        full_model_name = data.get('model', model_id)
        if 'llama' in full_model_name:
            max_output_tokens = 2048
        elif 'deepseek' in full_model_name:
            max_output_tokens = 8192
        elif 'titan' in full_model_name:
            max_output_tokens = 8000
        elif 'nova' in full_model_name:
            max_output_tokens = 5000
        elif 'ministral' in full_model_name:
            max_output_tokens = 8192
        elif 'mistral' in full_model_name:
            max_output_tokens = 4096
        elif 'qwen' in full_model_name:
            max_output_tokens = 8192
        elif 'claude' in full_model_name:
            if '3.5' in full_model_name or '3.7' in full_model_name or '4' in full_model_name:
                max_output_tokens = 8192
            else:
                max_output_tokens = 4096
        elif 'o1' in full_model_name or 'o3' in full_model_name or 'o4' in full_model_name:
            max_output_tokens = 100000
        elif '4o' in full_model_name:
            max_output_tokens = 16384
        elif 'gpt' in full_model_name:
            max_output_tokens = 4096
        elif 'gemini' in full_model_name:
            max_output_tokens = 8192
        else:
            max_output_tokens = 'Unknown'

        # Extract results for each interference level
        results = []
        if "results" in data and isinstance(data["results"], list):
            for level_result in data["results"]:
                if "interference_level" in level_result and "accuracy" in level_result:
                    # Extract token usage information
                    token_usage = level_result.get("token_usage", {})
                    input_tokens = token_usage.get("input_tokens", None)
                    output_tokens = token_usage.get("output_tokens", None)
                    total_tokens = token_usage.get("total_tokens", None)
                    was_truncated = token_usage.get("was_truncated", False)
                    error = token_usage.get("error", None)

                    # IMPORTANT: Also check batch_response for error messages
                    # Some models put errors in batch_response instead of token_usage.error
                    batch_response = level_result.get("batch_response", "")
                    if batch_response and isinstance(batch_response, str):
                        if batch_response.startswith("Error:") or "error" in batch_response.lower()[:100]:
                            # Extract error from batch_response if not already in error field
                            if error is None:
                                error = batch_response[:200]  # First 200 chars of error message

                    # Determine if this is a valid data point
                    # Invalid if: has error OR was truncated OR accuracy is 0 with missing_count = total_count
                    is_valid = (error is None) and (not was_truncated)

                    # Additional check: if accuracy is 0 and all values are missing, mark as potentially invalid
                    if level_result.get("accuracy", 0) == 0 and level_result.get("missing_count", 0) == level_result.get("total_count", 0):
                        # Only mark invalid if there's an obvious error indicator
                        if error or was_truncated:
                            is_valid = False

                    results.append({
                        "model_id": model_id,
                        "model_name": model_name,
                        "input_max_tokens": input_max_tokens,
                        "max_output_tokens": max_output_tokens,
                        "interference_level": level_result["interference_level"],
                        "accuracy": level_result["accuracy"],
                        "correct_count": level_result.get("correct_count", 0),
                        "total_count": level_result.get("total_count", 0),
                        "missing_count": level_result.get("missing_count", 0),
                        "missing_rate": level_result.get("missing_rate", 0.0),
                        "sequence_length": level_result.get("sequence_length", 0),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                        "was_truncated": was_truncated,
                        "error": error,
                        "is_valid": is_valid
                    })

        if not results:
            print(f"⚠️  No valid results found in {file_path.name}")
            return []

        # Check if ALL levels have accuracy = 0
        # If so, exclude this model completely
        all_zero = all(r["accuracy"] == 0 for r in results)
        if all_zero:
            print(f"🚫 Excluding {model_id}: accuracy = 0 for all {len(results)} levels")
            return []

        return results

    except Exception as e:
        print(f"❌ Error reading {file_path.name}: {e}")
        return []


def main():
    """Main function to export results to Excel."""
    print("🔍 Finding result files...")
    result_files = find_result_files()

    if not result_files:
        print("❌ No result files found!")
        return

    print(f"✓ Found {len(result_files)} result files")

    # Collect all results
    all_results = []
    models_with_data = set()

    for file_path in sorted(result_files):
        print(f"  Processing {file_path.name}...")
        results = load_results_from_file(file_path)
        if results:
            all_results.extend(results)
            models_with_data.add(results[0]["model_id"])

    if not all_results:
        print("❌ No valid results extracted!")
        return

    print(f"\n✓ Extracted {len(all_results)} data points from {len(models_with_data)} models")

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Sort by model_id and interference_level
    df = df.sort_values(['model_id', 'interference_level'])

    # Create output filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"experiment_results_{timestamp}.xlsx"

    # Export to Excel
    print(f"\n📊 Exporting to {output_file}...")
    df.to_excel(output_file, index=False, sheet_name="Results")

    # Print summary statistics
    print(f"\n✓ Export complete!")
    print(f"\n📈 Summary:")
    print(f"  Total models: {df['model_id'].nunique()}")
    print(f"  Total rows: {len(df)}")
    print(f"  Valid data points: {df['is_valid'].sum()}/{len(df)} ({100*df['is_valid'].sum()/len(df):.1f}%)")
    print(f"  Invalid (errors/truncated): {(~df['is_valid']).sum()}")
    print(f"  Interference levels: {sorted(df['interference_level'].unique())}")
    print(f"\n  Models included:")
    for model in sorted(df['model_id'].unique()):
        model_df = df[df['model_id'] == model]
        count = len(model_df)
        valid_count = model_df['is_valid'].sum()
        invalid_count = count - valid_count
        if invalid_count > 0:
            print(f"    - {model}: {count} levels ({valid_count} valid, {invalid_count} invalid)")
        else:
            print(f"    - {model}: {count} levels")

    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
