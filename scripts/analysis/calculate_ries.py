"""
Calculate Retrospective Interference Endurance Score (RIES) for all models.

RIES is defined as the area under the curve of accuracy vs log10(interference_level + 1)
using trapezoidal integration, following the methodology from "Unable to Forget" paper (IES).

Terminology:
- IES (paper): Interference Endurance Score for Proactive Interference (recall LAST value)
- RIES (our study): Retrospective Interference Endurance Score (recall FIRST value)

Higher RIES = better resistance to retrospective interference

Usage:
    python calculate_ries.py --input experiment_results_20251226_130754.xlsx --output ries_analysis.xlsx
"""

import pandas as pd
import numpy as np
from scipy import integrate
import argparse
from pathlib import Path


def calculate_ries(model_data: pd.DataFrame) -> float:
    """
    Calculate RIES for a single model using trapezoidal integration.

    Args:
        model_data: DataFrame with 'interference_level' and 'accuracy' columns

    Returns:
        RIES score (area under curve)
    """
    # Sort by interference level
    model_data = model_data.sort_values('interference_level')

    # Convert interference levels to log scale
    x = np.log10(model_data['interference_level'].values + 1)

    # Accuracy values (0-100 scale)
    y = model_data['accuracy'].values

    # Calculate area under curve using trapezoidal rule
    ries = integrate.trapezoid(y, x)

    return ries


def extract_model_size(model_id: str) -> float:
    """
    Extract model size in billions of parameters from model ID.

    Args:
        model_id: Model identifier

    Returns:
        Model size in billions (B), or NaN if not extractable
    """
    import re

    # Common patterns: "llama-3.1-70b", "gpt-4", "claude-4.5-sonnet"
    # Extract numbers followed by 'b' (case insensitive)
    # IMPORTANT: For Llama 4 models, use total parameters, not active parameters
    match = re.search(r'(\d+(?:\.\d+)?)[bB]', model_id)
    if match:
        size = float(match.group(1))
        # Special handling for Llama 4 Scout/Maverick (MoE models with many experts)
        if 'llama-4-scout' in model_id.lower():
            return 109  # 109B total params (16 experts, 17B active)
        elif 'llama-4-maverick' in model_id.lower():
            return 400  # 400B total params (128 experts, 17B active)
        return size

    # Special cases based on known model sizes
    size_mapping = {
        # GPT models (estimated from public information)
        'gpt-3.5-turbo': 175,
        'gpt-4': 1760,  # Estimated (mixture of experts)
        'gpt-4-turbo': 1760,
        'gpt-4o': 1760,
        'gpt-4o-mini': 8,
        'gpt-4.1': 2000,  # Estimated next-gen
        'gpt-4.1-mini': 10,
        'gpt-4.1-nano': 1,
        'gpt-5': 2500,  # Estimated next-gen
        'gpt-5-mini': 20,
        'gpt-5-nano': 2,

        # O-series (reasoning models, estimated similar to GPT-4)
        'o1': 1760,
        'o1-mini': 1760,
        'o1-preview': 1760,
        'o1-pro': 1760,
        'o3': 2000,
        'o3-mini': 20,
        'o4-mini': 30,

        # Claude models (estimated from public info)
        'claude-3-opus': 175,
        'claude-3-sonnet': 100,
        'claude-3.5-haiku': 10,
        'claude-3.5-sonnet': 100,
        'claude-3.7-sonnet': 120,
        'claude-4-sonnet': 200,
        'claude-4-opus': 200,
        'claude-4.5-haiku': 15,
        'claude-4.5-sonnet': 250,
        'claude-4.5-opus': 300,

        # Gemini models (estimated)
        'gemini-1.5-pro': 250,
        'gemini-1.5-flash': 80,
        'gemini-2.0-flash': 100,
        'gemini-2.0-flash-exp': 100,
        'gemini-2.0-flash-lite': 30,
        'gemini-2.5-pro': 300,
        'gemini-2.5-flash-lite': 40,

        # Amazon Nova models
        'nova-micro': 1,
        'nova-lite': 7,
        'nova-pro': 70,
        'nova-premier': 200,

        # Amazon Titan models
        'titan-lite': 20,
        'titan-express': 50,
        'titan-premier': 100,

        # Mistral models
        'mistral-7b': 7,
        'mistral-8x7b': 47,  # 8 experts × 7B, ~47B active parameters
        'mistral-large': 123,
        'mistral-small': 22,
        'ministral-3b': 3,
        'ministral-8b': 8,

        # DeepSeek
        'deepseek-r1': 671,  # 671B parameters

        # Qwen
        'qwen-turbo': 72,
        'qwen-plus': 235,
    }

    # Try to find match in mapping
    for key, size in size_mapping.items():
        if key in model_id.lower():
            return size

    return np.nan


def parse_context_limit(context_str: str) -> int:
    """
    Parse context limit string to integer tokens.

    Args:
        context_str: String like '8K', '128K', '1M', '42K chars'

    Returns:
        Integer token count
    """
    if pd.isna(context_str):
        return np.nan

    context_str = str(context_str).upper().strip()

    # Remove 'chars' suffix if present (convert chars to approx tokens)
    if 'CHARS' in context_str:
        context_str = context_str.replace('CHARS', '').strip()
        # Approximate: 1 token ≈ 4 characters
        chars = int(context_str.replace('K', '')) * 1000
        return chars // 4

    # Parse K (thousands)
    if 'K' in context_str:
        return int(context_str.replace('K', '')) * 1000

    # Parse M (millions) - handle decimals like 3.5M
    if 'M' in context_str:
        return int(float(context_str.replace('M', '')) * 1000000)

    # Already a number
    return int(context_str)


def main():
    parser = argparse.ArgumentParser(description='Calculate RIES scores for all models')
    parser.add_argument(
        '--input',
        type=str,
        default='experiment_results_20251226_130754.xlsx',
        help='Input Excel file with experiment results'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='ries_analysis.xlsx',
        help='Output Excel file with RIES scores and analysis'
    )
    parser.add_argument(
        '--max-level',
        type=int,
        default=400,
        help='Maximum interference level to include in RIES calculation (default: 400)'
    )

    args = parser.parse_args()

    # Load data
    print(f"📊 Loading data from {args.input}...")
    df = pd.read_excel(args.input)

    print(f"   Found {len(df)} data points for {df['model_id'].nunique()} models")
    print(f"   Interference levels: {sorted(df['interference_level'].unique())}")

    # Filter data based on criteria
    print(f"\n🔍 Filtering data...")
    print(f"   Excluding interference level > {args.max_level} (keeping levels 3-{args.max_level})")

    original_count = len(df)

    # Step 1: Keep only levels up to max_level
    df = df[df['interference_level'] <= args.max_level].copy()
    print(f"   After level filter: {len(df)} data points ({original_count - len(df)} excluded)")

    # Step 2: Exclude data points with accuracy=0 AND any error
    # Keep data points where accuracy=0 but NO error (real interference)
    before_error_filter = len(df)

    # Identify rows to exclude: accuracy=0 AND error is not null/empty
    error_mask = (
        (df['accuracy'] == 0) &
        (df['error'].notna()) &
        (df['error'].astype(str).str.strip() != '')
    )

    df_excluded = df[error_mask].copy()
    df = df[~error_mask].copy()

    print(f"   Excluded {len(df_excluded)} data points with accuracy=0 AND error:")
    if len(df_excluded) > 0:
        error_summary = df_excluded.groupby(['model_id', 'interference_level']).size()
        for (model, level), count in error_summary.items():
            print(f"     - {model} level {level}")

    print(f"   After error filter: {len(df)} data points")
    print(f"   Total excluded: {original_count - len(df)} ({100*(original_count - len(df))/original_count:.1f}%)")
    print(f"   Interference levels in filtered data: {sorted(df['interference_level'].unique())}")

    # Calculate RIES for each model
    print(f"\n🔬 Calculating RIES scores...")
    ries_results = []

    # Determine required number of levels based on max_level
    expected_levels = [3, 10, 50, 100, 200, 300, 400, 500]
    required_levels = [l for l in expected_levels if l <= args.max_level]
    num_required_levels = len(required_levels)

    print(f"\n   Required levels for inclusion: {required_levels}")
    print(f"   Models must have ALL {num_required_levels} levels for fair comparison\n")

    for model_id in df['model_id'].unique():
        model_data = df[df['model_id'] == model_id].copy()

        # Skip models without ALL required levels (for fair comparison)
        if len(model_data) < num_required_levels:
            print(f"   ⚠️  {model_id:30s} SKIPPED (only {len(model_data)}/{num_required_levels} levels)")
            continue

        # Get model info
        model_name = model_data['model_name'].iloc[0]
        context_limit = model_data['input_max_tokens'].iloc[0]

        # Calculate RIES
        ries = calculate_ries(model_data)

        # Extract model size
        model_size = extract_model_size(model_id)

        # Parse context limit to integer
        context_tokens = parse_context_limit(context_limit)

        # Get accuracy statistics
        accuracies = model_data['accuracy'].values
        num_levels = len(accuracies)
        max_accuracy = accuracies.max()
        min_accuracy = accuracies.min()
        mean_accuracy = accuracies.mean()

        # Get list of interference levels used for this model
        levels_used = sorted(model_data['interference_level'].unique())

        ries_results.append({
            'model_id': model_id,
            'model_name': model_name,
            'ries_score': ries,
            'model_size_b': model_size,
            'context_tokens': context_tokens,
            'num_levels_tested': num_levels,
            'levels_used': str(levels_used),  # Store which levels were used
            'max_accuracy': max_accuracy,
            'min_accuracy': min_accuracy,
            'mean_accuracy': mean_accuracy,
            'accuracy_drop': max_accuracy - min_accuracy,
        })

        print(f"   ✓ {model_id:30s} RIES={ries:6.1f} (n={num_levels}, accuracy: {min_accuracy:.1f}% → {max_accuracy:.1f}%)")

    # Create results DataFrame
    ries_df = pd.DataFrame(ries_results)

    # Sort by RIES (descending)
    ries_df = ries_df.sort_values('ries_score', ascending=False)

    # Calculate correlations
    print(f"\n📈 Correlation Analysis:")

    # RIES vs model size
    valid_size = ries_df[ries_df['model_size_b'].notna()]
    if len(valid_size) > 1:
        corr_size = valid_size['ries_score'].corr(valid_size['model_size_b'])
        r2_size = corr_size ** 2
        print(f"   RIES vs Model Size:         r = {corr_size:+.3f}, R² = {r2_size:.3f} (n={len(valid_size)})")

        # Log scale correlation (like the paper)
        corr_size_log = valid_size['ries_score'].corr(np.log10(valid_size['model_size_b']))
        r2_size_log = corr_size_log ** 2
        print(f"   RIES vs log10(Size):        r = {corr_size_log:+.3f}, R² = {r2_size_log:.3f}")
        print(f"   Paper's IES vs Size:        R² = 0.75 (from Figure 6a)")

    # RIES vs context length
    valid_context = ries_df[ries_df['context_tokens'].notna()]
    if len(valid_context) > 1:
        corr_context = valid_context['ries_score'].corr(valid_context['context_tokens'])
        r2_context = corr_context ** 2
        print(f"   RIES vs Context Length:     r = {corr_context:+.3f}, R² = {r2_context:.3f} (n={len(valid_context)})")

        # Log scale correlation
        corr_context_log = valid_context['ries_score'].corr(np.log10(valid_context['context_tokens']))
        r2_context_log = corr_context_log ** 2
        print(f"   RIES vs log10(Context):     r = {corr_context_log:+.3f}, R² = {r2_context_log:.3f}")
        print(f"   Paper's IES vs Context:     R² ≈ 0 (from Figure 6b)")

    # Interpretation
    print(f"\n💡 Interpretation:")
    if 'valid_size' in locals() and len(valid_size) > 1:
        if r2_size > 0.3:
            print(f"   ✓ Model size DOES predict RIES (R² = {r2_size:.3f})")
        else:
            print(f"   ⚠ Model size weakly predicts RIES (R² = {r2_size:.3f})")

    if 'valid_context' in locals() and len(valid_context) > 1:
        if r2_context < 0.1:
            print(f"   ✓ Context length does NOT predict RIES (R² = {r2_context:.3f}) - matches paper!")
        else:
            print(f"   ⚠ Context length shows unexpected correlation (R² = {r2_context:.3f})")

    # Save results
    print(f"\n💾 Saving results to {args.output}...")

    # Create Excel writer with multiple sheets
    with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
        # Sheet 1: RIES scores and model info
        ries_df.to_excel(writer, sheet_name='RIES_Scores', index=False)

        # Sheet 2: Original data with model info merged
        df_merged = df.merge(
            ries_df[['model_id', 'ries_score', 'model_size_b']],
            on='model_id',
            how='left'
        )
        df_merged.to_excel(writer, sheet_name='Full_Data', index=False)

        # Sheet 3: Summary statistics by model family
        ries_df['model_family'] = ries_df['model_id'].apply(
            lambda x: x.split('-')[0] if '-' in x else x.split('_')[0]
        )
        family_stats = ries_df.groupby('model_family').agg({
            'ries_score': ['mean', 'std', 'min', 'max', 'count'],
            'model_size_b': 'mean',
            'mean_accuracy': 'mean'
        }).round(2)
        family_stats.to_excel(writer, sheet_name='Family_Stats')

    print(f"   ✓ Saved {len(ries_df)} model RIES scores")
    print(f"   ✓ Saved {len(df)} raw data points")
    print(f"   ✓ Saved summary statistics by model family")

    # Print top 10 models
    print(f"\n🏆 Top 10 Models by RIES (Retrospective Interference Endurance):")
    print(f"{'Rank':<6} {'Model':<35} {'RIES':<8} {'Size (B)':<10} {'Accuracy Range'}")
    print("=" * 80)
    for idx, (i, row) in enumerate(ries_df.head(10).iterrows(), 1):
        size_str = f"{row['model_size_b']:.0f}B" if not pd.isna(row['model_size_b']) else "Unknown"
        acc_range = f"{row['min_accuracy']:.1f}% → {row['max_accuracy']:.1f}%"
        print(f"{idx:<6} {row['model_id']:<35} {row['ries_score']:<8.1f} {size_str:<10} {acc_range}")

    print(f"\n✅ Analysis complete! Check {args.output} for full results.")
    print(f"\n📝 Next steps:")
    print(f"   1. Generate visualizations (decay curves, scatter plots)")
    print(f"   2. Error classification analysis (parse JSON results)")
    print(f"   3. Compare RIES patterns to paper's IES patterns")


if __name__ == '__main__':
    main()
