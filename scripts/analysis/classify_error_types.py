#!/usr/bin/env python3
"""
Error Type Classification for Retroactive Interference Experiments

Classifies incorrect model extractions into 5 cognitive failure modes:
1. Same-Key Interference: Earlier value from SAME category
2. Cross-Key Interference: Value from DIFFERENT category
3. Partial Match: Close but not exact match
4. Hallucination: Value NEVER presented in prompt
5. Retrieval Failure: No value extracted (null/None/empty)

Based on methodology from "Unable to Forget: Persistent Memory in Language Models"
(Section 2.2.1: Incorrect Extractions Analysis)
"""

import json
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd
from typing import Dict, List, Tuple, Optional
import re

def normalize_value(value: str) -> str:
    """Normalize value for comparison (lowercase, strip whitespace)."""
    if value is None:
        return ""
    return str(value).lower().strip()

def fuzzy_match(extracted: str, target: str, threshold: float = 0.7) -> bool:
    """
    Check if extracted is a partial match of target.

    Uses multiple strategies:
    - Exact substring match
    - Word overlap for multi-word values
    - Common prefix/suffix
    """
    if not extracted or not target:
        return False

    extracted_norm = normalize_value(extracted)
    target_norm = normalize_value(target)

    # Exact match (shouldn't happen, but just in case)
    if extracted_norm == target_norm:
        return True

    # Substring match (one contains the other)
    if extracted_norm in target_norm or target_norm in extracted_norm:
        if len(extracted_norm) >= 3 or len(target_norm) >= 3:  # At least 3 chars
            return True

    # Word overlap for multi-word values
    extracted_words = set(extracted_norm.split())
    target_words = set(target_norm.split())

    if len(extracted_words) > 0 and len(target_words) > 0:
        overlap = len(extracted_words & target_words)
        max_words = max(len(extracted_words), len(target_words))

        if overlap > 0 and (overlap / max_words) >= threshold:
            return True

    return False

def load_dataset(dataset_path: str) -> Dict:
    """Load the interleaved dataset."""
    with open(dataset_path, 'r') as f:
        return json.load(f)

def get_all_values_for_category(dataset: Dict, level: int, category: str) -> List[Dict]:
    """
    Get all values presented for a category at a given interference level.
    Returns list of dicts with 'value' and 'position' (1-indexed from first appearance).
    """
    level_key = str(level)
    if level_key not in dataset['levels']:
        return []

    level_data = dataset['levels'][level_key]

    # Extract all updates for this category in order
    category_updates = []
    for i, update in enumerate(level_data['interleaved_sequence'], 1):
        if update['category'] == category:
            category_updates.append({
                'value': update['value'],
                'position': len(category_updates) + 1,  # Position within category updates (1-indexed)
                'global_position': i  # Position in full sequence
            })

    return category_updates

def get_all_values_other_categories(dataset: Dict, level: int, exclude_category: str) -> List[str]:
    """Get all values from OTHER categories at this interference level."""
    level_key = str(level)
    if level_key not in dataset['levels']:
        return []

    level_data = dataset['levels'][level_key]

    other_values = []
    for update in level_data['interleaved_sequence']:
        if update['category'] != exclude_category:
            other_values.append(update['value'])

    return other_values

def classify_error(extracted: str, expected: str, category: str,
                   category_values: List[Dict], other_category_values: List[str]) -> Tuple[str, Dict]:
    """
    Classify an incorrect extraction into one of 5 error types.

    Returns:
        error_type: One of ['same_key_interference', 'cross_key_interference',
                           'partial_match', 'hallucination', 'retrieval_failure']
        details: Dict with additional info (position, matched_value, etc.)
    """

    extracted_norm = normalize_value(extracted)
    expected_norm = normalize_value(expected)

    # Case 1: Retrieval Failure (null/None/empty)
    if not extracted_norm or extracted_norm in ["none", "null", "n/a", "na"]:
        return 'retrieval_failure', {
            'reason': 'No value extracted',
            'raw_value': extracted
        }

    # Case 2: Same-Key Interference (earlier value from SAME category)
    # Check for exact matches first
    for i, val_info in enumerate(category_values):
        val_norm = normalize_value(val_info['value'])

        # Skip the expected value (position 1 = baseline/first/target)
        if val_info['position'] == 1:
            continue

        # Exact match to a later update
        if extracted_norm == val_norm:
            return 'same_key_interference', {
                'matched_value': val_info['value'],
                'position': val_info['position'],
                'total_updates': len(category_values),
                'recency': len(category_values) - val_info['position'] + 1,  # How many positions from end
                'is_most_recent': (val_info['position'] == len(category_values))
            }

    # Case 3: Cross-Key Interference (exact match to value from DIFFERENT category)
    for other_val in other_category_values:
        other_val_norm = normalize_value(other_val)

        if extracted_norm == other_val_norm:
            return 'cross_key_interference', {
                'matched_value': other_val,
                'source': 'other_category'
            }

    # Case 4: Partial Match (fuzzy match to ANY presented value)
    # Check same category first
    for val_info in category_values:
        if fuzzy_match(extracted, val_info['value']):
            return 'partial_match', {
                'matched_value': val_info['value'],
                'extracted_value': extracted,
                'match_type': 'same_category',
                'position': val_info['position']
            }

    # Check other categories
    for other_val in other_category_values:
        if fuzzy_match(extracted, other_val):
            return 'partial_match', {
                'matched_value': other_val,
                'extracted_value': extracted,
                'match_type': 'other_category'
            }

    # Case 5: Hallucination (value never presented anywhere)
    return 'hallucination', {
        'extracted_value': extracted,
        'reason': 'Value not found in any presented updates'
    }

def analyze_model_errors(result_file: Path, dataset: Dict, output_dir: Path, max_level: int = 300) -> pd.DataFrame:
    """Analyze all errors for a single model across all interference levels."""

    with open(result_file, 'r') as f:
        data = json.load(f)

    model_id = data['model']

    # Check if model has API errors (all extractions are None at level 3)
    # This excludes models with context limit or API errors, NOT models that are just bad
    level_3_results = next((r for r in data['results'] if r['interference_level'] == 3), None)
    if level_3_results:
        all_none = all(
            item.get('extracted') is None
            for item in level_3_results.get('results', [])
        )
        if all_none:
            print(f"\n  EXCLUDING {model_id}: API errors (all extractions None at level 3)")
            return pd.DataFrame()

    print(f"\n{'='*60}")
    print(f"Analyzing: {model_id}")
    print(f"{'='*60}")

    all_errors = []
    total_errors = 0

    for level_result in data['results']:
        level = level_result['interference_level']

        # Skip levels beyond max_level
        if level > max_level:
            continue

        accuracy = level_result['accuracy']
        total = level_result['total_count']
        correct = level_result['correct_count']
        error_count = total - correct

        total_errors += error_count

        # Get all incorrect extractions
        incorrect = [r for r in level_result['results'] if not r['correct']]

        if not incorrect:
            print(f"  Level {level:3d}: {accuracy:5.1f}% accuracy | 0 errors")
            continue

        print(f"  Level {level:3d}: {accuracy:5.1f}% accuracy | {error_count} errors")

        for item in incorrect:
            category = item['category']
            expected = item['expected']
            extracted = item.get('extracted', None)

            # Get all values for this category at this level
            category_values = get_all_values_for_category(dataset, level, category)
            other_values = get_all_values_other_categories(dataset, level, category)

            # Classify the error
            error_type, details = classify_error(
                extracted, expected, category, category_values, other_values
            )

            all_errors.append({
                'model': model_id,
                'interference_level': level,
                'accuracy': accuracy,
                'category': category,
                'expected': expected,
                'extracted': extracted,
                'error_type': error_type,
                **details
            })

    print(f"\nTotal errors: {total_errors}")
    return pd.DataFrame(all_errors)

def summarize_error_patterns(df: pd.DataFrame, output_dir: Path):
    """Generate summary statistics and export."""

    print("\n" + "="*60)
    print("OVERALL ERROR TYPE DISTRIBUTION")
    print("="*60)

    error_counts = df['error_type'].value_counts()
    error_pcts = df['error_type'].value_counts(normalize=True) * 100

    total_errors = len(df)

    print(f"\nTotal errors analyzed: {total_errors}\n")
    for error_type in ['same_key_interference', 'cross_key_interference',
                       'partial_match', 'hallucination', 'retrieval_failure']:
        if error_type in error_counts.index:
            count = error_counts[error_type]
            pct = error_pcts[error_type]
            print(f"  {error_type:25s}: {count:5d} ({pct:5.1f}%)")

    # Error distribution by interference level
    print("\n" + "="*60)
    print("ERROR DISTRIBUTION BY INTERFERENCE LEVEL (%)")
    print("="*60)

    pivot = pd.crosstab(df['interference_level'], df['error_type'], normalize='index') * 100

    # Reorder columns
    desired_order = ['same_key_interference', 'cross_key_interference',
                     'partial_match', 'hallucination', 'retrieval_failure']
    available_cols = [col for col in desired_order if col in pivot.columns]
    pivot = pivot[available_cols]

    print(pivot.round(1).to_string())

    # Same-Key Interference position analysis
    ski_errors = df[df['error_type'] == 'same_key_interference'].copy()
    if len(ski_errors) > 0:
        print("\n" + "="*60)
        print("SAME-KEY INTERFERENCE: POSITION ANALYSIS")
        print("="*60)
        print("(Position 1 = first/target, higher = more recent updates)\n")

        # Count most recent vs earlier
        recency_bias = ski_errors['is_most_recent'].sum()
        recency_pct = (recency_bias / len(ski_errors)) * 100

        print(f"  Most recent value (recency bias): {recency_bias}/{len(ski_errors)} ({recency_pct:.1f}%)")
        print()

        # Group by interference level
        for level in sorted(ski_errors['interference_level'].unique()):
            level_ski = ski_errors[ski_errors['interference_level'] == level]
            avg_position = level_ski['position'].mean()
            avg_recency = level_ski['recency'].mean()
            recent_count = level_ski['is_most_recent'].sum()
            recent_pct = (recent_count / len(level_ski)) * 100

            print(f"  Level {level:3d}: Avg pos={avg_position:.1f}, "
                  f"Avg recency={avg_recency:.1f} steps back, "
                  f"Most recent={recent_count}/{len(level_ski)} ({recent_pct:.0f}%)")

    # Export detailed results
    output_file = output_dir / 'error_classification_detailed.xlsx'

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Sheet 1: All errors with full details
        df_export = df.copy()
        df_export.to_excel(writer, sheet_name='All_Errors', index=False)

        # Sheet 2: Summary by level (counts)
        summary_counts = df.groupby(['interference_level', 'error_type']).size().reset_index(name='count')
        summary_pivot = summary_counts.pivot(index='interference_level',
                                              columns='error_type',
                                              values='count').fillna(0).astype(int)
        summary_pivot = summary_pivot[available_cols]
        summary_pivot.to_excel(writer, sheet_name='Counts_By_Level')

        # Sheet 3: Summary by level (percentages)
        pivot.to_excel(writer, sheet_name='Percentages_By_Level')

        # Sheet 4: Summary by model
        summary_by_model = df.groupby(['model', 'error_type']).size().reset_index(name='count')
        summary_model_pivot = summary_by_model.pivot(index='model',
                                                      columns='error_type',
                                                      values='count').fillna(0).astype(int)
        if len(summary_model_pivot.columns) > 0:
            model_cols = [col for col in available_cols if col in summary_model_pivot.columns]
            summary_model_pivot = summary_model_pivot[model_cols]
        summary_model_pivot.to_excel(writer, sheet_name='Counts_By_Model')

        # Sheet 5: Same-Key position analysis
        if len(ski_errors) > 0:
            ski_summary = ski_errors.groupby('interference_level').agg({
                'position': ['mean', 'std', 'min', 'max'],
                'recency': ['mean', 'std'],
                'is_most_recent': ['sum', 'mean']
            }).round(2)
            ski_summary.to_excel(writer, sheet_name='SameKey_Position_Analysis')

    print(f"\n{'='*60}")
    print(f"✅ Results saved: {output_file}")
    print(f"{'='*60}")

    return df

def main():
    parser = argparse.ArgumentParser(description='Classify error types in RI experiments')
    parser.add_argument('--results-dirs', type=str, nargs='+',
                       default=[
                           'results/run1_claude_gpt_gemini_results',
                           'results/archived/bedrock_responses_deepseek/bedrock_results'
                       ],
                       help='Directories containing result JSON files')
    parser.add_argument('--dataset', type=str,
                       default='data/interleaved_dataset_meaningful.json',
                       help='Path to interleaved dataset')
    parser.add_argument('--output-dir', type=str,
                       default='results/error_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--models', type=str, nargs='+',
                       help='Specific models to analyze (default: all)')
    parser.add_argument('--max-level', type=int, default=300,
                       help='Maximum interference level to analyze (default: 300)')

    args = parser.parse_args()

    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset)
    print(f"✅ Loaded dataset with {len(dataset['levels'])} interference levels")

    # Find result files from all directories
    result_files = []
    for results_dir in args.results_dirs:
        dir_path = Path(results_dir)
        if dir_path.exists():
            files = list(dir_path.glob('*.json'))
            result_files.extend(files)
            print(f"Found {len(files)} result files in {results_dir}")

    if not result_files:
        print("\n❌ No result files found!")
        return

    # Filter by specific models if requested
    if args.models:
        result_files = [f for f in result_files
                       if any(model.lower() in f.stem.lower() for model in args.models)]
        print(f"\n✅ Filtered to {len(result_files)} files matching: {args.models}")

    print(f"\n{'='*60}")
    print(f"ANALYZING {len(result_files)} MODELS")
    print(f"{'='*60}")

    # Analyze all models
    all_dfs = []
    for i, result_file in enumerate(sorted(result_files), 1):
        try:
            print(f"\n[{i}/{len(result_files)}]")
            df = analyze_model_errors(result_file, dataset, output_dir, max_level=args.max_level)

            if len(df) > 0:
                # Filter to max_level already done in analyze_model_errors
                all_dfs.append(df)
        except Exception as e:
            print(f"\n❌ Error analyzing {result_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_dfs:
        print("\n❌ No errors found to analyze")
        return

    # Combine all results
    combined_df = pd.concat(all_dfs, ignore_index=True)

    print(f"\n{'='*60}")
    print(f"COMBINED ANALYSIS: {len(combined_df)} TOTAL ERRORS")
    print(f"Models: {combined_df['model'].nunique()}")
    print(f"{'='*60}")

    # Generate summary
    summarize_error_patterns(combined_df, output_dir)

    print("\n✅ Error classification complete!")

if __name__ == '__main__':
    main()
