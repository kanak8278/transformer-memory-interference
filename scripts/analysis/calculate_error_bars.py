#!/usr/bin/env python3
"""
Calculate error bars (mean ± std, 95% CI) from multiple RI experiment runs.

Input:
- data/raw_ri_claude_gpt_gemini/*.json (3 runs per model)
- data/raw_ri_bedrock/*.json (3 runs per model)

Output:
- results/key_results/ries_with_error_bars.xlsx
  - Sheet 1: Error bars by level (mean, std, CI at each interference level)
  - Sheet 2: Variance analysis (test if variance differs by model size)
  - Sheet 3: Summary statistics (overall mean RIES with confidence intervals)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats
import sys
import os

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def load_experiment_runs(data_dir):
    """
    Load all experiment JSON files from directory, group by model_id.

    Returns:
        dict: {model_id: [run1_data, run2_data, run3_data]}
    """
    data_dir = Path(data_dir)
    runs_by_model = defaultdict(list)

    for json_file in sorted(data_dir.glob("*.json")):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Extract model_id from filename
            # Format: {model_id}_levels_{levels}_{timestamp}.json
            filename = json_file.stem
            model_id = filename.split("_levels_")[0]

            # Normalize model_id (remove bedrock- prefix if present)
            model_id = model_id.replace("bedrock-", "")

            runs_by_model[model_id].append(data)

        except Exception as e:
            print(f"Warning: Error loading {json_file}: {e}")

    return runs_by_model


def calculate_statistics(runs_list):
    """
    Calculate mean, std, and 95% CI from multiple runs.

    Args:
        runs_list: List of experiment result dicts (each has 'results' key with list of level results)

    Returns:
        pd.DataFrame with columns: interference_level, mean_accuracy, std_accuracy,
                                   n_runs, sem, ci_lower, ci_upper, ci_width
    """
    # Extract accuracy by level for each run
    level_accuracies = defaultdict(list)

    for run_data in runs_list:
        for level_result in run_data['results']:
            level = level_result['interference_level']
            accuracy = level_result['accuracy']
            level_accuracies[level].append(accuracy)

    # Calculate statistics for each level
    stats_list = []
    for level in sorted(level_accuracies.keys()):
        accuracies = level_accuracies[level]
        n = len(accuracies)

        if n == 0:
            continue

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies, ddof=1) if n > 1 else 0.0
        sem = std_acc / np.sqrt(n) if n > 1 else 0.0

        # 95% confidence interval using t-distribution
        if n > 1:
            ci_width = stats.t.ppf(0.975, df=n-1) * sem
            ci_lower = mean_acc - ci_width
            ci_upper = mean_acc + ci_width
        else:
            ci_width = 0.0
            ci_lower = mean_acc
            ci_upper = mean_acc

        stats_list.append({
            'interference_level': level,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'n_runs': n,
            'sem': sem,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_width
        })

    return pd.DataFrame(stats_list)


def calculate_ries_with_ci(stats_df):
    """
    Calculate RIES score with confidence interval from statistics DataFrame.

    Uses trapezoidal integration on log-scale x-axis.
    """
    # Sort by interference level
    stats_df = stats_df.sort_values('interference_level')

    levels = stats_df['interference_level'].values
    mean_acc = stats_df['mean_accuracy'].values
    ci_lower = stats_df['ci_lower'].values
    ci_upper = stats_df['ci_upper'].values

    # Log-scale x-axis
    x = np.log10(levels + 1)

    # Trapezoidal integration
    ries_mean = np.trapz(mean_acc, x)
    ries_lower = np.trapz(ci_lower, x)
    ries_upper = np.trapz(ci_upper, x)

    return {
        'ries_mean': ries_mean,
        'ries_ci_lower': ries_lower,
        'ries_ci_upper': ries_upper,
        'ries_ci_width': ries_upper - ries_lower
    }


def main():
    print("="*80)
    print("Calculating Error Bars from Multiple RI Runs")
    print("="*80)

    # Paths
    data_dir1 = project_root / "data" / "raw_ri_claude_gpt_gemini"
    data_dir2 = project_root / "data" / "raw_ri_bedrock"
    output_dir = project_root / "results" / "key_results"
    output_file = output_dir / "ries_with_error_bars.xlsx"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load runs from both directories
    print(f"\nLoading runs from:")
    print(f"  {data_dir1}")
    print(f"  {data_dir2}")

    runs_by_model = {}

    if data_dir1.exists():
        runs1 = load_experiment_runs(data_dir1)
        runs_by_model.update(runs1)
        print(f"  Loaded {len(runs1)} models from directory 1")

    if data_dir2.exists():
        runs2 = load_experiment_runs(data_dir2)
        runs_by_model.update(runs2)
        print(f"  Loaded {len(runs2)} models from directory 2")

    print(f"\nTotal unique models: {len(runs_by_model)}")

    # Check run counts
    run_counts = {model: len(runs) for model, runs in runs_by_model.items()}
    print(f"\nRun count distribution:")
    for count in sorted(set(run_counts.values())):
        models_with_count = [m for m, c in run_counts.items() if c == count]
        print(f"  {count} runs: {len(models_with_count)} models")

    # Calculate statistics for each model
    all_stats = []
    ries_summary = []

    print(f"\nCalculating statistics for each model...")
    for model_id, runs in sorted(runs_by_model.items()):
        print(f"  {model_id}: {len(runs)} runs")

        # Calculate statistics at each level
        stats_df = calculate_statistics(runs)
        stats_df['model_id'] = model_id
        all_stats.append(stats_df)

        # Calculate RIES with CI
        ries_with_ci = calculate_ries_with_ci(stats_df)
        ries_summary.append({
            'model_id': model_id,
            'n_runs': len(runs),
            **ries_with_ci
        })

    # Combine all statistics
    all_stats_df = pd.concat(all_stats, ignore_index=True)
    ries_summary_df = pd.DataFrame(ries_summary)

    # Variance analysis: Test if variance differs by model size
    print(f"\nPerforming variance analysis...")

    # For simplicity, we'll look at variance at key levels (10, 100, 300)
    variance_analysis = []

    for level in [10, 100, 300]:
        level_data = all_stats_df[all_stats_df['interference_level'] == level].copy()

        if len(level_data) > 0:
            variance_analysis.append({
                'interference_level': level,
                'mean_std': level_data['std_accuracy'].mean(),
                'median_std': level_data['std_accuracy'].median(),
                'min_std': level_data['std_accuracy'].min(),
                'max_std': level_data['std_accuracy'].max(),
                'n_models': len(level_data)
            })

    variance_df = pd.DataFrame(variance_analysis)

    # Save to Excel
    print(f"\nSaving results to: {output_file}")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Sheet 1: Error bars by level for each model
        all_stats_df.to_excel(writer, sheet_name='Error_Bars_by_Level', index=False)

        # Sheet 2: RIES summary with confidence intervals
        ries_summary_df.to_excel(writer, sheet_name='RIES_with_CI', index=False)

        # Sheet 3: Variance analysis
        variance_df.to_excel(writer, sheet_name='Variance_Analysis', index=False)

    print(f"\n{'='*80}")
    print("Summary Statistics")
    print(f"{'='*80}")
    print(f"\nRIES with 95% Confidence Intervals:")
    print(f"  Mean RIES: {ries_summary_df['ries_mean'].mean():.2f} ± {ries_summary_df['ries_ci_width'].mean():.2f}")
    print(f"  Range: [{ries_summary_df['ries_ci_lower'].mean():.2f}, {ries_summary_df['ries_ci_upper'].mean():.2f}]")

    print(f"\nVariance Analysis (Mean Std Dev by Level):")
    for _, row in variance_df.iterrows():
        print(f"  Level {int(row['interference_level']):3d}: {row['mean_std']:.2f}% (range: {row['min_std']:.2f}-{row['max_std']:.2f}%)")

    print(f"\n{'='*80}")
    print("✓ Error bars calculation complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
