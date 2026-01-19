"""
Compare Retroactive Interference (RI) vs Proactive Interference (PI) across all models.

This script performs comprehensive comparison analysis:
1. Correlation analysis (RIES vs PIES)
2. Asymmetry testing (paired t-test, Cohen's d)
3. Scatter plots and visualizations
4. Decay pattern comparison
5. Model family analysis

Usage:
    python compare_ri_vs_pi.py --ries results/key_results/ries_analysis.xlsx --pies results/key_results/pies_analysis.xlsx --output results/ri_vs_pi_comparison/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import argparse
from typing import Tuple, Dict


def load_data(ries_file: str, pies_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load RIES and PIES analysis results.

    Args:
        ries_file: Path to RIES analysis Excel file
        pies_file: Path to PIES analysis Excel file

    Returns:
        Tuple of (ries_df, pies_df)
    """
    print(f"\n📊 Loading data...")
    print(f"   RI (RIES):  {ries_file}")
    print(f"   PI (PIES): {pies_file}")

    ries_df = pd.read_excel(ries_file, sheet_name='RIES_Scores')
    pies_df = pd.read_excel(pies_file, sheet_name='PIES_Scores')

    print(f"\n✓ Loaded RIES data: {len(ries_df)} models")
    print(f"✓ Loaded PIES data:  {len(pies_df)} models")

    return ries_df, pies_df


def merge_datasets(ries_df: pd.DataFrame, pies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge RIES and PIES datasets on model_id.

    Args:
        ries_df: RIES scores DataFrame
        pies_df: PIES scores DataFrame

    Returns:
        Merged DataFrame with both RIES and PIES scores
    """
    print(f"\n🔗 Merging RI and PI datasets on model_id...")

    # Rename columns to distinguish RI and PI (RIES file already has 'ries_score', PIES has 'pies_score')
    # No renaming needed - they already have correct column names

    # Merge on model_id
    # Select available columns (not all columns exist in both files)
    ries_cols = ['model_id', 'model_name', 'ries_score']
    if 'model_size_b' in ries_df.columns:
        ries_cols.append('model_size_b')

    merged = pd.merge(
        ries_df[ries_cols],
        pies_df[['model_id', 'pies_score']],
        on='model_id',
        how='inner',
        suffixes=('_ri', '_pi')
    )

    # Rename for consistency
    if 'model_size_b' in merged.columns:
        merged = merged.rename(columns={'model_size_b': 'size_billions'})

    print(f"✓ Matched {len(merged)} common models")

    # Calculate difference (asymmetry)
    merged['ri_pi_diff'] = merged['ries_score'] - merged['pies_score']
    merged['ri_pi_ratio'] = merged['ries_score'] / merged['pies_score']

    return merged


def correlation_analysis(merged: pd.DataFrame) -> Dict:
    """
    Perform correlation analysis between RIES and PIES.

    Args:
        merged: DataFrame with both RIES and PIES scores

    Returns:
        Dictionary of correlation statistics
    """
    print(f"\n📈 Correlation Analysis: RIES vs PIES")
    print(f"=" * 70)

    # Pearson correlation
    r_pearson, p_pearson = stats.pearsonr(merged['ries_score'], merged['pies_score'])

    # Spearman correlation (rank-based, robust to outliers)
    r_spearman, p_spearman = stats.spearmanr(merged['ries_score'], merged['pies_score'])

    # R-squared
    r_squared = r_pearson ** 2

    print(f"\n✓ Pearson Correlation:  r = {r_pearson:+.3f}, p = {p_pearson:.4f}, R² = {r_squared:.3f}")
    print(f"✓ Spearman Correlation: r = {r_spearman:+.3f}, p = {p_spearman:.4f}")

    # Interpretation
    print(f"\n💡 Interpretation:")
    if r_squared > 0.7:
        print(f"   Strong correlation (R² = {r_squared:.3f}) - RI and PI measure similar construct")
    elif r_squared > 0.4:
        print(f"   Moderate correlation (R² = {r_squared:.3f}) - RI and PI partially related")
    else:
        print(f"   Weak correlation (R² = {r_squared:.3f}) - RI and PI measure different constructs")

    if p_pearson < 0.001:
        print(f"   Highly significant (p < 0.001) ***")
    elif p_pearson < 0.01:
        print(f"   Very significant (p < 0.01) **")
    elif p_pearson < 0.05:
        print(f"   Significant (p < 0.05) *")
    else:
        print(f"   Not significant (p = {p_pearson:.3f})")

    return {
        'r_pearson': r_pearson,
        'p_pearson': p_pearson,
        'r_squared': r_squared,
        'r_spearman': r_spearman,
        'p_spearman': p_spearman
    }


def asymmetry_test(merged: pd.DataFrame) -> Dict:
    """
    Test for asymmetry between RI and PI (RIES ≠ PIES).

    Args:
        merged: DataFrame with both RIES and PIES scores

    Returns:
        Dictionary of asymmetry statistics
    """
    print(f"\n⚖️  Asymmetry Test: RIES vs PIES")
    print(f"=" * 70)

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(merged['ries_score'], merged['pies_score'])

    # Effect size (Cohen's d for paired samples)
    diff = merged['ries_score'] - merged['pies_score']
    mean_diff = diff.mean()
    std_diff = diff.std()
    cohens_d = mean_diff / std_diff

    # Summary statistics
    ries_mean = merged['ries_score'].mean()
    pies_mean = merged['pies_score'].mean()
    ries_std = merged['ries_score'].std()
    pies_std = merged['pies_score'].std()

    print(f"\n📊 Descriptive Statistics:")
    print(f"   RI (RIES):  Mean = {ries_mean:.1f} ± {ries_std:.1f}")
    print(f"   PI (PIES):  Mean = {pies_mean:.1f} ± {pies_std:.1f}")
    print(f"   Difference: Mean = {mean_diff:+.1f} ± {std_diff:.1f}")

    print(f"\n🔬 Paired t-test:")
    print(f"   t({len(merged)-1}) = {t_stat:+.3f}, p = {p_value:.4f}")
    print(f"   Cohen's d = {cohens_d:+.3f}")

    # Effect size interpretation
    print(f"\n💡 Interpretation:")
    if abs(cohens_d) < 0.2:
        effect = "negligible"
    elif abs(cohens_d) < 0.5:
        effect = "small"
    elif abs(cohens_d) < 0.8:
        effect = "medium"
    else:
        effect = "large"

    if mean_diff > 0:
        direction = "RI > PI (retroactive interference HARDER to resist)"
    elif mean_diff < 0:
        direction = "PI > RI (proactive interference HARDER to resist)"
    else:
        direction = "RI = PI (symmetric)"

    print(f"   Effect size: {effect} ({abs(cohens_d):.3f})")
    print(f"   Direction: {direction}")

    if p_value < 0.001:
        print(f"   Significance: Highly significant (p < 0.001) ***")
    elif p_value < 0.01:
        print(f"   Significance: Very significant (p < 0.01) **")
    elif p_value < 0.05:
        print(f"   Significance: Significant (p < 0.05) *")
    else:
        print(f"   Significance: Not significant (p = {p_value:.3f})")

    # Count winners
    ri_better = (merged['ries_score'] > merged['pies_score']).sum()
    pi_better = (merged['pies_score'] > merged['ries_score']).sum()
    tied = (merged['ries_score'] == merged['pies_score']).sum()

    print(f"\n📊 Model-by-Model Comparison:")
    print(f"   RI better than PI: {ri_better}/{len(merged)} ({100*ri_better/len(merged):.1f}%)")
    print(f"   PI better than RI: {pi_better}/{len(merged)} ({100*pi_better/len(merged):.1f}%)")
    print(f"   Tied:              {tied}/{len(merged)} ({100*tied/len(merged):.1f}%)")

    return {
        't_stat': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'ries_mean': ries_mean,
        'pies_mean': pies_mean,
        'ri_better_count': ri_better,
        'pi_better_count': pi_better
    }


def generate_scatter_plot(merged: pd.DataFrame, output_dir: Path, stats: Dict):
    """
    Generate RIES vs PIES scatter plot.

    Args:
        merged: DataFrame with both scores
        output_dir: Output directory for plot
        stats: Correlation statistics
    """
    print(f"\n📊 Generating scatter plot: RIES vs PIES...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by model family
    merged['family'] = merged['model_id'].apply(lambda x: x.split('-')[0] if '-' in x else x.split('_')[0] if '_' in x else 'other')
    families = merged['family'].unique()
    colors = sns.color_palette('husl', len(families))
    family_colors = dict(zip(families, colors))

    # Scatter plot with family colors
    for family in families:
        family_data = merged[merged['family'] == family]
        ax.scatter(
            family_data['ries_score'],
            family_data['pies_score'],
            label=family.capitalize(),
            alpha=0.7,
            s=100,
            color=family_colors[family],
            edgecolors='black',
            linewidths=0.5
        )

    # Perfect correlation line (y=x)
    min_val = min(merged['ries_score'].min(), merged['pies_score'].min())
    max_val = max(merged['ries_score'].max(), merged['pies_score'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Perfect correlation (y=x)')

    # Regression line
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(merged['ries_score'], merged['pies_score'])
    x_line = np.array([min_val, max_val])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', alpha=0.5, linewidth=2, label=f'Regression (r={stats["r_pearson"]:.3f})')

    # Labels and title
    ax.set_xlabel('RIES (Retroactive Interference Score)', fontsize=12, fontweight='bold')
    ax.set_ylabel('PIES (Proactive Interference Endurance Score)', fontsize=12, fontweight='bold')
    ax.set_title(f'RI vs PI Correlation Across {len(merged)} Models\nR² = {stats["r_squared"]:.3f}, p = {stats["p_pearson"]:.4f}',
                 fontsize=14, fontweight='bold')

    # Add statistics box
    textstr = f'Pearson r = {stats["r_pearson"]:+.3f}\nR² = {stats["r_squared"]:.3f}\np = {stats["p_pearson"]:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Save
    output_file = output_dir / 'ries_vs_pies_scatter.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to {output_file}")
    plt.close()


def generate_bland_altman_plot(merged: pd.DataFrame, output_dir: Path, stats: Dict):
    """
    Generate Bland-Altman agreement plot.

    Args:
        merged: DataFrame with both scores
        output_dir: Output directory for plot
        stats: Asymmetry statistics
    """
    print(f"\n📊 Generating Bland-Altman plot...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate mean and difference
    mean_score = (merged['ries_score'] + merged['pies_score']) / 2
    diff_score = merged['ries_score'] - merged['pies_score']

    # Scatter plot
    ax.scatter(mean_score, diff_score, alpha=0.6, s=80, edgecolors='black', linewidths=0.5)

    # Mean difference line
    mean_diff = diff_score.mean()
    ax.axhline(mean_diff, color='red', linestyle='-', linewidth=2, label=f'Mean difference = {mean_diff:+.1f}')

    # Limits of agreement (±1.96 SD)
    std_diff = diff_score.std()
    upper_loa = mean_diff + 1.96 * std_diff
    lower_loa = mean_diff - 1.96 * std_diff
    ax.axhline(upper_loa, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'±1.96 SD ({upper_loa:+.1f}, {lower_loa:+.1f})')
    ax.axhline(lower_loa, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    # Zero line
    ax.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.5, label='Perfect agreement (0)')

    # Labels and title
    ax.set_xlabel('Mean of RIES and PIES', fontsize=12, fontweight='bold')
    ax.set_ylabel('Difference (RIES - PIES)', fontsize=12, fontweight='bold')
    ax.set_title(f'Bland-Altman Agreement Plot: RI vs PI\nMean Difference = {mean_diff:+.1f} ± {std_diff:.1f}',
                 fontsize=14, fontweight='bold')

    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Save
    output_file = output_dir / 'bland_altman_plot.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to {output_file}")
    plt.close()


def generate_comparison_table(merged: pd.DataFrame, output_dir: Path):
    """
    Generate side-by-side comparison table.

    Args:
        merged: DataFrame with both scores
        output_dir: Output directory
    """
    print(f"\n📋 Generating comparison table...")

    # Sort by RIES score
    cols_to_include = ['model_id', 'model_name', 'ries_score', 'pies_score', 'ri_pi_diff', 'ri_pi_ratio']
    if 'size_billions' in merged.columns:
        cols_to_include.insert(2, 'size_billions')

    comparison = merged[cols_to_include].copy()
    comparison = comparison.sort_values('ries_score', ascending=False)

    # Add rankings
    comparison['ries_rank'] = comparison['ries_score'].rank(ascending=False, method='min').astype(int)
    comparison['pies_rank'] = comparison['pies_score'].rank(ascending=False, method='min').astype(int)
    comparison['rank_diff'] = comparison['ries_rank'] - comparison['pies_rank']

    # Format for display
    if 'size_billions' in comparison.columns:
        comparison['size_billions'] = comparison['size_billions'].apply(lambda x: f"{x:.1f}B" if pd.notna(x) else "Unknown")
    comparison['ries_score'] = comparison['ries_score'].apply(lambda x: f"{x:.1f}")
    comparison['pies_score'] = comparison['pies_score'].apply(lambda x: f"{x:.1f}")
    comparison['ri_pi_diff'] = comparison['ri_pi_diff'].apply(lambda x: f"{x:+.1f}")
    comparison['ri_pi_ratio'] = comparison['ri_pi_ratio'].apply(lambda x: f"{x:.3f}")

    # Save to Excel
    output_file = output_dir / 'ri_vs_pi_comparison_table.xlsx'
    comparison.to_excel(output_file, index=False, sheet_name='RI vs PI Comparison')
    print(f"   ✓ Saved to {output_file}")

    # Print top 10
    print(f"\n🏆 Top 10 Models by RIES (with PI comparison):")
    if 'size_billions' in comparison.columns:
        print(f"{'Rank':<6} {'Model':<30} {'Size':<10} {'RIES':<8} {'PIES':<8} {'Diff':<8} {'Ratio':<8}")
        print("=" * 85)
        for idx, row in comparison.head(10).iterrows():
            print(f"{row['ries_rank']:<6} {row['model_id']:<30} {row['size_billions']:<10} "
                  f"{row['ries_score']:<8} {row['pies_score']:<8} {row['ri_pi_diff']:<8} {row['ri_pi_ratio']:<8}")
    else:
        print(f"{'Rank':<6} {'Model':<30} {'RIES':<8} {'PIES':<8} {'Diff':<8} {'Ratio':<8}")
        print("=" * 75)
        for idx, row in comparison.head(10).iterrows():
            print(f"{row['ries_rank']:<6} {row['model_id']:<30} "
                  f"{row['ries_score']:<8} {row['pies_score']:<8} {row['ri_pi_diff']:<8} {row['ri_pi_ratio']:<8}")


def main():
    parser = argparse.ArgumentParser(description='Compare RI vs PI interference patterns')
    parser.add_argument('--ries', type=str, default='results/key_results/ries_analysis.xlsx',
                       help='RIES analysis file (default: results/key_results/ries_analysis.xlsx)')
    parser.add_argument('--pies', type=str, default='results/key_results/pies_analysis.xlsx',
                       help='PIES analysis file (default: results/key_results/pies_analysis.xlsx)')
    parser.add_argument('--output', type=str, default='results/ri_vs_pi_comparison/',
                       help='Output directory (default: results/ri_vs_pi_comparison/)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RI vs PI COMPARISON ANALYSIS")
    print("=" * 70)

    # Load data
    ries_df, pies_df = load_data(args.ries, args.pies)

    # Merge datasets
    merged = merge_datasets(ries_df, pies_df)

    # Correlation analysis
    corr_stats = correlation_analysis(merged)

    # Asymmetry test
    asym_stats = asymmetry_test(merged)

    # Generate visualizations
    generate_scatter_plot(merged, output_dir, corr_stats)
    generate_bland_altman_plot(merged, output_dir, asym_stats)
    generate_comparison_table(merged, output_dir)

    # Save merged data
    merged_file = output_dir / 'merged_ri_pi_data.xlsx'
    merged.to_excel(merged_file, index=False, sheet_name='Merged Data')
    print(f"\n💾 Saved merged data to {merged_file}")

    # Summary
    print(f"\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n📁 Output files saved to: {output_dir}")
    print(f"   - ries_vs_pies_scatter.png")
    print(f"   - bland_altman_plot.png")
    print(f"   - ri_vs_pi_comparison_table.xlsx")
    print(f"   - merged_ri_pi_data.xlsx")

    print(f"\n🔑 Key Findings:")
    print(f"   • Correlation: R² = {corr_stats['r_squared']:.3f}, p = {corr_stats['p_pearson']:.4f}")
    print(f"   • Asymmetry: Mean diff = {asym_stats['mean_diff']:+.1f}, Cohen's d = {asym_stats['cohens_d']:+.3f}, p = {asym_stats['p_value']:.4f}")
    print(f"   • Models where RI > PI: {asym_stats['ri_better_count']}/{len(merged)}")
    print(f"   • Models where PI > RI: {asym_stats['pi_better_count']}/{len(merged)}")


if __name__ == "__main__":
    main()
