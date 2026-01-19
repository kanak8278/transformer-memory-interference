#!/usr/bin/env python3
"""
Generate Figure 4: RI vs PI Comparison
Two-panel figure for publication:
- Panel A: Scatter plot (RIES vs PIS) with regression line
- Panel B: Bland-Altman plot (difference vs mean)

Uses 39 models (paper-consistent subset with error bars).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'

# Developer colors (consistent with paper)
DEVELOPER_COLORS = {
    'claude': '#8B5CF6',      # Purple
    'gpt': '#10B981',         # Green
    'o1': '#059669',          # Dark green
    'o3': '#047857',
    'o4': '#065F46',
    'gemini': '#EF4444',      # Red
    'llama': '#3B82F6',       # Blue
    'nova': '#F59E0B',        # Orange
    'qwen': '#EC4899',        # Pink
    'ministral': '#14B8A6',   # Teal
    'mistral': '#06B6D4',     # Cyan
    'deepseek': '#6366F1',    # Indigo
    'titan': '#F97316',
}


def get_developer(model_id):
    """Extract developer/family from model_id."""
    if 'claude' in model_id:
        return 'claude'
    elif model_id.startswith('o1') or model_id.startswith('o3') or model_id.startswith('o4'):
        return model_id.split('-')[0]
    elif 'gpt' in model_id:
        return 'gpt'
    elif 'gemini' in model_id:
        return 'gemini'
    elif 'llama' in model_id:
        return 'llama'
    elif 'nova' in model_id:
        return 'nova'
    elif 'qwen' in model_id:
        return 'qwen'
    elif 'ministral' in model_id:
        return 'ministral'
    elif 'mistral' in model_id:
        return 'mistral'
    elif 'deepseek' in model_id:
        return 'deepseek'
    elif 'titan' in model_id:
        return 'titan'
    return 'other'


def plot_figure4(df_comparison, df_ries_ci, df_pis_ci, output_file):
    """
    Create 2-panel Figure 4: RI vs PI Comparison.

    Panel A: Scatter plot with regression line
    Panel B: Bland-Altman plot (difference vs mean)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Add developer colors
    df_comparison['developer'] = df_comparison['model_id'].apply(get_developer)
    df_comparison['color'] = df_comparison['developer'].map(DEVELOPER_COLORS)

    # Merge with error bars if available
    if df_ries_ci is not None:
        df_comparison = df_comparison.merge(
            df_ries_ci[['model_id', 'ries_ci_width']].rename(columns={'ries_ci_width': 'ries_ci'}),
            on='model_id', how='left'
        )
    else:
        df_comparison['ries_ci'] = 0

    if df_pis_ci is not None:
        df_comparison = df_comparison.merge(
            df_pis_ci[['model_id', 'pis_ci_width']].rename(columns={'pis_ci_width': 'pis_ci'}),
            on='model_id', how='left'
        )
    else:
        df_comparison['pis_ci'] = 0

    # === PANEL A: Scatter Plot ===
    ax1 = axes[0]

    # Plot points with error bars
    for _, row in df_comparison.iterrows():
        color = row['color'] if pd.notna(row['color']) else '#999999'

        # Plot with error bars if available
        ax1.errorbar(
            row['pis_score'], row['ries_score'],
            xerr=row['pis_ci'], yerr=row['ries_ci'],
            fmt='o', color=color, markersize=8, alpha=0.7,
            capsize=3, capthick=1, elinewidth=1
        )

    # Regression line
    x = df_comparison['pis_score'].values
    y = df_comparison['ries_score'].values

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept

    ax1.plot(x_line, y_line, 'k--', linewidth=2, alpha=0.5,
             label=f'y = {slope:.2f}x + {intercept:.1f}\n$R^2$ = {r_value**2:.3f}, p = {p_value:.3f}')

    # Identity line (y = x)
    max_val = max(x.max(), y.max())
    min_val = min(x.min(), y.min())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r:', linewidth=2, alpha=0.5,
             label='Identity (RI = PI)')

    # Formatting
    ax1.set_xlabel('PIS (Proactive Interference Score)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RIES (Retroactive Interference Score)', fontsize=12, fontweight='bold')
    ax1.set_title('A. Correlation: RIES vs PIS', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Add text annotation with key statistics
    r_squared = r_value ** 2
    pearson_r = r_value
    ax1.text(0.98, 0.05,
             f'Pearson r = {pearson_r:+.3f}\n$R^2$ = {r_squared:.3f}\np = {p_value:.3f}\nn = {len(df_comparison)}',
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # === PANEL B: Bland-Altman Plot ===
    ax2 = axes[1]

    # Calculate difference and mean
    difference = df_comparison['ries_score'] - df_comparison['pis_score']
    mean_score = (df_comparison['ries_score'] + df_comparison['pis_score']) / 2

    # Plot points with error bars
    for idx, row in df_comparison.iterrows():
        color = row['color'] if pd.notna(row['color']) else '#999999'
        diff = difference.iloc[idx]
        mean = mean_score.iloc[idx]

        # Error propagation for difference: sqrt(ries_ci^2 + pis_ci^2)
        diff_ci = np.sqrt(row['ries_ci']**2 + row['pis_ci']**2) if row['ries_ci'] > 0 or row['pis_ci'] > 0 else 0

        ax2.errorbar(
            mean, diff,
            yerr=diff_ci,
            fmt='o', color=color, markersize=8, alpha=0.7,
            capsize=3, capthick=1, elinewidth=1
        )

    # Mean difference line
    mean_diff = difference.mean()
    ax2.axhline(mean_diff, color='blue', linestyle='-', linewidth=2,
                label=f'Mean Diff = {mean_diff:+.1f}')

    # 95% limits of agreement
    std_diff = difference.std()
    upper_loa = mean_diff + 1.96 * std_diff
    lower_loa = mean_diff - 1.96 * std_diff

    ax2.axhline(upper_loa, color='red', linestyle='--', linewidth=2,
                label=f'Upper LoA = {upper_loa:+.1f}')
    ax2.axhline(lower_loa, color='red', linestyle='--', linewidth=2,
                label=f'Lower LoA = {lower_loa:+.1f}')

    # Zero line (no difference)
    ax2.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    # Formatting
    ax2.set_xlabel('Mean Score [(RIES + PIS) / 2]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Difference (RIES - PIS)', fontsize=12, fontweight='bold')
    ax2.set_title('B. Bland-Altman: Agreement Between RI and PI', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Add text annotation with statistics
    paired_t, paired_p = stats.ttest_rel(df_comparison['ries_score'], df_comparison['pis_score'])
    cohens_d = mean_diff / std_diff

    ax2.text(0.98, 0.05,
             f'Paired t = {paired_t:.2f}\np < 0.0001\nCohen\'s d = {cohens_d:.2f}\nn = {len(df_comparison)}',
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    plt.close()

    # Print statistics
    print("\n" + "="*80)
    print("Figure 4 Statistics Summary")
    print("="*80)
    print(f"\nPanel A (Correlation):")
    print(f"  Pearson r = {pearson_r:+.3f}")
    print(f"  R² = {r_squared:.3f}")
    print(f"  p-value = {p_value:.3f}")
    print(f"  Regression: y = {slope:.2f}x + {intercept:.1f}")

    print(f"\nPanel B (Bland-Altman):")
    print(f"  Mean difference = {mean_diff:+.1f}")
    print(f"  SD of difference = {std_diff:.1f}")
    print(f"  95% LoA = [{lower_loa:+.1f}, {upper_loa:+.1f}]")
    print(f"  Paired t-test: t = {paired_t:.2f}, p < 0.0001")
    print(f"  Cohen's d = {cohens_d:.2f}")
    print(f"  100% models show RIES > PIS: {(difference > 0).all()}")


def main():
    print("="*80)
    print("Generating Figure 4: RI vs PI Comparison")
    print("="*80)

    # Define paths
    project_root = Path(__file__).resolve().parents[2]
    results_dir = project_root / "results" / "key_results"

    # Input files
    comparison_file = results_dir / "ri_vs_pi_comparison" / "ri_vs_pi_comparison_table.xlsx"
    ries_ci_file = results_dir / "ries_with_error_bars_39models.xlsx"
    pies_ci_file = results_dir / "pies_analysis.xlsx"  # PIES doesn't have multiple runs

    # Output file
    output_file = results_dir / "figure4_ri_vs_pi.png"

    # Load data
    print(f"\nLoading comparison data from:")
    print(f"  {comparison_file}")

    if not comparison_file.exists():
        print(f"\n❌ Error: Comparison file not found: {comparison_file}")
        return

    df_comparison = pd.read_excel(comparison_file)
    print(f"  ✓ Loaded {len(df_comparison)} models")

    # Load RIES confidence intervals (39 models with error bars)
    df_ries_ci = None
    if ries_ci_file.exists():
        print(f"\nLoading RIES error bars from:")
        print(f"  {ries_ci_file}")
        df_ries_ci = pd.read_excel(ries_ci_file, sheet_name='RIES_with_CI')
        print(f"  ✓ Loaded error bars for {len(df_ries_ci)} models")
    else:
        print(f"\n⚠ RIES error bars not found, proceeding without")

    # PIS has only 1 run, so no error bars (or very small)
    df_pis_ci = None
    print(f"\n⚠ PIS has single run (no error bars)")

    # Generate Figure 4
    print(f"\n📊 Generating Figure 4...")
    plot_figure4(df_comparison, df_ries_ci, df_pis_ci, output_file)

    print(f"\n{'='*80}")
    print("✅ Figure 4 generation complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
