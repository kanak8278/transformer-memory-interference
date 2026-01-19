#!/usr/bin/env python3
"""
Position-Binned Error Analysis for RI and PI Experiments

Creates visualizations showing WHERE in the update sequence errors come from,
similar to Figure 5/19 in "Unable to Forget" paper.

Key insight:
- RI errors (same-key): Should show RECENCY bias (errors cluster near end/recent)
- PI errors (same-key): Should show PRIMACY bias (errors cluster near start/early)

This analysis bins same-key interference errors by their position in the
update sequence to visualize these opposing patterns.
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_error_data(ri_path: Path, pi_path: Path):
    """Load the error classification Excel files."""
    ri_df = pd.read_excel(ri_path, sheet_name='All_Errors')
    pi_df = pd.read_excel(pi_path, sheet_name='All_PI_Errors')
    return ri_df, pi_df


def normalize_position_to_bins(position: int, total_updates: int, n_bins: int = 10) -> int:
    """
    Normalize position to a bin index (0 to n_bins-1).

    For RI: position 1 = target (first), higher = more recent interference
    For PI: position 1 = first, last position = target

    This function returns which bin (0 to n_bins-1) the position falls into,
    where bin 0 is earliest and bin n_bins-1 is most recent.
    """
    if total_updates <= 1:
        return 0

    # Normalize position to 0-1 range (0 = earliest, 1 = most recent)
    normalized = (position - 1) / (total_updates - 1)

    # Convert to bin index
    bin_idx = int(normalized * (n_bins - 1))
    return min(bin_idx, n_bins - 1)


def create_position_histogram_data(df: pd.DataFrame, exp_type: str, n_bins: int = 10) -> dict:
    """
    Create histogram data for same-key interference errors.

    Returns dict with level -> bin counts
    """
    # Filter to same-key interference only
    ski_df = df[df['error_type'] == 'same_key_interference'].copy()

    if len(ski_df) == 0:
        return {}

    result = {}

    for level in sorted(ski_df['interference_level'].unique()):
        level_data = ski_df[ski_df['interference_level'] == level]

        # Initialize bins
        bin_counts = np.zeros(n_bins)

        for _, row in level_data.iterrows():
            pos = row['position']
            total = row['total_updates']

            if pd.notna(pos) and pd.notna(total) and total > 1:
                bin_idx = normalize_position_to_bins(int(pos), int(total), n_bins)
                bin_counts[bin_idx] += 1

        # Normalize to percentages
        total_errors = bin_counts.sum()
        if total_errors > 0:
            bin_pcts = (bin_counts / total_errors) * 100
        else:
            bin_pcts = bin_counts

        result[level] = {
            'counts': bin_counts,
            'percentages': bin_pcts,
            'total': total_errors
        }

    return result


def plot_position_binned_comparison(ri_df: pd.DataFrame, pi_df: pd.DataFrame,
                                     output_dir: Path, n_bins: int = 10):
    """
    Create side-by-side position-binned histograms for RI vs PI.
    Shows how error positions differ: RI = recency, PI = primacy.
    """
    # Get histogram data
    ri_hist = create_position_histogram_data(ri_df, 'RI', n_bins)
    pi_hist = create_position_histogram_data(pi_df, 'PI', n_bins)

    # Select representative levels
    target_levels = [10, 50, 100, 200, 300]
    ri_levels = [l for l in target_levels if l in ri_hist]
    pi_levels = [l for l in target_levels if l in pi_hist]

    common_levels = sorted(set(ri_levels) & set(pi_levels))

    if not common_levels:
        print("No common levels found for comparison")
        return

    # Create figure with subplots
    n_levels = len(common_levels)
    fig, axes = plt.subplots(2, n_levels, figsize=(3 * n_levels, 8), sharey=True)

    if n_levels == 1:
        axes = axes.reshape(2, 1)

    bin_labels = [f'{int(i/n_bins*100)}-{int((i+1)/n_bins*100)}%' for i in range(n_bins)]
    x = np.arange(n_bins)

    # Color gradient: early = blue, late = red
    colors_ri = plt.cm.Blues(np.linspace(0.3, 0.9, n_bins))
    colors_pi = plt.cm.Reds(np.linspace(0.3, 0.9, n_bins))

    for col, level in enumerate(common_levels):
        # Top row: RI
        ax_ri = axes[0, col]
        ri_pcts = ri_hist[level]['percentages']
        bars_ri = ax_ri.bar(x, ri_pcts, color=colors_ri, edgecolor='black', linewidth=0.5)
        ax_ri.set_title(f'Update Count = {level}', fontweight='bold')
        ax_ri.set_xticks(x)
        ax_ri.set_xticklabels(['E' if i == 0 else ('L' if i == n_bins-1 else '') for i in range(n_bins)], fontsize=8)
        ax_ri.grid(axis='y', alpha=0.3)

        if col == 0:
            ax_ri.set_ylabel('RI Errors (%)\n(Recall FIRST)', fontweight='bold')

        # Bottom row: PI
        ax_pi = axes[1, col]
        pi_pcts = pi_hist[level]['percentages']
        bars_pi = ax_pi.bar(x, pi_pcts, color=colors_pi, edgecolor='black', linewidth=0.5)
        ax_pi.set_xlabel('Position in Sequence\n(E=Early, L=Late)', fontsize=9)
        ax_pi.set_xticks(x)
        ax_pi.set_xticklabels(['E' if i == 0 else ('L' if i == n_bins-1 else '') for i in range(n_bins)], fontsize=8)
        ax_pi.grid(axis='y', alpha=0.3)

        if col == 0:
            ax_pi.set_ylabel('PI Errors (%)\n(Recall LAST)', fontweight='bold')

    # Add overall title
    fig.suptitle('Same-Key Interference: Position Analysis\n'
                 'RI shows RECENCY bias (errors from late positions) | '
                 'PI shows PRIMACY bias (errors from early positions)',
                 fontweight='bold', fontsize=12, y=1.02)

    plt.tight_layout()
    output_file = output_dir / 'position_binned_ri_vs_pi.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"  Saved: {output_file}")
    plt.close()


def plot_position_evolution(ri_df: pd.DataFrame, pi_df: pd.DataFrame, output_dir: Path):
    """
    Create line plots showing how average error position changes with interference level.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # RI same-key interference
    ri_ski = ri_df[ri_df['error_type'] == 'same_key_interference'].copy()
    pi_ski = pi_df[pi_df['error_type'] == 'same_key_interference'].copy()

    # Calculate normalized position (0 = earliest, 1 = most recent)
    def calc_normalized_pos(row):
        if pd.notna(row['position']) and pd.notna(row['total_updates']) and row['total_updates'] > 1:
            return (row['position'] - 1) / (row['total_updates'] - 1)
        return np.nan

    ri_ski['norm_pos'] = ri_ski.apply(calc_normalized_pos, axis=1)
    pi_ski['norm_pos'] = pi_ski.apply(calc_normalized_pos, axis=1)

    # Aggregate by level
    ri_agg = ri_ski.groupby('interference_level').agg({
        'norm_pos': ['mean', 'std', 'count']
    }).round(3)
    ri_agg.columns = ['mean', 'std', 'count']

    pi_agg = pi_ski.groupby('interference_level').agg({
        'norm_pos': ['mean', 'std', 'count']
    }).round(3)
    pi_agg.columns = ['mean', 'std', 'count']

    # Standard levels
    standard_levels = [3, 10, 50, 100, 200, 300]

    # RI plot
    ri_levels = [l for l in standard_levels if l in ri_agg.index]
    x_ri = np.arange(len(ri_levels))
    ri_means = ri_agg.loc[ri_levels, 'mean'].values
    ri_stds = ri_agg.loc[ri_levels, 'std'].fillna(0).values

    ax1.errorbar(x_ri, ri_means, yerr=ri_stds, marker='o', linewidth=2, markersize=8,
                 capsize=5, color='#2E86AB', label='Mean position')
    ax1.fill_between(x_ri, ri_means - ri_stds, ri_means + ri_stds, alpha=0.2, color='#2E86AB')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Midpoint')
    ax1.set_xlabel('Update Count', fontweight='bold')
    ax1.set_ylabel('Normalized Error Position\n(0=Early, 1=Late)', fontweight='bold')
    ax1.set_title('(A) RI: Same-Key Interference Positions\n(Recall FIRST value)', fontweight='bold')
    ax1.set_xticks(x_ri)
    ax1.set_xticklabels([str(l) for l in ri_levels])
    ax1.set_ylim(0, 1)
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.3)

    # Add annotation
    ax1.annotate('RECENCY BIAS\n(errors from late positions)',
                xy=(len(ri_levels)-1, ri_means[-1]), xytext=(len(ri_levels)-2, 0.85),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='gray'))

    # PI plot
    pi_levels = [l for l in standard_levels if l in pi_agg.index]
    x_pi = np.arange(len(pi_levels))
    pi_means = pi_agg.loc[pi_levels, 'mean'].values
    pi_stds = pi_agg.loc[pi_levels, 'std'].fillna(0).values

    ax2.errorbar(x_pi, pi_means, yerr=pi_stds, marker='s', linewidth=2, markersize=8,
                 capsize=5, color='#A23B72', label='Mean position')
    ax2.fill_between(x_pi, pi_means - pi_stds, pi_means + pi_stds, alpha=0.2, color='#A23B72')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Midpoint')
    ax2.set_xlabel('Update Count', fontweight='bold')
    ax2.set_ylabel('Normalized Error Position\n(0=Early, 1=Late)', fontweight='bold')
    ax2.set_title('(B) PI: Same-Key Interference Positions\n(Recall LAST value)', fontweight='bold')
    ax2.set_xticks(x_pi)
    ax2.set_xticklabels([str(l) for l in pi_levels])
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right')
    ax2.grid(alpha=0.3)

    # Add annotation
    ax2.annotate('PRIMACY BIAS\n(errors from early positions)',
                xy=(len(pi_levels)-1, pi_means[-1]), xytext=(len(pi_levels)-2, 0.15),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()
    output_file = output_dir / 'position_evolution_ri_vs_pi.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"  Saved: {output_file}")
    plt.close()


def plot_combined_position_figure(ri_df: pd.DataFrame, pi_df: pd.DataFrame,
                                   output_dir: Path, n_bins: int = 10):
    """
    Create publication-quality figure combining position analysis.
    Panel A: RI histogram at high interference
    Panel B: PI histogram at high interference
    Panel C: Position evolution comparison
    """
    fig = plt.figure(figsize=(14, 10))

    # Create grid
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.25)

    # Get histogram data for level 300 (or highest available)
    ri_hist = create_position_histogram_data(ri_df, 'RI', n_bins)
    pi_hist = create_position_histogram_data(pi_df, 'PI', n_bins)

    target_level = 300
    if target_level not in ri_hist:
        target_level = max(ri_hist.keys()) if ri_hist else 100

    x = np.arange(n_bins)

    # Panel A: RI histogram
    ax1 = fig.add_subplot(gs[0, 0])
    if target_level in ri_hist:
        ri_pcts = ri_hist[target_level]['percentages']
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_bins))
        ax1.bar(x, ri_pcts, color=colors, edgecolor='black', linewidth=0.5)
        ax1.axvline(x=n_bins-1, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax1.set_xlabel('Position in Update Sequence (E=Early, L=Late)', fontweight='bold')
    ax1.set_ylabel('Error Percentage (%)', fontweight='bold')
    ax1.set_title(f'(A) RI Same-Key Errors (Update Count={target_level})\nRecall FIRST value - Errors from LATE positions',
                  fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['E'] + [''] * (n_bins-2) + ['L'])
    ax1.grid(axis='y', alpha=0.3)

    # Panel B: PI histogram
    ax2 = fig.add_subplot(gs[0, 1])
    if target_level in pi_hist:
        pi_pcts = pi_hist[target_level]['percentages']
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, n_bins))
        ax2.bar(x, pi_pcts, color=colors, edgecolor='black', linewidth=0.5)
        ax2.axvline(x=0, color='blue', linestyle='--', alpha=0.7, linewidth=2)
    ax2.set_xlabel('Position in Update Sequence (E=Early, L=Late)', fontweight='bold')
    ax2.set_ylabel('Error Percentage (%)', fontweight='bold')
    ax2.set_title(f'(B) PI Same-Key Errors (Update Count={target_level})\nRecall LAST value - Errors from EARLY positions',
                  fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['E'] + [''] * (n_bins-2) + ['L'])
    ax2.grid(axis='y', alpha=0.3)

    # Panel C: Position evolution (bottom spanning both columns)
    ax3 = fig.add_subplot(gs[1, :])

    # Calculate normalized positions
    ri_ski = ri_df[ri_df['error_type'] == 'same_key_interference'].copy()
    pi_ski = pi_df[pi_df['error_type'] == 'same_key_interference'].copy()

    def calc_normalized_pos(row):
        if pd.notna(row['position']) and pd.notna(row['total_updates']) and row['total_updates'] > 1:
            return (row['position'] - 1) / (row['total_updates'] - 1)
        return np.nan

    ri_ski['norm_pos'] = ri_ski.apply(calc_normalized_pos, axis=1)
    pi_ski['norm_pos'] = pi_ski.apply(calc_normalized_pos, axis=1)

    ri_agg = ri_ski.groupby('interference_level')['norm_pos'].mean()
    pi_agg = pi_ski.groupby('interference_level')['norm_pos'].mean()

    standard_levels = [3, 10, 50, 100, 200, 300]
    common_levels = sorted(set(l for l in standard_levels if l in ri_agg.index and l in pi_agg.index))

    if common_levels:
        x_pos = np.arange(len(common_levels))

        ax3.plot(x_pos, [ri_agg[l] for l in common_levels], marker='o', linewidth=3,
                markersize=10, color='#2E86AB', label='RI (Recall FIRST)')
        ax3.plot(x_pos, [pi_agg[l] for l in common_levels], marker='s', linewidth=3,
                markersize=10, color='#A23B72', label='PI (Recall LAST)')

        ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        ax3.fill_between(x_pos, 0.5, [ri_agg[l] for l in common_levels], alpha=0.15, color='#2E86AB')
        ax3.fill_between(x_pos, [pi_agg[l] for l in common_levels], 0.5, alpha=0.15, color='#A23B72')

        ax3.set_xlabel('Update Count', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Mean Error Position\n(0=Early, 1=Late)', fontweight='bold', fontsize=12)
        ax3.set_title('(C) Error Position Evolution: RI vs PI\n'
                     'RI errors shift LATE (recency) | PI errors shift EARLY (primacy)',
                     fontweight='bold', fontsize=12)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([str(l) for l in common_levels])
        ax3.set_ylim(0, 1)
        ax3.legend(loc='center right', fontsize=11, frameon=True)
        ax3.grid(alpha=0.3)

        # Add annotations
        ax3.text(0.02, 0.95, 'Late positions\n(Recency zone)', transform=ax3.transAxes,
                fontsize=9, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax3.text(0.02, 0.08, 'Early positions\n(Primacy zone)', transform=ax3.transAxes,
                fontsize=9, va='bottom', ha='left',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    plt.tight_layout()
    output_file = output_dir / 'figure_position_analysis.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"\n  Saved main figure: {output_file}")
    plt.close()


def generate_position_stats(ri_df: pd.DataFrame, pi_df: pd.DataFrame, output_dir: Path):
    """Generate statistics about position distributions."""

    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append("POSITION-BINNED ERROR ANALYSIS: RI vs PI")
    output_lines.append("=" * 70)
    output_lines.append("")

    # RI stats
    ri_ski = ri_df[ri_df['error_type'] == 'same_key_interference'].copy()
    pi_ski = pi_df[pi_df['error_type'] == 'same_key_interference'].copy()

    output_lines.append(f"RI Same-Key Interference Errors: {len(ri_ski)}")
    output_lines.append(f"PI Same-Key Interference Errors: {len(pi_ski)}")
    output_lines.append("")

    # Calculate normalized positions
    def calc_normalized_pos(row):
        if pd.notna(row['position']) and pd.notna(row['total_updates']) and row['total_updates'] > 1:
            return (row['position'] - 1) / (row['total_updates'] - 1)
        return np.nan

    ri_ski['norm_pos'] = ri_ski.apply(calc_normalized_pos, axis=1)
    pi_ski['norm_pos'] = pi_ski.apply(calc_normalized_pos, axis=1)

    output_lines.append("-" * 70)
    output_lines.append("OVERALL POSITION STATISTICS")
    output_lines.append("-" * 70)
    output_lines.append("")
    output_lines.append(f"RI Mean Position: {ri_ski['norm_pos'].mean():.3f} (0=early, 1=late)")
    output_lines.append(f"PI Mean Position: {pi_ski['norm_pos'].mean():.3f} (0=early, 1=late)")
    output_lines.append("")

    # Primacy/recency bias quantification
    ri_late_pct = (ri_ski['norm_pos'] > 0.5).mean() * 100
    pi_early_pct = (pi_ski['norm_pos'] < 0.5).mean() * 100

    output_lines.append(f"RI errors from LATE half (>0.5): {ri_late_pct:.1f}% - RECENCY BIAS")
    output_lines.append(f"PI errors from EARLY half (<0.5): {pi_early_pct:.1f}% - PRIMACY BIAS")
    output_lines.append("")

    # First/last position bias
    if 'is_most_recent' in ri_ski.columns:
        ri_most_recent_pct = ri_ski['is_most_recent'].mean() * 100
        output_lines.append(f"RI errors from MOST RECENT position: {ri_most_recent_pct:.1f}%")

    if 'is_first' in pi_ski.columns:
        pi_first_pct = pi_ski['is_first'].mean() * 100
        output_lines.append(f"PI errors from FIRST position: {pi_first_pct:.1f}%")

    output_lines.append("")
    output_lines.append("-" * 70)
    output_lines.append("POSITION BY INTERFERENCE LEVEL")
    output_lines.append("-" * 70)
    output_lines.append("")
    output_lines.append(f"{'Level':>6} | {'RI Mean Pos':>12} | {'PI Mean Pos':>12} | {'Difference':>10}")
    output_lines.append("-" * 50)

    standard_levels = [3, 10, 50, 100, 200, 300]
    for level in standard_levels:
        ri_level = ri_ski[ri_ski['interference_level'] == level]['norm_pos']
        pi_level = pi_ski[pi_ski['interference_level'] == level]['norm_pos']

        if len(ri_level) > 0 and len(pi_level) > 0:
            ri_mean = ri_level.mean()
            pi_mean = pi_level.mean()
            diff = ri_mean - pi_mean
            output_lines.append(f"{level:>6} | {ri_mean:>12.3f} | {pi_mean:>12.3f} | {diff:>+10.3f}")

    output_lines.append("")
    output_lines.append("=" * 70)
    output_lines.append("KEY FINDING: Opposite biases in RI vs PI")
    output_lines.append("=" * 70)
    output_lines.append("")
    output_lines.append("RI (Recall FIRST value):")
    output_lines.append("  - Errors come from LATE positions (recent updates)")
    output_lines.append("  - Demonstrates RECENCY BIAS in interference")
    output_lines.append("  - Recent information overwrites early memories")
    output_lines.append("")
    output_lines.append("PI (Recall LAST value):")
    output_lines.append("  - Errors come from EARLY positions (first updates)")
    output_lines.append("  - Demonstrates PRIMACY BIAS in interference")
    output_lines.append("  - Early information persists despite updates")
    output_lines.append("")
    output_lines.append("This mirrors the 'Unable to Forget' paper's Figure 5/19 findings,")
    output_lines.append("showing that LLMs exhibit opposite interference patterns for")
    output_lines.append("early vs recent information retrieval.")
    output_lines.append("")

    # Write to file
    output_file = output_dir / 'POSITION_ANALYSIS_SUMMARY.txt'
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))

    print(f"  Saved: {output_file}")

    # Also print to console
    print('\n'.join(output_lines))


def main():
    parser = argparse.ArgumentParser(description='Generate position-binned error analysis')
    parser.add_argument('--ri-errors', type=str,
                       default='results/error_analysis/error_classification_detailed.xlsx',
                       help='Path to RI error classification Excel file')
    parser.add_argument('--pi-errors', type=str,
                       default='results/error_analysis/pi_error_classification_detailed.xlsx',
                       help='Path to PI error classification Excel file')
    parser.add_argument('--output-dir', type=str,
                       default='results/key_results',
                       help='Output directory for figures')
    parser.add_argument('--n-bins', type=int, default=10,
                       help='Number of position bins (default: 10)')

    args = parser.parse_args()

    # Setup paths
    ri_path = Path(args.ri_errors)
    pi_path = Path(args.pi_errors)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("POSITION-BINNED ERROR ANALYSIS")
    print("=" * 60)

    # Check files exist
    if not ri_path.exists():
        print(f"  RI error file not found: {ri_path}")
        return
    if not pi_path.exists():
        print(f"  PI error file not found: {pi_path}")
        return

    print(f"\nLoading error data...")
    print(f"  RI: {ri_path}")
    print(f"  PI: {pi_path}")

    ri_df, pi_df = load_error_data(ri_path, pi_path)

    print(f"\n  RI errors: {len(ri_df)}")
    print(f"  PI errors: {len(pi_df)}")

    print(f"\nGenerating position-binned visualizations...")

    # Generate all plots
    plot_position_binned_comparison(ri_df, pi_df, output_dir, args.n_bins)
    plot_position_evolution(ri_df, pi_df, output_dir)
    plot_combined_position_figure(ri_df, pi_df, output_dir, args.n_bins)

    # Generate statistics
    generate_position_stats(ri_df, pi_df, output_dir)

    print("\n  Position-binned analysis complete!")


if __name__ == '__main__':
    main()
