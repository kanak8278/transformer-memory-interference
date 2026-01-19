#!/usr/bin/env python3
"""
Generate RI vs PI Error Comparison Figures

Creates publication-quality visualizations comparing error patterns between:
- Retroactive Interference (RI): recall FIRST value
- Proactive Interference (PI): recall LAST value

Key contrast:
- RI errors show RECENCY bias (return recent values instead of first)
- PI errors show HALLUCINATION dominance (confabulate instead of recall last)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

# Publication style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['font.family'] = 'sans-serif'

# Color palette for error types
ERROR_COLORS = {
    'same_key_interference': '#E74C3C',      # Red
    'cross_key_interference': '#F39C12',     # Orange
    'partial_match': '#3498DB',              # Blue
    'hallucination': '#9B59B6',              # Purple
    'retrieval_failure': '#95A5A6'           # Gray
}

ERROR_LABELS = {
    'same_key_interference': 'Same-Key\nInterference',
    'cross_key_interference': 'Cross-Key\nInterference',
    'partial_match': 'Partial\nMatch',
    'hallucination': 'Hallucination',
    'retrieval_failure': 'Retrieval\nFailure'
}

ERROR_LABELS_SHORT = {
    'same_key_interference': 'Same-Key',
    'cross_key_interference': 'Cross-Key',
    'partial_match': 'Partial',
    'hallucination': 'Hallucin.',
    'retrieval_failure': 'Retrieval Fail'
}


def load_error_data(ri_path: Path, pi_path: Path) -> tuple:
    """Load RI and PI error classification data."""
    print(f"Loading RI errors from: {ri_path}")
    ri_all = pd.read_excel(ri_path, sheet_name='All_Errors')
    ri_pcts = pd.read_excel(ri_path, sheet_name='Percentages_By_Level', index_col=0)

    print(f"Loading PI errors from: {pi_path}")
    pi_all = pd.read_excel(pi_path, sheet_name='All_PI_Errors')
    pi_pcts = pd.read_excel(pi_path, sheet_name='Percentages_By_Level', index_col=0)

    print(f"  RI: {len(ri_all)} errors across {ri_all['model'].nunique()} models")
    print(f"  PI: {len(pi_all)} errors across {pi_all['model'].nunique()} models")

    return ri_all, ri_pcts, pi_all, pi_pcts


def plot_error_comparison_bars(ri_all: pd.DataFrame, pi_all: pd.DataFrame, output_dir: Path):
    """
    Create side-by-side bar chart comparing overall error distributions.
    This is Figure A: Overall Error Type Distribution
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate overall percentages
    ri_dist = ri_all['error_type'].value_counts(normalize=True) * 100
    pi_dist = pi_all['error_type'].value_counts(normalize=True) * 100

    error_types = ['same_key_interference', 'retrieval_failure', 'hallucination',
                   'cross_key_interference', 'partial_match']

    x = np.arange(len(error_types))
    width = 0.35

    ri_vals = [ri_dist.get(et, 0) for et in error_types]
    pi_vals = [pi_dist.get(et, 0) for et in error_types]

    bars1 = ax.bar(x - width/2, ri_vals, width, label='RI (Retroactive)',
                   color='#2E86AB', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, pi_vals, width, label='PI (Proactive)',
                   color='#A23B72', edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars1, ri_vals):
        if val > 2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    for bar, val in zip(bars2, pi_vals):
        if val > 2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_ylabel('Percentage of Errors (%)', fontweight='bold')
    ax.set_xlabel('Error Type', fontweight='bold')
    ax.set_title('Error Type Distribution: RI vs PI\n(All Models, Levels 3-300)',
                fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([ERROR_LABELS_SHORT[et] for et in error_types], rotation=0)
    ax.legend(loc='upper right', frameon=True)
    ax.set_ylim(0, max(max(ri_vals), max(pi_vals)) * 1.15)
    ax.grid(axis='y', alpha=0.3)

    # Add annotations for key findings
    # Hallucination difference
    hall_idx = error_types.index('hallucination')
    ax.annotate('', xy=(hall_idx + width/2, pi_vals[hall_idx]),
               xytext=(hall_idx + width/2, pi_vals[hall_idx] + 8),
               arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(hall_idx + width/2 + 0.15, pi_vals[hall_idx] + 10,
           f'+{pi_vals[hall_idx] - ri_vals[hall_idx]:.0f}%\nvs RI',
           fontsize=8, color='red', fontweight='bold')

    plt.tight_layout()
    output_file = output_dir / 'ri_vs_pi_error_comparison_bars.png'
    plt.savefig(output_file, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_error_trajectories_comparison(ri_pcts: pd.DataFrame, pi_pcts: pd.DataFrame, output_dir: Path):
    """
    Create dual-panel line plot showing error evolution at each interference level.
    Panel A: RI error trajectories
    Panel B: PI error trajectories
    Uses discrete update counts on x-axis for consistency with other plots.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    error_types = ['same_key_interference', 'retrieval_failure', 'hallucination',
                   'cross_key_interference', 'partial_match']

    # Use standard interference levels
    standard_levels = [3, 10, 50, 100, 200, 300]
    common_levels = [l for l in standard_levels if l in ri_pcts.index and l in pi_pcts.index]
    x_positions = np.arange(len(common_levels))

    # Panel A: RI
    for error_type in error_types:
        if error_type in ri_pcts.columns:
            vals = ri_pcts.loc[common_levels, error_type]
            ax1.plot(x_positions, vals.values, marker='o', linewidth=2, markersize=6,
                    label=ERROR_LABELS_SHORT[error_type], color=ERROR_COLORS[error_type])

    ax1.set_xlabel('Update Count', fontweight='bold')
    ax1.set_ylabel('Error Type Percentage (%)', fontweight='bold')
    ax1.set_title('(A) RI: Error Type Evolution\n(Recall FIRST value)', fontweight='bold')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([str(l) for l in common_levels])
    ax1.legend(loc='upper left', frameon=True, fontsize=8)
    ax1.grid(alpha=0.3)

    # Panel B: PI
    for error_type in error_types:
        if error_type in pi_pcts.columns:
            vals = pi_pcts.loc[common_levels, error_type]
            ax2.plot(x_positions, vals.values, marker='s', linewidth=2, markersize=6,
                    label=ERROR_LABELS_SHORT[error_type], color=ERROR_COLORS[error_type])

    ax2.set_xlabel('Update Count', fontweight='bold')
    ax2.set_title('(B) PI: Error Type Evolution\n(Recall LAST value)', fontweight='bold')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([str(l) for l in common_levels])
    ax2.legend(loc='upper left', frameon=True, fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / 'ri_vs_pi_error_trajectories.png'
    plt.savefig(output_file, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_same_key_interference_contrast(ri_pcts: pd.DataFrame, pi_pcts: pd.DataFrame, output_dir: Path):
    """
    Create focused comparison of same-key interference: RI (recency) vs PI (primacy).
    Key finding: RI shows recency bias, PI shows primacy bias in same-key errors
    Uses linear x-axis with discrete update counts for consistency with other plots.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use standard interference levels as discrete points
    standard_levels = [3, 10, 50, 100, 200, 300]
    common_levels = [l for l in standard_levels if l in ri_pcts.index and l in pi_pcts.index]

    ri_ski = ri_pcts.loc[common_levels, 'same_key_interference'] if 'same_key_interference' in ri_pcts.columns else pd.Series([0]*len(common_levels), index=common_levels)
    pi_ski = pi_pcts.loc[common_levels, 'same_key_interference'] if 'same_key_interference' in pi_pcts.columns else pd.Series([0]*len(common_levels), index=common_levels)

    # Use index positions for even spacing
    x_positions = np.arange(len(common_levels))

    # Plot lines
    ax.plot(x_positions, ri_ski.values, marker='o', linewidth=3, markersize=10,
           label='RI (Recency Bias)', color='#2E86AB')
    ax.fill_between(x_positions, 0, ri_ski.values, alpha=0.2, color='#2E86AB')

    ax.plot(x_positions, pi_ski.values, marker='s', linewidth=3, markersize=10,
           label='PI (Primacy Bias)', color='#A23B72')
    ax.fill_between(x_positions, 0, pi_ski.values, alpha=0.2, color='#A23B72')

    ax.set_xlabel('Update Count', fontweight='bold')
    ax.set_ylabel('Same-Key Interference (%)', fontweight='bold')
    ax.set_title('Same-Key Interference: RI vs PI\n(Recency vs Primacy Bias)', fontweight='bold', pad=15)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(l) for l in common_levels])
    ax.legend(loc='upper right', frameon=True, fontsize=11)
    ax.grid(alpha=0.3)

    # Add annotation box
    textstr = ('RI errors: Return RECENT values\n'
              '(recency intrusion)\n\n'
              'PI errors: Return EARLY values\n'
              '(primacy intrusion)')
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gray')
    ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='left', bbox=props)

    plt.tight_layout()
    output_file = output_dir / 'ri_vs_pi_samekey_contrast.png'
    plt.savefig(output_file, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_combined_figure(ri_all: pd.DataFrame, pi_all: pd.DataFrame,
                         ri_pcts: pd.DataFrame, pi_pcts: pd.DataFrame, output_dir: Path):
    """
    Create a single publication figure with 2 panels:
    (A) Overall error type distribution (bar chart)
    (B) Same-key interference by update count (shows recency vs primacy bias)
    Uses discrete update counts on x-axis for consistency.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ===== Panel A: Overall Distribution =====
    ri_dist = ri_all['error_type'].value_counts(normalize=True) * 100
    pi_dist = pi_all['error_type'].value_counts(normalize=True) * 100

    error_types = ['same_key_interference', 'retrieval_failure', 'hallucination',
                   'cross_key_interference', 'partial_match']

    x = np.arange(len(error_types))
    width = 0.35

    ri_vals = [ri_dist.get(et, 0) for et in error_types]
    pi_vals = [pi_dist.get(et, 0) for et in error_types]

    bars1 = ax1.bar(x - width/2, ri_vals, width, label='RI',
                   color='#2E86AB', edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, pi_vals, width, label='PI',
                   color='#A23B72', edgecolor='black', linewidth=0.5)

    for bar, val in zip(bars1, ri_vals):
        if val > 3:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.0f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')

    for bar, val in zip(bars2, pi_vals):
        if val > 3:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.0f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax1.set_ylabel('Percentage of Errors (%)', fontweight='bold')
    ax1.set_xlabel('Error Type', fontweight='bold')
    ax1.set_title('(A) Error Type Distribution', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([ERROR_LABELS_SHORT[et] for et in error_types], rotation=15, ha='right')
    ax1.legend(loc='upper right', frameon=True)
    ax1.set_ylim(0, max(max(ri_vals), max(pi_vals)) * 1.2)
    ax1.grid(axis='y', alpha=0.3)

    # ===== Panel B: Same-Key Interference by Update Count =====
    standard_levels = [3, 10, 50, 100, 200, 300]
    common_levels = [l for l in standard_levels if l in ri_pcts.index and l in pi_pcts.index]
    x_positions = np.arange(len(common_levels))

    ri_ski = ri_pcts.loc[common_levels, 'same_key_interference'] if 'same_key_interference' in ri_pcts.columns else pd.Series([0]*len(common_levels), index=common_levels)
    pi_ski = pi_pcts.loc[common_levels, 'same_key_interference'] if 'same_key_interference' in pi_pcts.columns else pd.Series([0]*len(common_levels), index=common_levels)

    ax2.plot(x_positions, ri_ski.values, marker='o', linewidth=2.5, markersize=8,
           label='RI (Recency Bias)', color='#2E86AB')
    ax2.fill_between(x_positions, 0, ri_ski.values, alpha=0.2, color='#2E86AB')

    ax2.plot(x_positions, pi_ski.values, marker='s', linewidth=2.5, markersize=8,
           label='PI (Primacy Bias)', color='#A23B72')
    ax2.fill_between(x_positions, 0, pi_ski.values, alpha=0.2, color='#A23B72')

    ax2.set_xlabel('Update Count', fontweight='bold')
    ax2.set_ylabel('Same-Key Interference (%)', fontweight='bold')
    ax2.set_title('(B) Same-Key Interference by Update Count', fontweight='bold')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([str(l) for l in common_levels])
    ax2.legend(loc='upper left', frameon=True)
    ax2.grid(alpha=0.3)

    # Annotation box
    textstr = 'RI: returns RECENT values\nPI: returns EARLY values'
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gray')
    ax2.text(0.97, 0.97, textstr, transform=ax2.transAxes, fontsize=9,
           verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    output_file = output_dir / 'figure_error_patterns.png'
    plt.savefig(output_file, bbox_inches='tight')
    print(f"\n  Saved main figure: {output_file}")
    plt.close()


def generate_summary_stats(ri_all: pd.DataFrame, pi_all: pd.DataFrame, output_dir: Path):
    """Generate text summary comparing RI vs PI error patterns."""

    summary = []
    summary.append("="*70)
    summary.append("RI vs PI ERROR PATTERN COMPARISON")
    summary.append("="*70)
    summary.append("")

    # Overall stats
    ri_total = len(ri_all)
    pi_total = len(pi_all)
    ri_dist = ri_all['error_type'].value_counts(normalize=True) * 100
    pi_dist = pi_all['error_type'].value_counts(normalize=True) * 100

    summary.append(f"RI Total Errors: {ri_total:,} ({ri_all['model'].nunique()} models)")
    summary.append(f"PI Total Errors: {pi_total:,} ({pi_all['model'].nunique()} models)")
    summary.append("")

    summary.append("ERROR TYPE COMPARISON:")
    summary.append("-" * 50)
    summary.append(f"{'Error Type':<25} {'RI %':>10} {'PI %':>10} {'Diff':>10}")
    summary.append("-" * 50)

    for et in ['same_key_interference', 'retrieval_failure', 'hallucination',
               'cross_key_interference', 'partial_match']:
        ri_pct = ri_dist.get(et, 0)
        pi_pct = pi_dist.get(et, 0)
        diff = pi_pct - ri_pct
        summary.append(f"{ERROR_LABELS_SHORT[et]:<25} {ri_pct:>9.1f}% {pi_pct:>9.1f}% {diff:>+9.1f}%")

    summary.append("")
    summary.append("="*70)
    summary.append("KEY FINDINGS")
    summary.append("="*70)
    summary.append("")

    ri_hall = ri_dist.get('hallucination', 0)
    pi_hall = pi_dist.get('hallucination', 0)
    ri_ret = ri_dist.get('retrieval_failure', 0)
    pi_ret = pi_dist.get('retrieval_failure', 0)

    summary.append(f"1. HALLUCINATION CONTRAST:")
    summary.append(f"   RI: {ri_hall:.1f}% (rare - models fail silently)")
    summary.append(f"   PI: {pi_hall:.1f}% (common - models confabulate)")
    summary.append(f"   --> PI hallucination rate is {pi_hall/max(ri_hall,0.1):.1f}x higher than RI")
    summary.append("")

    summary.append(f"2. RETRIEVAL FAILURE:")
    summary.append(f"   RI: {ri_ret:.1f}% (dominant failure mode)")
    summary.append(f"   PI: {pi_ret:.1f}%")
    summary.append("")

    summary.append(f"3. MECHANISTIC INTERPRETATION:")
    summary.append(f"   RI failures are 'passive': cannot access initial encoding")
    summary.append(f"   PI failures are 'active': generate plausible but wrong values")
    summary.append("")

    summary.append("="*70)

    summary_text = "\n".join(summary)
    output_file = output_dir / 'RI_VS_PI_ERROR_COMPARISON.txt'
    with open(output_file, 'w') as f:
        f.write(summary_text)

    print(f"\n  Summary saved: {output_file}\n")
    print(summary_text)


def main():
    parser = argparse.ArgumentParser(description='Generate RI vs PI error comparison figures')
    parser.add_argument('--ri-errors', type=str,
                       default='results/error_analysis/error_classification_detailed.xlsx',
                       help='Path to RI error classification Excel')
    parser.add_argument('--pi-errors', type=str,
                       default='results/error_analysis/pi_error_classification_detailed.xlsx',
                       help='Path to PI error classification Excel')
    parser.add_argument('--output-dir', type=str,
                       default='results/key_results',
                       help='Output directory for figures')

    args = parser.parse_args()

    ri_path = Path(args.ri_errors)
    pi_path = Path(args.pi_errors)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    ri_all, ri_pcts, pi_all, pi_pcts = load_error_data(ri_path, pi_path)

    print("\nGenerating comparison visualizations...")

    # Generate all plots
    plot_error_comparison_bars(ri_all, pi_all, output_dir)
    plot_error_trajectories_comparison(ri_pcts, pi_pcts, output_dir)
    plot_same_key_interference_contrast(ri_pcts, pi_pcts, output_dir)
    plot_combined_figure(ri_all, pi_all, ri_pcts, pi_pcts, output_dir)

    # Generate summary
    generate_summary_stats(ri_all, pi_all, output_dir)

    print("\n  All RI vs PI error comparison figures complete!")


if __name__ == '__main__':
    main()
