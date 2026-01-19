#!/usr/bin/env python3
"""
Visualize Error Type Patterns

Creates publication-quality visualizations of the 5 error types across interference levels.
Shows the three-stage progression from "Unable to Forget" paper.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

# Set publication style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Color palette for 5 error types
ERROR_COLORS = {
    'same_key_interference': '#E74C3C',      # Red - most severe
    'cross_key_interference': '#F39C12',     # Orange
    'partial_match': '#3498DB',              # Blue
    'hallucination': '#9B59B6',              # Purple
    'retrieval_failure': '#95A5A6'           # Gray - missing
}

ERROR_LABELS = {
    'same_key_interference': 'Same-Key Interference',
    'cross_key_interference': 'Cross-Key Interference',
    'partial_match': 'Partial Match',
    'hallucination': 'Hallucination',
    'retrieval_failure': 'Retrieval Failure'
}

def load_error_data(excel_path: Path) -> tuple:
    """Load error classification data from Excel."""
    df_all = pd.read_excel(excel_path, sheet_name='All_Errors')
    df_pcts = pd.read_excel(excel_path, sheet_name='Percentages_By_Level', index_col=0)
    df_counts = pd.read_excel(excel_path, sheet_name='Counts_By_Level', index_col=0)

    return df_all, df_pcts, df_counts

def plot_error_distribution_stacked(df_pcts: pd.DataFrame, output_dir: Path):
    """Create stacked bar chart showing error type distribution by level."""

    fig, ax = plt.subplots(figsize=(10, 6))

    # Reorder columns
    error_types = ['same_key_interference', 'cross_key_interference',
                   'partial_match', 'hallucination', 'retrieval_failure']
    available = [col for col in error_types if col in df_pcts.columns]
    df_plot = df_pcts[available]

    # Create stacked bar chart
    bottom = np.zeros(len(df_plot))
    for error_type in available:
        values = df_plot[error_type].values
        ax.bar(df_plot.index, values, bottom=bottom,
               label=ERROR_LABELS[error_type],
               color=ERROR_COLORS[error_type],
               edgecolor='white', linewidth=0.5)
        bottom += values

    ax.set_xlabel('Number of Interfering Updates', fontweight='bold')
    ax.set_ylabel('Error Type Distribution (%)', fontweight='bold')
    ax.set_title('Error Type Distribution Across Update Counts\n(All 57 Models, Levels 3-300)',
                 fontweight='bold', pad=15)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True)
    ax.grid(axis='y', alpha=0.3)

    # Add three-stage annotations
    ax.axvline(x=25, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=125, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    ax.text(15, 95, 'Stage 1:\nTightly Focused', ha='center', va='top',
            fontsize=8, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax.text(75, 95, 'Stage 2:\nDispersed', ha='center', va='top',
            fontsize=8, style='italic', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax.text(225, 95, 'Stage 3:\nHallucinatory', ha='center', va='top',
            fontsize=8, style='italic', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

    plt.tight_layout()
    output_file = output_dir / 'error_distribution_stacked.png'
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()

def plot_error_trajectories(df_pcts: pd.DataFrame, output_dir: Path):
    """Create line plot showing how each error type evolves."""

    fig, ax = plt.subplots(figsize=(10, 6))

    error_types = ['same_key_interference', 'cross_key_interference',
                   'partial_match', 'hallucination', 'retrieval_failure']
    available = [col for col in error_types if col in df_pcts.columns]

    for error_type in available:
        ax.plot(df_pcts.index, df_pcts[error_type],
                marker='o', linewidth=2, markersize=6,
                label=ERROR_LABELS[error_type],
                color=ERROR_COLORS[error_type])

    ax.set_xlabel('Number of Interfering Updates', fontweight='bold')
    ax.set_ylabel('Error Type Percentage (%)', fontweight='bold')
    ax.set_title('Error Type Trajectories Across Update Counts',
                 fontweight='bold', pad=15)
    ax.legend(loc='best', frameon=True)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / 'error_trajectories.png'
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()

def plot_same_key_position_heatmap(df_all: pd.DataFrame, output_dir: Path):
    """Create heatmap showing which positions are retrieved in same-key errors."""

    # Filter to same-key interference only
    ski_df = df_all[df_all['error_type'] == 'same_key_interference'].copy()

    if len(ski_df) == 0:
        print("⚠️  No same-key interference errors to plot")
        return

    # Create position bins (normalized 0-100%)
    ski_df['position_pct'] = (ski_df['position'] / ski_df['total_updates']) * 100

    # Bin positions
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    bin_labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%',
                  '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    ski_df['position_bin'] = pd.cut(ski_df['position_pct'], bins=bins, labels=bin_labels)

    # Create heatmap data
    heatmap_data = ski_df.groupby(['interference_level', 'position_bin']).size().unstack(fill_value=0)

    # Normalize by row (each interference level sums to 100%)
    heatmap_data_pct = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(heatmap_data_pct, cmap='YlOrRd', annot=True, fmt='.1f',
                cbar_kws={'label': 'Percentage of SKI Errors (%)'},
                ax=ax, linewidths=0.5, linecolor='white')

    ax.set_xlabel('Position in Update Sequence\n(1st update = 0%, Last update = 100%)',
                  fontweight='bold')
    ax.set_ylabel('Number of Interfering Updates', fontweight='bold')
    ax.set_title('Same-Key Interference: Position Distribution Heatmap\n(Which earlier values are retrieved?)',
                 fontweight='bold', pad=15)

    plt.tight_layout()
    output_file = output_dir / 'same_key_position_heatmap.png'
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()

def plot_hallucination_phase_transition(df_pcts: pd.DataFrame, df_counts: pd.DataFrame, output_dir: Path):
    """Show hallucination rate increase (phase transition at high interference)."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Hallucination percentage
    if 'hallucination' in df_pcts.columns:
        ax1.plot(df_pcts.index, df_pcts['hallucination'],
                marker='o', linewidth=3, markersize=8,
                color=ERROR_COLORS['hallucination'])
        ax1.fill_between(df_pcts.index, 0, df_pcts['hallucination'],
                        alpha=0.3, color=ERROR_COLORS['hallucination'])

        ax1.set_xlabel('Interference Level', fontweight='bold')
        ax1.set_ylabel('Hallucination Rate (%)', fontweight='bold')
        ax1.set_title('Hallucination Rate vs Interference\n(Phase Transition at High Load)',
                     fontweight='bold')
        ax1.grid(alpha=0.3)

    # Right plot: Retrieval failure vs Same-key interference (tradeoff)
    if 'retrieval_failure' in df_pcts.columns and 'same_key_interference' in df_pcts.columns:
        ax2.plot(df_pcts.index, df_pcts['retrieval_failure'],
                marker='s', linewidth=2, markersize=6,
                label='Retrieval Failure', color=ERROR_COLORS['retrieval_failure'])
        ax2.plot(df_pcts.index, df_pcts['same_key_interference'],
                marker='o', linewidth=2, markersize=6,
                label='Same-Key Interference', color=ERROR_COLORS['same_key_interference'])

        ax2.set_xlabel('Interference Level', fontweight='bold')
        ax2.set_ylabel('Error Type Percentage (%)', fontweight='bold')
        ax2.set_title('Tradeoff: Retrieval Failure vs Same-Key Interference',
                     fontweight='bold')
        ax2.legend(loc='best', frameon=True)
        ax2.grid(alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / 'hallucination_phase_transition.png'
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()

def generate_summary_stats(df_all: pd.DataFrame, df_pcts: pd.DataFrame, output_dir: Path):
    """Generate text summary of key findings."""

    summary = []
    summary.append("="*70)
    summary.append("ERROR TYPE CLASSIFICATION - KEY FINDINGS")
    summary.append("="*70)
    summary.append("")

    # Overall distribution
    total_errors = len(df_all)
    error_dist = df_all['error_type'].value_counts()
    error_pct = (error_dist / total_errors * 100).round(1)

    summary.append(f"Total Errors Analyzed: {total_errors:,}")
    summary.append(f"Models: {df_all['model'].nunique()}")
    summary.append(f"Interference Levels: {sorted(df_all['interference_level'].unique())}")
    summary.append("")
    summary.append("OVERALL ERROR DISTRIBUTION:")
    for error_type in ['same_key_interference', 'cross_key_interference',
                       'partial_match', 'hallucination', 'retrieval_failure']:
        if error_type in error_dist.index:
            summary.append(f"  {ERROR_LABELS[error_type]:30s}: {error_dist[error_type]:5d} ({error_pct[error_type]:5.1f}%)")

    # Three-stage progression
    summary.append("")
    summary.append("="*70)
    summary.append("THREE-STAGE PROGRESSION (from 'Unable to Forget' Paper)")
    summary.append("="*70)
    summary.append("")

    # Stage 1: Low interference (3, 10)
    stage1_levels = [3, 10]
    stage1_data = df_pcts.loc[stage1_levels].mean()
    summary.append("STAGE 1 - LOW INTERFERENCE (Levels 3, 10): Tightly Focused Errors")
    summary.append(f"  Same-Key Interference: {stage1_data.get('same_key_interference', 0):.1f}%")
    summary.append(f"  Cross-Key Interference: {stage1_data.get('cross_key_interference', 0):.1f}%")
    summary.append(f"  Hallucination: {stage1_data.get('hallucination', 0):.1f}%")
    summary.append(f"  Retrieval Failure: {stage1_data.get('retrieval_failure', 0):.1f}%")
    summary.append("")

    # Stage 2: Moderate interference (50, 100)
    stage2_levels = [50, 100]
    stage2_data = df_pcts.loc[stage2_levels].mean()
    summary.append("STAGE 2 - MODERATE INTERFERENCE (Levels 50, 100): Dispersed Errors")
    summary.append(f"  Same-Key Interference: {stage2_data.get('same_key_interference', 0):.1f}%")
    summary.append(f"  Cross-Key Interference: {stage2_data.get('cross_key_interference', 0):.1f}%")
    summary.append(f"  Hallucination: {stage2_data.get('hallucination', 0):.1f}%")
    summary.append(f"  Retrieval Failure: {stage2_data.get('retrieval_failure', 0):.1f}%")
    summary.append("")

    # Stage 3: High interference (200, 300)
    stage3_levels = [200, 300]
    stage3_data = df_pcts.loc[stage3_levels].mean()
    summary.append("STAGE 3 - HIGH INTERFERENCE (Levels 200, 300): Hallucinatory Responses")
    summary.append(f"  Same-Key Interference: {stage3_data.get('same_key_interference', 0):.1f}%")
    summary.append(f"  Cross-Key Interference: {stage3_data.get('cross_key_interference', 0):.1f}%")
    summary.append(f"  Hallucination: {stage3_data.get('hallucination', 0):.1f}%")
    summary.append(f"  Retrieval Failure: {stage3_data.get('retrieval_failure', 0):.1f}%")
    summary.append("")

    # Same-Key position analysis
    ski_df = df_all[df_all['error_type'] == 'same_key_interference']
    if len(ski_df) > 0:
        summary.append("="*70)
        summary.append("SAME-KEY INTERFERENCE POSITION ANALYSIS")
        summary.append("="*70)
        summary.append("")
        summary.append("(Position 1 = first/target, higher = more recent)")
        summary.append("")

        for level in sorted(ski_df['interference_level'].unique()):
            level_ski = ski_df[ski_df['interference_level'] == level]
            avg_pos = level_ski['position'].mean()
            avg_recency = level_ski['recency'].mean()
            most_recent = level_ski['is_most_recent'].sum()
            total = len(level_ski)
            recent_pct = (most_recent / total * 100) if total > 0 else 0

            summary.append(f"  Level {level:3d}: Avg position={avg_pos:5.1f}, "
                         f"Recency={avg_recency:5.1f} steps back, "
                         f"Most recent={most_recent}/{total} ({recent_pct:.0f}%)")

    summary.append("")
    summary.append("="*70)

    # Save summary
    summary_text = "\n".join(summary)
    output_file = output_dir / 'ERROR_ANALYSIS_SUMMARY.txt'
    with open(output_file, 'w') as f:
        f.write(summary_text)

    print(f"\n✅ Summary saved: {output_file}\n")
    print(summary_text)

def main():
    parser = argparse.ArgumentParser(description='Visualize error type patterns')
    parser.add_argument('--input', type=str,
                       default='results/error_analysis/error_classification_detailed.xlsx',
                       help='Path to error classification Excel file')
    parser.add_argument('--output-dir', type=str,
                       default='results/error_analysis',
                       help='Output directory for visualizations')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading error data from: {input_path}")
    df_all, df_pcts, df_counts = load_error_data(input_path)

    print(f"\nGenerating visualizations...\n")

    # Generate all plots
    plot_error_distribution_stacked(df_pcts, output_dir)
    plot_error_trajectories(df_pcts, output_dir)
    plot_same_key_position_heatmap(df_all, output_dir)
    plot_hallucination_phase_transition(df_pcts, df_counts, output_dir)

    # Generate summary
    generate_summary_stats(df_all, df_pcts, output_dir)

    print("\n✅ All visualizations complete!")

if __name__ == '__main__':
    main()
