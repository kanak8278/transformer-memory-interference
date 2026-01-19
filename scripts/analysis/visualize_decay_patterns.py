#!/usr/bin/env python3
"""
Visualize decay patterns for retroactive interference across model groups.

Creates publication-quality figures with:
1. Mean trend lines with shaded confidence intervals
2. Faint individual model lines in background
3. Horizontal panel layouts for better comparison
4. Statistical annotations (mean RIES, n models)
5. Clean integrated legends

Publication-quality figures for ICML 2026.
Matches the style of "Unable to Forget" (Falk et al., 2024) Figure 1.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# Set ICML publication style
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.linewidth': 1.0,
    'lines.linewidth': 2.0,
    'lines.markersize': 5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# Color scheme for size tiers (gradient from light to dark)
SIZE_TIER_COLORS = {
    'XS': '#FFB3BA',  # Light pink
    'S': '#FFDFBA',   # Light orange
    'M': '#BAE1FF',   # Light blue
    'L': '#BAFFC9',   # Light green
}

SIZE_TIER_COLORS_DARK = {
    'XS': '#E74C3C',  # Red
    'S': '#F39C12',   # Orange
    'M': '#3498DB',   # Blue
    'L': '#27AE60',   # Green
}

# Colors for binary comparisons
REASONING_COLORS = {
    True: '#E74C3C',   # Red for Reasoning
    False: '#3498DB',  # Blue for Non-Reasoning
}

MOE_COLORS = {
    True: '#9B59B6',   # Purple for MoE
    False: '#2ECC71',  # Green for Dense
}


def categorize_models(df_ries):
    """Categorize models into different groups."""
    categories = []

    for _, row in df_ries.iterrows():
        model_id = row['model_id']
        size = row['model_size_b']

        # Size tier
        if size < 10:
            size_tier = 'XS'
        elif size < 100:
            size_tier = 'S'
        elif size < 500:
            size_tier = 'M'
        else:
            size_tier = 'L'

        # Reasoning models
        is_reasoning = any(x in model_id for x in ['o1', 'o3', 'o4', 'deepseek-r1'])

        # MoE models
        moe_models = ['o1', 'o3', 'o4', 'gpt-4.1', 'gpt-5', 'gpt-3.5-turbo',
                      'gemini-2.0', 'gemini-2.5', 'llama-4-maverick', 'llama-4-scout',
                      'qwen-3-vl', 'qwen-3-coder', 'ministral', 'mistral-8x7b']
        is_moe = any(x in model_id for x in moe_models)

        categories.append({
            'model_id': model_id,
            'size_tier': size_tier,
            'is_reasoning': is_reasoning,
            'is_moe': is_moe,
            'model_size_b': size,
            'ries_score': row['ries_score']
        })

    return pd.DataFrame(categories)


def compute_group_stats(df_full, model_ids):
    """Compute mean and std for a group of models at each interference level."""
    group_data = df_full[df_full['model_id'].isin(model_ids)]

    stats = group_data.groupby('interference_level').agg({
        'accuracy': ['mean', 'std', 'count']
    }).reset_index()
    stats.columns = ['interference_level', 'mean', 'std', 'count']

    # Compute 95% CI
    stats['ci'] = 1.96 * stats['std'] / np.sqrt(stats['count'])
    stats['ci'] = stats['ci'].fillna(0)

    return stats


def plot_decay_by_size_tiers(df_full, df_categories, output_file):
    """
    Create horizontal 4-panel plot for size tiers with aggregated trends.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    size_tiers = ['XS', 'S', 'M', 'L']

    update_counts = [3, 10, 50, 100, 200, 300]

    for idx, tier in enumerate(size_tiers):
        ax = axes[idx]

        # Get models in this tier
        tier_models = df_categories[df_categories['size_tier'] == tier]
        model_ids = tier_models['model_id'].tolist()

        if len(model_ids) == 0:
            ax.text(0.5, 0.5, f'No {tier} models', ha='center', va='center', transform=ax.transAxes)
            continue

        # Plot individual models as faint background lines
        for model_id in model_ids:
            model_data = df_full[df_full['model_id'] == model_id].sort_values('interference_level')
            if len(model_data) > 0:
                ax.plot(model_data['interference_level'], model_data['accuracy'],
                       color=SIZE_TIER_COLORS[tier], alpha=0.3, linewidth=1, zorder=1)

        # Compute and plot aggregated statistics
        stats = compute_group_stats(df_full, model_ids)

        # Plot shaded confidence interval
        ax.fill_between(stats['interference_level'],
                       stats['mean'] - stats['ci'],
                       stats['mean'] + stats['ci'],
                       color=SIZE_TIER_COLORS_DARK[tier], alpha=0.2, zorder=2)

        # Plot mean line
        ax.plot(stats['interference_level'], stats['mean'],
               color=SIZE_TIER_COLORS_DARK[tier], linewidth=2.5,
               marker='o', markersize=6, zorder=3)

        # Formatting
        ax.set_xscale('log')
        ax.set_xlim(2, 400)
        ax.set_ylim(-5, 105)
        ax.set_xticks(update_counts)
        ax.set_xticklabels([str(x) for x in update_counts])
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

        # Calculate mean RIES for annotation
        mean_ries = tier_models['ries_score'].mean()
        std_ries = tier_models['ries_score'].std()
        n_models = len(tier_models)

        # Title with tier info
        ax.set_title(f'{tier} (<{[10, 100, 500, "∞"][idx]}B)\nn={n_models}',
                    fontsize=11, fontweight='bold')

        # Statistical annotation in corner
        ax.text(0.95, 0.95, f'RIES: {mean_ries:.1f}±{std_ries:.1f}',
               transform=ax.transAxes, ha='right', va='top',
               fontsize=9, bbox=dict(boxstyle='round,pad=0.3',
                                     facecolor='white', alpha=0.8, edgecolor='gray'))

        ax.set_xlabel('Update Count', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_file}")

    # Also save PDF
    plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_file.with_suffix('.pdf')}")
    plt.close()


def plot_decay_reasoning_vs_nonreasoning(df_full, df_categories, output_file):
    """
    Create horizontal 2-panel plot for Reasoning vs Non-Reasoning.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    update_counts = [3, 10, 50, 100, 200, 300]
    groups = [(True, 'Reasoning (CoT)'), (False, 'Standard')]

    for idx, (is_reasoning, group_name) in enumerate(groups):
        ax = axes[idx]

        # Get models in this group
        group_models = df_categories[df_categories['is_reasoning'] == is_reasoning]
        model_ids = group_models['model_id'].tolist()
        color = REASONING_COLORS[is_reasoning]

        if len(model_ids) == 0:
            ax.text(0.5, 0.5, f'No {group_name} models', ha='center', va='center', transform=ax.transAxes)
            continue

        # Plot individual models as faint background lines
        for model_id in model_ids:
            model_data = df_full[df_full['model_id'] == model_id].sort_values('interference_level')
            if len(model_data) > 0:
                ax.plot(model_data['interference_level'], model_data['accuracy'],
                       color=color, alpha=0.15, linewidth=1, zorder=1)

        # Compute and plot aggregated statistics
        stats = compute_group_stats(df_full, model_ids)

        # Plot shaded confidence interval
        ax.fill_between(stats['interference_level'],
                       stats['mean'] - stats['ci'],
                       stats['mean'] + stats['ci'],
                       color=color, alpha=0.2, zorder=2)

        # Plot mean line
        ax.plot(stats['interference_level'], stats['mean'],
               color=color, linewidth=2.5, marker='o', markersize=6, zorder=3)

        # Formatting
        ax.set_xscale('log')
        ax.set_xlim(2, 400)
        ax.set_ylim(-5, 105)
        ax.set_xticks(update_counts)
        ax.set_xticklabels([str(x) for x in update_counts])
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

        # Calculate stats for annotation
        mean_ries = group_models['ries_score'].mean()
        std_ries = group_models['ries_score'].std()
        n_models = len(group_models)

        ax.set_title(f'{group_name}\nn={n_models}', fontsize=11, fontweight='bold')

        # Statistical annotation
        ax.text(0.95, 0.95, f'RIES: {mean_ries:.1f}±{std_ries:.1f}',
               transform=ax.transAxes, ha='right', va='top',
               fontsize=9, bbox=dict(boxstyle='round,pad=0.3',
                                     facecolor='white', alpha=0.8, edgecolor='gray'))

        ax.set_xlabel('Update Count', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_file}")
    plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_file.with_suffix('.pdf')}")
    plt.close()


def plot_decay_moe_vs_dense(df_full, df_categories, output_file):
    """
    Create horizontal 2-panel plot for MoE vs Dense.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    update_counts = [3, 10, 50, 100, 200, 300]
    groups = [(True, 'MoE'), (False, 'Dense')]

    for idx, (is_moe, group_name) in enumerate(groups):
        ax = axes[idx]

        # Get models in this group
        group_models = df_categories[df_categories['is_moe'] == is_moe]
        model_ids = group_models['model_id'].tolist()
        color = MOE_COLORS[is_moe]

        if len(model_ids) == 0:
            ax.text(0.5, 0.5, f'No {group_name} models', ha='center', va='center', transform=ax.transAxes)
            continue

        # Plot individual models as faint background lines
        for model_id in model_ids:
            model_data = df_full[df_full['model_id'] == model_id].sort_values('interference_level')
            if len(model_data) > 0:
                ax.plot(model_data['interference_level'], model_data['accuracy'],
                       color=color, alpha=0.15, linewidth=1, zorder=1)

        # Compute and plot aggregated statistics
        stats = compute_group_stats(df_full, model_ids)

        # Plot shaded confidence interval
        ax.fill_between(stats['interference_level'],
                       stats['mean'] - stats['ci'],
                       stats['mean'] + stats['ci'],
                       color=color, alpha=0.2, zorder=2)

        # Plot mean line
        ax.plot(stats['interference_level'], stats['mean'],
               color=color, linewidth=2.5, marker='o', markersize=6, zorder=3)

        # Formatting
        ax.set_xscale('log')
        ax.set_xlim(2, 400)
        ax.set_ylim(-5, 105)
        ax.set_xticks(update_counts)
        ax.set_xticklabels([str(x) for x in update_counts])
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

        # Calculate stats for annotation
        mean_ries = group_models['ries_score'].mean()
        std_ries = group_models['ries_score'].std()
        n_models = len(group_models)

        ax.set_title(f'{group_name}\nn={n_models}', fontsize=11, fontweight='bold')

        # Statistical annotation
        ax.text(0.95, 0.95, f'RIES: {mean_ries:.1f}±{std_ries:.1f}',
               transform=ax.transAxes, ha='right', va='top',
               fontsize=9, bbox=dict(boxstyle='round,pad=0.3',
                                     facecolor='white', alpha=0.8, edgecolor='gray'))

        ax.set_xlabel('Update Count', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_file}")
    plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_file.with_suffix('.pdf')}")
    plt.close()


def plot_combined_size_comparison(df_full, df_categories, output_file):
    """
    Create a single figure comparing all size tiers on one plot.
    Similar to Figure 1 in "Unable to Forget".
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    update_counts = [3, 10, 50, 100, 200, 300]
    size_tiers = ['XS', 'S', 'M', 'L']

    for tier in size_tiers:
        tier_models = df_categories[df_categories['size_tier'] == tier]
        model_ids = tier_models['model_id'].tolist()
        color = SIZE_TIER_COLORS_DARK[tier]

        if len(model_ids) == 0:
            continue

        # Compute aggregated statistics
        stats = compute_group_stats(df_full, model_ids)

        # Plot shaded confidence interval
        ax.fill_between(stats['interference_level'],
                       stats['mean'] - stats['ci'],
                       stats['mean'] + stats['ci'],
                       color=color, alpha=0.15, zorder=1)

        # Plot mean line with label
        mean_ries = tier_models['ries_score'].mean()
        n_models = len(tier_models)
        ax.plot(stats['interference_level'], stats['mean'],
               color=color, linewidth=2.5, marker='o', markersize=5,
               label=f'{tier} (n={n_models}, RIES={mean_ries:.0f})', zorder=2)

    # Formatting
    ax.set_xscale('log')
    ax.set_xlim(2, 400)
    ax.set_ylim(-5, 105)
    ax.set_xticks(update_counts)
    ax.set_xticklabels([str(x) for x in update_counts])
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    ax.set_xlabel('Update Count', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Retroactive Interference by Model Size', fontsize=12, fontweight='bold')

    # Legend inside plot
    ax.legend(loc='lower left', frameon=True, fancybox=True,
             framealpha=0.9, edgecolor='gray')

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_file}")
    plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_file.with_suffix('.pdf')}")
    plt.close()


def plot_reasoning_comparison(df_full, df_categories, output_file):
    """
    Single plot comparing Reasoning vs Standard models.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    update_counts = [3, 10, 50, 100, 200, 300]
    groups = [(True, 'Reasoning (CoT)'), (False, 'Standard')]

    for is_reasoning, group_name in groups:
        group_models = df_categories[df_categories['is_reasoning'] == is_reasoning]
        model_ids = group_models['model_id'].tolist()
        color = REASONING_COLORS[is_reasoning]

        if len(model_ids) == 0:
            continue

        # Compute aggregated statistics
        stats = compute_group_stats(df_full, model_ids)

        # Plot shaded confidence interval
        ax.fill_between(stats['interference_level'],
                       stats['mean'] - stats['ci'],
                       stats['mean'] + stats['ci'],
                       color=color, alpha=0.15, zorder=1)

        # Plot mean line
        mean_ries = group_models['ries_score'].mean()
        n_models = len(group_models)
        ax.plot(stats['interference_level'], stats['mean'],
               color=color, linewidth=2.5, marker='o', markersize=5,
               label=f'{group_name} (n={n_models}, RIES={mean_ries:.0f})', zorder=2)

    # Formatting
    ax.set_xscale('log')
    ax.set_xlim(2, 400)
    ax.set_ylim(-5, 105)
    ax.set_xticks(update_counts)
    ax.set_xticklabels([str(x) for x in update_counts])
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    ax.set_xlabel('Update Count', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Retroactive Interference: Reasoning vs Standard Models', fontsize=12, fontweight='bold')

    ax.legend(loc='lower left', frameon=True, fancybox=True,
             framealpha=0.9, edgecolor='gray')

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_file}")
    plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_file.with_suffix('.pdf')}")
    plt.close()


def plot_individual_model_curves(df_full, df_ries, output_file, top_n=15):
    """
    Create a plot showing individual model decay curves for top/bottom performers.
    Shows top N and bottom N models by RIES score.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    update_counts = [3, 10, 50, 100, 200, 300]

    # Sort by RIES
    df_sorted = df_ries.sort_values('ries_score', ascending=False)

    # Top performers
    ax = axes[0]
    top_models = df_sorted.head(top_n)

    colors = plt.cm.Greens(np.linspace(0.3, 0.9, top_n))
    for i, (_, row) in enumerate(top_models.iterrows()):
        model_id = row['model_id']
        model_data = df_full[df_full['model_id'] == model_id].sort_values('interference_level')
        if len(model_data) > 0:
            ax.plot(model_data['interference_level'], model_data['accuracy'],
                   color=colors[i], linewidth=1.5, alpha=0.8,
                   label=f"{model_id[:20]} ({row['ries_score']:.0f})")

    ax.set_xscale('log')
    ax.set_xlim(2, 400)
    ax.set_ylim(-5, 105)
    ax.set_xticks(update_counts)
    ax.set_xticklabels([str(x) for x in update_counts])
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel('Update Count', fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title(f'Top {top_n} Models (Highest RIES)', fontsize=11, fontweight='bold')
    ax.legend(loc='lower left', fontsize=7, ncol=1, framealpha=0.9)

    # Bottom performers
    ax = axes[1]
    bottom_models = df_sorted.tail(top_n)

    colors = plt.cm.Reds(np.linspace(0.3, 0.9, top_n))
    for i, (_, row) in enumerate(bottom_models.iterrows()):
        model_id = row['model_id']
        model_data = df_full[df_full['model_id'] == model_id].sort_values('interference_level')
        if len(model_data) > 0:
            ax.plot(model_data['interference_level'], model_data['accuracy'],
                   color=colors[i], linewidth=1.5, alpha=0.8,
                   label=f"{model_id[:20]} ({row['ries_score']:.0f})")

    ax.set_xscale('log')
    ax.set_xlim(2, 400)
    ax.set_ylim(-5, 105)
    ax.set_xticks(update_counts)
    ax.set_xticklabels([str(x) for x in update_counts])
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel('Update Count', fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title(f'Bottom {top_n} Models (Lowest RIES)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=7, ncol=1, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_file}")
    plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_file.with_suffix('.pdf')}")
    plt.close()


def plot_pi_decay_curves(df_pi_full, df_pi_scores, output_file):
    """
    Create PI decay curves similar to RI, showing size tier comparison.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    update_counts = [3, 10, 50, 100, 200, 300]

    # Categorize PI models by size
    size_tiers = {'XS': [], 'S': [], 'M': [], 'L': []}
    for _, row in df_pi_scores.iterrows():
        size = row['model_size_b']
        if size < 10:
            size_tiers['XS'].append(row['model_id'])
        elif size < 100:
            size_tiers['S'].append(row['model_id'])
        elif size < 500:
            size_tiers['M'].append(row['model_id'])
        else:
            size_tiers['L'].append(row['model_id'])

    for tier, model_ids in size_tiers.items():
        if len(model_ids) == 0:
            continue

        color = SIZE_TIER_COLORS_DARK[tier]

        # Compute stats
        group_data = df_pi_full[df_pi_full['model_id'].isin(model_ids)]
        stats = group_data.groupby('interference_level').agg({
            'accuracy': ['mean', 'std', 'count']
        }).reset_index()
        stats.columns = ['interference_level', 'mean', 'std', 'count']
        stats['ci'] = 1.96 * stats['std'] / np.sqrt(stats['count'])
        stats['ci'] = stats['ci'].fillna(0)

        # Plot
        ax.fill_between(stats['interference_level'],
                       stats['mean'] - stats['ci'],
                       stats['mean'] + stats['ci'],
                       color=color, alpha=0.15, zorder=1)

        tier_pis = df_pi_scores[df_pi_scores['model_id'].isin(model_ids)]['pis_score'].mean()
        ax.plot(stats['interference_level'], stats['mean'],
               color=color, linewidth=2.5, marker='o', markersize=5,
               label=f'{tier} (n={len(model_ids)}, PIS={tier_pis:.0f})', zorder=2)

    ax.set_xscale('log')
    ax.set_xlim(2, 400)
    ax.set_ylim(-5, 105)
    ax.set_xticks(update_counts)
    ax.set_xticklabels([str(x) for x in update_counts])
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    ax.set_xlabel('Update Count', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Proactive Interference (PI) by Model Size', fontsize=12, fontweight='bold')
    ax.legend(loc='lower left', frameon=True, fancybox=True, framealpha=0.9, edgecolor='gray')

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_file}")
    plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_file.with_suffix('.pdf')}")
    plt.close()


def plot_ri_vs_pi_comparison(df_ri_full, df_pi_full, df_ri_scores, df_pi_scores, output_file):
    """
    Create side-by-side comparison of RI vs PI decay curves.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    update_counts = [3, 10, 50, 100, 200, 300]

    # Find common models
    ri_models = set(df_ri_scores['model_id'])
    pi_models = set(df_pi_scores['model_id'])
    common_models = ri_models.intersection(pi_models)

    datasets = [
        (axes[0], df_ri_full, df_ri_scores, 'Retroactive Interference (RI)', 'RIES'),
        (axes[1], df_pi_full, df_pi_scores, 'Proactive Interference (PI)', 'PIS'),
    ]

    for ax, df_full, df_scores, title, score_name in datasets:
        # Filter to common models only
        df_scores_common = df_scores[df_scores['model_id'].isin(common_models)]
        df_full_common = df_full[df_full['model_id'].isin(common_models)]

        # Compute overall mean
        stats = df_full_common.groupby('interference_level').agg({
            'accuracy': ['mean', 'std', 'count']
        }).reset_index()
        stats.columns = ['interference_level', 'mean', 'std', 'count']
        stats['ci'] = 1.96 * stats['std'] / np.sqrt(stats['count'])

        color = '#E74C3C' if 'Retroactive' in title else '#3498DB'

        ax.fill_between(stats['interference_level'],
                       stats['mean'] - stats['ci'],
                       stats['mean'] + stats['ci'],
                       color=color, alpha=0.2, zorder=1)

        mean_score = df_scores_common[score_name.lower() + '_score'].mean()
        ax.plot(stats['interference_level'], stats['mean'],
               color=color, linewidth=2.5, marker='o', markersize=6, zorder=2)

        ax.set_xscale('log')
        ax.set_xlim(2, 400)
        ax.set_ylim(-5, 105)
        ax.set_xticks(update_counts)
        ax.set_xticklabels([str(x) for x in update_counts])
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

        ax.set_xlabel('Update Count', fontsize=10)
        ax.set_title(f'{title}\nn={len(common_models)}, Mean {score_name}={mean_score:.1f}',
                    fontsize=11, fontweight='bold')

        # Add annotation for key insight
        if 'Retroactive' in title:
            ax.set_ylabel('Accuracy (%)', fontsize=11)
            ax.text(0.95, 0.05, 'Recall INITIAL value',
                   transform=ax.transAxes, ha='right', va='bottom',
                   fontsize=9, style='italic', color='gray')
        else:
            ax.text(0.95, 0.05, 'Recall LATEST value',
                   transform=ax.transAxes, ha='right', va='bottom',
                   fontsize=9, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_file}")
    plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_file.with_suffix('.pdf')}")
    plt.close()


def plot_error_patterns(error_file, output_file):
    """
    Create error pattern visualization showing error type distribution.
    """
    df_pcts = pd.read_excel(error_file, sheet_name='Percentages_By_Level', index_col=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Error colors
    ERROR_COLORS = {
        'same_key_interference': '#E74C3C',
        'cross_key_interference': '#F39C12',
        'partial_match': '#3498DB',
        'hallucination': '#9B59B6',
        'retrieval_failure': '#95A5A6'
    }
    ERROR_LABELS = {
        'same_key_interference': 'Same-Key Interference',
        'cross_key_interference': 'Cross-Key Interference',
        'partial_match': 'Partial Match',
        'hallucination': 'Hallucination',
        'retrieval_failure': 'Retrieval Failure'
    }

    error_types = ['same_key_interference', 'cross_key_interference',
                   'partial_match', 'hallucination', 'retrieval_failure']
    available = [col for col in error_types if col in df_pcts.columns]

    # Left: Stacked area chart
    ax = axes[0]
    ax.stackplot(df_pcts.index,
                 [df_pcts[et].values for et in available],
                 labels=[ERROR_LABELS[et] for et in available],
                 colors=[ERROR_COLORS[et] for et in available],
                 alpha=0.8)

    ax.set_xlabel('Update Count', fontsize=10, fontweight='bold')
    ax.set_ylabel('Error Distribution (%)', fontsize=10, fontweight='bold')
    ax.set_title('Error Type Distribution by Update Count', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)

    # Right: Line chart showing trajectories
    ax = axes[1]
    for et in available:
        ax.plot(df_pcts.index, df_pcts[et],
               marker='o', linewidth=2, markersize=5,
               label=ERROR_LABELS[et], color=ERROR_COLORS[et])

    ax.set_xlabel('Update Count', fontsize=10, fontweight='bold')
    ax.set_ylabel('Error Type Percentage (%)', fontsize=10, fontweight='bold')
    ax.set_title('Error Type Trajectories', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_file}")
    plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_file.with_suffix('.pdf')}")
    plt.close()


def print_category_summary(df_categories):
    """Print summary statistics for each category."""
    print("\n" + "=" * 60)
    print("MODEL CATEGORIZATION SUMMARY")
    print("=" * 60)

    print("\nSize Tiers:")
    for tier in ['XS', 'S', 'M', 'L']:
        tier_models = df_categories[df_categories['size_tier'] == tier]
        if len(tier_models) > 0:
            print(f"   {tier}: {len(tier_models)} models (RIES: {tier_models['ries_score'].mean():.1f} ± {tier_models['ries_score'].std():.1f})")

    print("\nReasoning vs Non-Reasoning:")
    for is_reasoning, label in [(True, "Reasoning"), (False, "Non-Reasoning")]:
        group = df_categories[df_categories['is_reasoning'] == is_reasoning]
        if len(group) > 0:
            print(f"   {label}: {len(group)} models (RIES: {group['ries_score'].mean():.1f} ± {group['ries_score'].std():.1f})")

    print("\nMoE vs Dense:")
    for is_moe, label in [(True, "MoE"), (False, "Dense")]:
        group = df_categories[df_categories['is_moe'] == is_moe]
        if len(group) > 0:
            print(f"   {label}: {len(group)} models (RIES: {group['ries_score'].mean():.1f} ± {group['ries_score'].std():.1f})")


def main():
    print("📊 Visualizing Decay Patterns for Retroactive Interference\n")
    print("=" * 60)

    # Set up paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    results_dir = project_root / 'results' / 'key_results'
    error_dir = project_root / 'results' / 'error_analysis'

    # Load RI data
    print("Loading RI data...")
    ries_file = results_dir / 'ries_analysis.xlsx'
    df_ries = pd.read_excel(ries_file, sheet_name='RIES_Scores')
    df_full = pd.read_excel(ries_file, sheet_name='Full_Data')

    print(f"   {len(df_ries)} models with RIES scores")
    print(f"   {len(df_full)} data points")

    # Load PI data (if available)
    pies_file = results_dir / 'pies_analysis.xlsx'
    df_pi_scores = None
    df_pi_full = None
    if pies_file.exists():
        print("\nLoading PI data...")
        df_pi_scores = pd.read_excel(pies_file, sheet_name='PIES_Scores')
        df_pi_full = pd.read_excel(pies_file, sheet_name='Full_Data')
        print(f"   {len(df_pi_scores)} models with PIES scores")
        print(f"   {len(df_pi_full)} PI data points")
    else:
        print(f"\nNote: PI data not found at {pies_file}")

    # Load error data (if available)
    error_file = error_dir / 'error_classification_detailed.xlsx'
    has_error_data = error_file.exists()
    if has_error_data:
        print(f"\nFound error data at {error_file}")
    else:
        print(f"\nNote: Error data not found at {error_file}")

    # Categorize models
    print("\nCategorizing models...")
    df_categories = categorize_models(df_ries)

    # Print summary
    print_category_summary(df_categories)

    # Generate visualizations
    print("\n📈 Generating visualizations...")

    # 1. Size tiers - horizontal panels
    print("\n1. Size Tiers (horizontal panels)...")
    plot_decay_by_size_tiers(df_full, df_categories,
                            results_dir / 'decay_curves_size_tiers.png')

    # 2. Reasoning vs Non-Reasoning - horizontal panels
    print("\n2. Reasoning vs Non-Reasoning (horizontal panels)...")
    plot_decay_reasoning_vs_nonreasoning(df_full, df_categories,
                                         results_dir / 'decay_curves_reasoning.png')

    # 3. MoE vs Dense - horizontal panels
    print("\n3. MoE vs Dense (horizontal panels)...")
    plot_decay_moe_vs_dense(df_full, df_categories,
                           results_dir / 'decay_curves_moe.png')

    # 4. Combined size comparison (single plot)
    print("\n4. Combined Size Comparison (single plot)...")
    plot_combined_size_comparison(df_full, df_categories,
                                  results_dir / 'decay_curves_size_combined.png')

    # 5. Reasoning comparison (single plot)
    print("\n5. Reasoning Comparison (single plot)...")
    plot_reasoning_comparison(df_full, df_categories,
                             results_dir / 'decay_curves_reasoning_combined.png')

    # 6. Individual model curves (top/bottom performers)
    print("\n6. Individual Model Curves (Top/Bottom performers)...")
    plot_individual_model_curves(df_full, df_ries,
                                 results_dir / 'decay_curves_individual_models.png',
                                 top_n=12)

    # 7. PI decay curves (if data available)
    if df_pi_scores is not None and df_pi_full is not None:
        print("\n7. PI Decay Curves by Size...")
        plot_pi_decay_curves(df_pi_full, df_pi_scores,
                            results_dir / 'pi_decay_curves_size.png')

        # 8. RI vs PI comparison
        print("\n8. RI vs PI Comparison...")
        plot_ri_vs_pi_comparison(df_full, df_pi_full, df_ries, df_pi_scores,
                                results_dir / 'ri_vs_pi_comparison.png')

    # 9. Error patterns (if data available)
    if has_error_data:
        print("\n9. Error Pattern Visualization...")
        plot_error_patterns(error_file, results_dir / 'error_patterns.png')

    # Save categorization
    df_categories.to_excel(results_dir / 'model_categories.xlsx', index=False)
    print(f"\n💾 Saved model categorization to {results_dir / 'model_categories.xlsx'}")

    print("\n" + "=" * 60)
    print("✅ All visualizations complete!")
    print(f"\nOutput directory: {results_dir}")


if __name__ == '__main__':
    main()
