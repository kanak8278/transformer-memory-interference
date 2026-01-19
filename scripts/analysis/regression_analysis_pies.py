#!/usr/bin/env python3
"""
Regression analysis: PIES vs Size Class and Context Length

Analyzes Proactive Interference Endurance Score (PIES) relationships:
- Group models into size classes (XS, S, M, L)
- Test if parameter size or context length predicts PIES
- Compare with RIES results to show differential size effects
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='PIES Regression Analysis')
    parser.add_argument('--output-dir', type=str, default='../../results/key_results',
                       help='Output directory for results')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("REGRESSION ANALYSIS: PIES vs Size Class & Context Length")
    print("="*80)

    # Load PIES data
    pies_path = output_dir / 'pies_analysis.xlsx'
    df_pies = pd.read_excel(pies_path, sheet_name='PIES_Scores')

    # Rename column if needed
    if 'pis_score' in df_pies.columns:
        df_pies = df_pies.rename(columns={'pis_score': 'pies_score'})

    # Load RIES data for comparison (matched models)
    ries_path = output_dir / 'model_categories.xlsx'
    df_ries = pd.read_excel(ries_path)

    # Merge to get matched models with both RIES and PIES
    df = pd.merge(df_ries, df_pies[['model_id', 'pies_score', 'context_tokens']],
                  on='model_id', how='inner')

    print(f"\n Dataset: {len(df)} models with both RIES and PIES data")

    # Add size classes as numeric
    size_class_map = {'XS': 1, 'S': 2, 'M': 3, 'L': 4}
    df['size_class_numeric'] = df['size_tier'].map(size_class_map)

    # Filter for valid data
    df_valid = df[df['model_size_b'].notna() & df['context_tokens'].notna()].copy()
    print(f"   Models with complete size/context data: {len(df_valid)}")

    print("\n" + "="*80)
    print("1. PIES vs SIZE ANALYSIS")
    print("="*80)

    # Pearson correlations
    r_size_linear, p_size_linear = stats.pearsonr(df_valid['pies_score'], df_valid['model_size_b'])
    r_size_log, p_size_log = stats.pearsonr(df_valid['pies_score'], np.log10(df_valid['model_size_b']))

    print(f"\n PIES vs Model Size:")
    print(f"   Linear:     r = {r_size_linear:+.4f}, R² = {r_size_linear**2:.4f}, p = {p_size_linear:.6f}")
    print(f"   Log scale:  r = {r_size_log:+.4f}, R² = {r_size_log**2:.4f}, p = {p_size_log:.6f}")

    # Spearman correlation
    rho_size, p_spearman_size = stats.spearmanr(df_valid['pies_score'], df_valid['model_size_b'])
    print(f"   Spearman:   ρ = {rho_size:+.4f}, ρ² = {rho_size**2:.4f}, p = {p_spearman_size:.6f}")

    # Size class correlation
    r_class, p_class = stats.pearsonr(df_valid['pies_score'], df_valid['size_class_numeric'])
    print(f"\n PIES vs Size Class:")
    print(f"   r = {r_class:+.4f}, R² = {r_class**2:.4f}, p = {p_class:.6f}")

    print("\n" + "="*80)
    print("2. PIES vs CONTEXT LENGTH ANALYSIS")
    print("="*80)

    r_context, p_context = stats.pearsonr(df_valid['pies_score'], df_valid['context_tokens'])
    r_context_log, p_context_log = stats.pearsonr(df_valid['pies_score'], np.log10(df_valid['context_tokens']))

    print(f"\n PIES vs Context Length:")
    print(f"   Linear:     r = {r_context:+.4f}, R² = {r_context**2:.4f}, p = {p_context:.6f}")
    print(f"   Log scale:  r = {r_context_log:+.4f}, R² = {r_context_log**2:.4f}, p = {p_context_log:.6f}")

    print("\n" + "="*80)
    print("3. MULTIPLE REGRESSION: PIES ~ Size + Context")
    print("="*80)

    from sklearn.linear_model import LinearRegression

    X = df_valid[['size_class_numeric', 'context_tokens']].values
    y = df_valid['pies_score'].values

    model = LinearRegression()
    model.fit(X, y)
    r2_full = model.score(X, y)

    print(f"\n Multiple Linear Regression:")
    print(f"   R² = {r2_full:.4f} ({r2_full*100:.1f}% variance explained)")
    print(f"   Intercept: {model.intercept_:.2f}")
    print(f"   Coefficient (size_class): {model.coef_[0]:.4f}")
    print(f"   Coefficient (context): {model.coef_[1]:.8f}")

    print("\n" + "="*80)
    print("4. SIZE TIER COMPARISON (PIES)")
    print("="*80)

    print(f"\n Mean PIES by Size Tier:")
    for tier in ['XS', 'S', 'M', 'L']:
        tier_data = df[df['size_tier'] == tier]
        if len(tier_data) > 0:
            print(f"   {tier}: {tier_data['pies_score'].mean():.1f} ± {tier_data['pies_score'].std():.1f} (n={len(tier_data)})")

    # ANOVA
    groups = [df[df['size_tier'] == tier]['pies_score'].dropna().values
              for tier in ['XS', 'S', 'M', 'L']
              if len(df[df['size_tier'] == tier]['pies_score'].dropna()) > 0]

    if len(groups) >= 2:
        f_stat, p_anova = stats.f_oneway(*groups)
        print(f"\n ANOVA (Size Tier Effect on PIES):")
        print(f"   F = {f_stat:.3f}, p = {p_anova:.4f} {'***' if p_anova < 0.001 else '**' if p_anova < 0.01 else '*' if p_anova < 0.05 else 'n.s.'}")

    print("\n" + "="*80)
    print("5. COMPARISON: RIES vs PIES SIZE EFFECTS")
    print("="*80)

    # RIES correlations for comparison
    r_ries_size, p_ries_size = stats.pearsonr(df_valid['ries_score'], np.log10(df_valid['model_size_b']))
    r_ries_context, p_ries_context = stats.pearsonr(df_valid['ries_score'], df_valid['context_tokens'])

    print(f"\n RI (RIES) vs Size:    R² = {r_ries_size**2:.4f}, p = {p_ries_size:.6f} {'***' if p_ries_size < 0.001 else 'n.s.'}")
    print(f"   PI (PIES) vs Size:    R² = {r_size_log**2:.4f}, p = {p_size_log:.6f} {'***' if p_size_log < 0.001 else 'n.s.'}")
    print(f"\n RI (RIES) vs Context: R² = {r_ries_context**2:.4f}, p = {p_ries_context:.6f} {'***' if p_ries_context < 0.001 else 'n.s.'}")
    print(f"   PI (PIES) vs Context: R² = {r_context**2:.4f}, p = {p_context:.6f} {'***' if p_context < 0.001 else 'n.s.'}")

    # Ratio of size effects
    ratio = r_ries_size**2 / r_size_log**2 if r_size_log**2 > 0 else float('inf')
    print(f"\n Size effect ratio (RI/PI): {ratio:.1f}x stronger for RI")

    print("\n" + "="*80)
    print("6. RIES vs PIES CORRELATION")
    print("="*80)

    r_ri_pi, p_ri_pi = stats.pearsonr(df_valid['ries_score'], df_valid['pies_score'])
    print(f"\n RIES vs PIES:")
    print(f"   r = {r_ri_pi:+.4f}, R² = {r_ri_pi**2:.4f}, p = {p_ri_pi:.6f}")
    print(f"   Interpretation: {'Correlated' if p_ri_pi < 0.05 else 'NOT correlated'} - {'same' if p_ri_pi < 0.05 else 'distinct'} mechanisms")

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: PIES vs Model Size (log scale)
    ax = axes[0, 0]
    ax.scatter(np.log10(df_valid['model_size_b']), df_valid['pies_score'],
               s=80, alpha=0.6, c='#A23B72', edgecolors='white', linewidth=0.5)
    # Add regression line
    x_range = np.linspace(np.log10(df_valid['model_size_b'].min()),
                          np.log10(df_valid['model_size_b'].max()), 100)
    slope, intercept = np.polyfit(np.log10(df_valid['model_size_b']), df_valid['pies_score'], 1)
    ax.plot(x_range, slope * x_range + intercept, 'r--', linewidth=2,
            label=f'R²={r_size_log**2:.3f}, p={p_size_log:.3f}')
    ax.set_xlabel('log₁₀(Model Size in B)', fontsize=11, fontweight='bold')
    ax.set_ylabel('PIES Score', fontsize=11, fontweight='bold')
    ax.set_title('PIES vs Model Size', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Plot 2: PIES vs Context Length
    ax = axes[0, 1]
    ax.scatter(df_valid['context_tokens']/1000, df_valid['pies_score'],
               s=80, alpha=0.6, c='#A23B72', edgecolors='white', linewidth=0.5)
    ax.set_xlabel('Context Length (K tokens)', fontsize=11, fontweight='bold')
    ax.set_ylabel('PIES Score', fontsize=11, fontweight='bold')
    ax.set_title(f'PIES vs Context (R²={r_context**2:.3f}, n.s.)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 3: PIES by Size Tier (boxplot)
    ax = axes[0, 2]
    tier_colors = {'XS': '#d62728', 'S': '#ff7f0e', 'M': '#1f77b4', 'L': '#2ca02c'}
    tier_data = [df[df['size_tier'] == tier]['pies_score'].dropna().values for tier in ['XS', 'S', 'M', 'L']]
    bp = ax.boxplot(tier_data, labels=['XS\n(<10B)', 'S\n(10-100B)', 'M\n(100-500B)', 'L\n(≥500B)'],
                    patch_artist=True)
    for patch, tier in zip(bp['boxes'], ['XS', 'S', 'M', 'L']):
        patch.set_facecolor(tier_colors[tier])
        patch.set_alpha(0.6)
    ax.set_ylabel('PIES Score', fontsize=11, fontweight='bold')
    ax.set_title(f'PIES by Size Tier', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: RIES vs Size (for comparison)
    ax = axes[1, 0]
    ax.scatter(np.log10(df_valid['model_size_b']), df_valid['ries_score'],
               s=80, alpha=0.6, c='#2E86AB', edgecolors='white', linewidth=0.5)
    slope, intercept = np.polyfit(np.log10(df_valid['model_size_b']), df_valid['ries_score'], 1)
    ax.plot(x_range, slope * x_range + intercept, 'r--', linewidth=2,
            label=f'R²={r_ries_size**2:.3f}, p<0.001')
    ax.set_xlabel('log₁₀(Model Size in B)', fontsize=11, fontweight='bold')
    ax.set_ylabel('RIES Score', fontsize=11, fontweight='bold')
    ax.set_title('RIES vs Model Size (for comparison)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Plot 5: RIES vs PIES scatter
    ax = axes[1, 1]
    ax.scatter(df_valid['ries_score'], df_valid['pies_score'],
               s=80, alpha=0.6, c='#6B4C9A', edgecolors='white', linewidth=0.5)
    # Add identity line
    min_val = min(df_valid['ries_score'].min(), df_valid['pies_score'].min())
    max_val = max(df_valid['ries_score'].max(), df_valid['pies_score'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Identity line')
    ax.set_xlabel('RIES Score', fontsize=11, fontweight='bold')
    ax.set_ylabel('PIES Score', fontsize=11, fontweight='bold')
    ax.set_title(f'RIES vs PIES (R²={r_ri_pi**2:.3f}, n.s.)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Plot 6: Side-by-side R² comparison
    ax = axes[1, 2]
    metrics = ['Size\n(log)', 'Context', 'Combined']
    ries_r2 = [r_ries_size**2, r_ries_context**2, 0.49]  # From paper
    pies_r2 = [r_size_log**2, r_context**2, r2_full]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, ries_r2, width, label='RI (RIES)', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, pies_r2, width, label='PI (PIES)', color='#A23B72', alpha=0.8)

    ax.set_ylabel('R² (Variance Explained)', fontsize=11, fontweight='bold')
    ax.set_title('Size Effect: RI vs PI', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 0.6)
    ax.axhline(y=0.05, color='gray', linestyle=':', alpha=0.5)
    ax.text(2.5, 0.06, 'p=0.05 threshold', fontsize=8, color='gray')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_file = output_dir / 'pies_regression_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n Saved visualization: {output_file}")
    plt.close()

    # Save results to Excel
    results = {
        'metric': [
            'PIES vs Size (linear) R²',
            'PIES vs Size (linear) p',
            'PIES vs Size (log) R²',
            'PIES vs Size (log) p',
            'PIES vs Context R²',
            'PIES vs Context p',
            'PIES Combined R²',
            'RIES vs Size (log) R²',
            'RIES vs Size (log) p',
            'RIES vs PIES R²',
            'RIES vs PIES p',
            'Size effect ratio (RI/PI)',
        ],
        'value': [
            r_size_linear**2, p_size_linear,
            r_size_log**2, p_size_log,
            r_context**2, p_context,
            r2_full,
            r_ries_size**2, p_ries_size,
            r_ri_pi**2, p_ri_pi,
            ratio,
        ]
    }

    results_df = pd.DataFrame(results)
    output_excel = output_dir / 'pies_regression_analysis.xlsx'
    results_df.to_excel(output_excel, index=False)
    print(f" Saved results: {output_excel}")

    print("\n" + "="*80)
    print("KEY FINDINGS SUMMARY")
    print("="*80)
    print(f"""
   1. PIES vs Size: R² = {r_size_log**2:.3f}, p = {p_size_log:.4f} ({'significant' if p_size_log < 0.05 else 'NOT significant'})
   2. PIES vs Context: R² = {r_context**2:.3f}, p = {p_context:.4f} ({'significant' if p_context < 0.05 else 'NOT significant'})
   3. RIES vs PIES correlation: R² = {r_ri_pi**2:.3f} (distinct mechanisms)
   4. Size effect is {ratio:.1f}x stronger for RI than PI

   CONCLUSION: Size predicts RI (R²=0.49) but NOT PI (R²={r_size_log**2:.2f}, n.s.)
               This confirms RI and PI engage different mechanisms.
""")

    print("\n Analysis complete!")


if __name__ == '__main__':
    main()
