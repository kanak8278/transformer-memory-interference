#!/usr/bin/env python3
"""
Regression analysis: RIES vs Size Class and Context Length

Replicates the analysis from "Unable to Forget" paper:
- Group models into size classes (XS, S, M, L)
- Test if parameter size or context length predicts RIES
- Compare reasoning vs non-reasoning models
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("="*80)
    print("REGRESSION ANALYSIS: RIES vs Size Class & Context Length")
    print("="*80)

    # Load data - model_categories already has all we need
    df = pd.read_excel('../../results/key_results/model_categories.xlsx')

    print(f"\n📊 Dataset: {len(df)} models with complete data (levels 3-300)")

    # Add size classes as numeric
    size_class_map = {'XS': 1, 'S': 2, 'M': 3, 'L': 4}
    df['size_class_numeric'] = df['size_tier'].map(size_class_map)

    # Add context tokens from RIES file
    df_ries = pd.read_excel('../../results/key_results/ries_analysis.xlsx', sheet_name='RIES_Scores')
    df = pd.merge(df, df_ries[['model_id', 'context_tokens']], on='model_id', how='left')

    print("\n" + "="*80)
    print("1. FULL DATASET ANALYSIS (All 39 Models)")
    print("="*80)

    # Linear regression: RIES ~ size_class + context_length
    from sklearn.linear_model import LinearRegression

    X_full = df[['size_class_numeric', 'context_tokens']].values
    y_full = df['ries_score'].values

    model_full = LinearRegression()
    model_full.fit(X_full, y_full)
    r2_full = model_full.score(X_full, y_full)

    print(f"\n📈 Multiple Linear Regression: RIES ~ size_class + context_length")
    print(f"   R² = {r2_full:.3f} ({r2_full*100:.1f}% variance explained)")
    print(f"   Intercept: {model_full.intercept_:.2f}")
    print(f"   Coefficient (size_class): {model_full.coef_[0]:.2f}")
    print(f"   Coefficient (context): {model_full.coef_[1]:.6f}")

    # T-tests for each coefficient
    from scipy.stats import t as t_dist

    # Residuals
    y_pred = model_full.predict(X_full)
    residuals = y_full - y_pred
    mse = np.mean(residuals**2)

    # Standard errors (simplified)
    n = len(y_full)
    p = 2  # number of predictors
    se = np.sqrt(mse / (n - p - 1))

    # Approximate t-statistics
    t_size = model_full.coef_[0] / (se / np.std(X_full[:, 0]))
    t_context = model_full.coef_[1] / (se / np.std(X_full[:, 1]))

    p_size = 2 * (1 - t_dist.cdf(abs(t_size), n - p - 1))
    p_context = 2 * (1 - t_dist.cdf(abs(t_context), n - p - 1))

    print(f"\n📊 Statistical Significance:")
    print(f"   Size class:      t = {t_size:.3f}, p = {p_size:.4f} {'***' if p_size < 0.001 else '**' if p_size < 0.01 else '*' if p_size < 0.05 else 'ns'}")
    print(f"   Context length:  t = {t_context:.3f}, p = {p_context:.4f} {'***' if p_context < 0.001 else '**' if p_context < 0.01 else '*' if p_context < 0.05 else 'ns'}")

    # Univariate correlations
    print(f"\n📊 Univariate Correlations:")
    r_size, p_size_uni = stats.pearsonr(df['size_class_numeric'], df['ries_score'])
    r_context, p_context_uni = stats.pearsonr(df['context_tokens'], df['ries_score'])

    print(f"   RIES vs size_class:      r = {r_size:+.3f}, R² = {r_size**2:.3f}, p = {p_size_uni:.4f}")
    print(f"   RIES vs context_length:  r = {r_context:+.3f}, R² = {r_context**2:.3f}, p = {p_context_uni:.4f}")

    # Spearman correlation (non-parametric, like paper)
    rho_size, p_spearman_size = stats.spearmanr(df['model_size_b'], df['ries_score'])
    rho_context, p_spearman_context = stats.spearmanr(df['context_tokens'], df['ries_score'])

    print(f"\n📊 Spearman Correlations (non-parametric):")
    print(f"   RIES vs model_size:      ρ = {rho_size:+.3f}, ρ² = {rho_size**2:.3f}, p = {p_spearman_size:.4f}")
    print(f"   RIES vs context_length:  ρ = {rho_context:+.3f}, ρ² = {rho_context**2:.3f}, p = {p_spearman_context:.4f}")

    print("\n" + "="*80)
    print("2. REASONING vs NON-REASONING MODELS")
    print("="*80)

    # Separate analysis
    df_reasoning = df[df['is_reasoning'] == True]
    df_nonreasoning = df[df['is_reasoning'] == False]

    print(f"\n🧠 Reasoning models (n={len(df_reasoning)}):")
    print(f"   Mean RIES: {df_reasoning['ries_score'].mean():.1f} ± {df_reasoning['ries_score'].std():.1f}")
    print(f"   Size range: {df_reasoning['model_size_b'].min():.0f}B - {df_reasoning['model_size_b'].max():.0f}B")

    print(f"\n💡 Non-Reasoning models (n={len(df_nonreasoning)}):")
    print(f"   Mean RIES: {df_nonreasoning['ries_score'].mean():.1f} ± {df_nonreasoning['ries_score'].std():.1f}")
    print(f"   Size range: {df_nonreasoning['model_size_b'].min():.0f}B - {df_nonreasoning['model_size_b'].max():.0f}B")

    # T-test
    t_stat, p_ttest = stats.ttest_ind(df_reasoning['ries_score'], df_nonreasoning['ries_score'])
    print(f"\n📊 T-test (Reasoning vs Non-Reasoning):")
    print(f"   t = {t_stat:.3f}, p = {p_ttest:.4f} {'***' if p_ttest < 0.001 else '**' if p_ttest < 0.01 else '*' if p_ttest < 0.05 else 'ns'}")
    print(f"   Cohen's d = {(df_reasoning['ries_score'].mean() - df_nonreasoning['ries_score'].mean()) / np.sqrt((df_reasoning['ries_score'].std()**2 + df_nonreasoning['ries_score'].std()**2) / 2):.3f}")

    # Regression for non-reasoning only
    print(f"\n📈 Non-Reasoning Models Only (n={len(df_nonreasoning)}):")
    X_nonreason = df_nonreasoning[['size_class_numeric', 'context_tokens']].values
    y_nonreason = df_nonreasoning['ries_score'].values

    model_nonreason = LinearRegression()
    model_nonreason.fit(X_nonreason, y_nonreason)
    r2_nonreason = model_nonreason.score(X_nonreason, y_nonreason)

    print(f"   R² = {r2_nonreason:.3f} (multiple regression)")

    r_size_nr, p_size_nr = stats.pearsonr(df_nonreasoning['size_class_numeric'], df_nonreasoning['ries_score'])
    r_context_nr, p_context_nr = stats.pearsonr(df_nonreasoning['context_tokens'], df_nonreasoning['ries_score'])

    print(f"   RIES vs size_class:      r = {r_size_nr:+.3f}, R² = {r_size_nr**2:.3f}, p = {p_size_nr:.4f}")
    print(f"   RIES vs context_length:  r = {r_context_nr:+.3f}, R² = {r_context_nr**2:.3f}, p = {p_context_nr:.4f}")

    print("\n" + "="*80)
    print("3. SIZE TIER COMPARISON")
    print("="*80)

    print(f"\n📊 Mean RIES by Size Tier:")
    for tier in ['XS', 'S', 'M', 'L']:
        tier_data = df[df['size_tier'] == tier]
        print(f"   {tier}: {tier_data['ries_score'].mean():.1f} ± {tier_data['ries_score'].std():.1f} (n={len(tier_data)})")

    # ANOVA
    groups = [df[df['size_tier'] == tier]['ries_score'].values for tier in ['XS', 'S', 'M', 'L']]
    f_stat, p_anova = stats.f_oneway(*groups)

    print(f"\n📊 ANOVA (Size Tier Effect):")
    print(f"   F = {f_stat:.3f}, p = {p_anova:.4f} {'***' if p_anova < 0.001 else '**' if p_anova < 0.01 else '*' if p_anova < 0.05 else 'ns'}")

    print("\n" + "="*80)
    print("4. COMPARISON WITH 'UNABLE TO FORGET' PAPER")
    print("="*80)

    print(f"\n📝 Paper (Proactive Interference, IES):")
    print(f"   Dataset: 30 open-source models, excluded reasoning models")
    print(f"   Size class predictor:    t = 3.03, p = 0.005 (significant)")
    print(f"   Context length:          t = -0.144, p = 0.886 (not significant)")
    print(f"   Combined R²:             0.261 (26.1% variance explained)")
    print(f"   Spearman (128k models):  ρ² = 0.673 (strong correlation)")

    print(f"\n📝 Our Study (Retroactive Interference, RIES):")
    print(f"   Dataset: 39 models (all available with complete data)")
    print(f"   Size class correlation:  r = {r_size:+.3f}, R² = {r_size**2:.3f}, p = {p_size_uni:.4f}")
    print(f"   Context length:          r = {r_context:+.3f}, R² = {r_context**2:.3f}, p = {p_context_uni:.4f}")
    print(f"   Combined R²:             {r2_full:.3f} ({r2_full*100:.1f}% variance explained)")
    print(f"   Spearman (model size):   ρ² = {rho_size**2:.3f}")

    print(f"\n💡 Key Findings:")
    if r_size**2 > 0.1 and p_size_uni < 0.05:
        print(f"   ✓ Size class DOES predict RIES (R² = {r_size**2:.3f}, p < 0.05)")
    else:
        print(f"   ⚠️  Size class weakly predicts RIES (R² = {r_size**2:.3f})")

    if r_context**2 < 0.1 or p_context_uni > 0.05:
        print(f"   ✓ Context length does NOT predict RIES - MATCHES PAPER")
    else:
        print(f"   ⚠️  Context length shows unexpected correlation")

    if t_stat > 2 and p_ttest < 0.05:
        print(f"   ✓ Reasoning models significantly outperform non-reasoning (p < 0.05)")
        print(f"      → Novel finding: CoT models resist retroactive interference!")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: RIES vs Size Class
    ax = axes[0, 0]
    for tier in ['XS', 'S', 'M', 'L']:
        tier_data = df[df['size_tier'] == tier]
        reasoning_mask = tier_data['is_reasoning']
        ax.scatter(tier_data[~reasoning_mask]['size_class_numeric'],
                  tier_data[~reasoning_mask]['ries_score'],
                  s=100, alpha=0.6, label=f'{tier} (Non-CoT)')
        ax.scatter(tier_data[reasoning_mask]['size_class_numeric'],
                  tier_data[reasoning_mask]['ries_score'],
                  s=100, marker='*', alpha=0.8, label=f'{tier} (CoT)')

    # Regression line
    X_plot = np.array([1, 2, 3, 4]).reshape(-1, 1)
    y_plot = model_full.intercept_ + model_full.coef_[0] * X_plot.flatten()
    ax.plot(X_plot, y_plot, 'r--', linewidth=2, label=f'Regression (R²={r2_full:.3f})')

    ax.set_xlabel('Size Class (1=XS, 2=S, 3=M, 4=L)', fontsize=12)
    ax.set_ylabel('RIES Score', fontsize=12)
    ax.set_title('RIES vs Size Class', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: RIES vs Context Length
    ax = axes[0, 1]
    reasoning_mask = df['is_reasoning']
    ax.scatter(df[~reasoning_mask]['context_tokens'],
              df[~reasoning_mask]['ries_score'],
              s=100, alpha=0.6, label='Non-CoT')
    ax.scatter(df[reasoning_mask]['context_tokens'],
              df[reasoning_mask]['ries_score'],
              s=100, marker='*', alpha=0.8, label='CoT')
    ax.set_xlabel('Context Length (tokens)', fontsize=12)
    ax.set_ylabel('RIES Score', fontsize=12)
    ax.set_title(f'RIES vs Context Length (r={r_context:.3f})', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: RIES by Tier (boxplot)
    ax = axes[1, 0]
    tier_data_list = [df[df['size_tier'] == tier]['ries_score'].values for tier in ['XS', 'S', 'M', 'L']]
    ax.boxplot(tier_data_list, labels=['XS', 'S', 'M', 'L'])
    ax.set_xlabel('Size Tier', fontsize=12)
    ax.set_ylabel('RIES Score', fontsize=12)
    ax.set_title(f'RIES Distribution by Size (ANOVA p={p_anova:.4f})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Reasoning vs Non-Reasoning
    ax = axes[1, 1]
    ax.boxplot([df_nonreasoning['ries_score'].values, df_reasoning['ries_score'].values],
               labels=['Non-CoT', 'CoT'])
    ax.set_ylabel('RIES Score', fontsize=12)
    ax.set_title(f'Reasoning vs Non-Reasoning (p={p_ttest:.4f})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('../../results/key_results/regression_analysis_results.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved visualization: ../../results/key_results/regression_analysis_results.png")

    # Save results
    results = {
        'metric': ['R² (full model)', 'R² (size only)', 'R² (context only)',
                   'Correlation (size)', 'Correlation (context)',
                   'Spearman ρ² (size)', 'Spearman ρ² (context)',
                   'Mean RIES (reasoning)', 'Mean RIES (non-reasoning)',
                   'T-test p-value', 'ANOVA p-value'],
        'value': [r2_full, r_size**2, r_context**2,
                  r_size, r_context,
                  rho_size**2, rho_context**2,
                  df_reasoning['ries_score'].mean(), df_nonreasoning['ries_score'].mean(),
                  p_ttest, p_anova]
    }
    pd.DataFrame(results).to_excel('../../results/key_results/regression_analysis_results.xlsx', index=False)
    print(f"💾 Saved results: ../../results/key_results/regression_analysis_results.xlsx")

    print("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()
