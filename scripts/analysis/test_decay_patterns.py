#!/usr/bin/env python3
"""
Test Multiple Decay Patterns for Retroactive Interference.

Tests various functional forms:
1. Log-Linear: accuracy = a - b * log(x+1)
2. Log-Quadratic: accuracy = a - b * log(x+1) - c * log²(x+1)
3. Exponential: accuracy = a * exp(-b * x)
4. Power Law: accuracy = a * x^(-b)
5. Logarithmic: accuracy = a - b * log(x)
6. Polynomial (quadratic): accuracy = a - b*x - c*x²

Compares models by AIC (Akaike Information Criterion) to find best fit.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")

def log_linear(x, a, b):
    """accuracy = a - b * log(x+1)"""
    return a - b * np.log10(x + 1)

def log_quadratic(x, a, b, c):
    """accuracy = a - b * log(x+1) - c * log²(x+1)"""
    log_x = np.log10(x + 1)
    return a - b * log_x - c * (log_x ** 2)

def exponential_decay(x, a, b):
    """accuracy = a * exp(-b * x)"""
    return a * np.exp(-b * x)

def power_law(x, a, b):
    """accuracy = a * (x+1)^(-b)"""
    return a * np.power(x + 1, -b)

def logarithmic(x, a, b):
    """accuracy = a - b * log(x+1)"""
    return a - b * np.log(x + 1)

def polynomial_quadratic(x, a, b, c):
    """accuracy = a - b*x - c*x²"""
    return a - b * x - c * (x ** 2)


def calculate_aic(residuals, n_params, n_points):
    """Calculate Akaike Information Criterion"""
    rss = np.sum(residuals ** 2)
    aic = n_points * np.log(rss / n_points) + 2 * n_params
    return aic


def fit_all_models(x, y):
    """
    Fit all functional forms and return best model by AIC.

    Returns:
        dict with fit results for each model
    """
    results = {}
    n_points = len(x)

    # 1. Log-Linear
    try:
        popt, _ = curve_fit(log_linear, x, y, maxfev=10000)
        y_pred = log_linear(x, *popt)
        residuals = y - y_pred
        r_squared = 1 - (np.sum(residuals**2) / np.sum((y - np.mean(y))**2))
        aic = calculate_aic(residuals, 2, n_points)
        results['log_linear'] = {
            'params': popt,
            'r_squared': r_squared,
            'aic': aic,
            'rmse': np.sqrt(np.mean(residuals**2)),
            'name': 'Log-Linear'
        }
    except:
        results['log_linear'] = None

    # 2. Log-Quadratic
    try:
        popt, _ = curve_fit(log_quadratic, x, y, maxfev=10000)
        y_pred = log_quadratic(x, *popt)
        residuals = y - y_pred
        r_squared = 1 - (np.sum(residuals**2) / np.sum((y - np.mean(y))**2))
        aic = calculate_aic(residuals, 3, n_points)
        results['log_quadratic'] = {
            'params': popt,
            'r_squared': r_squared,
            'aic': aic,
            'rmse': np.sqrt(np.mean(residuals**2)),
            'name': 'Log-Quadratic'
        }
    except:
        results['log_quadratic'] = None

    # 3. Exponential
    try:
        # Normalize x for numerical stability
        x_norm = x / x.max()
        popt, _ = curve_fit(exponential_decay, x_norm, y, maxfev=10000, p0=[100, 1])
        y_pred = exponential_decay(x_norm, *popt)
        residuals = y - y_pred
        r_squared = 1 - (np.sum(residuals**2) / np.sum((y - np.mean(y))**2))
        aic = calculate_aic(residuals, 2, n_points)
        results['exponential'] = {
            'params': popt,
            'r_squared': r_squared,
            'aic': aic,
            'rmse': np.sqrt(np.mean(residuals**2)),
            'name': 'Exponential'
        }
    except:
        results['exponential'] = None

    # 4. Power Law
    try:
        popt, _ = curve_fit(power_law, x, y, maxfev=10000, p0=[100, 0.5])
        y_pred = power_law(x, *popt)
        residuals = y - y_pred
        r_squared = 1 - (np.sum(residuals**2) / np.sum((y - np.mean(y))**2))
        aic = calculate_aic(residuals, 2, n_points)
        results['power_law'] = {
            'params': popt,
            'r_squared': r_squared,
            'aic': aic,
            'rmse': np.sqrt(np.mean(residuals**2)),
            'name': 'Power Law'
        }
    except:
        results['power_law'] = None

    # 5. Polynomial Quadratic
    try:
        # Normalize x for numerical stability
        x_norm = x / x.max()
        popt, _ = curve_fit(polynomial_quadratic, x_norm, y, maxfev=10000)
        y_pred = polynomial_quadratic(x_norm, *popt)
        residuals = y - y_pred
        r_squared = 1 - (np.sum(residuals**2) / np.sum((y - np.mean(y))**2))
        aic = calculate_aic(residuals, 3, n_points)
        results['polynomial'] = {
            'params': popt,
            'r_squared': r_squared,
            'aic': aic,
            'rmse': np.sqrt(np.mean(residuals**2)),
            'name': 'Polynomial'
        }
    except:
        results['polynomial'] = None

    # Find best model by AIC (lower is better)
    valid_models = {k: v for k, v in results.items() if v is not None}
    if valid_models:
        best_model = min(valid_models.keys(), key=lambda k: valid_models[k]['aic'])
        results['best_model'] = best_model
    else:
        results['best_model'] = None

    return results


def main():
    print("=" * 80)
    print("Testing Multiple Decay Patterns for Retroactive Interference")
    print("=" * 80)

    # Load data
    print("\n📊 Loading data...")
    df = pd.read_excel('ries_analysis_levels_3_400.xlsx', sheet_name='Full_Data')

    print(f"   Found {len(df)} data points for {df['model_id'].nunique()} models")

    # Analyze each model
    print("\n🔬 Fitting multiple decay patterns...")
    print("=" * 80)

    all_results = []

    for model_id in sorted(df['model_id'].unique()):
        model_data = df[df['model_id'] == model_id].copy()

        if len(model_data) < 4:  # Need at least 4 points for quadratic models
            continue

        x = model_data['interference_level'].values
        y = model_data['accuracy'].values

        # Fit all models
        fit_results = fit_all_models(x, y)

        if fit_results['best_model'] is None:
            continue

        best = fit_results[fit_results['best_model']]

        # Collect results
        result_row = {
            'model_id': model_id,
            'best_pattern': best['name'],
            'best_r_squared': best['r_squared'],
            'best_aic': best['aic'],
            'best_rmse': best['rmse'],
            'n_points': len(model_data),
        }

        # Add R² for each pattern
        for pattern_name in ['log_linear', 'log_quadratic', 'exponential', 'power_law', 'polynomial']:
            if fit_results.get(pattern_name):
                result_row[f'{pattern_name}_r2'] = fit_results[pattern_name]['r_squared']
                result_row[f'{pattern_name}_aic'] = fit_results[pattern_name]['aic']
            else:
                result_row[f'{pattern_name}_r2'] = np.nan
                result_row[f'{pattern_name}_aic'] = np.nan

        all_results.append(result_row)

        # Print summary
        print(f"{model_id:30s} Best: {best['name']:15s} R²={best['r_squared']:.3f} AIC={best['aic']:.1f}")

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Summary
    print("\n" + "=" * 80)
    print("Summary: Best-Fit Pattern Distribution")
    print("=" * 80)

    pattern_counts = results_df['best_pattern'].value_counts()
    for pattern, count in pattern_counts.items():
        pct = 100 * count / len(results_df)
        print(f"{pattern:20s}: {count:2d} models ({pct:5.1f}%)")

    # Average R² by pattern
    print("\n" + "=" * 80)
    print("Average R² by Pattern (across all models)")
    print("=" * 80)

    for pattern in ['log_linear', 'log_quadratic', 'exponential', 'power_law', 'polynomial']:
        col = f'{pattern}_r2'
        if col in results_df.columns:
            mean_r2 = results_df[col].mean()
            print(f"{pattern:20s}: R² = {mean_r2:.3f}")

    # Save results
    results_df.to_excel('visualizations/statistical_tests/decay_pattern_comparison.xlsx', index=False)
    print(f"\n✓ Saved: visualizations/statistical_tests/decay_pattern_comparison.xlsx")

    print("\n✅ Pattern analysis complete!")
    print("\n📝 Next: Generating individual decay curve visualizations...")


if __name__ == '__main__':
    main()
