"""
Evaluation utilities for interference experiments.

Provides error classification and bootstrap confidence intervals.
Used by all behavioral sweep scripts and experiments.

Usage:
    from core.evaluation import classify_error, bootstrap_ci

    error_type = classify_error(predicted, expected, initial, final, all_values, "PI")
    # → "correct", "primacy_intrusion", "recency_intrusion",
    #   "intermediate_intrusion", or "garbage"

    mean, ci_lo, ci_hi = bootstrap_ci([True, False, True, True])
    # → (0.75, 0.25, 1.0)
"""

import numpy as np


def classify_error(predicted, expected, initial_value, final_value,
                   all_values, condition):
    """Classify a model's answer into an error category.

    Error types (outcome labels — describe WHAT the model output, not WHY):
      - correct: answer matches expected value
      - primacy_intrusion (PI only): model output the initial value when asked for final
      - recency_intrusion (RI only): model output the final value when asked for initial
      - intermediate_intrusion: model output some intermediate value from the stream
      - garbage: model output something not in the value stream at all

    Note: "primacy_intrusion" and "recency_intrusion" are outcome classifications,
    not mechanism claims. Whether primacy intrusion is caused by initial-value promotion
    or final-value suppression is determined by mechanistic experiments (exp 12, 14, 21a).

    Args:
        predicted: Model's generated answer string.
        expected: The correct answer string.
        initial_value: First value assigned to the test category.
        final_value: Last value assigned to the test category.
        all_values: All values for the test category in sequence order.
        condition: "RI" or "PI".

    Returns:
        Error type string.
    """
    pred_lower = predicted.lower().strip()
    exp_lower = expected.lower().strip()

    if exp_lower in pred_lower or pred_lower.startswith(exp_lower):
        return "correct"

    init_lower = initial_value.lower()
    final_lower = final_value.lower()

    if condition == "PI" and (init_lower in pred_lower or pred_lower.startswith(init_lower)):
        return "primacy_intrusion"

    if condition == "RI" and (final_lower in pred_lower or pred_lower.startswith(final_lower)):
        return "recency_intrusion"

    for val in all_values:
        if val.lower() != exp_lower and (
            val.lower() in pred_lower or pred_lower.startswith(val.lower())
        ):
            return "intermediate_intrusion"

    return "garbage"


def bootstrap_ci(data, n_bootstrap=2000, ci=0.95):
    """Compute bootstrap confidence interval for a list of binary outcomes.

    Args:
        data: List of True/False or 1/0 values.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (default 0.95 → 95% CI).

    Returns:
        (mean, ci_lower, ci_upper)
    """
    if not data:
        return 0.0, 0.0, 0.0

    arr = np.array(data, dtype=float)
    mean = arr.mean()

    if len(arr) < 3:
        return mean, 0.0, 1.0

    rng_np = np.random.RandomState(42)
    boot_means = [
        rng_np.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_bootstrap)
    ]
    alpha = (1 - ci) / 2
    return (
        mean,
        np.percentile(boot_means, alpha * 100),
        np.percentile(boot_means, (1 - alpha) * 100),
    )
