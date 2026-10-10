
import math
import statistics

from scipy.stats import t


def interpret_analysis(analysis, practical_threshold=0.01, exploratory=True):
    """Separate observed effects from interval-supported practical effects."""
    if practical_threshold <= 0:
        raise ValueError("Practical threshold must be positive.")
    mean = analysis["mean_difference"]
    lower, upper = analysis["ci_lower"], analysis["ci_upper"]
    # Zero variance does not establish inferential evidence.
    valid = analysis["corrected_se"] > 0
    return {
        "observed_winner": "random_forest" if mean > 0 else "logistic_regression" if mean < 0 else "tie",
        "practically_meaningful": abs(mean) >= practical_threshold,
        "statistical_direction": (
            "random_forest" if valid and lower > 0 else
            "logistic_regression" if valid and upper < 0 else "inconclusive"
        ),
        "practical_improvement_supported": valid and (
            lower > practical_threshold or upper < -practical_threshold
        ),
        "practical_threshold": practical_threshold,
        "exploratory": exploratory,
        "limitations": "Approximate dependence correction on one reused dataset; independent validation is required.",
    }


def analyze_results(
    results: list[dict],
    n_train: int,
    n_test: int,
    confidence: float = 0.95,
):
    """
    Analyze paired model accuracy differences using
    the Nadeau-Bengio corrected resampled t-test.

    Assumes repeated random train/test splits with
    identical split sizes and paired model scores.
    """

    differences = [row["difference"] for row in results]
    n = len(differences)

    if n < 2:
        raise ValueError("At least 2 paired runs are required.")

    if n_train <= 0 or n_test <= 0:
        raise ValueError("Train/test sizes must be positive.")

    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1.")

    mean_diff = statistics.mean(differences)
    std_diff = statistics.stdev(differences)

    # Standard error assuming independent runs
    naive_se = std_diff / math.sqrt(n)

    # Corrected standard error for repeated holdout
    correction = (1 / n) + (n_test / n_train)

    corrected_se = std_diff * math.sqrt(correction)

    degrees_freedom = n - 1
    alpha = 1 - confidence

    critical_value = t.ppf(
        1 - alpha / 2,
        df=degrees_freedom
    )

    margin = critical_value * corrected_se

    ci_lower = mean_diff - margin
    ci_upper = mean_diff + margin

    if corrected_se == 0:
        # Degenerate case: no observed variability.
        # Do not report an inferential p-value.
        t_statistic = None
        p_value = None
    else:
        t_statistic = mean_diff / corrected_se

        # Two-sided test
        p_value = 2 * t.sf(
            abs(t_statistic),
            df=degrees_freedom
        )

    return {
        "runs": n,
        "mean_difference": mean_diff,
        "std_difference": std_diff,
        "naive_se": naive_se,
        "corrected_se": corrected_se,
        "correction_factor": correction,
        "t_statistic": t_statistic,
        "p_value": p_value,
        "confidence": confidence,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "method": "Nadeau-Bengio corrected resampled t-test",
    }
