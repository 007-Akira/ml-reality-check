
import pytest

from mlreality.statistics import analyze_results, interpret_analysis


def test_identical_models():
    results = [
        {"difference": 0.0},
        {"difference": 0.0},
        {"difference": 0.0},
    ]

    analysis = analyze_results(
        results,
        n_train=80,
        n_test=20,
    )

    assert analysis["mean_difference"] == 0
    assert analysis["corrected_se"] == 0
    assert analysis["p_value"] is None


def test_negative_difference():
    results = [
        {"difference": -0.01},
        {"difference": -0.02},
        {"difference": -0.03},
        {"difference": -0.01},
    ]

    analysis = analyze_results(
        results,
        n_train=80,
        n_test=20,
    )

    assert analysis["mean_difference"] < 0
    assert analysis["corrected_se"] > analysis["naive_se"]
    assert 0 <= analysis["p_value"] <= 1


def test_single_run_rejected():
    with pytest.raises(ValueError):
        analyze_results(
            [{"difference": -0.01}],
            n_train=80,
            n_test=20,
        )


def test_direction_does_not_establish_practical_improvement():
    interpretation = interpret_analysis({
        "mean_difference": -0.01754,
        "ci_lower": -0.03251,
        "ci_upper": -0.00258,
        "corrected_se": 0.00715,
    })
    assert interpretation["observed_winner"] == "logistic_regression"
    assert interpretation["statistical_direction"] == "logistic_regression"
    assert interpretation["practically_meaningful"]
    assert not interpretation["practical_improvement_supported"]
