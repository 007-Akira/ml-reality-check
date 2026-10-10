from mlreality.benchmark import run_benchmark


def test_repeatability():
    first_results, first_meta = run_benchmark(seeds=2)
    second_results, second_meta = run_benchmark(seeds=2)
    assert first_results == second_results
    assert first_meta["dataset"]["sha256"] == second_meta["dataset"]["sha256"]
    assert first_meta["splits"] == second_meta["splits"]


def test_paired_difference():
    results, _ = run_benchmark(seeds=2)
    for row in results:
        assert abs(row["difference"] - (row["random_forest"] - row["logistic_regression"])) < 1e-12
