import json

import pandas as pd
import pytest

from mlreality.validation import load_validated_run


@pytest.fixture
def saved_run(tmp_path):
    df = pd.DataFrame({'seed': [42, 43], 'logistic_regression': [0.95, 0.96],
                       'random_forest': [0.94, 0.97], 'difference': [-0.01, 0.01]})
    df.to_csv(tmp_path / 'benchmark.csv', index=False)
    (tmp_path / 'provenance.json').write_text(json.dumps({
        'experiment': {'splitting_seeds': [42, 43]}, 'splits': [{}, {}]}))
    (tmp_path / 'analysis.json').write_text(json.dumps({
        'analysis': {'runs': 2, 'mean_difference': 0.0}}))
    return tmp_path


def test_valid_report(saved_run):
    df, _, _ = load_validated_run(saved_run)
    assert len(df) == 2


@pytest.mark.parametrize('column,values', [
    ('difference', [0.5, 0.01]), ('seed', [42, 42]),
    ('random_forest', [1.1, 0.97]), ('difference', [float('inf'), 0.01]),
    ('logistic_regression', [None, 0.96]),
])
def test_invalid_measurements(saved_run, column, values):
    path = saved_run / 'benchmark.csv'
    df = pd.read_csv(path)
    df[column] = values
    df.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_validated_run(saved_run)


@pytest.mark.parametrize('field,value', [('runs', 3), ('mean_difference', 0.2)])
def test_analysis_mismatch(saved_run, field, value):
    path = saved_run / 'analysis.json'
    report = json.loads(path.read_text())
    report['analysis'][field] = value
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        load_validated_run(saved_run)


def test_seed_mismatch(saved_run):
    path = saved_run / 'provenance.json'
    provenance = json.loads(path.read_text())
    provenance['experiment']['splitting_seeds'] = [43, 42]
    path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError):
        load_validated_run(saved_run)


def test_missing_artifact(saved_run):
    (saved_run / 'analysis.json').unlink()
    with pytest.raises(ValueError):
        load_validated_run(saved_run)
