from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from mlreality.statistics import analyze_results, interpret_analysis

DASHBOARD = Path(__file__).resolve().parents[1] / 'dashboard.py'


def dashboard_app(tmp_path):
    run = tmp_path / 'results' / 'sample'
    run.mkdir(parents=True)
    for filename in ('benchmark.csv', 'analysis.json', 'provenance.json'):
        (run / filename).write_text('sample download')
    source = DASHBOARD.read_text().replace(
        "RESULTS_DIR = Path(__file__).resolve().parent / 'results'",
        f'RESULTS_DIR = Path({str(run.parent)!r})')
    return AppTest.from_string(source)


def test_saved_experiment_dashboard(tmp_path):
    df = pd.DataFrame({'seed': [42, 43], 'logistic_regression': [0.95, 0.96],
                       'random_forest': [0.94, 0.97], 'difference': [-0.01, 0.01]})
    analysis = analyze_results(df.to_dict('records'), n_train=80, n_test=20)
    report = {'analysis': analysis, 'interpretation': interpret_analysis(analysis)}
    app = dashboard_app(tmp_path)
    with patch('mlreality.validation.load_validated_run', return_value=(df, {}, report)):
        app.run(timeout=20)
    assert not app.exception
    assert [metric.value for metric in app.metric] == ['95.50%', '95.50%', '+0.000 pp', '2']
    assert len(app.get('plotly_chart')) == 3
    assert len(app.get('download_button')) == 3
    assert len(app.json) == 2


def test_invalid_report_displays_error(tmp_path):
    app = dashboard_app(tmp_path)
    with patch('mlreality.validation.load_validated_run', side_effect=ValueError('Invalid report')):
        app.run(timeout=20)
    assert not app.exception
    assert app.error[0].value == 'Invalid report'
    assert not app.metric
    assert not app.get('plotly_chart')
