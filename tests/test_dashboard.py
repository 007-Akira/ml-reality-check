from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from mlreality.dashboard_ui import accuracy_figure, difference_figure, interval_figure
from mlreality.statistics import analyze_results, interpret_analysis

DASHBOARD = Path(__file__).resolve().parents[1] / 'dashboard.py'


@pytest.fixture
def records():
    df = pd.DataFrame({'seed': [42, 43], 'logistic_regression': [0.95, 0.96],
                       'random_forest': [0.94, 0.97], 'difference': [-0.01, 0.01]})
    analysis = analyze_results(df.to_dict('records'), n_train=80, n_test=20)
    report = {'analysis': analysis, 'interpretation': interpret_analysis(analysis)}
    return df, {}, report


def dashboard_app(tmp_path, *, empty=False, png=True):
    root = tmp_path / 'results'
    root.mkdir()
    if not empty:
        run = root / 'sample'
        run.mkdir()
        for filename in ('benchmark.csv', 'analysis.json', 'provenance.json'):
            (run / filename).write_text('sample download')
        if png:
            (run / 'paired_differences.png').write_bytes(b'sample PNG payload')
    source = DASHBOARD.read_text().replace(
        "RESULTS_DIR = Path(__file__).resolve().parent / 'results'",
        f'RESULTS_DIR = Path({str(root)!r})')
    return AppTest.from_string(source)


def test_saved_experiment_dashboard(tmp_path, records):
    app = dashboard_app(tmp_path)
    with patch('mlreality.validation.load_validated_run', return_value=records):
        app.run(timeout=20)
    assert not app.exception
    assert [metric.value for metric in app.metric] == ['95.50%', '95.50%', '+0.000 pp', '2']
    assert len(app.get('plotly_chart')) == 2
    assert len(app.get('download_button')) == 4


@pytest.mark.parametrize('section,chart_count', [
    ('Performance analysis', 5), ('Statistical evidence', 1), ('Experiment provenance', 0)
])
def test_navigation_sections(tmp_path, records, section, chart_count):
    app = dashboard_app(tmp_path)
    with patch('mlreality.validation.load_validated_run', return_value=records):
        app.run(timeout=20)
        app.sidebar.radio[0].set_value(section).run(timeout=20)
    assert not app.exception
    assert len(app.get('plotly_chart')) == chart_count
    assert len(app.get('download_button')) == 4


def test_invalid_report_displays_error(tmp_path):
    app = dashboard_app(tmp_path)
    with patch('mlreality.validation.load_validated_run', side_effect=ValueError('Invalid report')):
        app.run(timeout=20)
    assert not app.exception
    assert app.error[0].value == 'Invalid report'
    assert not app.metric
    assert not app.get('plotly_chart')


def test_incomplete_analysis_displays_error(tmp_path, records):
    records[2]['analysis'].pop('ci_lower')
    app = dashboard_app(tmp_path)
    with patch('mlreality.validation.load_validated_run', return_value=records):
        app.run(timeout=20)
    assert not app.exception
    assert 'ci_lower' in app.error[0].value
    assert not app.metric


def test_empty_results(tmp_path):
    app = dashboard_app(tmp_path, empty=True).run(timeout=20)
    assert not app.exception
    assert 'No saved experiment' in app.info[0].value
    assert not app.metric


def test_missing_optional_png(tmp_path, records):
    app = dashboard_app(tmp_path, png=False)
    with patch('mlreality.validation.load_validated_run', return_value=records):
        app.run(timeout=20)
    assert not app.exception
    assert len(app.get('download_button')) == 3
    assert any('PNG unavailable' in caption.value for caption in app.caption)


def test_chart_values_match_saved_data(records):
    df, _, report = records
    chart = accuracy_figure(df)
    assert len(chart.data) == 2
    for trace, model in zip(chart.data, ('logistic_regression', 'random_forest')):
        assert list(trace.x) == df['seed'].tolist()
        assert list(trace.y) == (df[model] * 100).tolist()
        assert trace.mode == 'lines+markers'
        assert trace.line.shape == 'linear'
    difference = difference_figure(df)
    assert list(difference.data[0].y) == (df['difference'] * 100).tolist()
    interval = interval_figure(report['analysis'], report['interpretation'])
    point = interval.data[0]
    analysis = report['analysis']
    assert point.x[0] == analysis['mean_difference'] * 100
    assert point.x[0] - point.error_x.arrayminus[0] == pytest.approx(analysis['ci_lower'] * 100)
    assert point.x[0] + point.error_x.array[0] == pytest.approx(analysis['ci_upper'] * 100)
    assert len(interval.layout.shapes) == 3


def test_legacy_serialized_boolean(tmp_path, records):
    records[2]['interpretation']['practical_improvement_supported'] = 'False'
    app = dashboard_app(tmp_path)
    with patch('mlreality.validation.load_validated_run', return_value=records):
        app.run(timeout=20)
        app.sidebar.radio[0].set_value('Statistical evidence').run(timeout=20)
    assert not app.exception and not app.error
    assert any('Interval supports practical improvement: False' in item.value for item in app.markdown)


def test_corrupted_provenance(tmp_path, records):
    records[1]['dataset'] = 'invalid object'
    app = dashboard_app(tmp_path)
    with patch('mlreality.validation.load_validated_run', return_value=records):
        app.run(timeout=20)
    assert not app.exception
    assert 'Invalid provenance field: dataset' in app.error[0].value
    assert not app.metric


def test_download_payloads(tmp_path):
    from unittest.mock import MagicMock
    from mlreality.dashboard_ui import downloads

    files = ('benchmark.csv', 'analysis.json', 'provenance.json', 'paired_differences.png')
    for filename in files:
        (tmp_path / filename).write_bytes(f'original {filename}'.encode())
    with patch('mlreality.dashboard_ui.st') as streamlit:
        streamlit.columns.return_value = [MagicMock() for _ in files]
        downloads(tmp_path)
    assert streamlit.download_button.call_count == 4
    for call, filename in zip(streamlit.download_button.call_args_list, files):
        assert call.args[1] == (tmp_path / filename).read_bytes()
        assert call.kwargs['file_name'] == f'{tmp_path.name}-{filename}'
