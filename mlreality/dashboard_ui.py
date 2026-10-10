"""Reusable presentation components for the saved-experiment dashboard."""
from html import escape
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

MODELS = ['logistic_regression', 'random_forest']
NAMES = {'logistic_regression': 'Logistic Regression', 'random_forest': 'Random Forest'}
COLORS = {'Logistic Regression': '#62D5C5', 'Random Forest': '#A895FF'}
SECTIONS = ['Overview', 'Performance analysis', 'Statistical evidence', 'Experiment provenance']
CHART_CONFIG = {'displaylogo': False, 'scrollZoom': True,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d']}


def apply_style():
    st.markdown('''<style>
    .stApp {background: #0b1120; color: #e7edf7;}
    [data-testid="stSidebar"] {background: #10192a; border-right: 1px solid #243149;}
    .block-container {max-width: 1480px; padding-top: 2.2rem; padding-bottom: 3rem;}
    h1 {letter-spacing: -.045em; font-weight: 750;}
    h2, h3 {letter-spacing: -.025em;}
    [data-testid="stMetric"] {background: linear-gradient(125deg, #17243a, #111b2c);
      border: 1px solid #2a3952; border-radius: 16px; padding: 20px; min-height: 128px;}
    [data-testid="stMetricValue"] {font-size: 2rem; font-variant-numeric: tabular-nums;}
    [data-testid="stMetricLabel"] {color: #aab9d0;}
    [data-testid="stVerticalBlockBorderWrapper"] {border-radius: 16px;}
    .eyebrow {font-size: .72rem; letter-spacing: .16em; color: #62d5c5; font-weight: 650;}
    .muted {color: #aab9d0; font-size: .9rem;}
    .insight {border: 1px solid #2d4e58; border-left: 3px solid #62d5c5;
      background: linear-gradient(115deg, #132b35, #141e30); border-radius: 12px;
      padding: 20px 24px; margin: 20px 0; line-height: 1.7;}
    .stDownloadButton button {border-radius: 10px; border-color: #30405b;}
    @media (max-width: 700px) {.block-container {padding: 1rem;} h1 {font-size: 2rem;}}
    </style>''', unsafe_allow_html=True)


def style_figure(figure, *, height=380, y_title=None):
    figure.update_layout(template='plotly_dark', height=height,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#111b2c',
        font=dict(family='Inter, system-ui, sans-serif', color='#b9c7dd', size=12),
        margin=dict(l=24, r=24, t=30, b=30),
        legend=dict(orientation='h', y=1.13, x=0, title_text=''),
        hoverlabel=dict(bgcolor='#22324b', font_size=13),
        hovermode='x unified', dragmode='zoom')
    figure.update_xaxes(gridcolor='#243149', zeroline=False)
    figure.update_yaxes(gridcolor='#243149', zeroline=False)
    if y_title:
        figure.update_yaxes(title=y_title)
    return figure


def accuracy_figure(df):
    """Every marker represents a saved score; lines use linear segments."""
    figure = go.Figure()
    for model in MODELS:
        name = NAMES[model]
        figure.add_trace(go.Scatter(x=df['seed'].tolist(), y=(df[model] * 100).tolist(),
            name=name, mode='lines+markers',
            line=dict(color=COLORS[name], width=2.5, shape='linear'),
            marker=dict(size=7, color=COLORS[name]),
            hovertemplate=f'Seed %{{x}}<br>{name}: %{{y:.6f}}%<extra></extra>'))
    figure.update_xaxes(title='Splitting seed')
    if len(df) <= 25:
        figure.update_xaxes(tickmode='array', tickvals=df['seed'].tolist())
    figure.update_yaxes(ticksuffix='%')
    return style_figure(figure, height=440, y_title='Accuracy (%)')


def difference_figure(df):
    figure = go.Figure(go.Scatter(x=df['seed'].tolist(), y=(df['difference'] * 100).tolist(),
        name='RF − LR', mode='lines+markers', line=dict(color='#7cb8ff', width=2, shape='linear'),
        marker=dict(size=7), hovertemplate='Seed %{x}<br>RF − LR: %{y:+.6f} pp<extra></extra>'))
    figure.add_hline(y=0, line_color='#c1ccdd', line_dash='dash', annotation_text='Equal accuracy')
    figure.update_xaxes(title='Splitting seed')
    return style_figure(figure, height=290, y_title='RF − LR (percentage points)')


def interval_figure(analysis, interpretation):
    mean, lower, upper = [analysis[key] * 100 for key in ('mean_difference', 'ci_lower', 'ci_upper')]
    figure = go.Figure(go.Scatter(x=[mean], y=['RF − LR'], mode='markers',
        marker=dict(size=13, color='#7cb8ff'),
        error_x=dict(type='data', symmetric=False, array=[upper - mean],
                     arrayminus=[mean - lower], color='#7cb8ff', thickness=3, width=9),
        customdata=[[lower, upper]],
        hovertemplate='Mean: %{x:+.6f} pp<br>Interval: [%{customdata[0]:+.6f}, %{customdata[1]:+.6f}] pp<extra></extra>'))
    figure.add_vline(x=0, line_dash='dash', line_color='#c1ccdd', annotation_text='Zero difference')
    threshold = interpretation.get('practical_threshold')
    if threshold is not None:
        for bound in (-threshold * 100, threshold * 100):
            figure.add_vline(x=bound, line_dash='dot', line_color='#f4bd6a',
                            annotation_text=f'{bound:+g} pp practical threshold')
    figure.update_xaxes(title='RF − LR (percentage points)')
    return style_figure(figure, height=290)


def show_chart(figure, key):
    st.plotly_chart(figure, width='stretch', config=CHART_CONFIG, key=key)


def validate_display_report(report):
    """Require renderable stored evidence; never synthesize missing values."""
    analysis = report.get('analysis')
    interpretation = report.get('interpretation')
    if not isinstance(analysis, dict) or not isinstance(interpretation, dict):
        raise ValueError('Missing analysis or interpretation object')
    for key in ('mean_difference', 'std_difference', 'corrected_se', 'ci_lower', 'ci_upper', 'confidence'):
        value = analysis.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Missing or invalid analysis field: {key}')
    if not 0 < analysis['confidence'] < 1:
        raise ValueError('Invalid confidence level')
    if analysis['std_difference'] < 0 or analysis['corrected_se'] < 0:
        raise ValueError('Negative statistical dispersion')
    if not analysis['ci_lower'] <= analysis['mean_difference'] <= analysis['ci_upper']:
        raise ValueError('Invalid confidence interval')
    if not isinstance(analysis.get('method'), str) or 'p_value' not in analysis:
        raise ValueError('Missing statistical method or p-value field')
    p_value = analysis['p_value']
    if p_value is not None and (isinstance(p_value, bool) or not isinstance(p_value, (int, float))
                                 or not math.isfinite(p_value) or not 0 <= p_value <= 1):
        raise ValueError('Invalid p-value')
    threshold = interpretation.get('practical_threshold')
    if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                                  or not math.isfinite(threshold) or threshold <= 0):
        raise ValueError('Invalid practical threshold')
    for key in ('observed_winner', 'statistical_direction'):
        if interpretation.get(key) not in (*MODELS, 'tie', 'inconclusive'):
            raise ValueError(f'Missing or invalid interpretation field: {key}')
    supported = interpretation.get('practical_improvement_supported')
    # Existing reports used default=str for NumPy booleans. Accept their exact
    # legacy spellings, without treating arbitrary strings as truthy values.
    if not isinstance(supported, (bool, np.bool_)) and supported not in ('True', 'False'):
        raise ValueError('Missing practical improvement interpretation')
    if not isinstance(interpretation.get('limitations'), str):
        raise ValueError('Missing statistical limitations')
    notes = report.get('notes', [])
    if not isinstance(notes, list) or not all(isinstance(note, str) for note in notes):
        raise ValueError('Invalid report notes')


def validate_display_provenance(provenance):
    for key in ('dataset', 'source', 'environment', 'experiment', 'models'):
        if key in provenance and not isinstance(provenance[key], dict):
            raise ValueError(f'Invalid provenance field: {key}')
    env = provenance.get('environment', {})
    if not isinstance(env.get('packages', {}), dict):
        raise ValueError('Invalid dependency metadata')


def sidebar_details(df, provenance):
    st.sidebar.caption('EXPERIMENT CONTEXT')
    st.sidebar.write(provenance.get('dataset', {}).get('name', 'Dataset not recorded'))
    st.sidebar.caption(f'{len(df)} paired runs · Accuracy')
    st.sidebar.caption(f"Recorded UTC: {provenance.get('created_at_utc', 'Not recorded')}")
    with st.sidebar.expander('Evaluation design'):
        st.write(provenance.get('experiment', {}))
    st.sidebar.divider()
    st.sidebar.caption('CREATE AN EXPERIMENT')
    st.sidebar.code('python -m mlreality.cli run --seeds 20', language='bash')
    st.sidebar.caption('The dashboard reads saved reports. No models are trained here.')


def overview(df, report):
    analysis, interpretation = report['analysis'], report['interpretation']
    cards = st.columns(4)
    for column, model in zip(cards, MODELS):
        column.metric(NAMES[model], f'{df[model].mean():.2%}')
    cards[2].metric('Mean difference · RF − LR', f"{analysis['mean_difference'] * 100:+.3f} pp")
    cards[3].metric('Paired runs', len(df))
    st.caption(
        f"Split outcomes · Logistic Regression wins: {int((df['difference'] < 0).sum())} · "
        f"Random Forest wins: {int((df['difference'] > 0).sum())} · "
        f"Ties: {int((df['difference'] == 0).sum())}"
    )
    winner = interpretation['observed_winner']
    descriptive = f'{NAMES[winner]} had the higher observed mean accuracy' if winner in NAMES else 'The observed mean accuracies were tied'
    st.markdown(f'<div class="insight"><strong>{escape(descriptive)}.</strong><br>'
        f'Mean paired difference: {analysis["mean_difference"] * 100:+.3f} percentage points (RF − LR). '
        f'<span class="muted">{escape(interpretation["limitations"])}</span></div>', unsafe_allow_html=True)
    st.subheader('Accuracy across splits')
    st.caption('Every point is a saved measurement. Click a legend entry to toggle a model; drag to zoom.')
    show_chart(accuracy_figure(df), 'accuracy_line')
    st.subheader('Paired accuracy difference')
    st.caption('Positive favors Random Forest · Negative favors Logistic Regression')
    show_chart(difference_figure(df), 'difference_line')


def performance(df):
    scores = df.melt(id_vars='seed', value_vars=MODELS, var_name='Model', value_name='Accuracy')
    scores['Model'] = scores['Model'].map(NAMES)
    scores['Accuracy'] *= 100
    left, right = st.columns(2)
    with left:
        st.subheader('Accuracy distribution')
        show_chart(style_figure(px.box(scores, x='Model', y='Accuracy', color='Model', points='all',
            color_discrete_map=COLORS, labels={'Accuracy': 'Accuracy (%)'})), 'box')
    with right:
        st.subheader('Score frequency')
        hist = px.histogram(scores, x='Accuracy', color='Model', barmode='overlay', opacity=0.65,
                            color_discrete_map=COLORS, labels={'Accuracy': 'Accuracy (%)'})
        show_chart(style_figure(hist), 'histogram')
    st.subheader('Per-run comparison')
    show_chart(style_figure(px.bar(scores, x='seed', y='Accuracy', color='Model', barmode='group',
        color_discrete_map=COLORS, labels={'seed': 'Splitting seed', 'Accuracy': 'Accuracy (%)'})), 'per_run')
    left, right = st.columns(2)
    with left:
        st.subheader('Split outcomes')
        wins = pd.DataFrame({'Outcome': ['Logistic Regression', 'Random Forest', 'Tie'],
            'Splits': [int((df['difference'] < 0).sum()), int((df['difference'] > 0).sum()),
                       int((df['difference'] == 0).sum())]})
        show_chart(style_figure(px.bar(wins, x='Outcome', y='Splits', color='Outcome',
            color_discrete_map={**COLORS, 'Tie': '#98a6be'}), height=300), 'wins')
    with right:
        st.subheader('Mean and variability')
        summary = pd.DataFrame([{'Model': NAMES[model], 'Mean accuracy (%)': df[model].mean() * 100,
            'Sample standard deviation (pp)': df[model].std(ddof=1) * 100} for model in MODELS])
        show_chart(style_figure(px.bar(summary, x='Model', y='Mean accuracy (%)', color='Model',
            color_discrete_map=COLORS), height=300), 'means')
        st.dataframe(summary, hide_index=True, width='stretch')
        st.caption('Standard deviation describes split variability; it is not a confidence interval.')
    st.subheader('Saved measurements')
    st.caption('Full-precision accuracy fractions from benchmark.csv. Sort columns to inspect individual runs.')
    st.dataframe(df, hide_index=True, width='stretch')


def evidence(report):
    analysis, interpretation = report['analysis'], report['interpretation']
    st.caption(analysis['method'])
    st.caption('Exploratory inference · Approximate dependence correction · Independent validation required')
    cards = st.columns(3)
    cards[0].metric('Mean paired difference', f"{analysis['mean_difference'] * 100:+.3f} pp")
    cards[1].metric('Difference standard deviation', f"{analysis['std_difference'] * 100:.3f} pp")
    cards[2].metric('Corrected standard error', f"{analysis['corrected_se'] * 100:.3f} pp")
    st.subheader(f"Approximate {analysis['confidence']:.0%} confidence interval")
    st.write(f"[{analysis['ci_lower'] * 100:+.3f}, {analysis['ci_upper'] * 100:+.3f}] percentage points")
    show_chart(interval_figure(analysis, interpretation), 'interval')
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader('Statistical direction')
            st.write(interpretation['statistical_direction'].replace('_', ' ').title())
            value = analysis['p_value']
            st.write(f'Approximate corrected p-value: {value:.6g}' if value is not None else
                     'Corrected p-value unavailable: no observed variability.')
            st.caption('A p-value is not the probability that a model is better.')
    with right:
        with st.container(border=True):
            st.subheader('Practical importance')
            threshold = interpretation.get('practical_threshold')
            st.write(f'Recorded threshold: {threshold * 100:g} pp' if threshold is not None else 'Practical threshold not recorded.')
            st.write(f"Interval supports practical improvement: {interpretation['practical_improvement_supported']}")
            st.caption('An observed mean above the threshold does not establish that the entire interval exceeds it.')
    st.warning('Repeated train/test splits overlap. The Nadeau–Bengio corrected resampled t-test '
               'uses an approximate dependence correction. These runs are not independent external validation.')
    st.caption(interpretation['limitations'])
    for note in report.get('notes', []):
        st.caption(note)


def provenance_view(provenance, report):
    dataset, source, env = [provenance.get(key, {}) for key in ('dataset', 'source', 'environment')]
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader('Dataset & design')
            st.write(dataset.get('name', 'Not recorded'))
            st.caption(f"Samples: {dataset.get('samples', 'Not recorded')} · Features: {dataset.get('features', 'Not recorded')}")
            st.caption('Dataset SHA-256')
            st.code(dataset.get('sha256', 'Not recorded'), language=None)
            st.write('Experiment timestamp (UTC)', provenance.get('created_at_utc', 'Not recorded'))
    with right:
        with st.container(border=True):
            st.subheader('Source & environment')
            st.caption('Git commit')
            st.code(source.get('git_commit') or 'Not recorded', language=None)
            dirty = source.get('git_dirty')
            st.write('Working tree:', 'Modified' if dirty is True else 'Clean' if dirty is False else 'Not recorded')
            st.write('Python:', env.get('python', 'Not recorded'))
            st.caption(env.get('platform', 'Platform not recorded'))
    with st.expander('Model configurations', expanded=True):
        for model, parameters in provenance.get('models', {}).items():
            st.write(NAMES.get(model, model))
            st.json(parameters)
    with st.expander('Random seeds and train/test splits'):
        st.json(provenance.get('experiment', {}))
        st.dataframe(pd.DataFrame(provenance.get('splits', [])), hide_index=True, width='stretch')
    with st.expander('Dependency versions'):
        st.dataframe(pd.DataFrame(list(env.get('packages', {}).items()), columns=['Package', 'Version']), hide_index=True)
    with st.expander('Complete audit records'):
        st.json(provenance)
        st.json(report)
    st.info('Provenance is an audit record, not a fully executable specification. Uncommitted source '
            'contents and every system-level numerical dependency are not preserved.')


def downloads(selected):
    st.divider()
    st.subheader('Export this experiment')
    for column, (filename, label, mime) in zip(st.columns(4), (
        ('benchmark.csv', 'Benchmark CSV', 'text/csv'),
        ('analysis.json', 'Analysis JSON', 'application/json'),
        ('provenance.json', 'Provenance JSON', 'application/json'),
        ('paired_differences.png', 'Difference chart PNG', 'image/png'),
    )):
        with column:
            try:
                data = (selected / filename).read_bytes()
            except OSError:
                st.caption(f'{label} unavailable')
            else:
                st.download_button(label, data, file_name=f'{selected.name}-{filename}', mime=mime,
                                   width='stretch')
