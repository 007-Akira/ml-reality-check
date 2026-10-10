"""Local graphical interface for validated saved experiments."""
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from mlreality.validation import load_validated_run

RESULTS_DIR = Path(__file__).resolve().parent / 'results'
st.set_page_config(page_title='ML Reality Check', page_icon='📊', layout='wide')
st.title('ML Reality Check')
st.caption('Reproducible machine learning experiment analysis · Local')
runs = sorted((p for p in RESULTS_DIR.glob('*') if p.is_dir()), reverse=True)
if not runs:
    st.info('Run: python -m mlreality.cli run --seeds 20')
    st.stop()
selected = st.sidebar.selectbox('Selected run', runs, format_func=lambda p: p.name)
try:
    df, provenance, report = load_validated_run(selected)
    analysis = report['analysis']
    interpretation = report['interpretation']
    for key in ('ci_lower', 'ci_upper', 'confidence', 'method', 'p_value'):
        if key not in analysis:
            raise ValueError(f'Missing analysis field: {key}')
except (ValueError, KeyError) as error:
    st.error(str(error))
    st.stop()

st.sidebar.write(f'{len(df)} paired runs')
st.sidebar.write(provenance.get('dataset', {}).get('name', 'Unknown dataset'))
st.sidebar.write(provenance.get('experiment', {}).get('type', 'Unknown evaluation'))
overview, performance, evidence, reproducibility = st.tabs([
    'Overview', 'Model performance', 'Statistical evidence', 'Reproducibility details'
])
with overview:
    columns = st.columns(4)
    models = [column for column in df.columns if column not in ('seed', 'difference')]
    for column, model in zip(columns, models):
        column.metric(model.replace('_', ' ').title(), f'{df[model].mean():.2%}')
    columns[2].metric('Mean difference (RF − LR)', f"{analysis['mean_difference'] * 100:+.3f} pp")
    columns[3].metric('Paired runs', len(df))
    st.dataframe(df, hide_index=True)
with performance:
    scores = df.melt(id_vars='seed', value_vars=models, var_name='Model', value_name='Accuracy')
    scores['Accuracy'] *= 100
    st.plotly_chart(px.box(scores, x='Model', y='Accuracy', points='all',
                           labels={'Accuracy': 'Accuracy (%)'}), width='stretch')
    differences = df.assign(difference_pp=df['difference'] * 100)
    figure = px.bar(differences, x='seed', y='difference_pp',
                    labels={'seed': 'Splitting seed', 'difference_pp': 'RF − LR (pp)'})
    figure.add_hline(y=0)
    st.plotly_chart(figure, width='stretch')
with evidence:
    st.write(analysis['method'])
    mean, lower, upper = [analysis[key] * 100 for key in ('mean_difference', 'ci_lower', 'ci_upper')]
    st.write(f"Corrected {analysis['confidence']:.0%} interval: [{lower:+.3f}, {upper:+.3f}] pp")
    figure = go.Figure(go.Scatter(x=[mean], y=['RF − LR'], mode='markers',
        error_x=dict(type='data', symmetric=False, array=[upper - mean], arrayminus=[mean - lower])))
    figure.add_vline(x=0)
    threshold = interpretation['practical_threshold'] * 100
    for boundary in (-threshold, threshold):
        figure.add_vline(x=boundary, line_dash='dash', line_color='orange')
    figure.update_layout(xaxis_title='Accuracy difference (percentage points)')
    st.plotly_chart(figure, width='stretch')
    st.write(f"Corrected p-value: {analysis['p_value']}")
    st.write(f"Observed winner: {interpretation['observed_winner']}")
    st.write(f"Practical improvement supported: {interpretation['practical_improvement_supported']}")
    st.warning(interpretation['limitations'])
    for note in report.get('notes', []):
        st.caption(note)
with reproducibility:
    st.caption('Provenance is an audit record; it is not a fully executable experiment specification.')
    st.json(provenance)
    st.json(report)
for filename, label, mime in (
    ('benchmark.csv', 'Download measurements', 'text/csv'),
    ('analysis.json', 'Download analysis', 'application/json'),
    ('provenance.json', 'Download provenance', 'application/json'),
):
    st.download_button(label, (selected / filename).read_bytes(),
                       file_name=f'{selected.name}-{filename}', mime=mime)
