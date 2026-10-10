"""Launch with: python -m streamlit run dashboard.py."""
from pathlib import Path

import streamlit as st

from mlreality.validation import load_validated_run
from mlreality.dashboard_ui import (
    SECTIONS, apply_style, downloads, evidence, overview, performance,
    provenance_view, sidebar_details, validate_display_report, validate_display_provenance,
)

RESULTS_DIR = Path(__file__).resolve().parent / 'results'


def main():
    st.set_page_config(page_title='ML Reality Check', page_icon='📊', layout='wide')
    apply_style()
    st.sidebar.markdown('### ML Reality Check')
    st.sidebar.caption('RESEARCH & REPRODUCIBILITY')
    runs = sorted((p for p in RESULTS_DIR.glob('*') if p.is_dir()), reverse=True)
    if not runs:
        st.title('Your experiments start here')
        st.info('No saved experiment directories found. Create one with the CLI.')
        st.code('python -m mlreality.cli run --seeds 20', language='bash')
        return
    selected = st.sidebar.selectbox('Selected experiment', runs, format_func=lambda p: p.name)
    section = st.sidebar.radio('Workspace', SECTIONS)
    try:
        df, provenance, report = load_validated_run(selected)
        validate_display_report(report)
        validate_display_provenance(provenance)
    except (ValueError, KeyError, TypeError) as error:
        st.error(str(error))
        st.info('This experiment cannot be displayed. Select another run or create a new report with the CLI.')
        return
    sidebar_details(df, provenance)
    st.markdown('<div class="eyebrow">EXPERIMENT EVALUATION · LOCAL WORKSPACE</div>', unsafe_allow_html=True)
    st.title('ML Reality Check' if section == 'Overview' else section)
    st.caption(f'{selected.name} · Saved experiment · Accuracy comparison')
    if section == 'Overview':
        overview(df, report)
    elif section == 'Performance analysis':
        performance(df)
    elif section == 'Statistical evidence':
        evidence(report)
    else:
        provenance_view(provenance, report)
    downloads(selected)


if __name__ == '__main__':
    main()
