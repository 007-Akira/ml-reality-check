# ML Reality Check

A reproducible machine learning benchmark with a local dashboard for inspecting saved paired experiments. The current comparison uses standardized Logistic Regression and Random Forest on the Breast Cancer Wisconsin Diagnostic dataset.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m mlreality.cli run --seeds 20
python -m streamlit run dashboard.py
```

Open http://localhost:8501. The dashboard uses a dark theme by default and reads `results/<run-id>/` without training models. The older `python -m streamlit run mlreality/dashboard.py` entry point also works.

## Explore an experiment

Choose a saved experiment and a section in the sidebar:

- **Overview:** mean accuracy cards, descriptive interpretation, interactive per-seed accuracy lines, and paired differences. Click a legend label to toggle a model, drag to zoom, or use the toolbar to pan and reset. Markers and straight line segments preserve every recorded measurement.
- **Performance analysis:** box plots, histograms, grouped per-run scores, split wins/losses/ties, mean accuracy, sample standard deviations, and sortable raw measurements.
- **Statistical evidence:** stored corrected standard error, difference standard deviation, approximate confidence interval and p-value, and recorded practical thresholds.
- **Experiment provenance:** dataset fingerprint, model parameters, random seeds, split sizes, Git status, Python and dependency versions, and experiment timestamp in UTC.

Download the benchmark CSV, statistical analysis JSON, provenance JSON, and existing difference-chart PNG from any section. Missing optional PNGs are marked unavailable. Empty result folders show CLI instructions; incomplete or inconsistent reports show an error before charts or metrics are rendered.

## Evidence and reproducibility

Differences are **Random Forest minus Logistic Regression**, expressed in percentage points in charts. The dashboard validates the saved CSV and metadata and displays stored analysis; it does not change the statistical method or replace missing evidence with calculated guesses.

Repeated train/test splits overlap. The Nadeau–Bengio corrected resampled t-test uses an approximate dependence correction, and a directional interval does not establish general model superiority. Practical importance depends on the recorded threshold and whether the entire interval supports it. Independent validation is still needed.

Provenance is an audit record, not a fully executable experiment specification. Dirty working-tree contents and all system-level numerical dependencies are not captured. Validation checks structural consistency; it does not independently prove the saved inference or verify dataset hashes against original data.

## Tests

```bash
python -m pytest -v
```

Tests cover repeatability, paired differences, statistical interpretation, report validation, dashboard navigation and error handling, and exact per-seed chart values and interval bounds.
