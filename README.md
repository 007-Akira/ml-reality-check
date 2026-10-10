# ML Reality Check

ML Reality Check is a Python project for comparing machine learning models across repeated train/test splits and preserving the evidence behind each experiment. It combines a command-line benchmark, statistical analysis, experiment provenance, report validation, and an interactive Streamlit dashboard.

The current implementation compares Logistic Regression and Random Forest on the Breast Cancer Wisconsin Diagnostic dataset. Its purpose is to make model comparisons inspectable and repeatable within a recorded environment, while clearly separating observed performance, statistical evidence, and practical importance.

## Capabilities

- Run paired model evaluations on identical stratified train/test splits.
- Preserve full-precision accuracy measurements in a separate directory for each experiment.
- Calculate an approximate corrected resampled t-test and confidence interval.
- Distinguish an observed advantage from an interval-supported practical improvement.
- Record dataset and split fingerprints, model parameters, dependency versions, and Git state.
- Validate saved report consistency before displaying results.
- Explore per-seed scores, distributions, paired differences, and provenance in a local dashboard.
- Download CSV measurements, JSON reports, and the generated PNG chart.

## Technology

| Component | Technology |
| --- | --- |
| Benchmarking and preprocessing | scikit-learn, NumPy |
| Statistical inference | SciPy, Python statistics module |
| Measurements and tables | pandas |
| Command-line interface | Typer |
| Interactive dashboard | Streamlit, Plotly |
| Static report chart | Matplotlib |
| Tests | pytest, Streamlit AppTest |

## Installation

Git and Python with virtual environment support are required. The current development environment uses Python 3.14.4. Other Python versions have not been verified by this project; compatibility also depends on the pinned packages in `requirements.txt`.

Clone the repository and install its dependencies in a virtual environment:

```bash
git clone https://github.com/007-Akira/ml-reality-check.git
cd ml-reality-check
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell instead of `source .venv/bin/activate`.

If the repository is already cloned, start from its root directory and reuse the existing virtual environment. All commands below assume that environment is active and the working directory is the repository root.

The benchmark loads the dataset bundled with scikit-learn. No separate dataset download, account, or API key is required.

## Run a benchmark

```bash
python -m mlreality.cli run --seeds 20
```

`--seeds` specifies the number of paired runs, not a particular random seed. The CLI defaults to five runs and requires at least two because its statistical analysis needs multiple paired measurements.

A 20-run experiment uses splitting seeds 42 through 61. Both models use the same training and test observations within each run. Model randomness remains fixed at seed 42.

For available commands and options:

```bash
python -m mlreality.cli --help
python -m mlreality.cli run --help
```

The CLI prints per-seed accuracy scores, mean accuracies, paired differences, split outcomes, and an exploratory statistical summary. Each invocation creates a new report directory instead of overwriting an earlier experiment.

### Current experiment configuration

| Setting | Value |
| --- | --- |
| Dataset | Breast Cancer Wisconsin Diagnostic |
| Dataset dimensions | 569 samples, 30 features |
| Task and metric | Binary classification, accuracy |
| Evaluation design | Repeated stratified holdout |
| Train/test allocation | 80% / 20%; 455 training and 114 test samples |
| Logistic Regression | StandardScaler pipeline; `max_iter=1000`, `random_state=42` |
| Random Forest | `n_estimators=100`, `random_state=42`, `n_jobs=-1` |
| Confidence level | 95% |
| Practical improvement threshold | 0.01 accuracy units, or 1 percentage point |

The scaler is fitted through the pipeline on the training data for each split. Dataset selection, model configurations, split ratio, confidence level, and practical threshold are currently defined in code rather than exposed as CLI options.

## Open the dashboard

```bash
python -m streamlit run dashboard.py
```

Open [http://localhost:8501](http://localhost:8501) and select a saved experiment in the sidebar. The dashboard reads existing `results/<run-id>/` directories; opening it does not train models or rerun statistical inference.

The dashboard uses a dark theme configured in `.streamlit/config.toml` and provides four sections:

| Section | Contents |
| --- | --- |
| Overview | Mean accuracy cards, descriptive interpretation, per-seed accuracy lines, and paired differences |
| Performance analysis | Box plots, histograms, grouped per-run scores, split wins and ties, mean scores, sample standard deviations, and raw measurements |
| Statistical evidence | Stored corrected standard error, confidence interval, p-value, practical threshold, and interpretation limitations |
| Experiment provenance | Dataset fingerprint, model parameters, splitting seeds, split sizes, source state, Python version, and package versions |

Each accuracy marker corresponds to a saved score. Straight line segments connect observations without smoothing or interpolation that changes the measurements. Hover to inspect values, click legend entries to toggle models, drag to zoom, or use the chart toolbar to pan and reset.

Report downloads are available from every section. A missing optional PNG is marked unavailable. Empty result directories show instructions for creating an experiment; incomplete or inconsistent reports show an error before metrics and charts are displayed.

The compatibility entry point `python -m streamlit run mlreality/dashboard.py` remains available.

## Experiment artifacts

Each run identifier combines a UTC timestamp and an eight-character random suffix:

```text
results/<UTC-timestamp>-<random-suffix>/
    benchmark.csv
    analysis.json
    provenance.json
    paired_differences.png
```

| Artifact | Contents |
| --- | --- |
| `benchmark.csv` | Splitting seed, both model accuracies, and their paired difference for every run |
| `analysis.json` | Numerical statistical results, interpretation, practical threshold, test designation, and caveats |
| `provenance.json` | Run identifier, UTC timestamp, dataset hash, model parameters, split fingerprints, environment, and Git metadata |
| `paired_differences.png` | Static paired-difference chart generated by the CLI |

CSV accuracies and differences use fractions, such as `0.95` for 95% accuracy. Terminal output and dashboard charts convert accuracy differences to percentage points. The `results/` directory is excluded from Git; retain or share experiment artifacts separately when needed.

## Statistical interpretation

Every paired difference follows the same convention:

```text
difference = random_forest_accuracy - logistic_regression_accuracy
```

Positive differences favor Random Forest on that split; negative differences favor Logistic Regression. A difference of `0.01` means one percentage point, not a 1% relative increase.

Repeated splits reuse observations, so their measurements are dependent. The analyzer applies the Nadeau–Bengio corrected resampled t-test with corrected standard error:

```text
corrected_se = sample_std(differences) * sqrt(1 / number_of_runs + n_test / n_train)
```

It reports a two-sided approximate p-value and confidence interval using `number_of_runs - 1` degrees of freedom. When observed variability is zero, the p-value and t-statistic are unavailable rather than presented as inferential evidence.

Interpretation separates three questions:

1. Which model had the higher observed mean accuracy?
2. Does the corrected interval exclude a zero difference?
3. Does the entire interval exceed the recorded practical threshold in one direction?

An observed mean above the practical threshold does not establish that the entire interval supports that improvement. A p-value is not the probability that a model is better, and additional splits of one dataset do not provide independent external validation. These results are exploratory and do not establish general model superiority.

## Provenance and validation

Dataset fingerprints cover both feature and label arrays. Array hashing includes shape, dtype, and contiguous bytes. Split fingerprints cover the ordered training and test index arrays.

Provenance also records effective scikit-learn parameters, selected package versions, Python and platform information, the Git commit, and working-tree status. A dirty working tree means the commit alone is insufficient to reconstruct the exact source used.

Before rendering an experiment, validation checks required CSV columns, numeric and finite values, accuracy bounds, unique integer seeds, paired-difference consistency, agreement of run counts and mean differences, and consistency with provenance seeds and split counts. Dashboard checks also require renderable statistical and provenance fields.

These checks establish structural consistency. They do not independently recompute the entire saved inference or verify recorded hashes against the original dataset.

Provenance is an audit record, not a fully executable experiment specification. Some parameter objects are stored as readable strings, uncommitted source contents are not preserved, and all hardware and system-level numerical dependencies are not captured. Exact reproduction across machines or library versions is therefore not guaranteed.

## Project structure

```text
ml-reality-check/
    dashboard.py                 Streamlit application entry point
    mlreality/
        __init__.py              Package marker
        benchmark.py             Dataset loading, models, paired evaluation, hashes
        cli.py                   Benchmark command and artifact orchestration
        dashboard.py             Compatibility dashboard entry point
        dashboard_ui.py          Reusable dashboard views, charts, and display checks
        provenance.py            Run directories, environment metadata, JSON output
        report.py                CSV and static PNG generation
        statistics.py            Corrected inference and practical interpretation
        validation.py            Saved report consistency checks
    tests/
        test_dashboard.py        Dashboard views, errors, chart data, downloads
        test_reproducibility.py  Repeatability and paired-difference checks
        test_statistics.py       Statistical and interpretation behavior
        test_validation.py       Valid and inconsistent report cases
    .streamlit/config.toml       Dashboard theme
    requirements.txt             Pinned environment dependencies
    results/                     Generated experiment artifacts, ignored by Git
```

## Testing

```bash
python -m pytest -v
```

The suite covers repeatability in the current environment, paired differences, statistical interpretation, saved report consistency, dashboard navigation, missing and corrupted reports, exact chart values and interval bounds, and download payloads. Dashboard smoke tests use Streamlit AppTest and do not require launching a browser.

Repeatability tests demonstrate consistent results for the same inputs, seeds, and installed environment. They do not prove identical results across operating systems, hardware, or dependency versions.

## Troubleshooting

- **Externally managed Python environment:** activate `.venv` and install packages with `python -m pip`. Avoid installing project dependencies into system Python.
- **Missing dependencies:** confirm the virtual environment is active and reinstall with `python -m pip install -r requirements.txt`.
- **No experiments in the dashboard:** run the CLI from the repository root. The CLI writes `results/` relative to its working directory, while the dashboard reads the repository's `results/` directory.
- **Invalid experiment report:** select another run or create a new experiment. Keep the CSV, analysis, and provenance from the same run together.
- **Port 8501 already in use:** launch with `python -m streamlit run dashboard.py --server.port 8502` and open the displayed address.
