"""Validate the saved two-model experiment contract before displaying it."""
import json
import math
from pathlib import Path

import pandas as pd


def load_validated_run(run_dir):
    run_dir = Path(run_dir)
    try:
        df = pd.read_csv(run_dir / 'benchmark.csv')
        provenance = json.loads((run_dir / 'provenance.json').read_text())
        report = json.loads((run_dir / 'analysis.json').read_text())
        required = {'seed', 'difference', 'logistic_regression', 'random_forest'}
        if not required.issubset(df.columns):
            raise ValueError('Required benchmark columns missing')
        if df.empty:
            raise ValueError('Benchmark contains no runs')
        if df.isna().any().any():
            raise ValueError('Benchmark contains missing values')
        for column in required:
            df[column] = pd.to_numeric(df[column], errors='raise')
            if not all(math.isfinite(float(value)) for value in df[column]):
                raise ValueError(f'Non-finite values in {column}')
        if df['seed'].duplicated().any():
            raise ValueError('Duplicate splitting seeds detected')
        if not (df['seed'] % 1 == 0).all():
            raise ValueError('Splitting seeds must be integers')
        for column in ['logistic_regression', 'random_forest']:
            if not df[column].between(0, 1).all():
                raise ValueError(f'Invalid accuracy values: {column}')
        expected = df['random_forest'] - df['logistic_regression']
        if not (df['difference'] - expected).abs().le(1e-10).all():
            raise ValueError('Paired differences are inconsistent')
        analysis = report.get('analysis', {})
        if analysis.get('runs') != len(df):
            raise ValueError('Analysis run count mismatch')
        if not math.isclose(analysis['mean_difference'], float(df['difference'].mean()),
                            rel_tol=0, abs_tol=1e-10):
            raise ValueError('Reported mean difference mismatch')
        for key in ('ci_lower', 'ci_upper', 'confidence', 'corrected_se'):
            if key in analysis and not math.isfinite(float(analysis[key])):
                raise ValueError(f'Non-finite analysis field: {key}')
        if 'confidence' in analysis and not 0 < analysis['confidence'] < 1:
            raise ValueError('Invalid confidence level')
        if 'ci_lower' in analysis and 'ci_upper' in analysis:
            if not analysis['ci_lower'] <= analysis['mean_difference'] <= analysis['ci_upper']:
                raise ValueError('Invalid confidence interval')
        if analysis.get('p_value') is not None and not 0 <= analysis['p_value'] <= 1:
            raise ValueError('Invalid p-value')
        if provenance.get('experiment', {}).get('splitting_seeds') != df['seed'].tolist():
            raise ValueError('Provenance seed list mismatch')
        splits = provenance.get('splits', [])
        if len(splits) != len(df):
            raise ValueError('Split metadata count mismatch')
        for split, seed in zip(splits, df['seed']):
            if 'seed' in split and split['seed'] != seed:
                raise ValueError('Split seed mismatch')
        return df, provenance, report
    except (OSError, KeyError, TypeError, AttributeError, ValueError) as error:
        raise ValueError(f'Invalid experiment report: {error}') from error
