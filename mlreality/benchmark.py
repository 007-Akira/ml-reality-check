
import hashlib

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score


def array_hash(array):
    """Produce a SHA-256 fingerprint of a NumPy array."""
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()

    digest.update(str(arr.shape).encode())
    digest.update(str(arr.dtype).encode())
    digest.update(arr.tobytes())

    return digest.hexdigest()


def build_models():
    """Define the exact model configurations."""
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
    }


def run_benchmark(seeds: int = 5):
    if seeds < 1:
        raise ValueError("At least one run is required.")

    X, y = load_breast_cancer(return_X_y=True)

    # Fingerprint both features and labels.
    dataset_hash = hashlib.sha256(
        (array_hash(X) + array_hash(y)).encode()
    ).hexdigest()

    # Capture effective sklearn parameter values.
    model_parameters = {
        name: model.get_params(deep=True)
        for name, model in build_models().items()
    }

    results = []
    split_records = []

    for seed in range(42, 42 + seeds):

        # Split indices first, so we can record them.
        indices = np.arange(len(y))

        train_idx, test_idx = train_test_split(
            indices,
            test_size=0.2,
            random_state=seed,
            stratify=y,
        )

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scores = {}

        for name, model in build_models().items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            scores[name] = accuracy_score(
                y_test, predictions
            )

        difference = (
            scores["random_forest"]
            - scores["logistic_regression"]
        )

        results.append({
            "seed": seed,
            "logistic_regression": scores["logistic_regression"],
            "random_forest": scores["random_forest"],
            "difference": difference,
        })

        split_records.append({
            "seed": seed,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "train_indices_hash": array_hash(train_idx),
            "test_indices_hash": array_hash(test_idx),
        })

    metadata = {
        "dataset": {
            "name": "Breast Cancer Wisconsin Diagnostic",
            "samples": len(y),
            "features": X.shape[1],
            "sha256": dataset_hash,
        },
        "experiment": {
            "type": "repeated_stratified_holdout",
            "test_size": 0.2,
            "splitting_seeds": list(range(42, 42 + seeds)),
            "model_seed": 42,
            "metric": "accuracy",
        },
        "models": model_parameters,
        "splits": split_records,
    }

    return results, metadata
