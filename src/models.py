"""
models.py
---------
Module 2B: Classification Models for Baseline Evaluation.

Trains four classifiers (KNN, GaussianNB, DecisionTree, RandomForest)
with dataset-specific hyperparameters matching Pathak et al. (2025),
and records Accuracy, Precision, Recall, F1-Score + timing.
"""

import time
import numpy as np
import pandas as pd
from sklearn.neighbors         import KNeighborsClassifier
from sklearn.naive_bayes       import GaussianNB
from sklearn.tree              import DecisionTreeClassifier
from sklearn.ensemble          import RandomForestClassifier
from sklearn.metrics           import (accuracy_score, precision_score,
                                       recall_score, f1_score)

# ---------------------------------------------------------------------------
# Per-dataset hyperparameters (matching Table III / results in the paper)
# ---------------------------------------------------------------------------
MODEL_PARAMS = {
    "dataset1": {
        "knn": {"n_neighbors": 5},
        "dt" : {"criterion": "entropy"},
        "rf" : {"criterion": "entropy"},
    },
    "dataset2": {
        "knn": {"n_neighbors": 4},
        "dt" : {"criterion": "gini"},
        "rf" : {"criterion": "gini"},
    },
    "dataset3": {
        "knn": {"n_neighbors": 4},
        "dt" : {"criterion": "gini"},
        "rf" : {"criterion": "entropy"},
    },
}


def _build_models(dataset_name: str, random_state: int = 42) -> dict:
    """Return a dict of {model_name: fitted_estimator_instance}."""
    p = MODEL_PARAMS[dataset_name]
    return {
        "KNN": KNeighborsClassifier(**p["knn"]),
        "GaussianNB": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(
            **p["dt"], random_state=random_state
        ),
        "RandomForest": RandomForestClassifier(
            **p["rf"], random_state=random_state
        ),
    }


def train_and_evaluate(
    X_train: np.ndarray,
    X_test : np.ndarray,
    y_train: np.ndarray,
    y_test : np.ndarray,
    dataset_name: str,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Train each of the four classifiers and compute evaluation metrics.

    Parameters
    ----------
    X_train, X_test : scaled feature arrays
    y_train, y_test : integer label arrays
    dataset_name    : 'dataset1' | 'dataset2' | 'dataset3'
    random_state    : seed for tree-based models

    Returns
    -------
    pd.DataFrame with one row per model and columns:
      Model | Accuracy | Precision | Recall | F1-Score |
      Train_Time_s | Inference_Time_s
    """
    models  = _build_models(dataset_name, random_state)
    records = []

    for model_name, clf in models.items():
        # ---- Training ----
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        # ---- Inference ----
        t0 = time.perf_counter()
        y_pred = clf.predict(X_test)
        infer_time = time.perf_counter() - t0

        # ---- Metrics (weighted average for multi-class safety) ----
        acc  = accuracy_score (y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score   (y_test, y_pred, average="weighted", zero_division=0)
        f1   = f1_score       (y_test, y_pred, average="weighted", zero_division=0)

        records.append({
            "Model"          : model_name,
            "Accuracy"       : round(acc  * 100, 4),
            "Precision"      : round(prec * 100, 4),
            "Recall"         : round(rec  * 100, 4),
            "F1-Score"       : round(f1   * 100, 4),
            "Train_Time_s"   : round(train_time, 4),
            "Inference_Time_s": round(infer_time, 6),
        })

    return pd.DataFrame(records).set_index("Model")
