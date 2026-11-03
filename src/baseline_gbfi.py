"""
baseline_gbfi.py
----------------
Module 2: Gradient Boosting Feature Importance (GBFI) — Baseline Feature Selection.

Implements Algorithm 1 from:
  Pathak et al. (2025) - "Machine learning approach to detect android malware
  using feature-selection based on feature importance score."

Algorithm 1 (GBFI):
  1. Train GradientBoostingClassifier on (X_train, y_train).
  2. Extract feature_importances_ array (FI scores, range [0, 1]).
  3. Discard all features where FI score == 0.
  4. Return reduced feature matrices using only the surviving features.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


def select_features_gbfi(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    feature_names: list,
    random_state: int = 42,
):
    """
    Apply Gradient Boosting Feature Importance (GBFI) selection.

    Steps:
      1. Train a GradientBoostingClassifier with default hyperparameters.
      2. Compute per-feature importance scores.
      3. Remove all features with importance == 0 (Algorithm 1, Pathak 2025).
      4. Return the reduced matrices and metadata.

    Parameters
    ----------
    X_train : np.ndarray  shape (n_train, n_features)
    X_test  : np.ndarray  shape (n_test,  n_features)
    y_train : np.ndarray  shape (n_train,)
    feature_names : list  length n_features
    random_state : int

    Returns
    -------
    X_train_reduced : np.ndarray  shape (n_train, k)
    X_test_reduced  : np.ndarray  shape (n_test,  k)
    selected_indices : np.ndarray  integer indices of retained features
    selected_names   : list        names of retained features
    fi_scores        : np.ndarray  importance scores for ALL original features
    """
    print("[GBFI] Training GradientBoostingClassifier for feature scoring ...")

    gb = GradientBoostingClassifier(random_state=random_state)
    gb.fit(X_train, y_train)

    fi_scores = gb.feature_importances_          # shape (n_features,)

    # Algorithm 1: keep only features where FI score > 0
    selected_indices = np.where(fi_scores > 0)[0]
    selected_names   = [feature_names[i] for i in selected_indices]

    n_orig     = X_train.shape[1]
    n_selected = len(selected_indices)
    n_removed  = n_orig - n_selected

    print(f"  Original features : {n_orig}")
    print(f"  Features removed  : {n_removed}  (FI score == 0)")
    print(f"  Features selected : {n_selected}")

    X_train_reduced = X_train[:, selected_indices]
    X_test_reduced  = X_test[:, selected_indices]

    return X_train_reduced, X_test_reduced, selected_indices, selected_names, fi_scores


def get_feature_importance_scores(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list,
    top_n: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Train a GradientBoostingClassifier and return a sorted DataFrame of
    feature importance scores (for reporting / verification).

    Parameters
    ----------
    X_train, y_train : training data
    feature_names    : column names
    top_n            : how many top features to show (default 30)
    random_state     : seed

    Returns
    -------
    pd.DataFrame with columns ['Feature', 'Importance'] sorted descending.
    Full list is returned; caller can slice to top_n for display.
    """
    gb = GradientBoostingClassifier(random_state=random_state)
    gb.fit(X_train, y_train)

    df_fi = pd.DataFrame({
        "Feature"   : feature_names,
        "Importance": gb.feature_importances_,
    })
    df_fi = df_fi.sort_values("Importance", ascending=False).reset_index(drop=True)

    return df_fi
