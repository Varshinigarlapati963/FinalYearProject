"""
preprocessing.py
----------------
Module 1: Data Loading & Preprocessing for the Baseline Reproduction Study.

Reference paper:
  Pathak et al. (2025) - "Machine learning approach to detect android malware
  using feature-selection based on feature importance score."

Handles three datasets:
  - dataset1: DefenseDroid  (1491 cols, target='class')
  - dataset2: Mendeley       (174  cols, target='Label')
  - dataset3: Lopez et al.  (331  cols, semicolon-separated, target='type')
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Dataset configuration table
# ---------------------------------------------------------------------------
DATASET_CONFIG = {
    "dataset1": {
        "path"    : os.path.join("data", "dataset1.csv"),
        "sep"     : ",",         # standard CSV
        "target"  : "class",
        # All columns are numeric in the cleaned file; no string metadata
        "meta_cols": [],
    },
    "dataset2": {
        "path"    : os.path.join("data", "dataset2.csv"),
        "sep"     : ",",
        "target"  : "Label",
        "meta_cols": [],
    },
    "dataset3": {
        "path"    : os.path.join("data", "dataset3.csv"),
        "sep"     : ";",         # semicolon-delimited
        "target"  : "type",
        "meta_cols": [],
    },
}


def load_and_preprocess(dataset_name: str, random_state: int = 42):
    """
    Load a dataset, drop metadata columns, separate features/target,
    scale features, and perform a stratified 80/20 train-test split.

    Parameters
    ----------
    dataset_name : str
        One of 'dataset1', 'dataset2', 'dataset3'.
    random_state : int
        Seed for reproducibility (default 42).

    Returns
    -------
    X_train, X_test : np.ndarray
        Scaled feature matrices.
    y_train, y_test : np.ndarray
        Integer class labels (0 = Benign, 1 = Malware).
    feature_names : list[str]
        Names of the retained feature columns.
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {list(DATASET_CONFIG.keys())}"
        )

    cfg = DATASET_CONFIG[dataset_name]

    # ------------------------------------------------------------------
    # 1. Load CSV
    # ------------------------------------------------------------------
    print(f"\n[Preprocessing] Loading {dataset_name} from '{cfg['path']}' ...")
    df = pd.read_csv(cfg["path"], sep=cfg["sep"], engine="python")
    print(f"  Raw shape: {df.shape}")

    # ------------------------------------------------------------------
    # 2. Drop explicit metadata columns (if any declared in config)
    # ------------------------------------------------------------------
    cols_to_drop = [c for c in cfg["meta_cols"] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"  Dropped {len(cols_to_drop)} metadata column(s): {cols_to_drop}")

    # ------------------------------------------------------------------
    # 3. Drop any remaining non-numeric columns (except the target)
    # ------------------------------------------------------------------
    target_col = cfg["target"]
    non_numeric = df.drop(columns=[target_col]).select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric:
        df = df.drop(columns=non_numeric)
        print(f"  Auto-dropped {len(non_numeric)} non-numeric column(s): {non_numeric[:5]}{'...' if len(non_numeric) > 5 else ''}")

    # ------------------------------------------------------------------
    # 4. Separate features (X) and target (y)
    # ------------------------------------------------------------------
    y = df[target_col].values.astype(int)
    X_df = df.drop(columns=[target_col])
    feature_names = X_df.columns.tolist()
    X = X_df.values.astype(float)

    unique, counts = np.unique(y, return_counts=True)
    print(f"  Features: {X.shape[1]} | Samples: {X.shape[0]}")
    print(f"  Class distribution: { {int(k): int(v) for k, v in zip(unique, counts)} }")

    # ------------------------------------------------------------------
    # 5. Stratified 80/20 train-test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=random_state,
        stratify=y,
    )

    # ------------------------------------------------------------------
    # 6. StandardScaler (fit on train, transform both)
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"[Preprocessing] Done — '{dataset_name}' ready.\n")

    return X_train, X_test, y_train, y_test, feature_names
