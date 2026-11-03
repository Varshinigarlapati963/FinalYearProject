"""
main_baseline.py
----------------
Baseline Reproduction Script
Paper: Pathak et al. (2025) — "Machine learning approach to detect android
       malware using feature-selection based on feature importance score."

Runs for all 3 datasets:
  1. Load & preprocess
  2. Apply GBFI feature selection
  3. Train 4 classifiers on FULL features
  4. Train 4 classifiers on REDUCED (GBFI) features
  5. Print comparison tables
  6. Save all results to /results/baseline_results.csv

Expected approximate benchmarks (from paper):
  Dataset-1 RF: ~91.48% (full)  ~91.39% (reduced)
  Dataset-2 RF: ~74.88% (full)  ~74.63% (reduced)
  Dataset-3 RF: ~93.96% (full & reduced)
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

# Force UTF-8 output on Windows so special characters never crash the script
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Ensure the script can be run from the project root
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing  import load_and_preprocess
from src.baseline_gbfi  import select_features_gbfi, get_feature_importance_scores
from src.models         import train_and_evaluate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASETS      = ["dataset1", "dataset2", "dataset3"]
RESULTS_DIR   = "results"
RESULTS_FILE  = os.path.join(RESULTS_DIR, "baseline_results.csv")
RANDOM_STATE  = 42

os.makedirs(RESULTS_DIR, exist_ok=True)

PAPER_BENCHMARKS = {
    "dataset1": {"full_rf_acc": 91.48, "reduced_rf_acc": 91.39},
    "dataset2": {"full_rf_acc": 74.88, "reduced_rf_acc": 74.63},
    "dataset3": {"full_rf_acc": 93.96, "reduced_rf_acc": 93.96},
}

# ---------------------------------------------------------------------------
# Helper: pretty-print a DataFrame table
# ---------------------------------------------------------------------------
def print_table(title: str, df: pd.DataFrame):
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(df.to_string())
    print(sep)


# ---------------------------------------------------------------------------
# Helper: compute % gain/loss in accuracy (reduced vs full)
# ---------------------------------------------------------------------------
def accuracy_delta(full_df: pd.DataFrame, reduced_df: pd.DataFrame) -> pd.Series:
    delta = reduced_df["Accuracy"] - full_df["Accuracy"]
    return delta.map(lambda d: f"{d:+.4f}%")  # positive = gain, negative = loss


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
all_results = []   # collect rows for final CSV

total_start = time.perf_counter()

for dataset_name in DATASETS:
    header = f"\n{'#' * 80}\n##  {dataset_name.upper()}  ##\n{'#' * 80}"
    print(header)

    # ------------------------------------------------------------------ #
    # 1. Load & preprocess
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess(
        dataset_name, random_state=RANDOM_STATE
    )

    n_full_features = X_train.shape[1]

    # ------------------------------------------------------------------ #
    # 2. GBFI feature selection (Module 2)
    # ------------------------------------------------------------------ #
    gbfi_start = time.perf_counter()
    (X_train_red, X_test_red,
     sel_idx, sel_names, fi_scores) = select_features_gbfi(
        X_train, X_test, y_train, feature_names, random_state=RANDOM_STATE
    )
    gbfi_time = time.perf_counter() - gbfi_start
    n_reduced_features = X_train_red.shape[1]

    print(f"\n  GBFI selection time: {gbfi_time:.2f}s")

    # Print top-10 most important features
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": fi_scores})
    fi_df = fi_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    print(f"\n  Top-10 Features by Importance:")
    print(fi_df.head(10).to_string(index=True))

    # ------------------------------------------------------------------ #
    # 3 & 4. Train/evaluate on FULL and REDUCED feature sets
    # ------------------------------------------------------------------ #
    print(f"\n  Training on FULL features ({n_full_features}) ...")
    t0 = time.perf_counter()
    results_full = train_and_evaluate(
        X_train, X_test, y_train, y_test, dataset_name, RANDOM_STATE
    )
    time_full = time.perf_counter() - t0

    print(f"  Training on REDUCED features ({n_reduced_features}) ...")
    t0 = time.perf_counter()
    results_reduced = train_and_evaluate(
        X_train_red, X_test_red, y_train, y_test, dataset_name, RANDOM_STATE
    )
    time_reduced = time.perf_counter() - t0

    # ------------------------------------------------------------------ #
    # 5. Print comparison tables
    # ------------------------------------------------------------------ #

    # Table A — Full feature metrics
    print_table(
        f"TABLE A — {dataset_name.upper()}  |  FULL Feature Set ({n_full_features} features)",
        results_full[["Accuracy", "Precision", "Recall", "F1-Score"]]
    )

    # Table B — Reduced feature metrics + % change
    results_reduced_display = results_reduced[["Accuracy", "Precision", "Recall", "F1-Score"]].copy()
    results_reduced_display["Delta Acc vs Full"] = accuracy_delta(
        results_full, results_reduced
    )
    print_table(
        f"TABLE B — {dataset_name.upper()}  |  REDUCED Feature Set ({n_reduced_features} features)",
        results_reduced_display
    )

    # ------------------------------------------------------------------ #
    # 6. Execution time summary
    # ------------------------------------------------------------------ #
    bench = PAPER_BENCHMARKS[dataset_name]
    rf_full_acc    = results_full.loc["RandomForest", "Accuracy"]
    rf_reduced_acc = results_reduced.loc["RandomForest", "Accuracy"]

    print(f"\n  [Timing] Execution Time Summary:")
    print(f"     GBFI selection    : {gbfi_time:8.2f} s")
    print(f"     Full training     : {time_full:8.2f} s")
    print(f"     Reduced training  : {time_reduced:8.2f} s")
    print(f"     Total (this ds)   : {gbfi_time + time_full + time_reduced:8.2f} s")

    print(f"\n  [RF vs Paper] RandomForest Accuracy vs Paper Benchmarks:")
    print(f"     Full   - Got: {rf_full_acc:.4f}%   |  Paper: {bench['full_rf_acc']:.2f}%")
    print(f"     Reduced- Got: {rf_reduced_acc:.4f}%   |  Paper: {bench['reduced_rf_acc']:.2f}%")

    # ------------------------------------------------------------------ #
    # Collect rows for final CSV
    # ------------------------------------------------------------------ #
    for model in results_full.index:
        for fs_label, res_df, n_feat in [
            ("Full",    results_full,    n_full_features),
            ("Reduced", results_reduced, n_reduced_features),
        ]:
            row = {
                "Dataset"         : dataset_name,
                "FeatureSet"      : fs_label,
                "N_Features"      : n_feat,
                "Model"           : model,
                "Accuracy"        : res_df.loc[model, "Accuracy"],
                "Precision"       : res_df.loc[model, "Precision"],
                "Recall"          : res_df.loc[model, "Recall"],
                "F1-Score"        : res_df.loc[model, "F1-Score"],
                "Train_Time_s"    : res_df.loc[model, "Train_Time_s"],
                "Inference_Time_s": res_df.loc[model, "Inference_Time_s"],
            }
            all_results.append(row)

# ---------------------------------------------------------------------------
# Save combined results
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(all_results)
results_df.to_csv(RESULTS_FILE, index=False)
print(f"\n\n{'=' * 80}")
print(f"  [DONE] All results saved to: {RESULTS_FILE}")
print(f"  [TIME] Total wall-clock time: {time.perf_counter() - total_start:.1f}s")
print(f"{'=' * 80}")

# ---------------------------------------------------------------------------
# Final summary table across all datasets x models
# ---------------------------------------------------------------------------
print("\n\n  === FINAL SUMMARY TABLE (Reduced Feature Set) ===\n")
summary = (
    results_df[results_df["FeatureSet"] == "Reduced"]
    .pivot_table(index="Model", columns="Dataset", values="Accuracy")
    .round(4)
)
print(summary.to_string())
print()
