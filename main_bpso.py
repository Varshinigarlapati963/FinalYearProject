"""
main_bpso.py
============
Main runner for Binary PSO (BPSO) feature selection using SwarmOptimizer.

Runs on all 3 datasets and saves results to:
    results/bpso_results.csv

Comparison targets (GBFI baseline):
    Dataset-1: 137 features, 91.65% RF accuracy
    Dataset-2: 105 features, 67.98% RF accuracy
    Dataset-3:  54 features, 93.75% RF accuracy
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing   import load_and_preprocess
from src.swarm_optimizer import SwarmOptimizer
from src.models          import train_and_evaluate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASETS     = ["dataset1", "dataset2", "dataset3"]
RESULTS_DIR  = "results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "bpso_results.csv")
RANDOM_STATE = 42

# BPSO hyper-parameters (matching SwarmOptimizer defaults)
N_PARTICLES  = 20
N_ITERATIONS = 30
W            = 0.5    # inertia
C1           = 1.5    # cognitive
C2           = 1.5    # social

# GBFI baseline for direct comparison
GBFI_FEATURES = {"dataset1": 137, "dataset2": 105, "dataset3": 54}
GBFI_ACCURACY = {"dataset1": 91.6493, "dataset2": 67.9833, "dataset3": 93.7500}

os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SEP = "=" * 80

def print_header(title):
    print(f"\n{'#'*80}")
    print(f"##  {title}")
    print(f"{'#'*80}\n")

def print_table(title, df):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(df.to_string())
    print(f"{'='*80}\n")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
all_results = []
total_start = time.perf_counter()

for dataset_name in DATASETS:
    print_header(f"DATASET: {dataset_name.upper()}  |  BPSO Feature Selection")

    # ------------------------------------------------------------------ #
    # 1. Load & preprocess
    # ------------------------------------------------------------------ #
    X_train_np, X_test_np, y_train, y_test, feature_names = load_and_preprocess(
        dataset_name, random_state=RANDOM_STATE
    )
    n_original = X_train_np.shape[1]

    # SwarmOptimizer uses X.iloc[], so wrap numpy arrays as DataFrames
    X_train_df = pd.DataFrame(X_train_np, columns=feature_names)
    X_test_df  = pd.DataFrame(X_test_np,  columns=feature_names)

    # ------------------------------------------------------------------ #
    # 2. Run BPSO feature selection
    # ------------------------------------------------------------------ #
    print(f"  Features: {n_original} | Samples (train): {len(y_train)}")
    print(f"  BPSO: {N_PARTICLES} particles x {N_ITERATIONS} iterations")
    print(f"  Fitness = CV_Accuracy - 0.20*(k/n)  |  Init: 20% P(select)\n")

    bpso_start = time.perf_counter()
    optimizer  = SwarmOptimizer(
        n_particles=N_PARTICLES,
        iterations=N_ITERATIONS,
        w=W, c1=C1, c2=C2,
    )
    optimizer.fit(X_train_df, y_train)
    bpso_time = time.perf_counter() - bpso_start

    selected_idx   = optimizer.selected_features          # numpy array of indices
    n_selected     = len(selected_idx)
    selected_names = [feature_names[i] for i in selected_idx]

    # ------------------------------------------------------------------ #
    # 3. Feature selection summary
    # ------------------------------------------------------------------ #
    print(f"\n  Feature Selection Summary:")
    print(f"    Original features : {n_original}")
    print(f"    BPSO selected     : {n_selected}  ({100*n_selected/n_original:.1f}% retained)")
    print(f"    Features removed  : {n_original - n_selected}  "
          f"({100*(n_original-n_selected)/n_original:.1f}% eliminated)")
    print(f"    Best Fitness      : {optimizer.best_score:.6f}")
    print(f"    BPSO runtime      : {bpso_time:.1f}s")

    print(f"\n  Top-10 Selected Features (by index):")
    for rank, idx in enumerate(selected_idx[:10], 1):
        print(f"    {rank:2d}. [{idx:4d}]  {feature_names[idx]}")

    # ------------------------------------------------------------------ #
    # 4. Build reduced train/test arrays
    # ------------------------------------------------------------------ #
    X_train_red = X_train_np[:, selected_idx]
    X_test_red  = X_test_np[:,  selected_idx]

    # ------------------------------------------------------------------ #
    # 5. Train & evaluate classifiers on BPSO reduced feature set
    # ------------------------------------------------------------------ #
    print(f"\n  Training classifiers on BPSO features ({n_selected}) ...")
    t0         = time.perf_counter()
    results_df = train_and_evaluate(
        X_train_red, X_test_red, y_train, y_test,
        dataset_name, RANDOM_STATE
    )
    classify_time = time.perf_counter() - t0

    print_table(
        f"BPSO Results — {dataset_name.upper()}  "
        f"| {n_selected} features ({n_original - n_selected} eliminated)",
        results_df[["Accuracy", "Precision", "Recall", "F1-Score"]]
    )

    # ------------------------------------------------------------------ #
    # 6. Direct comparison vs GBFI baseline
    # ------------------------------------------------------------------ #
    gbfi_n   = GBFI_FEATURES.get(dataset_name)
    gbfi_acc = GBFI_ACCURACY.get(dataset_name)
    bpso_rf  = results_df.loc["RandomForest", "Accuracy"]

    print(f"  {'='*62}")
    print(f"  DIRECT COMPARISON vs GBFI BASELINE — {dataset_name.upper()}")
    print(f"  {'='*62}")
    print(f"  {'Metric':<30} {'GBFI':>10} {'BPSO':>10}  Result")
    print(f"  {'-'*62}")
    if gbfi_n:
        feat_verdict = "BETTER (fewer)" if n_selected < gbfi_n else "WORSE (more)"
        feat_delta   = n_selected - gbfi_n
        print(f"  {'Features selected':<30} {gbfi_n:>10d} {n_selected:>10d}  "
              f"{feat_verdict}  (delta {feat_delta:+d})")
        print(f"  {'Reduction %':<30} "
              f"{100*(n_original-gbfi_n)/n_original:>9.1f}% "
              f"{100*(n_original-n_selected)/n_original:>9.1f}%")
        print(f"  {'-'*62}")
    if gbfi_acc:
        acc_delta   = bpso_rf - gbfi_acc
        acc_verdict = f"{'BETTER' if acc_delta >= 0 else 'WORSE'} ({acc_delta:+.4f}%)"
        print(f"  {'RF Accuracy':<30} {gbfi_acc:>9.4f}% {bpso_rf:>9.4f}%  {acc_verdict}")
    print(f"  {'='*62}\n")

    print(f"  Classification time: {classify_time:.2f}s")

    # ------------------------------------------------------------------ #
    # 7. Collect results for CSV
    # ------------------------------------------------------------------ #
    for _, row in results_df.iterrows():
        all_results.append({
            "Dataset"       : dataset_name,
            "Method"        : "BPSO",
            "N_Original"    : n_original,
            "N_Selected"    : n_selected,
            "Reduction_pct" : round(100*(n_original-n_selected)/n_original, 2),
            "Best_Fitness"  : round(optimizer.best_score, 6),
            "BPSO_Time_s"   : round(bpso_time, 2),
            "Model"         : row.name,
            "Accuracy"      : row["Accuracy"],
            "Precision"     : row["Precision"],
            "Recall"        : row["Recall"],
            "F1-Score"      : row["F1-Score"],
            "GBFI_Features" : gbfi_n,
            "GBFI_RF_Acc"   : gbfi_acc,
            "Feat_vs_GBFI"  : (n_selected - gbfi_n) if gbfi_n else None,
            "Acc_vs_GBFI"   : round(bpso_rf - gbfi_acc, 4) if gbfi_acc else None,
        })

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
results_df_all = pd.DataFrame(all_results)
try:
    results_df_all.to_csv(RESULTS_FILE, index=False)
    print(f"\n  [DONE] Results saved to: {RESULTS_FILE}")
except PermissionError:
    fallback = RESULTS_FILE.replace(".csv", f"_{int(time.time())}.csv")
    results_df_all.to_csv(fallback, index=False)
    print(f"\n  [DONE] File locked — saved to: {fallback}")

# ---------------------------------------------------------------------------
# Final summary table
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("  FINAL SUMMARY — BPSO RandomForest Accuracy vs GBFI Baseline")
print(SEP)
summary = (
    results_df_all[results_df_all["Model"] == "RandomForest"]
    [["Dataset", "N_Original", "GBFI_Features", "N_Selected",
      "Reduction_pct", "Accuracy", "Acc_vs_GBFI", "BPSO_Time_s"]]
    .set_index("Dataset")
)
print(summary.to_string())
print(SEP)
print(f"\n  [TIME] Total wall-clock time: {time.perf_counter() - total_start:.1f}s")
print(SEP)
