"""
main_quantum_swarm.py
---------------------
Integration script for Hybrid Q-BPSO Feature Selection + Classification.

For each of the 3 datasets:
  1. Load & preprocess data
  2. Run Q-BPSO to find the optimal feature subset
  3. Train KNN / GaussianNB / DecisionTree / RandomForest on selected features
  4. Compute Accuracy, Precision, Recall, F1-Score
  5. Compare against GBFI baseline results
  6. Save results CSV and convergence curve plots

Expected runtime:
  Dataset-1 (~1490 features, 9580 training samples): ~15-25 min
  Dataset-2 (~173 features, 24000 training samples): ~8-12 min
  Dataset-3 (~330 features, 318 training samples):   <1 min
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe on all OS)
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Force UTF-8 on Windows cp1252 terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Path setup — allow running from project root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing  import load_and_preprocess
from src.quantum_swarm  import run_qbpso, apply_mask
from src.models         import train_and_evaluate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASETS      = ["dataset1", "dataset2", "dataset3"]
RESULTS_DIR   = "results"
RESULTS_FILE  = os.path.join(RESULTS_DIR, "quantum_swarm_results_v4_strict.csv")
RANDOM_STATE  = 42
N_PARTICLES   = 20
N_ITERATIONS  = 50    # v4: 50 iterations, sparse init, strong penalty

# GBFI baseline feature counts (from baseline run) — used for direct comparison
GBFI_FEATURES = {
    "dataset1": 137,   # GBFI reduced 1490 -> 137
    "dataset2": 105,   # GBFI reduced  173 -> 105
    "dataset3":  54,   # GBFI reduced  330 ->  54
}
GBFI_ACCURACY = {
    "dataset1": 91.6493,
    "dataset2": 67.9833,
    "dataset3": 93.7500,
}

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load baseline results for comparison (if available)
BASELINE_FILE = os.path.join(RESULTS_DIR, "baseline_results.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_sep(char="=", width=80):
    print(char * width)

def print_header(text, char="#"):
    print_sep(char)
    print(f"{char}{char}  {text}")
    print_sep(char)

def print_table(title: str, df: pd.DataFrame):
    print_sep()
    print(f"  {title}")
    print_sep()
    print(df.to_string())
    print_sep()


def plot_convergence(convergence: list, dataset_name: str, n_selected: int,
                     n_original: int):
    """
    Save a convergence curve (Fitness vs Iteration) to /results/.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    iterations = list(range(1, len(convergence) + 1))
    ax.plot(iterations, convergence, color="#2563EB", linewidth=2.0,
            marker="o", markersize=3.5, markerfacecolor="#1D4ED8", label="gBest Fitness")
    ax.fill_between(iterations, convergence, alpha=0.12, color="#2563EB")

    ax.set_title(
        f"Q-BPSO Convergence — {dataset_name.upper()}\n"
        f"Features: {n_original} → {n_selected} selected",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Best Fitness", fontsize=11)
    ax.set_xlim(1, len(convergence))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"convergence_{dataset_name}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [Plot] Convergence curve saved -> {out_path}")


def load_baseline_rf(dataset_name: str, feature_set: str = "Reduced") -> float | None:
    """Retrieve baseline RF accuracy for comparison, returns None if file missing."""
    if not os.path.exists(BASELINE_FILE):
        return None
    try:
        df = pd.read_csv(BASELINE_FILE)
        row = df[(df["Dataset"] == dataset_name) &
                 (df["Model"] == "RandomForest") &
                 (df["FeatureSet"] == feature_set)]
        return float(row["Accuracy"].values[0]) if len(row) > 0 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
all_results  = []
total_start  = time.perf_counter()

for dataset_name in DATASETS:
    print_header(f"DATASET: {dataset_name.upper()}  |  Q-BPSO Feature Selection")

    # ------------------------------------------------------------------ #
    # 1. Load & preprocess
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess(
        dataset_name, random_state=RANDOM_STATE
    )
    n_original = X_train.shape[1]

    # ------------------------------------------------------------------ #
    # 2. Run Q-BPSO
    # ------------------------------------------------------------------ #
    print(f"  Running Q-BPSO v2: {N_PARTICLES} particles x {N_ITERATIONS} iterations ...")
    print(f"  Fitness = 0.95 * CV_Accuracy - 0.05 * (n_selected / {n_original})")
    print(f"  Rotation: adaptive delta_max pi/5 -> pi/20 over {N_ITERATIONS} iters")
    print()

    qbpso_start = time.perf_counter()
    result = run_qbpso(
        X_train       = X_train,
        y_train       = y_train,
        feature_names = feature_names,
        n_particles   = N_PARTICLES,
        n_iterations  = N_ITERATIONS,
        c1            = 2.0,
        c2            = 2.0,
        mutation_rate = 0.05,          # v3: 5% bit-flip mutation
        random_state  = RANDOM_STATE,
        verbose       = True,
    )
    qbpso_time = time.perf_counter() - qbpso_start

    selected_indices = result["selected_indices"]
    selected_names   = result["selected_names"]
    n_selected       = result["n_selected"]
    gbest_fitness    = result["gbest_fitness"]
    convergence      = result["convergence"]

    X_train_red = result["X_train_reduced"]
    X_test_red  = apply_mask(X_test, result["gbest_mask"])

    # ------------------------------------------------------------------ #
    # 3. Print feature selection summary
    # ------------------------------------------------------------------ #
    print(f"\n  Feature Selection Summary:")
    print(f"    Original features : {n_original}")
    print(f"    Q-BPSO selected   : {n_selected}  "
          f"({100 * n_selected / n_original:.1f}% retained)")
    print(f"    Features removed  : {n_original - n_selected}  "
          f"({100 * (n_original - n_selected) / n_original:.1f}% eliminated)")
    print(f"    Best Fitness      : {gbest_fitness:.6f}")
    print(f"    Q-BPSO runtime    : {qbpso_time:.1f}s")

    print(f"\n  Top-10 Selected Features (by index):")
    for rank, (idx, name) in enumerate(
            zip(selected_indices[:10], selected_names[:10]), start=1):
        print(f"    {rank:2d}. [{idx:4d}]  {name}")

    # ------------------------------------------------------------------ #
    # 4. Plot convergence curve
    # ------------------------------------------------------------------ #
    plot_convergence(convergence, dataset_name, n_selected, n_original)

    # ------------------------------------------------------------------ #
    # 5. Train & evaluate classifiers on Q-BPSO reduced feature set
    # ------------------------------------------------------------------ #
    print(f"\n  Training classifiers on Q-BPSO features ({n_selected}) ...")
    t0 = time.perf_counter()
    results_df = train_and_evaluate(
        X_train_red, X_test_red, y_train, y_test,
        dataset_name, RANDOM_STATE
    )
    classify_time = time.perf_counter() - t0

    print_table(
        f"Q-BPSO Results — {dataset_name.upper()}  "
        f"| {n_selected} features ({n_original - n_selected} eliminated)",
        results_df[["Accuracy", "Precision", "Recall", "F1-Score"]]
    )

    # ------------------------------------------------------------------ #
    # 6. Compare against GBFI baseline (accuracy + feature count)
    # ------------------------------------------------------------------ #
    gbfi_n       = GBFI_FEATURES.get(dataset_name)
    gbfi_acc     = GBFI_ACCURACY.get(dataset_name)
    qbpso_rf_acc = results_df.loc["RandomForest", "Accuracy"]

    print(f"\n  ={'='*60}")
    print(f"  DIRECT COMPARISON vs GBFI BASELINE — {dataset_name.upper()}")
    print(f"  {'='*60}")
    print(f"  {'Metric':<28} {'GBFI':>12} {'Q-BPSO v4':>12} {'Result':>12}")
    print(f"  {'-'*64}")

    # Feature count comparison
    if gbfi_n is not None:
        feat_delta   = n_selected - gbfi_n
        feat_verdict = "BETTER (fewer)" if n_selected < gbfi_n else "WORSE (more)"
        print(f"  {'Features selected':<28} {gbfi_n:>12d} {n_selected:>12d}  "
              f"{feat_verdict}")
        print(f"  {'Features eliminated':<28} "
              f"{n_original - gbfi_n:>12d} {n_original - n_selected:>12d}  "
              f"(delta: {feat_delta:+d})")
        print(f"  {'Reduction %':<28} "
              f"{100*(n_original-gbfi_n)/n_original:>11.1f}% "
              f"{100*(n_original-n_selected)/n_original:>11.1f}%")
        print(f"  {'-'*64}")

    # Accuracy comparison
    if gbfi_acc is not None:
        acc_delta   = qbpso_rf_acc - gbfi_acc
        acc_verdict = f"{'BETTER' if acc_delta >= 0 else 'WORSE'} ({acc_delta:+.4f}%)"
        print(f"  {'RF Accuracy':<28} {gbfi_acc:>11.4f}% {qbpso_rf_acc:>11.4f}%  "
              f"{acc_verdict}")
    print(f"  {'='*60}\n")

    # ------------------------------------------------------------------ #
    # 7. Collect rows for final CSV
    # ------------------------------------------------------------------ #
    for model in results_df.index:
        row = {
            "Dataset"          : dataset_name,
            "Method"           : "Q-BPSO",
            "N_Original"       : n_original,
            "N_Selected"       : n_selected,
            "Reduction_pct"    : round(100 * (n_original - n_selected) / n_original, 2),
            "Best_Fitness"     : round(gbest_fitness, 6),
            "QBPSO_Time_s"     : round(qbpso_time, 2),
            "Model"            : model,
            "Accuracy"         : results_df.loc[model, "Accuracy"],
            "Precision"        : results_df.loc[model, "Precision"],
            "Recall"           : results_df.loc[model, "Recall"],
            "F1-Score"         : results_df.loc[model, "F1-Score"],
            "Train_Time_s"     : results_df.loc[model, "Train_Time_s"],
            "Inference_Time_s" : results_df.loc[model, "Inference_Time_s"],
        }
        all_results.append(row)

    print(f"\n  Classification time: {classify_time:.2f}s")

# ---------------------------------------------------------------------------
# Save all results to CSV
# ---------------------------------------------------------------------------
results_df_all = pd.DataFrame(all_results)
try:
    results_df_all.to_csv(RESULTS_FILE, index=False)
    _saved_to = RESULTS_FILE
except PermissionError:
    import time as _t
    _fallback = os.path.join(
        RESULTS_DIR, f"quantum_swarm_results_v2_{int(_t.time())}.csv"
    )
    results_df_all.to_csv(_fallback, index=False)
    _saved_to = _fallback
    print(f"  [WARN] Primary file locked — saved to: {_fallback}")

# ---------------------------------------------------------------------------
# Final summary table
# ---------------------------------------------------------------------------
print()
print_sep()
print("  FINAL SUMMARY — Q-BPSO RandomForest Accuracy Across All Datasets")
print_sep()

summary = (
    results_df_all[results_df_all["Model"] == "RandomForest"]
    [["Dataset", "N_Original", "N_Selected", "Reduction_pct",
      "Best_Fitness", "Accuracy", "F1-Score", "QBPSO_Time_s"]]
    .set_index("Dataset")
)
print(summary.to_string())
print_sep()

print(f"\n  [DONE] Results saved to: {_saved_to}")
print(f"  [TIME] Total wall-clock time: {time.perf_counter() - total_start:.1f}s")
print_sep()
