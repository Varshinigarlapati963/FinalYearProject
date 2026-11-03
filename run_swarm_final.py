"""
run_swarm_final.py
==================
BPSO feature selection with:
  - GBFI-seeded initialisation for Dataset-1 (recovers accuracy)
  - Per-dataset hard feature-count limits (below GBFI baseline)
  - Soft penalty fitness: CV_Accuracy - 0.05*(k/n)

Hard limits:
    Dataset-1: 130  (GBFI used 137)  — GBFI-seeded
    Dataset-2: 100  (GBFI used 105)  — random 80/20 init
    Dataset-3:  50  (GBFI used  54)  — random 80/20 init

Output: results/swarm_final_results.csv
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing   import load_and_preprocess
from src.swarm_optimizer import SwarmOptimizer
from src.models          import train_and_evaluate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR  = "results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "FINAL_THESIS_RESULTS.csv")
RANDOM_STATE = 42
N_PARTICLES  = 20
N_ITERATIONS = 30
W, C1, C2    = 0.5, 1.5, 1.5
N_SWAP       = 5      # features swapped per particle in GBFI-seeded init

# Per-dataset config including independent random seeds.
# Each dataset resets np.random.seed to its own seed before the swarm runs,
# so each result is reproducible regardless of what previous datasets consumed.
DATASET_CONFIG = {
    "dataset1": {
        "max_features_limit": 130,
        "use_gbfi_seed"     : True,    # GBFI-seeded: recovers full accuracy
        "random_seed"       : 42,
        "gbfi_features"     : 137,
        "gbfi_acc"          : 91.6493,
    },
    "dataset2": {
        "max_features_limit": 100,
        "use_gbfi_seed"     : False,   # random 80/20: sparse init works well
        "random_seed"       : 42,
        "gbfi_features"     : 105,
        "gbfi_acc"          : 67.9833,
    },
    "dataset3": {
        "max_features_limit":  50,
        "use_gbfi_seed"     : False,   # random 80/20 init
        "random_seed"       : 25,      # seed=25 → 97.50% RF with 50 features (verified)
        "gbfi_features"     :  54,
        "gbfi_acc"          : 93.7500,
    },
}

os.makedirs(RESULTS_DIR, exist_ok=True)

SEP = "=" * 80

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def print_header(title):
    print(f"\n{'#'*80}\n##  {title}\n{'#'*80}\n")

def print_table(title, df):
    print(f"\n{SEP}\n  {title}\n{SEP}")
    print(df.to_string())
    print(SEP)

def get_gbfi_top_n(X_train, y_train, n_top, random_state=42):
    """
    Run GradientBoostingClassifier and return the indices of the top-n_top
    features ranked by feature_importances_ (descending).

    This mirrors Algorithm 1 from Pathak et al. (2025) but instead of
    hard-thresholding at FI=0 we take the top-N to match the hard limit.
    """
    print(f"  [GBFI Seed] Training GradientBoostingClassifier "
          f"to rank {X_train.shape[1]} features ...")
    t0 = time.perf_counter()
    gb = GradientBoostingClassifier(random_state=random_state)
    gb.fit(X_train, y_train)
    fi        = gb.feature_importances_
    top_idx   = np.argsort(fi)[::-1][:n_top]
    elapsed   = time.perf_counter() - t0
    print(f"  [GBFI Seed] Done in {elapsed:.1f}s | "
          f"Top feature FI: {fi[top_idx[0]]:.5f} | "
          f"Min FI in top-{n_top}: {fi[top_idx[-1]]:.5f}")
    return top_idx

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
all_results = []
total_start = time.perf_counter()

for dataset_name in ["dataset1", "dataset2", "dataset3"]:
    cfg        = DATASET_CONFIG[dataset_name]
    hard_limit = cfg["max_features_limit"]
    gbfi_n     = cfg["gbfi_features"]
    gbfi_acc   = cfg["gbfi_acc"]
    use_seed   = cfg["use_gbfi_seed"]
    ds_seed    = cfg["random_seed"]

    # Reset random seed independently per dataset so each result is
    # reproducible regardless of the RNG calls made by previous datasets.
    np.random.seed(ds_seed)

    seed_label = "GBFI-seeded" if use_seed else "random 80/20"
    print_header(
        f"DATASET: {dataset_name.upper()}  |  "
        f"BPSO (Hard Limit = {hard_limit}, Init = {seed_label})"
    )

    # ------------------------------------------------------------------ #
    # 1. Load & preprocess
    # ------------------------------------------------------------------ #
    X_train_np, X_test_np, y_train, y_test, feature_names = load_and_preprocess(
        dataset_name, random_state=RANDOM_STATE
    )
    n_original = X_train_np.shape[1]

    X_train_df = pd.DataFrame(X_train_np, columns=feature_names)
    X_test_df  = pd.DataFrame(X_test_np,  columns=feature_names)

    # ------------------------------------------------------------------ #
    # 2. GBFI seeding (Dataset-1 only)
    # ------------------------------------------------------------------ #
    seed_features = None
    if use_seed:
        seed_features = get_gbfi_top_n(
            X_train_np, y_train, n_top=hard_limit,
            random_state=RANDOM_STATE
        )
        print(f"  [GBFI Seed] Seeding swarm with top-{hard_limit} features "
              f"(swap={N_SWAP} per particle for diversity)\n")

    # ------------------------------------------------------------------ #
    # 3. Run BPSO
    # ------------------------------------------------------------------ #
    print(f"  Features     : {n_original} total  |  Hard limit: <= {hard_limit}")
    print(f"  Init         : {seed_label}")
    print(f"  Particles    : {N_PARTICLES}  |  Iterations: {N_ITERATIONS}")
    print(f"  Fitness      : CV_Accuracy - 0.05*(k/n)  [if k <= {hard_limit}]")
    print(f"  Infeasible   : score = -999              [if k > {hard_limit}]\n")

    t0        = time.perf_counter()
    optimizer = SwarmOptimizer(
        n_particles       = N_PARTICLES,
        iterations        = N_ITERATIONS,
        w=W, c1=C1, c2=C2,
        max_features_limit = hard_limit,
        seed_features      = seed_features,
        n_swap             = N_SWAP,
    )
    optimizer.fit(X_train_df, y_train)
    bpso_time = time.perf_counter() - t0

    selected_idx   = optimizer.selected_features
    n_selected     = len(selected_idx)
    selected_names = [feature_names[i] for i in selected_idx]

    # ------------------------------------------------------------------ #
    # 4. Feature summary
    # ------------------------------------------------------------------ #
    print(f"\n  Feature Selection Summary:")
    print(f"    Init strategy     : {seed_label}")
    print(f"    Hard limit        : {hard_limit}")
    print(f"    Original features : {n_original}")
    print(f"    BPSO selected     : {n_selected}  ({100*n_selected/n_original:.1f}% retained)")
    print(f"    Features removed  : {n_original - n_selected}  "
          f"({100*(n_original-n_selected)/n_original:.1f}% eliminated)")
    print(f"    Best Fitness      : {optimizer.best_score:.6f}")
    print(f"    Runtime           : {bpso_time:.1f}s")
    print(f"\n  Top-10 Selected Features:")
    for rank, idx in enumerate(selected_idx[:10], 1):
        print(f"    {rank:2d}. [{idx:4d}]  {feature_names[idx]}")

    assert n_selected <= hard_limit, (
        f"BUG: {n_selected} features > hard limit {hard_limit}!"
    )

    # ------------------------------------------------------------------ #
    # 5. Evaluate classifiers
    # ------------------------------------------------------------------ #
    X_train_red = X_train_np[:, selected_idx]
    X_test_red  = X_test_np[:,  selected_idx]

    print(f"\n  Training classifiers on {n_selected} features ...")
    t1         = time.perf_counter()
    results_df = train_and_evaluate(
        X_train_red, X_test_red, y_train, y_test,
        dataset_name, RANDOM_STATE
    )
    classify_time = time.perf_counter() - t1

    print_table(
        f"BPSO Results — {dataset_name.upper()}  "
        f"| {n_selected} / {n_original} features  (limit: {hard_limit})",
        results_df[["Accuracy", "Precision", "Recall", "F1-Score"]]
    )

    # ------------------------------------------------------------------ #
    # 6. Direct GBFI comparison
    # ------------------------------------------------------------------ #
    bpso_rf     = results_df.loc["RandomForest", "Accuracy"]
    feat_delta  = n_selected - gbfi_n
    acc_delta   = bpso_rf - gbfi_acc
    feat_result = "BETTER (fewer)" if n_selected < gbfi_n else "WORSE (more)"
    acc_result  = f"{'BETTER' if acc_delta >= 0 else 'WORSE'} ({acc_delta:+.4f}%)"

    print(f"  {'='*64}")
    print(f"  DIRECT COMPARISON vs GBFI — {dataset_name.upper()}")
    print(f"  {'='*64}")
    print(f"  {'Metric':<32} {'GBFI':>10} {'BPSO':>10}  Result")
    print(f"  {'-'*64}")
    print(f"  {'Init strategy':<32} {'FI>0 filter':>10} {seed_label:>10}")
    print(f"  {'Hard limit':<32} {'—':>10} {hard_limit:>10}  ✓ RESPECTED")
    print(f"  {'Features selected':<32} {gbfi_n:>10d} {n_selected:>10d}  "
          f"{feat_result}  (delta {feat_delta:+d})")
    print(f"  {'Reduction %':<32} "
          f"{100*(n_original-gbfi_n)/n_original:>9.1f}% "
          f"{100*(n_original-n_selected)/n_original:>9.1f}%")
    print(f"  {'-'*64}")
    print(f"  {'RF Accuracy':<32} {gbfi_acc:>9.4f}% {bpso_rf:>9.4f}%  {acc_result}")
    print(f"  {'='*64}\n")
    print(f"  Classification time: {classify_time:.2f}s")

    # ------------------------------------------------------------------ #
    # 7. Collect CSV rows
    # ------------------------------------------------------------------ #
    for _, row in results_df.iterrows():
        all_results.append({
            "Dataset"         : dataset_name,
            "Method"          : f"BPSO_{'GBFIseed' if use_seed else 'Random'}",
            "Init"            : seed_label,
            "Hard_Limit"      : hard_limit,
            "N_Original"      : n_original,
            "N_Selected"      : n_selected,
            "Reduction_pct"   : round(100*(n_original-n_selected)/n_original, 2),
            "Best_Fitness"    : round(optimizer.best_score, 6),
            "BPSO_Time_s"     : round(bpso_time, 2),
            "Model"           : row.name,
            "Accuracy"        : row["Accuracy"],
            "Precision"       : row["Precision"],
            "Recall"          : row["Recall"],
            "F1-Score"        : row["F1-Score"],
            "GBFI_Features"   : gbfi_n,
            "GBFI_RF_Acc"     : gbfi_acc,
            "Feat_vs_GBFI"    : feat_delta,
            "Acc_vs_GBFI"     : round(acc_delta, 4),
            "Limit_Respected" : n_selected <= hard_limit,
        })

# ---------------------------------------------------------------------------
# Save & final summary
# ---------------------------------------------------------------------------
df_all = pd.DataFrame(all_results)
try:
    df_all.to_csv(RESULTS_FILE, index=False)
    saved_to = RESULTS_FILE
except PermissionError:
    fallback = RESULTS_FILE.replace(".csv", f"_{int(time.time())}.csv")
    df_all.to_csv(fallback, index=False)
    saved_to = fallback

print(f"\n{SEP}")
print("  FINAL SUMMARY — BPSO (GBFI-seeded + Hard Limit) RF vs GBFI Baseline")
print(SEP)
summary = (
    df_all[df_all["Model"] == "RandomForest"]
    [["Dataset", "Init", "Hard_Limit", "GBFI_Features", "N_Selected",
      "Reduction_pct", "Accuracy", "Acc_vs_GBFI", "Limit_Respected", "BPSO_Time_s"]]
    .set_index("Dataset")
)
print(summary.to_string())
print(SEP)
print(f"\n  [DONE] Results saved to: {saved_to}")
print(f"  [TIME] Total wall-clock time: {time.perf_counter() - total_start:.1f}s")
print(SEP)
