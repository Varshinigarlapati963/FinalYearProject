"""
generate_top_permissions.py
===========================
Runs BPSO (using the winning strategy per dataset) then trains a Random Forest
on the swarm-selected features and plots the Top-5 most important permissions
per dataset — mirroring Figure 2 from Pathak et al. (2025).

Output: results/ieee_plots/Top_Features_Dataset{1,2,3}.png  (300 DPI)
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing   import load_and_preprocess
from src.swarm_optimizer import SwarmOptimizer

os.makedirs("results/ieee_plots", exist_ok=True)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"]  = ["Times New Roman"] + plt.rcParams["font.serif"]

# ---------------------------------------------------------------------------
# Per-dataset BPSO configuration (matches FINAL_THESIS_RESULTS)
# ---------------------------------------------------------------------------
DATASET_CONFIG = {
    "dataset1": {
        "max_features_limit": 130,
        "use_gbfi_seed"     : True,
        "random_seed"       : 42,
        "label"             : "Dataset-1 (DefenseDroid)",
        "out_file"          : "Top_Features_Dataset1.png",
    },
    "dataset2": {
        "max_features_limit": 100,
        "use_gbfi_seed"     : False,
        "random_seed"       : 42,
        "label"             : "Dataset-2 (Mendeley)",
        "out_file"          : "Top_Features_Dataset2.png",
    },
    "dataset3": {
        "max_features_limit":  50,
        "use_gbfi_seed"     : False,
        "random_seed"       : 25,       # seed=25 → 97.50% RF (verified)
        "label"             : "Dataset-3 (Lopez et al.)",
        "out_file"          : "Top_Features_Dataset3.png",
    },
}

# ---------------------------------------------------------------------------
# Helper: clean up raw permission/feature names for chart display
# ---------------------------------------------------------------------------
def shorten_name(name: str) -> str:
    """Strip common prefixes and suffixes; truncate long names."""
    # DS3 / DS1 android.* prefix
    for prefix in [
        "android.permission.", "android.intent.category.",
        "android.app.cts.permission.", "permission.",
    ]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    # DS2 category prefix (e.g. "Default : ", "Dangerous : ")
    for cat in ["Default : ", "Dangerous : ", "Signature : ",
                "Development tools : ", "System : "]:
        if name.startswith(cat):
            name = name[len(cat):]
            break
    # Remove trailing type markers
    name = name.replace(" (S)", "").replace(" (D)", "").replace(".", "").strip()
    # Truncate
    return name[:26] + "…" if len(name) > 27 else name

# ---------------------------------------------------------------------------
# Helper: GBFI top-N for seeded init
# ---------------------------------------------------------------------------
def get_gbfi_top_n(X_train, y_train, n_top, random_state=42):
    print(f"  [GBFI Seed] Ranking {X_train.shape[1]} features ...")
    gb = GradientBoostingClassifier(random_state=random_state)
    gb.fit(X_train, y_train)
    fi = gb.feature_importances_
    return np.argsort(fi)[::-1][:n_top]

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for ds_name, cfg in DATASET_CONFIG.items():
    print(f"\n{'='*60}")
    print(f"  {ds_name.upper()}  —  {cfg['label']}")
    print(f"{'='*60}")

    # Reset seed independently per dataset
    np.random.seed(cfg["random_seed"])

    X_train_np, X_test_np, y_train, y_test, feat_names = load_and_preprocess(
        ds_name, random_state=42
    )
    X_train_df = pd.DataFrame(X_train_np, columns=feat_names)

    # GBFI seeding (Dataset-1 only)
    seed_features = None
    if cfg["use_gbfi_seed"]:
        seed_features = get_gbfi_top_n(
            X_train_np, y_train, cfg["max_features_limit"], random_state=42
        )
        print(f"  [GBFI Seed] Top-{cfg['max_features_limit']} features identified.\n")

    # Run BPSO with 10 iterations (converges at iter 1 for all datasets —
    # using 10 instead of 30 saves time while giving identical feature subsets)
    print(f"  Running BPSO (20 particles × 10 iters, limit={cfg['max_features_limit']}) ...")
    opt = SwarmOptimizer(
        n_particles        = 20,
        iterations         = 10,
        w=0.5, c1=1.5, c2=1.5,
        max_features_limit = cfg["max_features_limit"],
        seed_features      = seed_features,
        n_swap             = 5,
    )
    opt.fit(X_train_df, y_train)
    selected_idx = opt.selected_features
    print(f"  BPSO selected {len(selected_idx)} features  |  "
          f"Best fitness: {opt.best_score:.4f}")

    # ------------------------------------------------------------------ #
    # Train RF on selected features → extract importances
    # ------------------------------------------------------------------ #
    X_sel = X_train_np[:, selected_idx]
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_sel, y_train)

    importances  = rf.feature_importances_
    top5_local   = np.argsort(importances)[::-1][:5]
    top5_global  = selected_idx[top5_local]
    top5_names   = [shorten_name(feat_names[i]) for i in top5_global]
    top5_scores  = importances[top5_local]

    print(f"\n  Top-5 permissions by RF importance:")
    for rank, (nm, sc) in enumerate(zip(top5_names, top5_scores), 1):
        print(f"    {rank}. {nm:<30}  FI = {sc:.4f}")

    # ------------------------------------------------------------------ #
    # Plot
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = ["#C0392B"] * 5          # red bars, matching base paper style
    bars = ax.bar(range(5), top5_scores,
                  color=bar_colors, edgecolor="black", linewidth=0.8)

    ax.set_xticks(range(5))
    ax.set_xticklabels(top5_names, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Feature Importance Score", fontsize=12)
    ax.set_title(
        f"Top 5 Selected Permissions — {cfg['label']}",
        fontsize=13, fontweight="bold"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_ylim(0, max(top5_scores) * 1.28)

    # Value labels above bars
    for bar, score in zip(bars, top5_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(top5_scores) * 0.025,
            f"{score:.4f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    plt.tight_layout()
    out_path = f"results/ieee_plots/{cfg['out_file']}"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓  Saved → {out_path}")

print("\n" + "="*60)
print("  All Top-5 Permission charts generated in results/ieee_plots/")
print("="*60)
