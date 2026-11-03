"""
quantum_swarm.py  (v2 — corrected & tightened)
------------------------------------------------
Module 3: Quantum-Inspired Binary Particle Swarm Optimization (Q-BPSO)
for Android Malware Detection Feature Selection.

Quantum Bit (Q-bit) Theory
---------------------------
Each feature j of particle i is encoded as a probability amplitude pair
(alpha_ij, beta_ij) satisfying the normalisation constraint:

    alpha_ij^2 + beta_ij^2 = 1          ... (1)

We use the equivalent ANGLE representation:

    theta_ij in [0, pi/2]
    alpha_ij = cos(theta_ij)   -> P(feature j dropped)  = cos^2(theta_ij)
    beta_ij  = sin(theta_ij)   -> P(feature j selected) = sin^2(theta_ij)

Equation (1) is automatically satisfied for ANY theta_ij by the Pythagorean
identity  cos^2 + sin^2 = 1.  Keeping theta_ij strictly inside [0, pi/2]
therefore guarantees BOTH:
    * 0 <= alpha_ij <= 1  (valid probability amplitude)
    * 0 <= beta_ij  <= 1  (valid probability amplitude)
    * alpha_ij^2 + beta_ij^2 = 1  (normalisation)

with NO separate normalisation step required.

Quantum Rotation Gate (updated)
---------------------------------
The rotation gate updates theta by computing how far each angle should move
toward the target (pBest or gBest) positions:

    pbest_theta_ij = (pi/2) * pbest_ij   # pi/2 if pBest bit=1, 0 if 0
    gbest_theta_j  = (pi/2) * gbest_j

    delta_ij = c1*r1*(pbest_theta_ij - theta_ij)   [cognitive]
             + c2*r2*(gbest_theta_j  - theta_ij)   [social]

    delta_ij = clip(delta_ij, -delta_max(t), +delta_max(t))   # bounded step

    theta_new_ij = clip(theta_ij + delta_ij, 0, pi/2)         # hard normalisation

Delta_max decays linearly over iterations (analogous to PSO inertia weight),
promoting broad exploration early and fine-grained exploitation later.

Fitness Function (v2)
-----------------------
    Fitness = 0.95 * CV_Accuracy  -  0.05 * (k / n)

    where k = features selected, n = total features.

The 5x stronger feature penalty compared to v1 (was 0.01) aggressively
drives the swarm toward compact, discriminative feature subsets.

Initialisation
--------------
    theta_initial = pi/6   ->  P(select) = sin^2(pi/6) = 0.25

Starting at 25% selection (vs 50% in v1) gives the swarm a sparser initial
population, which combined with the stronger penalty drives toward leaner subsets.

Reference:
  Hybrid Quantum-Inspired and Swarm Intelligence-Based Optimization for
  Accurate Malware Classification — Final Year Project.
"""

import sys
import numpy as np
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Ensure UTF-8 safe output on Windows cp1252 terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
DEFAULT_N_PARTICLES   = 20
DEFAULT_N_ITERATIONS  = 50            # v4: 50 iterations for aggressive convergence
DEFAULT_C1            = 2.0           # cognitive acceleration
DEFAULT_C2            = 2.0           # social acceleration
DEFAULT_MUTATION_RATE = 0.02          # v4: 2% (down from 5%) — sparse swarm can't
                                      #      absorb 5% flips without bloating count

# v4 Sparse Initialization
# alpha=cos(theta), beta=sin(theta), P(select)=sin^2(theta)
# To get P(select)=0.04 (4%): theta = arcsin(0.2) ≈ 0.2014 rad
# This gives normalised amplitudes: alpha≈0.9798, beta=0.2, alpha²+beta²=1 ✓
THETA_INIT            = np.arcsin(0.2)  # P(select)=sin²(θ)=0.04 → ~4% features/particle
THETA_MAX             = np.pi / 2       # upper bound on theta
DELTA_MAX_INIT        = np.pi / 5       # rotation step at iteration 1  (~36 deg)
DELTA_MAX_FINAL       = np.pi / 20      # rotation step at iteration N  (  9 deg)

FITNESS_ALPHA         = 0.90          # v4: accuracy weight (down from 0.95)
FITNESS_BETA          = 0.10          # v4: feature-count penalty (up from 0.05)
                                      #     Penalty for 137/1490 features = 0.0092
                                      #     Penalty for 489/1490 features = 0.0328
CV_FOLDS              = 3             # cross-validation folds inside fitness
RF_ESTIMATORS         = 30            # estimators for the fitness RF
RF_MAX_DEPTH          = 10            # max tree depth in fitness RF


# ===========================================================================
# Section 1: Normalisation utilities
# ===========================================================================

def theta_to_amplitudes(theta: np.ndarray):
    """
    Convert angle matrix to (alpha, beta) amplitude matrices.

    Parameters
    ----------
    theta : np.ndarray  shape (n_particles, n_features), values in [0, pi/2]

    Returns
    -------
    alpha : np.ndarray  cos(theta)  ->  P(drop feature)   in [0, 1]
    beta  : np.ndarray  sin(theta)  ->  P(keep feature)   in [0, 1]

    Normalisation guarantee: alpha**2 + beta**2 == 1  (Pythagorean identity)
    """
    alpha = np.cos(theta)
    beta  = np.sin(theta)
    return alpha, beta


def verify_normalisation(theta: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Verify alpha^2 + beta^2 == 1 for all Q-bits.
    Returns True if all errors are below tol (always True for valid theta).
    """
    alpha, beta = theta_to_amplitudes(theta)
    errors = np.abs(alpha**2 + beta**2 - 1.0)
    return bool(np.all(errors < tol))


# ===========================================================================
# Section 2: Fitness Function
# ===========================================================================

def _evaluate_fitness(
    mask        : np.ndarray,
    X_train     : np.ndarray,
    y_train     : np.ndarray,
    n_features  : int,
    rng         : np.random.Generator,
    cv_folds    : int = CV_FOLDS,
) -> float:
    """
    Evaluate a binary feature mask.

    Fitness = 0.95 * CV_Accuracy  -  0.05 * (n_selected / n_total)

    Returns -inf for infeasible masks (zero features selected).
    """
    selected   = np.where(mask == 1)[0]
    n_selected = len(selected)

    if n_selected == 0:
        return -np.inf

    X_sub = X_train[:, selected]

    clf = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,           # v3: bounded depth for speed + generalisation
        random_state=int(rng.integers(0, 10_000)),
        n_jobs=-1,
    )
    skf = StratifiedKFold(
        n_splits=cv_folds, shuffle=True,
        random_state=int(rng.integers(0, 10_000))
    )
    scores = cross_val_score(clf, X_sub, y_train,
                             cv=skf, scoring="accuracy", n_jobs=-1)
    cv_acc = float(scores.mean())

    fitness = FITNESS_ALPHA * cv_acc - FITNESS_BETA * (n_selected / n_features)
    return fitness


# ===========================================================================
# Section 3: Q-bit Observation  (collapse to binary)
# ===========================================================================

def observe(theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Collapse Q-bits to a binary feature mask.

        P(feature_j selected) = sin^2(theta_j)  = beta_j^2

    Steps:
      1. Compute selection probabilities p = sin^2(theta)
      2. Draw uniform random matrix U
      3. feature selected iff U < p

    Parameters
    ----------
    theta : np.ndarray  shape (n_particles, n_features)  values in [0, pi/2]
    rng   : numpy Generator

    Returns
    -------
    binary : np.ndarray  shape (n_particles, n_features)  dtype int8
    """
    # Verify normalisation at observation time (debugging guard)
    assert verify_normalisation(theta), \
        "Q-bit normalisation violated before observation — check theta bounds."

    _, beta  = theta_to_amplitudes(theta)
    p_select = beta ** 2                        # P(select) per feature per particle
    rand     = rng.random(theta.shape)          # uniform [0, 1)
    return (rand < p_select).astype(np.int8)


# ===========================================================================
# Section 4: Quantum Rotation Gate  (fully vectorised, bounded, normalised)
# ===========================================================================

def quantum_rotation_gate(
    theta       : np.ndarray,   # (n_particles, n_features)
    pbest_mask  : np.ndarray,   # (n_particles, n_features)  binary personal bests
    gbest_mask  : np.ndarray,   # (n_features,)              binary global best
    c1          : float,
    c2          : float,
    delta_max   : float,        # current (decayed) max rotation step
    rng         : np.random.Generator,
) -> np.ndarray:
    """
    Apply the Quantum Rotation Gate to update all particle Q-bit angles.

    Rotation rule:
        pbest_theta_ij = (pi/2) * pbest_ij   [target angle for cognitive pull]
        gbest_theta_j  = (pi/2) * gbest_j    [target angle for social pull]

        delta_ij = c1*r1*(pbest_theta_ij - theta_ij)    [cognitive component]
                 + c2*r2*(gbest_theta_j  - theta_ij)    [social component]

        delta_ij = clip(delta_ij, -delta_max, +delta_max)   [bounded step]
        theta_new_ij = clip(theta_ij + delta_ij, 0, pi/2)   [hard bounds]

    Normalisation guarantee:
        After clipping theta_new to [0, pi/2]:
            alpha = cos(theta_new) in [0, 1]
            beta  = sin(theta_new) in [0, 1]
            alpha^2 + beta^2 = 1  (Pythagorean identity — always satisfied)

    Parameters
    ----------
    theta      : current angle matrix  (n_particles, n_features)
    pbest_mask : personal-best binary mask per particle (n_particles, n_features)
    gbest_mask : global-best binary mask (n_features,)
    c1, c2     : cognitive / social acceleration coefficients
    delta_max  : maximum rotation step this iteration (adaptive / decayed)
    rng        : numpy Generator

    Returns
    -------
    theta_new : np.ndarray  shape (n_particles, n_features)  guaranteed in [0, pi/2]
    """
    n_particles, n_features = theta.shape

    # Map binary masks to target angles
    # bit=1 -> pi/2  (high probability of selecting feature)
    # bit=0 -> 0     (low probability of selecting feature)
    pbest_theta = pbest_mask.astype(np.float64) * THETA_MAX   # (n_particles, n_features)
    gbest_theta = gbest_mask.astype(np.float64) * THETA_MAX   # (n_features,)

    # Draw random coefficients  (uniform [0,1] per bit per particle)
    r1 = rng.random((n_particles, n_features))
    r2 = rng.random((n_particles, n_features))

    # Cognitive pull: toward each particle's personal best angle
    cognitive = c1 * r1 * (pbest_theta - theta)

    # Social pull: toward the global best angle (broadcast n_features -> n_particles)
    social    = c2 * r2 * (gbest_theta[np.newaxis, :] - theta)

    delta_theta = cognitive + social

    # --- Bound the rotation step ---
    # This prevents any single update from overshooting by more than delta_max
    delta_theta = np.clip(delta_theta, -delta_max, delta_max)

    # --- Apply rotation and enforce hard theta bounds ---
    # PRIMARY clip: keeps theta in valid Q-bit range [0, pi/2]
    # This is the ONLY normalisation step needed — it guarantees
    # cos^2(theta) + sin^2(theta) = 1 for all resulting angles.
    theta_new = np.clip(theta + delta_theta, 0.0, THETA_MAX)

    # --- Numerical safety: second clip against floating-point drift ---
    theta_new = np.clip(theta_new, 0.0, THETA_MAX)

    return theta_new


# ===========================================================================
# Section 5: Adaptive delta_max schedule  (linear decay)
# ===========================================================================

def _get_delta_max(iteration: int, n_iterations: int) -> float:
    """
    Linearly decay the rotation step from DELTA_MAX_INIT to DELTA_MAX_FINAL.

    Iteration 1     ->  delta_max = DELTA_MAX_INIT  (broad exploration)
    Iteration N     ->  delta_max = DELTA_MAX_FINAL (fine exploitation)

    Analogy: inertia weight decay in classic PSO.
    """
    t   = (iteration - 1) / max(n_iterations - 1, 1)   # 0 at iter 1, 1 at iter N
    return DELTA_MAX_INIT + t * (DELTA_MAX_FINAL - DELTA_MAX_INIT)


# ===========================================================================
# Section 6: Main Q-BPSO Algorithm
# ===========================================================================

def run_qbpso(
    X_train       : np.ndarray,
    y_train       : np.ndarray,
    feature_names : list,
    n_particles   : int   = DEFAULT_N_PARTICLES,
    n_iterations  : int   = DEFAULT_N_ITERATIONS,
    c1            : float = DEFAULT_C1,
    c2            : float = DEFAULT_C2,
    mutation_rate : float = DEFAULT_MUTATION_RATE,  # v3: bit-flip mutation
    random_state  : int   = 42,
    verbose       : bool  = True,
) -> dict:
    """
    Run Quantum-Inspired Binary Particle Swarm Optimization (Q-BPSO v3).

    Key improvements vs v2:
      * Fitness RF: n_estimators 10->30, max_depth=10 (more reliable signal)
      * Iterations: 30->40 (more time to converge on Dataset-1)
      * Bit-flip mutation (default 5%): after each observe(), randomly flip
        mutation_rate fraction of bits across all particles to escape local optima

    Inherited from v2:
      * Fitness penalty 0.05 (aggressive feature pruning)
      * Initial theta = pi/6 (P(select) = 25%)
      * Adaptive delta_max: decays pi/5 -> pi/20 over iterations
      * Explicit normalisation verification at observation step
      * Double-clipping of theta after rotation for numerical safety

    Parameters
    ----------
    X_train, y_train : training data (already scaled)
    feature_names    : list of feature name strings
    n_particles      : swarm size (default 20)
    n_iterations     : number of iterations (default 30)
    c1, c2           : cognitive / social acceleration coefficients
    random_state     : seed for full reproducibility
    verbose          : print per-iteration progress

    Returns
    -------
    dict:
        selected_indices  : np.ndarray  indices of retained features
        selected_names    : list        names of retained features
        X_train_reduced   : np.ndarray  training data on selected features
        gbest_mask        : np.ndarray  final binary mask (n_features,)
        gbest_fitness     : float       best fitness achieved
        convergence       : list        best fitness per iteration
        n_selected        : int         number of features selected
    """
    rng         = np.random.default_rng(random_state)
    n_features  = X_train.shape[1]

    # ------------------------------------------------------------------ #
    # 1. Initialise Q-bits
    #    theta = pi/6 -> P(select) = sin^2(pi/6) = 0.25 (sparse start)
    #    This gives approximately n_features/4 features per particle initially.
    # ------------------------------------------------------------------ #
    theta = np.full((n_particles, n_features), THETA_INIT, dtype=np.float64)

    if verbose:
        init_p = np.sin(THETA_INIT) ** 2
        print(f"  [Q-BPSO v4] Swarm init: {n_particles} particles | "
              f"{n_features} features")
        print(f"  [Q-BPSO v4] Initial theta=arcsin(0.2): "
              f"P(select)=sin^2(theta)={init_p:.4f}  "
              f"~{int(init_p * n_features)} features/particle")
        print(f"  [Q-BPSO v4] alpha~0.9798, beta=0.2  (normalised: alpha^2+beta^2=1)")
        print(f"  [Q-BPSO v4] Fitness = {FITNESS_ALPHA}*Acc - {FITNESS_BETA}*(k/n) "
              f"| RF(n={RF_ESTIMATORS}, depth={RF_MAX_DEPTH})")
        print(f"  [Q-BPSO v4] delta_max: {DELTA_MAX_INIT:.4f} -> "
              f"{DELTA_MAX_FINAL:.4f} (adaptive decay)")
        print(f"  [Q-BPSO v4] Mutation: {mutation_rate:.0%} bit-flip per iter"
              f" | {n_iterations} iterations\n")

    # ------------------------------------------------------------------ #
    # 2. Initial observation
    # ------------------------------------------------------------------ #
    positions = observe(theta, rng)    # (n_particles, n_features)

    # ------------------------------------------------------------------ #
    # 3. Evaluate initial fitness
    # ------------------------------------------------------------------ #
    fitness = np.array([
        _evaluate_fitness(positions[i], X_train, y_train, n_features, rng)
        for i in range(n_particles)
    ], dtype=np.float64)

    # ------------------------------------------------------------------ #
    # 4. Initialise pBest and gBest
    # ------------------------------------------------------------------ #
    pbest_mask    = positions.copy()           # (n_particles, n_features)
    pbest_fitness = fitness.copy()             # (n_particles,)

    gbest_idx     = int(np.argmax(pbest_fitness))
    gbest_mask    = pbest_mask[gbest_idx].copy()
    gbest_fitness = float(pbest_fitness[gbest_idx])

    convergence = []

    # ------------------------------------------------------------------ #
    # 5. Main swarm loop
    # ------------------------------------------------------------------ #
    for iteration in range(1, n_iterations + 1):

        # Adaptive rotation step (large early, small late)
        delta_max = _get_delta_max(iteration, n_iterations)

        # --- 5a. Quantum Rotation Gate ---
        theta = quantum_rotation_gate(
            theta, pbest_mask, gbest_mask,
            c1, c2, delta_max, rng
        )

        # --- 5b. Observe new binary positions (with normalisation check) ---
        positions = observe(theta, rng)

        # --- 5b2. Bit-flip mutation (v3) ---
        # Randomly flip mutation_rate fraction of bits across all particles.
        # This injects diversity, preventing the swarm from converging to a
        # local optimum where redundant correlated features are all selected.
        # Mutation is applied ONLY to current positions — pBest/gBest are
        # unchanged, so the swarm never regresses below its best known solution.
        if mutation_rate > 0.0:
            flip_mask = rng.random(positions.shape) < mutation_rate
            positions = np.where(flip_mask, 1 - positions, positions).astype(np.int8)

        # --- 5c. Evaluate fitness ---
        fitness = np.array([
            _evaluate_fitness(positions[i], X_train, y_train, n_features, rng)
            for i in range(n_particles)
        ], dtype=np.float64)

        # --- 5d. Update personal bests ---
        improved              = fitness > pbest_fitness
        pbest_mask[improved]  = positions[improved]
        pbest_fitness[improved] = fitness[improved]

        # --- 5e. Update global best ---
        best_idx = int(np.argmax(pbest_fitness))
        if pbest_fitness[best_idx] > gbest_fitness:
            gbest_fitness = float(pbest_fitness[best_idx])
            gbest_mask    = pbest_mask[best_idx].copy()

        n_sel = int(gbest_mask.sum())
        convergence.append(gbest_fitness)

        if verbose:
            print(f"  Iter {iteration:3d}/{n_iterations} | "
                  f"Best Fitness = {gbest_fitness:.6f} | "
                  f"Features = {n_sel:4d} / {n_features} | "
                  f"delta_max = {delta_max:.4f}")

    # ------------------------------------------------------------------ #
    # 6. Package results
    # ------------------------------------------------------------------ #
    selected_indices = np.where(gbest_mask == 1)[0]
    selected_names   = [feature_names[i] for i in selected_indices]
    X_train_reduced  = X_train[:, selected_indices]

    if verbose:
        print(f"\n  [Q-BPSO v4] Done. Final: {len(selected_indices)} features | "
              f"Fitness: {gbest_fitness:.6f}")
        # Final normalisation audit
        ok = verify_normalisation(theta)
        print(f"  [Q-BPSO v4] Normalisation check (alpha^2+beta^2=1): "
              f"{'PASS' if ok else 'FAIL'}\n")

    return {
        "selected_indices" : selected_indices,
        "selected_names"   : selected_names,
        "X_train_reduced"  : X_train_reduced,
        "gbest_mask"       : gbest_mask,
        "gbest_fitness"    : gbest_fitness,
        "convergence"      : convergence,
        "n_selected"       : len(selected_indices),
    }


# ===========================================================================
# Section 7: Apply mask to test set
# ===========================================================================

def apply_mask(X_test: np.ndarray, gbest_mask: np.ndarray) -> np.ndarray:
    """
    Apply the final gBest binary mask to a test matrix.

    Parameters
    ----------
    X_test     : (n_test, n_features)
    gbest_mask : (n_features,)  binary mask from run_qbpso()

    Returns
    -------
    X_test_reduced : (n_test, k)
    """
    selected = np.where(gbest_mask == 1)[0]
    return X_test[:, selected]
