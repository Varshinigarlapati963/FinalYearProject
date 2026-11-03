import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


class SwarmOptimizer:
    """
    Binary PSO with:
      - Optional GBFI-seeded initialization (seed_features + n_swap diversity)
      - Hard feature-count limit (max_features_limit)
      - Soft penalty fitness: CV_Accuracy - 0.05*(k/n)
    """

    def __init__(self, n_particles=20, iterations=30, w=0.5, c1=1.5, c2=1.5,
                 max_features_limit=None, seed_features=None, n_swap=5):
        self.n_particles       = n_particles
        self.iterations        = iterations
        self.w                 = w              # Inertia weight
        self.c1                = c1             # Cognitive component (personal best)
        self.c2                = c2             # Social component (global best)
        self.max_features_limit = max_features_limit  # Hard upper bound on features
        self.seed_features     = seed_features  # numpy array of GBFI top-N feature indices
        self.n_swap            = n_swap         # Diversity: features swapped per particle

    def fit(self, X, y):
        n_features = X.shape[1]

        # ------------------------------------------------------------------ #
        # 1. Initialise positions
        # ------------------------------------------------------------------ #
        if self.seed_features is not None:
            positions = self._seeded_init(n_features)
        else:
            positions = self._random_init(n_features)

        velocities = np.random.uniform(-1, 1, (self.n_particles, n_features))

        # Personal bests
        pbest_positions = positions.copy()
        pbest_scores    = np.full(self.n_particles, -np.inf)

        # Global best
        gbest_position = None
        gbest_score    = -np.inf

        seed_tag = " [GBFI-seeded]" if self.seed_features is not None else ""
        print(f"Starting Swarm Optimization{seed_tag} "
              f"(Particles: {self.n_particles}, Iterations: {self.iterations})")

        # ------------------------------------------------------------------ #
        # 2. Main PSO loop
        # ------------------------------------------------------------------ #
        for iteration in range(self.iterations):
            for i in range(self.n_particles):
                selected_features = np.where(positions[i] == 1)[0]

                # Empty particle
                if len(selected_features) == 0:
                    score = -999

                # Hard limit violated → infeasible
                elif (self.max_features_limit is not None and
                      len(selected_features) > self.max_features_limit):
                    score = -999

                else:
                    X_subset = X.iloc[:, selected_features]
                    clf      = RandomForestClassifier(
                        n_estimators=30, max_depth=10,
                        random_state=42, n_jobs=-1
                    )
                    accuracy        = cross_val_score(clf, X_subset, y, cv=3,
                                                      scoring='accuracy').mean()
                    feature_penalty = 0.05 * (len(selected_features) / n_features)
                    score           = accuracy - feature_penalty

                # Update personal best
                if score > pbest_scores[i]:
                    pbest_scores[i]    = score
                    pbest_positions[i] = positions[i].copy()

                # Update global best
                if score > gbest_score:
                    gbest_score    = score
                    gbest_position = positions[i].copy()

            limit_str = (f" | Limit: {self.max_features_limit}"
                         if self.max_features_limit is not None else "")
            print(f"Iteration {iteration+1}/{self.iterations} | "
                  f"Best Fitness: {gbest_score:.4f} | "
                  f"Features Selected: {int(np.sum(gbest_position))}{limit_str}")

            # Velocity & position update
            r1         = np.random.rand(self.n_particles, n_features)
            r2         = np.random.rand(self.n_particles, n_features)
            velocities = (self.w  * velocities
                          + self.c1 * r1 * (pbest_positions - positions)
                          + self.c2 * r2 * (gbest_position  - positions))

            sigmoid   = 1 / (1 + np.exp(-velocities))
            positions = (np.random.rand(self.n_particles, n_features) < sigmoid).astype(int)

        self.selected_features = np.where(gbest_position == 1)[0]
        self.gbest_position    = gbest_position.copy()   # full binary mask
        self.best_score        = gbest_score
        return self

    # ------------------------------------------------------------------ #
    # Helper: random 80/20 init with hard-limit clipping
    # ------------------------------------------------------------------ #
    def _random_init(self, n_features):
        positions = np.random.choice(
            [0, 1], size=(self.n_particles, n_features), p=[0.8, 0.2]
        )
        # Ensure no fully-empty particle
        for i in range(self.n_particles):
            if positions[i].sum() == 0:
                positions[i][np.random.randint(n_features)] = 1

        # Clip to hard limit (random drop of excess)
        if self.max_features_limit is not None:
            for i in range(self.n_particles):
                sel = np.where(positions[i] == 1)[0]
                if len(sel) > self.max_features_limit:
                    drop = np.random.choice(
                        sel,
                        size=len(sel) - self.max_features_limit,
                        replace=False,
                    )
                    positions[i][drop] = 0
        return positions

    # ------------------------------------------------------------------ #
    # Helper: GBFI-seeded init with per-particle swap diversity
    # ------------------------------------------------------------------ #
    def _seeded_init(self, n_features):
        """
        Each particle starts with exactly the GBFI top-N seed features set to 1.
        Then n_swap features are randomly swapped (drop a seed, add a non-seed)
        to give each particle a unique neighbourhood to explore.

        This gives the swarm a strong, gradient-informed starting point while
        preserving diversity so the swarm actually explores.
        """
        seed      = np.asarray(self.seed_features, dtype=int)
        seed_set  = set(seed.tolist())
        non_seed  = np.array([i for i in range(n_features) if i not in seed_set])
        n_swap    = min(self.n_swap, len(seed), len(non_seed))

        positions = np.zeros((self.n_particles, n_features), dtype=int)

        for i in range(self.n_particles):
            # Start with all GBFI seeds selected
            positions[i, seed] = 1

            if n_swap > 0:
                # Drop n_swap random seed features
                drop_from_seed = np.random.choice(seed,     size=n_swap, replace=False)
                # Add  n_swap random non-seed features
                add_from_non   = np.random.choice(non_seed, size=n_swap, replace=False)
                positions[i, drop_from_seed] = 0
                positions[i, add_from_non]   = 1

            # Sanity: each particle still has exactly len(seed) features selected
            # (n_swap dropped + n_swap added = net zero change)

        print(f"  [GBFI-seeded init] {self.n_particles} particles | "
              f"Seed size: {len(seed)} | Swap per particle: {n_swap}")
        print(f"  [GBFI-seeded init] Each particle explores a unique "
              f"{len(seed)}-feature neighbourhood")
        return positions
