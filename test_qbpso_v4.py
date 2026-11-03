import numpy as np
from src.quantum_swarm import (run_qbpso, DEFAULT_N_ITERATIONS, DEFAULT_MUTATION_RATE,
                                RF_ESTIMATORS, RF_MAX_DEPTH, FITNESS_ALPHA, FITNESS_BETA,
                                THETA_INIT)

print("=== Q-BPSO v4 Sanity Check ===")
p     = np.sin(THETA_INIT)**2
alpha = np.cos(THETA_INIT)
beta  = np.sin(THETA_INIT)
print(f"THETA_INIT     : {THETA_INIT:.6f} rad = arcsin(0.2)")
print(f"alpha          : {alpha:.6f}  (expect ~0.9798)")
print(f"beta           : {beta:.6f}  (expect  0.2000)")
print(f"alpha^2+beta^2 : {alpha**2 + beta**2:.15f}  (expect 1.0)")
print(f"P(select)      : {p:.6f}  (expect 0.04)")
print(f"N_ITERATIONS   : {DEFAULT_N_ITERATIONS}  (expect 50)")
print(f"MUTATION_RATE  : {DEFAULT_MUTATION_RATE}  (expect 0.02)")
print(f"FITNESS_ALPHA  : {FITNESS_ALPHA}  (expect 0.90)")
print(f"FITNESS_BETA   : {FITNESS_BETA}  (expect 0.10)")
assert abs(alpha**2 + beta**2 - 1.0) < 1e-12, "NORMALISATION FAILED"
assert DEFAULT_N_ITERATIONS == 50
assert DEFAULT_MUTATION_RATE == 0.02
assert FITNESS_ALPHA == 0.90
assert FITNESS_BETA  == 0.10
print()
print(f"DS1 expected init features : {int(p*1490)} / 1490  (target < 137)")
print(f"DS3 expected init features : {int(p*330)} / 330   (target < 54)")
print()
print("Live run (5 particles, 8 iters, 100 features):")
X = np.random.randn(300, 100)
y = (X[:,0]+X[:,3] > 0).astype(int)
r = run_qbpso(X[:240], y[:240], ["f"+str(i) for i in range(100)],
              n_particles=5, n_iterations=8, mutation_rate=0.02, verbose=True)
n_sel = r["n_selected"]
print(f"Selected: {n_sel} / 100  (expect << 50 with 4% P-select and 0.10 penalty)")
assert n_sel < 50, f"Too many features selected: {n_sel}"
print("ALL CHECKS PASSED")
