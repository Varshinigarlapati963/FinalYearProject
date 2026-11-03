import numpy as np
from src.quantum_swarm import (observe, quantum_rotation_gate,
                                theta_to_amplitudes, verify_normalisation,
                                _get_delta_max, run_qbpso, apply_mask,
                                THETA_INIT, THETA_MAX)

print("=== Q-BPSO v2 Normalisation Tests ===")

# Test 1: initial theta -> alpha^2 + beta^2 = 1 everywhere
theta = np.full((5, 10), THETA_INIT)
alpha, beta = theta_to_amplitudes(theta)
norms = alpha**2 + beta**2
print(f"Test 1 - alpha^2+beta^2 (all pi/6): min={norms.min():.15f}  max={norms.max():.15f}")
assert np.allclose(norms, 1.0), "FAIL: normalisation violated at init"
print("Test 1 PASSED")

# Test 2: after rotation gate, still normalised
rng = np.random.default_rng(42)
pbest = np.random.randint(0, 2, (5, 10)).astype(np.int8)
gbest = np.random.randint(0, 2, (10,)).astype(np.int8)
theta2 = quantum_rotation_gate(theta, pbest, gbest, 2.0, 2.0, np.pi/5, rng)
ok = verify_normalisation(theta2)
print(f"Test 2 - theta after rotation: min={theta2.min():.6f}  max={theta2.max():.6f}")
assert ok, "FAIL: normalisation violated after rotation"
print("Test 2 PASSED")

# Test 3: 30-iter stress test -- theta never escapes [0, pi/2]
theta3 = np.full((20, 50), THETA_INIT)
for i in range(30):
    p = np.random.randint(0, 2, (20, 50)).astype(np.int8)
    g = np.random.randint(0, 2, (50,)).astype(np.int8)
    dm = _get_delta_max(i+1, 30)
    theta3 = quantum_rotation_gate(theta3, p, g, 2.0, 2.0, dm, rng)
    assert theta3.min() >= 0.0 and theta3.max() <= THETA_MAX + 1e-12
print(f"Test 3 PASSED - 30-iter stress: [{theta3.min():.6f}, {theta3.max():.6f}]")

# Test 4: adaptive delta_max decaying
dm1  = _get_delta_max(1,  30)
dm15 = _get_delta_max(15, 30)
dm30 = _get_delta_max(30, 30)
print(f"Test 4 - delta_max: iter1={dm1:.4f}  iter15={dm15:.4f}  iter30={dm30:.4f}")
assert dm1 > dm15 > dm30, "FAIL: delta_max not decaying"
print("Test 4 PASSED")

# Test 5: initial P(select) = 0.25
p_select = np.sin(THETA_INIT)**2
print(f"Test 5 - Initial P(select) = sin^2(pi/6) = {p_select:.6f} (expect 0.25)")
assert abs(p_select - 0.25) < 1e-10
print("Test 5 PASSED")

# Test 6: live run on synthetic data
print("\n=== Live synthetic run (5 particles, 5 iters, 40 features) ===")
X = np.random.randn(200, 40)
y = (X[:, 0] + X[:, 5] > 0).astype(int)
result = run_qbpso(X[:160], y[:160], ["f"+str(i) for i in range(40)],
                   n_particles=5, n_iterations=5, verbose=True)
n_sel = result["n_selected"]
print("Selected", n_sel, "/ 40 features")
assert n_sel < 30, f"Expected fewer than 30 features with 0.05 penalty, got {n_sel}"
print("ALL TESTS PASSED")
