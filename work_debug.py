import numpy as np
import math
from itertools import product

# copy your functions (or import them) exactly here
def discretize_params(k: int):
    step = 2 ** (-k)
    grid = np.arange(start=0, stop=1 + step, step=step, dtype=np.float64)
    return grid[(grid > 0.0) & (grid < 1.0)]

def generate_counts_fast(n_samples: int, theta_X: float, theta_Y_given_X: dict):
    theta_Y0 = theta_Y_given_X[0]
    theta_Y1 = theta_Y_given_X[1]
    p_00 = (1 - theta_X) * (1 - theta_Y0)
    p_01 = (1 - theta_X) * theta_Y0
    p_10 = theta_X * (1 - theta_Y1)
    p_11 = theta_X * theta_Y1
    counts = np.random.multinomial(n_samples, [p_11, p_10, p_01, p_00])
    # counts order: [p11, p10, p01, p00]
    n_Y1_X1 = counts[0]
    n_Y0_X1 = counts[1]
    n_Y1_X0 = counts[2]
    n_Y0_X0 = counts[3]
    n_X1 = n_Y1_X1 + n_Y0_X1
    n_X0 = n_Y1_X0 + n_Y0_X0
    # returned order matches your code:
    return (n_X1, n_X0, n_Y1_X0, n_Y0_X0, n_Y1_X1, n_Y0_X1)

def log_likelihood_fast(counts, theta_X, theta_Y0, theta_Y1):
    params = [
        theta_X,        # n_X1
        1 - theta_X,    # n_X0
        theta_Y0,       # n_Y1_X0
        1 - theta_Y0,   # n_Y0_X0
        theta_Y1,       # n_Y1_X1
        1 - theta_Y1,   # n_Y0_X1
    ]
    ll = 0.0
    for c, p in zip(counts, params):
        if c > 0:
            if p <= 0:
                return -np.inf
            ll += c * math.log(p)
    return ll

def fit_best_grid_full_grid(counts, grid):
    best_ll = -np.inf
    best_params = None
    for theta_X, theta_Y0, theta_Y1 in product(grid, grid, grid):
        ll = log_likelihood_fast(counts, theta_X, theta_Y0, theta_Y1)
        if ll > best_ll:
            best_ll = ll
            best_params = (theta_X, theta_Y0, theta_Y1)
    return best_params, best_ll

def joint_from_forward(theta_X, theta_Y0, theta_Y1):
    p11 = theta_X * theta_Y1
    p10 = theta_X * (1 - theta_Y1)
    p01 = (1 - theta_X) * theta_Y0
    p00 = (1 - theta_X) * (1 - theta_Y0)
    return np.array([p11, p10, p01, p00])

# Settings (choose k small to make full search cheap)
np.random.seed(0)
k = 2
grid = discretize_params(k)
n_samples = 40000

# Try many trials until we find a reverse win
for trial in range(1, 20001):
    theta_X = float(np.random.choice(grid))
    theta_Y0 = float(np.random.choice(grid))
    theta_Y1 = float(np.random.choice(grid))
    counts = generate_counts_fast(n_samples, theta_X, {0: theta_Y0, 1: theta_Y1})
    # forward full-grid fit
    params_XY_found, ll_XY_found = fit_best_grid_full_grid(counts, grid)
    # compute ll at the true params (this MUST equal ll_XY_found)
    ll_true = log_likelihood_fast(counts, theta_X, theta_Y0, theta_Y1)

    # Build relabeled counts for Y->X exactly as your function does
    n_X1, n_X0, n_Y1_X0, n_Y0_X0, n_Y1_X1, n_Y0_X1 = counts
    n_Y1 = n_Y1_X1 + n_Y1_X0
    n_Y0 = n_Y0_X1 + n_Y0_X0
    n_X1_Y0 = n_Y0_X1
    n_X0_Y0 = n_Y0_X0
    n_X1_Y1 = n_Y1_X1
    n_X0_Y1 = n_Y1_X0
    counts_rev = (n_Y1, n_Y0, n_X1_Y0, n_X0_Y0, n_X1_Y1, n_X0_Y1)

    params_YX_found, ll_YX_found = fit_best_grid_full_grid(counts_rev, grid)

    diff = ll_XY_found - ll_YX_found

    # If reverse wins decisively, print diagnostics and break
    if diff < -1e-12:
        print("=== Found reverse win trial ===")
        print("trial:", trial)
        print("true params: theta_X, theta_Y0, theta_Y1 =", theta_X, theta_Y0, theta_Y1)
        print("counts (n_X1, n_X0, n_Y1_X0, n_Y0_X0, n_Y1_X1, n_Y0_X1):", counts)
        print("ll_true (at true params):", ll_true)
        print("forward best params:", params_XY_found, "ll:", ll_XY_found)
        print("reverse best params (on relabeled counts):", params_YX_found, "ll:", ll_YX_found)
        print("ll diff (XY - YX):", diff)
        # Show joint cell probabilities for clarity
        jt = joint_from_forward(theta_X, theta_Y0, theta_Y1)
        print("true joint p11,p10,p01,p00:", jt)
        print("joint implied by forward best params:", joint_from_forward(*params_XY_found))
        print("joint implied by reverse best params:", joint_from_forward(*params_YX_found))
        # sanity checks
        print("Is ll_true == ll_XY_found? ->", abs(ll_true - ll_XY_found) < 1e-12)
        break
else:
    print("No decisive reverse win found in search; try increasing trial limit or lowering tie threshold.")
