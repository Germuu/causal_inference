import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from numpy.typing import NDArray

def generate_data(n_samples: int, theta_X: float, theta_Y_given_X: dict) -> tuple[np.ndarray, np.ndarray]:
    X = np.random.binomial(1, theta_X, size=n_samples)
    Y = np.array([np.random.binomial(1, theta_Y_given_X[x]) for x in X])
    return X, Y

def discretize_params(k:int) -> NDArray[np.float64]: 
    step: float = 2 ** (-k)
    return np.arange(0, 1 + step, step, dtype=np.float64)

def log_likelihood_fast(counts: tuple[int, int, int, int, int, int], theta_X: float, theta_Y0: float, theta_Y1: float) -> float:

    eps = 1e-12

    n_X1, n_X0, n_Y1_X0, n_Y0_X0, n_Y1_X1, n_Y0_X1 = counts

    ll_X = n_X1 * np.log(theta_X + eps) + n_X0 * np.log(1 - theta_X + eps)
    ll_Y0 = n_Y1_X0 * np.log(theta_Y0 + eps) + n_Y0_X0 * np.log(1 - theta_Y0 + eps)
    ll_Y1 = n_Y1_X1 * np.log(theta_Y1 + eps) + n_Y0_X1 * np.log(1 - theta_Y1 + eps)

    return ll_X + ll_Y0 + ll_Y1

def fit_direction_fast(X, Y, k, direction='X->Y'):
    grid = discretize_params(k)

    if direction == 'X->Y':
        X0 = (X == 0)
        X1 = (X == 1)
        n_X1 = np.sum(X1)
        n_X0 = len(X) - n_X1
        n_Y1_X0 = np.sum(Y[X0])
        n_Y0_X0 = n_X0 - n_Y1_X0
        n_Y1_X1 = np.sum(Y[X1])
        n_Y0_X1 = n_X1 - n_Y1_X1
        counts = (n_X1, n_X0, n_Y1_X0, n_Y0_X0, n_Y1_X1, n_Y0_X1)

        # Compute MLEs
        mle_theta_X = n_X1 / (n_X1 + n_X0) if (n_X1 + n_X0) > 0 else 0.0
        mle_theta_Y0 = n_Y1_X0 / (n_Y1_X0 + n_Y0_X0) if (n_Y1_X0 + n_Y0_X0) > 0 else 0.0
        mle_theta_Y1 = n_Y1_X1 / (n_Y1_X1 + n_Y0_X1) if (n_Y1_X1 + n_Y0_X1) > 0 else 0.0

        # Helper: find closest grid point(s) to mle, including neighbors if needed
        def neighbors(val):
            idx = np.searchsorted(grid, val)
            candidates = []
            if idx > 0:
                candidates.append(grid[idx-1])
            if idx < len(grid):
                candidates.append(grid[min(idx, len(grid)-1)])
            return list(set(candidates))  # unique

        theta_X_candidates = neighbors(mle_theta_X)
        theta_Y0_candidates = neighbors(mle_theta_Y0)
        theta_Y1_candidates = neighbors(mle_theta_Y1)

        best_ll = -np.inf
        best_params = None

        # Since parameters are independent, just check all combinations of closest neighbors (usually 2 each)
        for theta_X, theta_Y0, theta_Y1 in product(theta_X_candidates, theta_Y0_candidates, theta_Y1_candidates):
            ll = log_likelihood_fast(counts, theta_X, theta_Y0, theta_Y1)
            if ll > best_ll:
                best_ll = ll
                best_params = (theta_X, theta_Y0, theta_Y1)

        return best_params, best_ll

    else:
        # Flip X, Y and run same logic
        return fit_direction_fast(Y, X, k, direction='X->Y')


def run_experiment(n_trials=100, sample_sizes=None, k_values=None):
    if sample_sizes is None:
        sample_sizes = [50, 100, 500, 1000, 5000, 10000]
    if k_values is None:
        k_values = range(2, 10)

    results = {}  # (k, n_samples) -> proportion X->Y wins

    for k in k_values:
        grid = discretize_params(k)
        print(f"Running for k={k}")
        for n_samples in sample_sizes:
            wins = 0
            for _ in range(n_trials):
                # Sample true parameters uniformly from grid
                theta_X = np.random.choice(grid)
                theta_Y0 = np.random.choice(grid)
                theta_Y1 = np.random.choice(grid)
                #true_params = (theta_X, theta_Y0, theta_Y1)

                theta_Y_given_X = {0: theta_Y0, 1: theta_Y1}
                X, Y = generate_data(n_samples, theta_X, theta_Y_given_X)

                # Fit both directions
                _, ll_XY = fit_direction_fast(X, Y, k, 'X->Y')
                _, ll_YX = fit_direction_fast(X, Y, k, 'Y->X')

                if ll_XY > ll_YX:
                    wins += 1

            proportion = wins / n_trials
            results[(k, n_samples)] = proportion
            print(f"  n={n_samples}: Prop correct (X->Y) = {proportion:.3f}")

    return results, sample_sizes, k_values

def plot_results(results, sample_sizes, k_values):
    plt.figure(figsize=(12, 7))

    for k in k_values:
        proportions = [results[(k, n)] for n in sample_sizes]
        plt.plot(sample_sizes, proportions, marker='o', label=f'k={k}')

    plt.xscale('log')
    plt.xlabel('Sample size (log scale)')
    plt.ylabel('Proportion correct (X→Y wins)')
    plt.title('Causal direction identification accuracy vs Sample size')
    plt.legend(title='Discretization granularity k')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    results, sample_sizes, k_values = run_experiment()
    plot_results(results, sample_sizes, k_values)
