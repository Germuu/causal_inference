import numpy as np
from itertools import product
import matplotlib.pyplot as plt

def generate_data(n_samples, theta_X, theta_Y_given_X):
    # theta_Y_given_X is a dict {0: p(Y=1|X=0), 1: p(Y=1|X=1)}
    X = np.random.binomial(1, theta_X, size=n_samples)
    Y = np.array([np.random.binomial(1, theta_Y_given_X[x]) for x in X])
    return X, Y

def discretize_params(k):
    step = 2 ** (-k)
    return np.arange(0, 1 + step, step)

def log_likelihood_fast(counts, params):
    theta_X, theta_Y0, theta_Y1 = params
    eps = 1e-12

    n_X1, n_X0, n_Y1_X0, n_Y0_X0, n_Y1_X1, n_Y0_X1 = counts

    # log P(X)
    ll_X = n_X1 * np.log(theta_X + eps) + n_X0 * np.log(1 - theta_X + eps)

    # log P(Y|X=0)
    ll_Y0 = n_Y1_X0 * np.log(theta_Y0 + eps) + n_Y0_X0 * np.log(1 - theta_Y0 + eps)

    # log P(Y|X=1)
    ll_Y1 = n_Y1_X1 * np.log(theta_Y1 + eps) + n_Y0_X1 * np.log(1 - theta_Y1 + eps)

    return ll_X + ll_Y0 + ll_Y1

def fit_direction(X, Y, k, direction='X->Y'):
    grid = discretize_params(k)
    best_ll = -np.inf
    best_params = None

    if direction == 'X->Y':
        # Precompute sufficient statistics
        X0 = (X == 0)
        X1 = (X == 1)
        n_X1 = np.sum(X1)
        n_X0 = len(X) - n_X1
        n_Y1_X0 = np.sum(Y[X0])
        n_Y0_X0 = n_X0 - n_Y1_X0
        n_Y1_X1 = np.sum(Y[X1])
        n_Y0_X1 = n_X1 - n_Y1_X1
        counts = (n_X1, n_X0, n_Y1_X0, n_Y0_X0, n_Y1_X1, n_Y0_X1)

        for params in product(grid, repeat=3):
            ll = log_likelihood_fast(counts, params)
            if ll > best_ll:
                best_ll = ll
                best_params = params

    else:  # Y->X
        # Just swap roles
        return fit_direction(Y, X, k, direction='X->Y')

    return best_params, best_ll


def main():
    np.random.seed(42)
    
    # True parameters
    theta_X = 0.3
    theta_Y_given_X = {0: 0.4, 1: 0.8}
    
    n_samples = 100
    X, Y = generate_data(n_samples, theta_X, theta_Y_given_X)
    
    k_values = range(2, 7)  # Try k = 2 to 6
    ll_XY_list = []
    ll_YX_list = []

    for k in k_values:
        params_XY, ll_XY = fit_direction(X, Y, k, 'X->Y')
        params_YX, ll_YX = fit_direction(X, Y, k, 'Y->X')

        ll_XY_list.append(ll_XY)
        ll_YX_list.append(ll_YX)

        print(f"k={k}:")
        print(f"  X->Y LL: {ll_XY:.2f}")
        print(f"  Y->X LL: {ll_YX:.2f}")
        print(f"  Preferred: {'X->Y' if ll_XY > ll_YX else 'Y->X'}\n")

    # Plot log-likelihoods
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, ll_XY_list, marker='o', label='X → Y')
    plt.plot(k_values, ll_YX_list, marker='s', label='Y → X')
    plt.title('Log-likelihood vs Discretization Granularity (k)')
    plt.xlabel('k (Grid granularity 2^-k)')
    plt.ylabel('Log-likelihood')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot log-likelihood difference
    plt.figure(figsize=(10, 4))
    ll_diff = np.array(ll_XY_list) - np.array(ll_YX_list)
    plt.plot(k_values, ll_diff, marker='^', color='purple')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title('Log-likelihood Difference (X→Y minus Y→X)')
    plt.xlabel('k')
    plt.ylabel('LL(X→Y) − LL(Y→X)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
