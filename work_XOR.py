import numpy as np
from itertools import product
import matplotlib.pyplot as plt

def generate_anm_binary(n_samples, theta_X, p_noise):
    X = np.random.binomial(1, theta_X, size=n_samples)
    noise = np.random.binomial(1, p_noise, size=n_samples)
    Y = np.bitwise_xor(X, noise)
    return X, Y

def discretize_params(k):
    step = 2 ** (-k)
    return np.arange(0, 1 + step, step)

def log_likelihood_fast(counts, params):
    theta_X, theta_Y0, theta_Y1 = params
    eps = 1e-12
    n_X1, n_X0, n_Y1_X0, n_Y0_X0, n_Y1_X1, n_Y0_X1 = counts
    ll_X = n_X1 * np.log(theta_X + eps) + n_X0 * np.log(1 - theta_X + eps)
    ll_Y0 = n_Y1_X0 * np.log(theta_Y0 + eps) + n_Y0_X0 * np.log(1 - theta_Y0 + eps)
    ll_Y1 = n_Y1_X1 * np.log(theta_Y1 + eps) + n_Y0_X1 * np.log(1 - theta_Y1 + eps)
    return ll_X + ll_Y0 + ll_Y1

def fit_direction(X, Y, k, direction='X->Y'):
    grid = discretize_params(k)
    best_ll = -np.inf
    best_params = None

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

        for params in product(grid, repeat=3):
            ll = log_likelihood_fast(counts, params)
            if ll > best_ll:
                best_ll = ll
                best_params = params
    else:
        return fit_direction(Y, X, k, direction='X->Y')

    return best_params, best_ll

def main():
    np.random.seed(42)

    n_samples = 10000
    theta_X = 0.3
    p_noise = 0.1

    X, Y = generate_anm_binary(n_samples, theta_X, p_noise)

    max_k = 6
    ks = list(range(2, max_k + 1))
    ll_XY_list = []
    ll_YX_list = []

    for k in ks:
        _, ll_XY = fit_direction(X, Y, k, 'X->Y')
        _, ll_YX = fit_direction(X, Y, k, 'Y->X')
        ll_XY_list.append(ll_XY)
        ll_YX_list.append(ll_YX)
        preferred = "X->Y" if ll_XY > ll_YX else "Y->X"
        print(f"k={k}:  X->Y LL: {ll_XY:.2f},  Y->X LL: {ll_YX:.2f},  Preferred: {preferred}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(ks, ll_XY_list, label='Log-Likelihood X→Y', marker='o')
    plt.plot(ks, ll_YX_list, label='Log-Likelihood Y→X', marker='o')
    plt.plot(ks, np.array(ll_XY_list) - np.array(ll_YX_list), label='LL Difference (X→Y - Y→X)', linestyle='--', color='black')
    plt.axhline(0, color='gray', linestyle=':')
    plt.xlabel('k (Discretization granularity: step = 2⁻ᵏ)')
    plt.ylabel('Log-Likelihood')
    plt.title('Causal Direction Preference vs. Granularity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
