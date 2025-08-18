import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Dict, Tuple
from math import fsum


def generate_counts_fast_4val(
    n_samples: int,
    theta_X: NDArray[np.float64],  # probabilities for X=0,1,2,3
    theta_Y_given_X: Dict[int, NDArray[np.float64]],  # P(Y|X=i) for i=0,1,2,3
) -> NDArray[np.int64]:
    """Generate counts for 4x4 contingency table using multinomial sampling."""
    # Create joint probability matrix P(X=i, Y=j)
    joint_probs = []
    for x in range(4):
        for y in range(4):
            joint_probs.append(theta_X[x] * theta_Y_given_X[x][y])
    
    # Sample from multinomial
    counts = np.random.multinomial(n_samples, joint_probs)
    
    # Reshape to 4x4 matrix where counts[i,j] = count of (X=i, Y=j)
    return counts.reshape(4, 4)


def discretize_params_simplex(k: int, dim: int = 4) -> NDArray[np.float64]:
    """Generate discretized points on the (dim-1)-simplex."""
    step = 2 ** (-k)
    
    # Generate all combinations that sum to 1
    valid_points = []
    
    def generate_simplex_points(remaining_sum, remaining_dims, current_point):
        if remaining_dims == 1:
            if 0 < remaining_sum < 1:  # Exclude boundary
                valid_points.append(current_point + [remaining_sum])
            return
        
        max_val = min(1 - step, remaining_sum - (remaining_dims - 1) * step)
        for val in np.arange(step, max_val + step, step):
            generate_simplex_points(
                remaining_sum - val, 
                remaining_dims - 1, 
                current_point + [val]
            )
    
    generate_simplex_points(1.0, dim, [])
    return np.array(valid_points)


def log_likelihood_fast_4val(
    counts: NDArray[np.int64],  # 4x4 count matrix
    theta_X: NDArray[np.float64],
    theta_Y_given_X: Dict[int, NDArray[np.float64]],
) -> float:
    """Compute log-likelihood for 4-valued variables."""
    ll = 0.0
    
    # Marginal likelihood for X
    marginal_X = np.sum(counts, axis=1)  # Sum over Y for each X
    for x in range(4):
        if marginal_X[x] > 0 and theta_X[x] > 0:
            ll += marginal_X[x] * np.log(theta_X[x])
    
    # Conditional likelihood for Y given X
    for x in range(4):
        if marginal_X[x] > 0:
            for y in range(4):
                if counts[x, y] > 0 and theta_Y_given_X[x][y] > 0:
                    ll += counts[x, y] * np.log(theta_Y_given_X[x][y])
    
    return ll


def mle_estimate_4val(counts: NDArray[np.int64]) -> Tuple[NDArray[np.float64], Dict[int, NDArray[np.float64]]]:
    """Compute MLE estimates from count matrix."""
    n_total = np.sum(counts)
    marginal_X = np.sum(counts, axis=1)
    
    # MLE for P(X)
    theta_X_mle = marginal_X / n_total
    
    # MLE for P(Y|X)
    theta_Y_given_X_mle = {}
    for x in range(4):
        if marginal_X[x] > 0:
            theta_Y_given_X_mle[x] = counts[x, :] / marginal_X[x]
        else:
            theta_Y_given_X_mle[x] = np.ones(4) / 4  # Uniform if no data
    
    return theta_X_mle, theta_Y_given_X_mle


def neighbors_simplex(val: NDArray[np.float64], grid: NDArray[np.float64], tolerance: float = 1e-6) -> list[NDArray[np.float64]]:
    """Find neighboring points on simplex grid."""
    distances = np.sum(np.abs(grid - val), axis=1)
    closest_indices = np.where(distances < tolerance)[0]
    
    if len(closest_indices) == 0:
        # Find closest point if exact match not found
        closest_idx = np.argmin(distances)
        return [grid[closest_idx]]
    
    return [grid[i] for i in closest_indices]


def fit_direction_from_counts_4val(
    counts: NDArray[np.int64],
    grid_simplex: NDArray[np.float64],
    direction: str = "X->Y",
) -> Tuple[Tuple[NDArray[np.float64], Dict[int, NDArray[np.float64]]], float]:
    """Fit 4-valued causal model in specified direction."""
    if direction == "Y->X":
        counts = counts.T  # Transpose to swap X and Y
    
    # Get MLE estimates
    theta_X_mle, theta_Y_given_X_mle = mle_estimate_4val(counts)
    
    # Find neighboring grid points
    theta_X_candidates = neighbors_simplex(theta_X_mle, grid_simplex)
    
    theta_Y_candidates = {}
    for x in range(4):
        theta_Y_candidates[x] = neighbors_simplex(theta_Y_given_X_mle[x], grid_simplex)
    
    best_ll = -np.inf
    best_params = None
    
    # Search over candidate parameters
    for theta_X in theta_X_candidates:
        for theta_Y_combo in product(*[theta_Y_candidates[x] for x in range(4)]):
            theta_Y_given_X = {x: theta_Y_combo[x] for x in range(4)}
            
            ll = log_likelihood_fast_4val(counts, theta_X, theta_Y_given_X)
            
            if ll > best_ll:
                best_ll = ll
                best_params = (theta_X, theta_Y_given_X)
    
    return best_params, best_ll


def run_experiment_4val(
    n_trials=10000, sample_sizes=None, k_values=None
) -> tuple[dict[tuple[int, int], dict[str, float]], list[int], range]:
    """Run causal direction identification experiment with 4-valued variables."""
    if sample_sizes is None:
        sample_sizes = [50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000, 40000]
    if k_values is None:
        k_values = range(2, 6)  # Smaller range due to computational complexity
    
    results = {}
    
    grids = {k: discretize_params_simplex(k, 4) for k in k_values}
    
    for k in k_values:
        grid = grids[k]
        print(f"Running for k={k}")
        
        for n_samples in sample_sizes:
            wins = 0
            ties = 0
            
            for trial in range(n_trials):
        
                # Sample random parameters from grid
                theta_X = grid[np.random.randint(len(grid))]
                theta_Y_given_X = {}
                for x in range(4):
                    theta_Y_given_X[x] = grid[np.random.randint(len(grid))]
                
                # Generate data
                counts = generate_counts_fast_4val(n_samples, theta_X, theta_Y_given_X)
                
                # Fit both directions
                try:
                    ll_XY = fit_direction_from_counts_4val(counts, grid, "X->Y")[1]
                    ll_YX = fit_direction_from_counts_4val(counts, grid, "Y->X")[1]
                    
                    if ll_XY > ll_YX:
                        wins += 1
                    elif ll_XY == ll_YX:
                        wins += 0.5
                        ties += 1
                except Exception as e:
                    continue
            
            prop_correct = wins / n_trials
            prop_ties = ties / n_trials
            results[(k, n_samples)] = {"accuracy": prop_correct, "ties": prop_ties}
            print(f"  n={n_samples}: Acc={prop_correct:.3f}, Ties={prop_ties:.3f}")

    
    return results, sample_sizes, k_values


def plot_results(results, sample_sizes, k_values) -> None:
    # Accuracy plot
    plt.figure(figsize=(12, 7))
    for k in k_values:
        accuracies = [results[(k, n)]["accuracy"] for n in sample_sizes]
        plt.plot(sample_sizes, accuracies, marker="o", label=f"k={k}")
    plt.xscale("log")
    plt.xlabel("Sample size (log scale)")
    plt.ylabel("Proportion correct (X→Y wins)")
    plt.title("Causal direction identification accuracy vs Sample size")
    plt.legend(title="Discretization granularity k")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Tie proportion plot
    plt.figure(figsize=(12, 7))
    for k in k_values:
        tie_rates = [results[(k, n)]["ties"] for n in sample_sizes]
        plt.plot(sample_sizes, tie_rates, marker="s", linestyle="--", label=f"k={k}")
    plt.xscale("log")
    plt.xlabel("Sample size (log scale)")
    plt.ylabel("Proportion of ties")
    plt.title("Proportion of likelihood ties vs Sample size")
    plt.legend(title="Discretization granularity k")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    results, sample_sizes, k_values = run_experiment_4val()
    # Use the same plotting function from original code
    plot_results(results, sample_sizes, k_values)
