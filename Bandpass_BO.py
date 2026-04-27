import numpy as np
import random
import matplotlib.pyplot as plt
import joblib
import os
import time
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


# IMPORT SHARED CIRCUIT + COST FUNCTIONS

from Bandpass_SA import (
    R_values, C_values,
    eval_cost,
    analyze_active_filter,
    F_TARGET, F_REJECT_LOW, F_REJECT_HIGH
)



# SEARCH SPACE


R_idx_bounds = (0, len(R_values) - 1)
C_idx_bounds = (0, len(C_values) - 1)
DIM = 8 




# HELPER FUNCTIONS


def decode_params(x):
    """Convert index vector to actual component values"""
    r_vals = [R_values[int(i)] for i in x[:6]]
    c_vals = [C_values[int(i)] for i in x[6:]]
    return (*r_vals, *c_vals)


def values_to_indices(params):
    """Convert component values to index vector"""
    indices = []
    for i in range(6):
        indices.append(R_values.index(params[i]))
    for i in range(6, 8):
        indices.append(C_values.index(params[i]))
    return np.array(indices)


def objective_real(x):
    """True expensive objective (PySpice simulation)"""
    r1, r2, r3, r4, r5, r6, c1, c2 = decode_params(x)
    return eval_cost(r1, r2, r3, r4, r5, r6, c1, c2)




# LOAD PRE-TRAINED SURROGATE


def load_surrogate_model(model_path="surrogate_model.pkl"):
    """Load pre-trained surrogate model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Surrogate model not found: {model_path}\n"
            "Run these first:\n"
            "  1. python generate_dataset.py\n"
            "  2. python train_surrogate.py"
        )
    
    model_data = joblib.load(model_path)
    print(f"Loaded surrogate model: {model_data['n_samples']} training samples")
    print(f"Model R²: GP={model_data['gp_r2']:.4f}, RF={model_data['rf_r2']:.4f}")
    return model_data




# SURROGATE PREDICTION


def surrogate_predict(X_values, model_data, return_std=False):
    """
    Predict cost using pre-trained surrogate.
    X_values: array of component values (not indices!)
    """
    scaler = model_data['scaler']
    gp = model_data['gp']
    use_log = model_data.get('use_log_transform', False)
    
    X_scaled = scaler.transform(X_values)
    
    if return_std:
        mu_log, sigma_log = gp.predict(X_scaled, return_std=True)
        if use_log:
            mu = np.expm1(mu_log)
            # Approximate std after exp transform
            sigma = mu * sigma_log
        else:
            mu, sigma = mu_log, sigma_log
        return mu, sigma
    else:
        mu_log = gp.predict(X_scaled)
        if use_log:
            return np.expm1(mu_log)
        return mu_log




# ACQUISITION FUNCTIONS


def expected_improvement(mu, sigma, y_best, xi=0.01):
    """Expected Improvement (maximize to find minimum cost)"""
    sigma = np.clip(sigma, 1e-9, None)
    imp = y_best - mu - xi
    Z = imp / sigma
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-9] = 0.0
    return ei


def upper_confidence_bound(mu, sigma, kappa=2.0):
    """UCB for minimization: lower is better"""
    return mu - kappa * sigma




# CANDIDATE GENERATION


def generate_candidates(n_samples, best_points=None, n_local=0):
    """Generate candidate points (as component values)"""
    candidates = []
    
    # Random global samples
    n_random = n_samples - n_local
    for _ in range(n_random):
        params = (
            [random.choice(R_values) for _ in range(6)] +
            [random.choice(C_values) for _ in range(2)]
        )
        candidates.append(params)
    
    # Local perturbations around best points
    if best_points is not None and n_local > 0:
        for i in range(n_local):
            base = best_points[i % len(best_points)]
            params = list(base).copy()
            
            # Perturb 1-3 components
            n_perturb = random.randint(1, 3)
            for _ in range(n_perturb):
                idx = random.randint(0, 7)
                if idx < 6:  # Resistor
                    curr_idx = R_values.index(params[idx])
                    delta = random.choice([-2, -1, 1, 2])
                    new_idx = max(0, min(len(R_values)-1, curr_idx + delta))
                    params[idx] = R_values[new_idx]
                else:  # Capacitor
                    curr_idx = C_values.index(params[idx])
                    delta = random.choice([-1, 1])
                    new_idx = max(0, min(len(C_values)-1, curr_idx + delta))
                    params[idx] = C_values[new_idx]
            
            candidates.append(params)
    
    return np.array(candidates)




# BAYESIAN OPTIMIZATION WITH PRE-TRAINED SURROGATE


def bayesian_optimization_surrogate(
    max_iters=100,
    n_candidates=10000,
    n_verify=5,
    acquisition='EI',
    early_stop=20,
    target_cost=0.01
):
    """
    Fast BO using pre-trained surrogate model.
    Only verifies top candidates with real simulation.
    
    Args:
        max_iters: Max optimization iterations
        n_candidates: Candidates per iteration (cheap - surrogate only)
        n_verify: Number of top candidates to verify with real simulation
        acquisition: 'EI' or 'UCB'
        early_stop: Stop if no improvement for this many iters
        target_cost: Stop if cost below this
    """
    # Load pre-trained model
    model_data = load_surrogate_model()
    
    # Initialize from best in training data
    best_params = model_data['best_params']
    best_cost = model_data['best_cost']
    
    # Track verified points
    verified_X = [best_params.copy()]
    verified_y = [best_cost]
    
    history = [best_cost]
    no_improve = 0
    n_real_evals = 0
    
    print("\n" + "=" * 60)
    print("BAYESIAN OPTIMIZATION (Pre-trained Surrogate)")
    print("=" * 60)
    print(f"Starting from dataset best: {best_cost:.6g}")
    print(f"Candidates/iter: {n_candidates}, Verify top: {n_verify}")
    print("-" * 60)
    
    start_time = time.time()
    
    for i in range(max_iters):
        # Generate candidates (cheap!)
        candidates = generate_candidates(
            n_candidates, 
            best_points=verified_X[:5],
            n_local=int(n_candidates * 0.3)
        )
        
        # Predict with surrogate (very fast!)
        mu, sigma = surrogate_predict(candidates, model_data, return_std=True)
        
        # Compute acquisition
        y_best = min(verified_y)
        if acquisition == 'EI':
            acq = expected_improvement(mu, sigma, y_best)
            top_indices = np.argsort(acq)[-n_verify:][::-1]  # Highest EI
        else:
            acq = upper_confidence_bound(mu, sigma)
            top_indices = np.argsort(acq)[:n_verify]  # Lowest UCB
        
        # Verify top candidates with real simulation
        improved = False
        for idx in top_indices:
            params = tuple(candidates[idx])
            
            # Skip if already verified
            skip = False
            for vx in verified_X:
                if np.allclose(params, vx):
                    skip = True
                    break
            if skip:
                continue
            
            # Real evaluation
            real_cost = eval_cost(*params)
            n_real_evals += 1
            
            verified_X.append(np.array(params))
            verified_y.append(real_cost)
            
            if real_cost < best_cost:
                best_cost = real_cost
                best_params = np.array(params)
                improved = True
                no_improve = 0
                print(f"Iter {i+1:3d} | NEW BEST: {best_cost:.6g} (predicted: {mu[idx]:.4g})")
        
        if not improved:
            no_improve += 1
        
        history.append(best_cost)
        
        # Progress update
        if (i + 1) % 10 == 0:
            print(f"Iter {i+1:3d} | Best: {best_cost:.6g} | Real evals: {n_real_evals} | No improve: {no_improve}")
        
        # Early stopping
        if best_cost < target_cost:
            print(f"\n*** Target cost {target_cost} reached! ***")
            break
        
        if no_improve >= early_stop:
            print(f"\n*** Early stop: no improvement for {early_stop} iters ***")
            break
    
    elapsed = time.time() - start_time
    
    print("-" * 60)
    print(f"Optimization complete!")
    print(f"Total real evaluations: {n_real_evals}")
    print(f"Time elapsed: {elapsed:.1f}s")
    
    return best_params, best_cost, history, n_real_evals




# MAIN


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    print("\n" + "=" * 60)
    print("BANDPASS FILTER OPTIMIZATION")
    print("Using Pre-trained Surrogate Model")
    print("=" * 60)
    print(f"Target: {F_TARGET} Hz bandpass")
    print(f"Reject: {F_REJECT_LOW} Hz and {F_REJECT_HIGH} Hz")
    print("=" * 60)

    # Run optimized BO with surrogate
    best_params, best_cost, hist, n_evals = bayesian_optimization_surrogate(
        max_iters=100,
        n_candidates=10000,
        n_verify=3,
        acquisition='EI',
        early_stop=25,
        target_cost=0.005
    )

    r1, r2, r3, r4, r5, r6, c1, c2 = best_params

    print("\n" + "=" * 60)
    print("OPTIMIZED COMPONENT VALUES (BO-Surrogate)")
    print("=" * 60)
    print(f"R1 = {r1/1e3:.2f} kΩ")
    print(f"R2 = {r2/1e3:.2f} kΩ")
    print(f"R3 = {r3/1e3:.2f} kΩ")
    print(f"R4 = {r4/1e3:.2f} kΩ")
    print(f"R5 = {r5/1e3:.2f} kΩ")
    print(f"R6 = {r6/1e3:.2f} kΩ")
    print(f"C1 = {c1*1e9:.2f} nF")
    print(f"C2 = {c2*1e9:.2f} nF")
    print(f"\nBest Cost = {best_cost:.6g}")
    print(f"Real Evaluations = {n_evals} (vs SA's 500)")
    print("=" * 60)

    # Validate final solution with real simulation
    print("\nValidating final solution...")
    freqs, vout = analyze_active_filter(
        r1, r2, r3, r4, r5, r6, c1, c2, debug=True
    )

    
    
    # PLOTS 
    
    
    plt.figure(figsize=(12, 5))

    # Plot 1: Cost History
    plt.subplot(1, 2, 1)
    plt.plot(hist, color='darkorange', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Best Cost (log scale)')
    plt.title(f'BO (Surrogate) Cost History\nFinal: {best_cost:.4g}, Evals: {n_evals}')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    # Plot 2: Filter Response
    plt.subplot(1, 2, 2)
    if freqs is not None and vout is not None:
        plt.plot(freqs, vout, color='crimson', linewidth=2, label='Optimized Response')
        plt.axvline(F_TARGET, color='green', linestyle='--', label=f'Target ({F_TARGET/1e3:.1f}kHz)')
        plt.axvline(F_REJECT_LOW, color='blue', linestyle=':', label='Reject Low')
        plt.axvline(F_REJECT_HIGH, color='blue', linestyle=':', label='Reject High')
    plt.xscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.title('Optimized Filter Response (Vo)')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.show()
