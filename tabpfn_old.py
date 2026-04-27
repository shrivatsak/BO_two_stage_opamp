# bandpass_BO_tabpfn.py
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# TabPFN import
from tabpfn import TabPFNRegressor

# Import shared circuit and cost functions
from Bandpass_SA import (
    R_values, C_values,
    eval_cost,
    analyze_active_filter,
    F_TARGET, F_REJECT_LOW, F_REJECT_HIGH
)

# Import optimized sampling functions for TabPFN (OLD versions)
from tabpfnsample_old import (
    lhs_initial_sampling,
    mixed_candidate_sampling,
    bootstrap_tabpfn_predict,
    select_candidate_by_acquisition,
    balanced_initial_sampling
)


def random_sample_circuit():
    """Generate a random circuit configuration."""
    return (
        [random.choice(R_values) for _ in range(6)] +
        [random.choice(C_values) for _ in range(2)]
    )


def generate_initial_dataset(n_samples=30):
    """Generate initial dataset using balanced sampling.
    
    Uses a mix of LHS, focused, and deliberate failure samples to ensure
    TabPFN sees the full range of filter quality. This is critical because
    TabPFN needs contrast between good and bad samples to learn properly.
    
    Without failure examples, TabPFN may predict uniformly optimistic costs.
    """
    print(f"Generating initial dataset ({n_samples} samples via balanced sampling)...")
    
    # Use balanced sampling with explicit failure injection
    X, y = balanced_initial_sampling(
        n_samples=n_samples,
        R_values=R_values,
        C_values=C_values,
        eval_cost_fn=eval_cost,
        seed=42,
        verbose=True
    )
    
    print(f"  Done! Best initial cost: {min(y):.6f}")
    return np.array(X), np.array(y)


def prepare_features(X_raw):
    """Convert raw component values to normalized log-scale features."""
    X = np.array(X_raw, dtype=np.float64)
    
    # Log10 transform
    X_log = np.log10(X)
    
    # Normalize to [0, 1] based on component ranges
    r_min, r_max = np.log10(min(R_values)), np.log10(max(R_values))
    c_min, c_max = np.log10(min(C_values)), np.log10(max(C_values))
    
    X_norm = np.zeros_like(X_log)
    X_norm[:, :6] = (X_log[:, :6] - r_min) / (r_max - r_min)  # Resistors
    X_norm[:, 6:] = (X_log[:, 6:] - c_min) / (c_max - c_min)  # Capacitors
    
    return X_norm


def generate_candidates(n_candidates, best_points=None):
    """Generate candidate circuits for acquisition."""
    candidates = []
    
    # Global random samples (70%)
    n_global = int(n_candidates * 0.7)
    for _ in range(n_global):
        candidates.append(random_sample_circuit())
    
    # Local perturbations around best points (30%)
    if best_points is not None and len(best_points) > 0:
        n_local = n_candidates - n_global
        for i in range(n_local):
            base = best_points[i % len(best_points)].copy()
            params = list(base)
            
            # Perturb 1-3 random components
            n_perturb = random.randint(1, 3)
            indices = random.sample(range(8), n_perturb)
            
            for idx in indices:
                if idx < 6:  # Resistor
                    curr_idx = R_values.index(params[idx])
                    delta = random.choice([-2, -1, 1, 2])
                    new_idx = max(0, min(len(R_values) - 1, curr_idx + delta))
                    params[idx] = R_values[new_idx]
                else:  # Capacitor
                    curr_idx = C_values.index(params[idx])
                    delta = random.choice([-1, 1])
                    new_idx = max(0, min(len(C_values) - 1, curr_idx + delta))
                    params[idx] = C_values[new_idx]
            
            candidates.append(params)
    else:
        # Fill with random if no best points
        for _ in range(n_candidates - n_global):
            candidates.append(random_sample_circuit())
    
    return np.array(candidates)


def bayesian_optimization_tabpfn(
    max_iters=200,
    n_initial=30,
    n_candidates=500,
    early_stop=50,
    target_cost=0.005,
    refit_every=5,  # Only refit model every N iterations
    use_bootstrap=True,  # New: use bootstrap for uncertainty estimation
    n_bootstrap=5,       # New: number of bootstrap iterations
    verbose=True
):
    """
    Bayesian Optimization using TabPFN surrogate with improved sampling.
    
    Args:
        max_iters: Maximum iterations
        n_initial: Initial random samples
        n_candidates: Candidate pool size per iteration
        early_stop: Stop if no improvement for this many iterations
        target_cost: Stop if cost drops below this
        refit_every: Refit model every N iterations (speeds up optimization)
        use_bootstrap: Use bootstrap for uncertainty estimation (pseudo-EI)
        n_bootstrap: Number of bootstrap iterations for uncertainty
        verbose: Print progress
        
    Returns:
        best_params, best_cost, history, n_evals
    """
    
    # Step 1: Initial dataset
    X_train, y_train = generate_initial_dataset(n_initial)
    n_evals = len(y_train)
    
    # Track best
    best_idx = np.argmin(y_train)
    best_cost = y_train[best_idx]
    best_params = X_train[best_idx].copy()
    
    # Top-k for local search
    top_k = 5
    sorted_idx = np.argsort(y_train)[:top_k]
    best_points = [X_train[i] for i in sorted_idx]
    
    # Track evaluated configs (as tuples for fast lookup)
    evaluated = set()
    for i in range(len(X_train)):
        evaluated.add(tuple(X_train[i]))
    
    history = [best_cost]
    
    if verbose:
        print(f"\n{'='*60}")
        print("BAYESIAN OPTIMIZATION WITH TABPFN")
        print(f"{'='*60}")
        print(f"Initial best: {best_cost:.6f}")
        print(f"Max iterations: {max_iters}")
        print(f"Bootstrap uncertainty: {use_bootstrap}")
        print(f"-"*60)
    
    # Step 2: Create TabPFN model
    if verbose:
        print("Loading TabPFN model...")
    model = TabPFNRegressor()
    if verbose:
        print("Model loaded! Starting optimization...")
    
    start_time = time.time()
    no_improve = 0
    need_refit = True
    
    # Cache for normalized training data
    X_norm = None
    y_log = None
    
    # Step 3: Main loop
    for iteration in range(max_iters):
        
        # Prepare training data and fit model (only when needed)
        if need_refit:
            X_norm = prepare_features(X_train)
            y_log = np.log1p(y_train)
            model.fit(X_norm, y_log)
            need_refit = False
        
        # Generate candidates using mixed sampling strategy
        # This replaces the original generate_candidates function
        candidates = mixed_candidate_sampling(
            n_candidates=n_candidates,
            best_points=best_points,
            R_values=R_values,
            C_values=C_values,
            X_train=X_train,
            y_train=y_train
        )
        
        # Filter out already evaluated candidates
        new_candidates = []
        for c in candidates:
            if tuple(c) not in evaluated:
                new_candidates.append(c)
        
        if len(new_candidates) == 0:
            # All candidates already evaluated, generate more random ones
            for _ in range(100):
                c = random_sample_circuit()
                if tuple(c) not in evaluated:
                    new_candidates.append(c)
            if len(new_candidates) == 0:
                continue
        
        candidates = np.array(new_candidates[:n_candidates])
        cand_norm = prepare_features(candidates)
        
        # Predict with or without bootstrap uncertainty
        if use_bootstrap and (iteration % 3 == 0):  # Bootstrap every 3rd iteration for speed
            mean_pred, std_pred = bootstrap_tabpfn_predict(
                model, X_norm, y_log, cand_norm, n_bootstrap=n_bootstrap
            )
            best_cost_log = np.log1p(best_cost)
            best_cand_idx = select_candidate_by_acquisition(
                candidates, mean_pred, std_pred, best_cost_log,
                strategy='ei', exploration_prob=0.05
            )
            y_pred = np.expm1(mean_pred)
        else:
            # Standard prediction without bootstrap
            y_pred_log = model.predict(cand_norm)
            y_pred = np.expm1(y_pred_log)
            best_cand_idx = np.argmin(y_pred)
            
            # 10% random exploration when not using bootstrap
            if random.random() < 0.1:
                best_cand_idx = random.randint(0, len(candidates) - 1)
        
        selected = candidates[best_cand_idx]
        
        # Evaluate with PySpice
        real_cost = eval_cost(*selected)
        n_evals += 1
        evaluated.add(tuple(selected))
        
        # Add to training set
        if real_cost < 1e5:
            X_train = np.vstack([X_train, selected])
            y_train = np.append(y_train, real_cost)
            # Schedule refit every N iterations or when we find improvement
            if (iteration + 1) % refit_every == 0:
                need_refit = True
        
        # Update best
        if real_cost < best_cost:
            best_cost = real_cost
            best_params = selected.copy()
            no_improve = 0
            need_refit = True  # Refit when we find improvement
            
            # Update best points
            sorted_idx = np.argsort(y_train)[:top_k]
            best_points = [X_train[i] for i in sorted_idx]
            
            if verbose:
                print(f"Iter {iteration+1:3d} | NEW BEST: {best_cost:.6f}")
        else:
            no_improve += 1
        
        history.append(best_cost)
        
        # Progress every 25 iterations
        if verbose and (iteration + 1) % 25 == 0:
            elapsed = time.time() - start_time
            print(f"Iter {iteration+1:3d} | Best: {best_cost:.6f} | Evals: {n_evals} | Time: {elapsed:.1f}s")
        
        # Early stopping
        if best_cost < target_cost:
            if verbose:
                print(f"\n*** Target cost {target_cost} reached! ***")
            break
        
        if no_improve >= early_stop:
            if verbose:
                print(f"\n*** Early stop: no improvement for {early_stop} iters ***")
            break
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"-"*60)
        print(f"Done! Iterations: {len(history)-1}, Evals: {n_evals}, Time: {elapsed:.1f}s")
    
    return best_params, best_cost, history, n_evals


def plot_results(history, best_params, best_cost, n_evals):
    """Plot optimization results."""
    r1, r2, r3, r4, r5, r6, c1, c2 = best_params
    freqs, vout = analyze_active_filter(r1, r2, r3, r4, r5, r6, c1, c2, debug=True)
    
    plt.figure(figsize=(12, 5))
    
    # Cost history
    plt.subplot(1, 2, 1)
    plt.plot(history, color='darkorange', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Best Cost (log)')
    plt.title(f'TabPFN-BO Convergence\nFinal: {best_cost:.4g}, Evals: {n_evals}')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Frequency response
    plt.subplot(1, 2, 2)
    if freqs is not None and vout is not None:
        plt.plot(freqs, vout, color='crimson', linewidth=2, label='Optimized')
        plt.axvline(F_TARGET, color='green', linestyle='--', label=f'Target ({F_TARGET}Hz)')
        plt.axvline(F_REJECT_LOW, color='blue', linestyle=':', label=f'Reject ({F_REJECT_LOW}Hz)')
        plt.axvline(F_REJECT_HIGH, color='blue', linestyle=':', label=f'Reject ({F_REJECT_HIGH}Hz)')
    plt.xscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.title('Optimized Filter Response')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('tabpfn_bo_result.png', dpi=150)
    print("\nPlot saved to: tabpfn_bo_result.png")
    plt.show()


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    
    print("\n" + "="*60)
    print("BANDPASS FILTER OPTIMIZATION")
    print("Bayesian Optimization with TabPFN Surrogate")
    print("Enhanced with LHS + Mixed Sampling + Bootstrap Uncertainty")
    print("="*60)
    print(f"Target: {F_TARGET} Hz bandpass")
    print(f"Reject: {F_REJECT_LOW} Hz and {F_REJECT_HIGH} Hz")
    print("="*60)
    
    # Run optimization with new parameters
    best_params, best_cost, history, n_evals = bayesian_optimization_tabpfn(
        max_iters=200,
        n_initial=30,
        n_candidates=500,
        early_stop=40,
        target_cost=0.01,
        refit_every=5,
        use_bootstrap=True,   # Enable bootstrap uncertainty
        n_bootstrap=5,        # Number of bootstrap samples
        verbose=True
    )
    
    r1, r2, r3, r4, r5, r6, c1, c2 = best_params
    
    # Print results
    print("\n" + "="*60)
    print("OPTIMIZED COMPONENT VALUES")
    print("="*60)
    print(f"R1 = {r1/1e3:.2f} kΩ")
    print(f"R2 = {r2/1e3:.2f} kΩ")
    print(f"R3 = {r3/1e3:.2f} kΩ")
    print(f"R4 = {r4/1e3:.2f} kΩ")
    print(f"R5 = {r5/1e3:.2f} kΩ")
    print(f"R6 = {r6/1e3:.2f} kΩ")
    print(f"C1 = {c1*1e9:.2f} nF")
    print(f"C2 = {c2*1e9:.2f} nF")
    print(f"\nBest Cost = {best_cost:.6g}")
    print(f"PySpice Evaluations = {n_evals}")
    print("="*60)
    
    # Plot
    print("\nGenerating plots...")
    plot_results(history, best_params, best_cost, n_evals)