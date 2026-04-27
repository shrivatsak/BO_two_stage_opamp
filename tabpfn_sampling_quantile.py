# tabpfn_sampling.py

import numpy as np
from scipy.stats import qmc
import random
from typing import List, Tuple, Optional


def lhs_initial_sampling(
    n_samples: int,
    R_values: List[float],
    C_values: List[float],
    seed: Optional[int] = None
) -> np.ndarray:
    
    # Dimension: 6 resistors + 2 capacitors = 8
    n_dims = 8
    
    # Create LHS sampler
    sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
    
    # Generate samples in [0, 1]^8
    unit_samples = sampler.random(n=n_samples)
    
    # Convert to log-space indices for each component
    # This maps uniform [0,1] to discrete component indices
    
    n_r = len(R_values)
    n_c = len(C_values)
    
    # Log-transform the available values for proper spacing
    log_R = np.log10(R_values)
    log_C = np.log10(C_values)
    
    samples = np.zeros((n_samples, n_dims))
    
    for i in range(n_samples):
        # Map unit samples to component indices via log-space interpolation
        for j in range(6):  # Resistors
            # Interpolate in log-space, then snap to nearest available value
            log_val = log_R[0] + unit_samples[i, j] * (log_R[-1] - log_R[0])
            idx = np.argmin(np.abs(log_R - log_val))
            samples[i, j] = R_values[idx]
        
        for j in range(2):  # Capacitors
            log_val = log_C[0] + unit_samples[i, 6 + j] * (log_C[-1] - log_C[0])
            idx = np.argmin(np.abs(log_C - log_val))
            samples[i, 6 + j] = C_values[idx]
    
    return samples


def _local_perturbation(
    base_params: np.ndarray,
    R_values: List[float],
    C_values: List[float],
    perturbation_strength: str = 'medium'
) -> List[float]:

    params = list(base_params.copy())
    
    # Determine perturbation magnitude based on strength
    strength_map = {
        'small': (1, 2, [-1, 1], [-1, 1]),
        'medium': (2, 3, [-2, -1, 1, 2], [-1, 1]),
        'large': (3, 4, [-3, -2, -1, 1, 2, 3], [-2, -1, 1, 2])
    }
    
    min_perturb, max_perturb, r_deltas, c_deltas = strength_map.get(
        perturbation_strength, strength_map['medium']
    )
    
    # Perturb random subset of components
    n_perturb = random.randint(min_perturb, max_perturb)
    indices = random.sample(range(8), min(n_perturb, 8))
    
    for idx in indices:
        if idx < 6:  # Resistor
            try:
                curr_idx = R_values.index(params[idx])
            except ValueError:
                curr_idx = np.argmin(np.abs(np.array(R_values) - params[idx]))
            delta = random.choice(r_deltas)
            new_idx = max(0, min(len(R_values) - 1, curr_idx + delta))
            params[idx] = R_values[new_idx]
        else:  # Capacitor
            try:
                curr_idx = C_values.index(params[idx])
            except ValueError:
                curr_idx = np.argmin(np.abs(np.array(C_values) - params[idx]))
            delta = random.choice(c_deltas)
            new_idx = max(0, min(len(C_values) - 1, curr_idx + delta))
            params[idx] = C_values[new_idx]
    
    return params


def _stratified_random_sampling(
    n_samples: int,
    R_values: List[float],
    C_values: List[float]
) -> List[List[float]]:
    
    samples = []
    
    # Divide R and C ranges into strata (low, mid, high)
    n_r = len(R_values)
    n_c = len(C_values)
    
    r_strata = [
        R_values[:n_r//3],           # Low
        R_values[n_r//3:2*n_r//3],   # Mid
        R_values[2*n_r//3:]          # High
    ]
    
    c_strata = [
        C_values[:n_c//3],
        C_values[n_c//3:2*n_c//3],
        C_values[2*n_c//3:]
    ]
    
    for _ in range(n_samples):
        params = []
        
        # For each resistor, randomly pick a stratum, then a value within it
        for _ in range(6):
            stratum = random.choice(r_strata)
            if len(stratum) > 0:
                params.append(random.choice(stratum))
            else:
                params.append(random.choice(R_values))
        
        # Same for capacitors
        for _ in range(2):
            stratum = random.choice(c_strata)
            if len(stratum) > 0:
                params.append(random.choice(stratum))
            else:
                params.append(random.choice(C_values))
        
        samples.append(params)
    
    return samples


def _hard_negative_exploration(
    n_samples: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    R_values: List[float],
    C_values: List[float]
) -> List[List[float]]:

    samples = []
    
    if len(X_train) == 0:
        # Fallback to random if no training data
        for _ in range(n_samples):
            params = [random.choice(R_values) for _ in range(6)]
            params += [random.choice(C_values) for _ in range(2)]
            samples.append(params)
        return samples
    
    # Strategy 1: Sample near median-cost points (boundary exploration)
    # These are points where the model is most uncertain
    median_cost = np.median(y_train)
    boundary_mask = np.abs(y_train - median_cost) < np.std(y_train)
    boundary_points = X_train[boundary_mask] if np.any(boundary_mask) else X_train
    
    n_boundary = n_samples // 2
    for i in range(n_boundary):
        if len(boundary_points) > 0:
            base = boundary_points[i % len(boundary_points)]
            # Large perturbation to explore boundary
            samples.append(_local_perturbation(base, R_values, C_values, 'large'))
    
    # Strategy 2: Sample from sparse regions (furthest from existing points)
    # Find the centroid of each component dimension and sample oppositely
    n_sparse = n_samples - n_boundary
    
    log_X = np.log10(X_train)
    centroids = np.median(log_X, axis=0)
    
    log_R = np.log10(R_values)
    log_C = np.log10(C_values)
    
    for _ in range(n_sparse):
        params = []
        
        # For each dimension, sample away from the centroid
        for j in range(6):  # Resistors
            if random.random() < 0.5:
                # Sample from the opposite end of the range
                if centroids[j] < np.median(log_R):
                    idx = random.randint(len(R_values)//2, len(R_values)-1)
                else:
                    idx = random.randint(0, len(R_values)//2)
            else:
                # Random with bias toward extremes
                if random.random() < 0.6:
                    idx = random.choice([0, 1, 2, len(R_values)-3, len(R_values)-2, len(R_values)-1])
                else:
                    idx = random.randint(0, len(R_values)-1)
            params.append(R_values[min(idx, len(R_values)-1)])
        
        for j in range(2):  # Capacitors
            if random.random() < 0.5:
                if centroids[6+j] < np.median(log_C):
                    idx = random.randint(len(C_values)//2, len(C_values)-1)
                else:
                    idx = random.randint(0, len(C_values)//2)
            else:
                if random.random() < 0.6:
                    idx = random.choice([0, len(C_values)-1])
                else:
                    idx = random.randint(0, len(C_values)-1)
            params.append(C_values[min(idx, len(C_values)-1)])
        
        samples.append(params)
    
    return samples


def mixed_candidate_sampling(
    n_candidates: int,
    best_points: Optional[List[np.ndarray]],
    R_values: List[float],
    C_values: List[float],
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None
) -> np.ndarray:
    
    candidates = []
    
    # Allocation
    n_local = int(n_candidates * 0.50)
    n_stratified = int(n_candidates * 0.30)
    n_hard = n_candidates - n_local - n_stratified
    
    # 1. Local perturbations (50%)
    if best_points is not None and len(best_points) > 0:
        for i in range(n_local):
            base = best_points[i % len(best_points)]
            # Vary perturbation strength
            if i % 3 == 0:
                strength = 'small'
            elif i % 3 == 1:
                strength = 'medium'
            else:
                strength = 'large'
            candidates.append(_local_perturbation(base, R_values, C_values, strength))
    else:
        # No best points yet; use LHS for local portion
        lhs_samples = lhs_initial_sampling(n_local, R_values, C_values)
        candidates.extend([list(s) for s in lhs_samples])
    
    # 2. Stratified random (30%)
    stratified = _stratified_random_sampling(n_stratified, R_values, C_values)
    candidates.extend(stratified)
    
    # 3. Hard-negative exploration (20%)
    if X_train is not None and y_train is not None and len(X_train) > 0:
        hard = _hard_negative_exploration(n_hard, X_train, y_train, R_values, C_values)
    else:
        # Fallback to pure random at extremes
        hard = []
        for _ in range(n_hard):
            params = []
            for _ in range(6):
                # Bias toward extreme values
                if random.random() < 0.5:
                    params.append(random.choice(R_values[:5] + R_values[-5:]))
                else:
                    params.append(random.choice(R_values))
            for _ in range(2):
                if random.random() < 0.5:
                    params.append(random.choice([C_values[0], C_values[-1]]))
                else:
                    params.append(random.choice(C_values))
            hard.append(params)
    candidates.extend(hard)
    
    return np.array(candidates)


def bootstrap_tabpfn_predict(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_bootstrap: int = 5,
    subsample_ratio: float = 0.8
) -> Tuple[np.ndarray, np.ndarray]:
    """DEPRECATED: Use quantile_tabpfn_predict instead.
    
    This function refits TabPFN n_bootstrap times, which is extremely slow
    because each refit is a full transformer forward pass. Kept for
    backward compatibility only.
    """
   
    n_train = len(X_train)
    n_subsample = max(10, int(n_train * subsample_ratio))
    
    predictions = []
    
    for b in range(n_bootstrap):
        # Subsample with replacement
        indices = np.random.choice(n_train, size=n_subsample, replace=True)
        
        # Shuffle order (TabPFN can be sensitive to input order)
        np.random.shuffle(indices)
        
        X_boot = X_train[indices]
        y_boot = y_train[indices]
        
        # Refit and predict
        model.fit(X_boot, y_boot)
        pred = model.predict(X_test)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    # Refit on full data for consistency
    model.fit(X_train, y_train)
    
    return mean_pred, std_pred


def quantile_tabpfn_predict(
    model,
    X_test: np.ndarray,
    quantiles: list = [0.1, 0.5, 0.9]
) -> Tuple[np.ndarray, np.ndarray]:
    """Get mean and uncertainty from TabPFN using native quantile predictions.
    
    Uses a SINGLE forward pass instead of multiple bootstrap refits.
    TabPFN natively outputs predictive distributions, so we extract
    uncertainty from the interquantile range without any refitting.
    
    Speed comparison:
        bootstrap (n=5): 6 transformer forward passes per call
        quantile:         0 extra passes (piggybacks on existing prediction)
    
    Args:
        model: Fitted TabPFNRegressor (must already be fit)
        X_test: Normalized test features
        quantiles: Quantiles to predict [low, median, high]
    
    Returns:
        mean_pred: Median prediction (q50)
        std_pred: Approximate std from interquantile range
    """
    q_preds = model.predict(X_test, output_type='quantiles', quantiles=quantiles)
    
    # q_preds is a list of arrays, one per quantile
    q_low = q_preds[0]    # 10th percentile
    q_med = q_preds[1]    # 50th percentile (median)
    q_high = q_preds[2]   # 90th percentile
    
    mean_pred = q_med
    # 10th-90th percentile range ≈ 2.56 standard deviations for normal dist
    std_pred = np.maximum((q_high - q_low) / 2.56, 1e-9)
    
    return mean_pred, std_pred


def balanced_initial_sampling(
    n_samples: int,
    R_values: List[float],
    C_values: List[float],
    eval_cost_fn,
    seed: Optional[int] = None,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate balanced initial dataset with mix of good, bad, and random samples.
    
    TabPFN needs contrast between good and bad samples to learn the cost
    landscape properly. Without failure examples, it may predict uniformly
    optimistic costs.
    
    Sampling strategy:
        - 50% LHS space-filling samples (broad coverage)
        - 25% focused samples near typical good filter regions
        - 25% deliberate extreme/failure samples (for contrast)
    
    Args:
        n_samples: Total number of samples to generate
        R_values: Available resistor values
        C_values: Available capacitor values
        eval_cost_fn: Cost function to evaluate each sample
        seed: Random seed for reproducibility
        verbose: Print progress info
    
    Returns:
        X: Array of component values (n_samples, 8)
        y: Array of costs (n_samples,)
    """
    if seed is not None:
        rng_state = random.getstate()
        np_state = np.random.get_state()
        random.seed(seed)
        np.random.seed(seed)
    
    X = []
    y = []
    
    # 1. LHS space-filling samples (50%)
    n_lhs = n_samples // 2
    lhs_samples = lhs_initial_sampling(n_lhs, R_values, C_values, seed=seed)
    
    if verbose:
        print(f"  LHS samples: {n_lhs}")
    
    for s in lhs_samples:
        cost = eval_cost_fn(*s)
        X.append(list(s))
        y.append(cost)
    
    # 2. Focused samples near typical bandpass regions (25%)
    n_focused = n_samples // 4
    mid_R = [r for r in R_values if 1e3 <= r <= 22e3]
    mid_C = [c for c in C_values if 4.7e-9 <= c <= 68e-9]
    
    if verbose:
        print(f"  Focused samples: {n_focused}")
    
    if not mid_R:
        mid_R = R_values
    if not mid_C:
        mid_C = C_values
    
    for _ in range(n_focused):
        params = [random.choice(mid_R) for _ in range(6)]
        params += [random.choice(mid_C) for _ in range(2)]
        cost = eval_cost_fn(*params)
        X.append(params)
        y.append(cost)
    
    # 3. Extreme/failure samples for contrast (25%)
    n_extreme = n_samples - n_lhs - n_focused
    
    if verbose:
        print(f"  Extreme samples: {n_extreme}")
    
    for _ in range(n_extreme):
        params = []
        for _ in range(6):
            # Bias toward extreme resistor values
            if random.random() < 0.6:
                params.append(random.choice(R_values[:3] + R_values[-3:]))
            else:
                params.append(random.choice(R_values))
        for _ in range(2):
            if random.random() < 0.6:
                params.append(random.choice([C_values[0], C_values[-1]]))
            else:
                params.append(random.choice(C_values))
        cost = eval_cost_fn(*params)
        X.append(params)
        y.append(cost)
    
    if seed is not None:
        random.setstate(rng_state)
        np.random.set_state(np_state)
    
    return np.array(X), np.array(y)


def pseudo_expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    best_cost: float,
    xi: float = 0.01
) -> np.ndarray:
    
    from scipy.stats import norm
    
    # Ensure std is not zero
    std = np.maximum(std, 1e-9)
    
    # Improvement over best (we're minimizing, so improvement = best - predicted)
    improvement = best_cost - mean
    
    # Z-score
    Z = (improvement - xi) / std
    
    # Expected Improvement
    ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
    
    # Handle edge cases
    ei = np.where(std < 1e-8, 0.0, ei)
    
    return ei


def select_candidate_by_acquisition(
    candidates: np.ndarray,
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    best_cost_log: float,
    strategy: str = 'ei',
    exploration_prob: float = 0.1
) -> int:
    
    # Random exploration with small probability
    if random.random() < exploration_prob:
        return random.randint(0, len(candidates) - 1)
    
    if strategy == 'ei':
        scores = pseudo_expected_improvement(mean_pred, std_pred, best_cost_log)
        return np.argmax(scores)
    
    elif strategy == 'ucb':
        # Lower confidence bound (since we're minimizing)
        # Select point with lowest (mean - kappa * std)
        kappa = 2.0
        lcb = mean_pred - kappa * std_pred
        return np.argmin(lcb)
    
    else:  # greedy
        return np.argmin(mean_pred)
