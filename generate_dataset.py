# generate_dataset.py
import random
import csv
import numpy as np
from Bandpass_SA import R_values, C_values, eval_cost
import time

# Configuration
N_RANDOM = 400        # Random samples for global coverage
N_FOCUSED = 200       # Focused samples in promising regions
N_TOTAL = N_RANDOM + N_FOCUSED

random.seed(42)
np.random.seed(42)

print("=" * 60)
print("DATASET GENERATION FOR SURROGATE MODEL")
print("=" * 60)
print(f"Random samples: {N_RANDOM}")
print(f"Focused samples: {N_FOCUSED}")
print(f"Total: {N_TOTAL}")
print("=" * 60)

#LHS
def latin_hypercube_indices(n_samples, n_r_values, n_c_values):
    """Generate LHS-distributed indices for better space coverage"""
    samples = []
    
    # Create intervals for each dimension
    r_intervals = np.linspace(0, n_r_values - 1, n_samples + 1).astype(int)
    c_intervals = np.linspace(0, n_c_values - 1, n_samples + 1).astype(int)
    
    # Shuffle interval assignments for each dimension
    r_perms = [np.random.permutation(n_samples) for _ in range(6)]
    c_perms = [np.random.permutation(n_samples) for _ in range(2)]
    
    for i in range(n_samples):
        sample = []
        # 6 resistor indices
        for dim in range(6):
            interval = r_perms[dim][i]
            low = r_intervals[interval]
            high = r_intervals[interval + 1]
            idx = random.randint(low, max(low, high - 1))
            idx = min(idx, n_r_values - 1)
            sample.append(idx)
        # 2 capacitor indices  
        for dim in range(2):
            interval = c_perms[dim][i]
            low = c_intervals[interval]
            high = c_intervals[interval + 1]
            idx = random.randint(low, max(low, high - 1))
            idx = min(idx, n_c_values - 1)
            sample.append(idx)
        samples.append(sample)
    
    return samples

#random sampling not used
def random_sample():
    """Pure random sample"""
    return (
        random.choice(R_values),
        random.choice(R_values),
        random.choice(R_values),
        random.choice(R_values),
        random.choice(R_values),
        random.choice(R_values),
        random.choice(C_values),
        random.choice(C_values),
    )


def focused_sample():
    """Sample from promising regions (mid-range R, larger C)"""
    # Resistors in 1k-50k range (good for audio filters)
    r_good = [r for r in R_values if 1e3 <= r <= 50e3]
    # Capacitors in 10nF-100nF range
    c_good = [c for c in C_values if 10e-9 <= c <= 100e-9]
    
    return (
        random.choice(r_good),
        random.choice(r_good),
        random.choice(r_good),
        random.choice(r_good),
        random.choice(r_good),
        random.choice(r_good),
        random.choice(c_good),
        random.choice(c_good),
    )


def indices_to_values(indices):
    """Convert index list to actual component values"""
    return tuple(
        [R_values[i] for i in indices[:6]] +
        [C_values[i] for i in indices[6:]]
    )


# Generate samples
print("\nGenerating samples...")
start_time = time.time()

all_samples = []
best_cost = 1e6
best_params = None

with open("bandpass_dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["r1", "r2", "r3", "r4", "r5", "r6", "c1", "c2", "cost"])
    
    # Phase 1: LHS-based random samples
    print("\n[Phase 1] Latin Hypercube Sampling...")
    lhs_indices = latin_hypercube_indices(N_RANDOM, len(R_values), len(C_values))
    
    for i, indices in enumerate(lhs_indices):
        params = indices_to_values(indices)
        cost = eval_cost(*params)
        writer.writerow(list(params) + [cost])
        all_samples.append((params, cost))
        
        if cost < best_cost:
            best_cost = cost
            best_params = params
        
        if (i + 1) % 50 == 0:
            print(f"  {i+1:3d}/{N_RANDOM} | Current best: {best_cost:.4g}")
    
    print(f"  Phase 1 complete. Best cost: {best_cost:.4g}")
    
    # Phase 2: Focused samples in promising regions
    print("\n[Phase 2] Focused Sampling...")
    for i in range(N_FOCUSED):
        params = focused_sample()
        cost = eval_cost(*params)
        writer.writerow(list(params) + [cost])
        all_samples.append((params, cost))
        
        if cost < best_cost:
            best_cost = cost
            best_params = params
        
        if (i + 1) % 50 == 0:
            print(f"  {i+1:3d}/{N_FOCUSED} | Current best: {best_cost:.4g}")
    
    print(f"  Phase 2 complete. Best cost: {best_cost:.4g}")

elapsed = time.time() - start_time

# Statistics
costs = [s[1] for s in all_samples]
valid_costs = [c for c in costs if c < 1e5]

print("\n" + "=" * 60)
print("DATASET GENERATION COMPLETE")
print("=" * 60)
print(f"Total samples: {len(all_samples)}")
print(f"Valid samples: {len(valid_costs)} ({100*len(valid_costs)/len(all_samples):.1f}%)")
print(f"Best cost found: {best_cost:.6g}")
print(f"Cost range: [{min(valid_costs):.4g}, {max(valid_costs):.4g}]")
print(f"Time elapsed: {elapsed:.1f}s")
print(f"\nDataset saved as: bandpass_dataset.csv")
print("=" * 60)

if best_params:
    print("\nBest parameters found during generation:")
    print(f"  R1={best_params[0]/1e3:.2f}k, R2={best_params[1]/1e3:.2f}k, R3={best_params[2]/1e3:.2f}k")
    print(f"  R4={best_params[3]/1e3:.2f}k, R5={best_params[4]/1e3:.2f}k, R6={best_params[5]/1e3:.2f}k")
    print(f"  C1={best_params[6]*1e9:.2f}nF, C2={best_params[7]*1e9:.2f}nF")
