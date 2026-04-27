# compare_optimizers3.py
"""
Compare ALL 4 optimizers for bandpass filter design:
  1. SA         - Simulated Annealing
  2. GP-BO      - Gaussian Process Bayesian Optimization (pre-trained surrogate)
  3. TabPFN-Old - TabPFN with bootstrap uncertainty (slow)
  4. TabPFN-New - TabPFN with native quantile uncertainty (fast)
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import time
import warnings
import os
warnings.filterwarnings('ignore')

# Shared circuit functions
from Bandpass_SA import (
    R_values, C_values,
    eval_cost,
    analyze_active_filter,
    simulated_annealing,
    F_TARGET, F_REJECT_LOW, F_REJECT_HIGH
)

# GP-BO (pre-trained surrogate)
from Bandpass_BO import bayesian_optimization_surrogate

# TabPFN Old (bootstrap)
from tabpfn_old import bayesian_optimization_tabpfn as tabpfn_old_bo

# TabPFN New (quantile)
from bandpass_BO_tabpfn import bayesian_optimization_tabpfn as tabpfn_new_bo


# ─── Colors & labels for each optimizer ───
OPTIMIZERS = {
    'SA':          {'color': 'steelblue',  'label': 'SA'},
    'GP-BO':       {'color': 'forestgreen','label': 'GP-BO'},
    'TabPFN-Old':  {'color': 'darkorange', 'label': 'TabPFN-Old (bootstrap)'},
    'TabPFN-New':  {'color': 'crimson',    'label': 'TabPFN-New (quantile)'},
}


def run_all(sa_iters=500, gp_iters=100, tabpfn_old_iters=200, tabpfn_new_iters=200):
    """Run all 4 optimizers and collect results."""

    print("=" * 75)
    print("  FULL OPTIMIZER COMPARISON: SA | GP-BO | TabPFN-Old | TabPFN-New")
    print("=" * 75)
    print(f"  Target: {F_TARGET} Hz bandpass")
    print(f"  Reject: {F_REJECT_LOW} Hz and {F_REJECT_HIGH} Hz")
    print("=" * 75)

    results = {}

    # ────────────────── 1. Simulated Annealing ──────────────────
    print("\n" + "=" * 75)
    print("[1/4] SIMULATED ANNEALING")
    print("=" * 75)

    random.seed(42)
    np.random.seed(42)

    t0 = time.time()
    sa_best, sa_cost, sa_history = simulated_annealing(
        iters=sa_iters, temp_initial=2.0, cooling_rate=0.95
    )
    sa_time = time.time() - t0

    r1, r2, r3, r4, r5, r6, c1, c2 = sa_best
    sa_freqs, sa_vout = analyze_active_filter(r1, r2, r3, r4, r5, r6, c1, c2)

    results['SA'] = {
        'params': sa_best,
        'cost':   sa_cost,
        'history': sa_history,
        'time':   sa_time,
        'evals':  sa_iters,
        'freqs':  sa_freqs,
        'vout':   sa_vout,
    }
    print(f"\n  SA  →  Cost={sa_cost:.6g}  Time={sa_time:.1f}s  Evals={sa_iters}")

    # ────────────────── 2. GP-BO (pre-trained surrogate) ──────────────────
    has_surrogate = os.path.exists('surrogate_model.pkl')
    if has_surrogate:
        print("\n" + "=" * 75)
        print("[2/4] GP-BO (Pre-trained Surrogate)")
        print("=" * 75)

        random.seed(42)
        np.random.seed(42)

        t0 = time.time()
        gp_best, gp_cost, gp_history, gp_evals = bayesian_optimization_surrogate(
            max_iters=gp_iters,
            n_candidates=10000,
            n_verify=3,
            acquisition='EI',
            early_stop=25,
            target_cost=0.005,
        )
        gp_time = time.time() - t0

        r1, r2, r3, r4, r5, r6, c1, c2 = gp_best
        gp_freqs, gp_vout = analyze_active_filter(r1, r2, r3, r4, r5, r6, c1, c2)

        results['GP-BO'] = {
            'params': gp_best,
            'cost':   gp_cost,
            'history': gp_history,
            'time':   gp_time,
            'evals':  gp_evals,
            'freqs':  gp_freqs,
            'vout':   gp_vout,
        }
        print(f"\n  GP-BO  →  Cost={gp_cost:.6g}  Time={gp_time:.1f}s  Evals={gp_evals}")
    else:
        print("\n" + "=" * 75)
        print("[2/4] GP-BO SKIPPED — surrogate_model.pkl not found")
        print("      Run:  python generate_dataset.py && python train_surrogate.py")
        print("=" * 75)

    # ────────────────── 3. TabPFN-Old (bootstrap) ──────────────────
    print("\n" + "=" * 75)
    print("[3/4] TabPFN-Old (Bootstrap Uncertainty)")
    print("=" * 75)

    random.seed(42)
    np.random.seed(42)

    t0 = time.time()
    told_best, told_cost, told_history, told_evals = tabpfn_old_bo(
        max_iters=tabpfn_old_iters,
        n_initial=30,
        n_candidates=500,
        early_stop=40,
        target_cost=0.005,
        refit_every=5,
        use_bootstrap=True,
        n_bootstrap=5,
        verbose=True,
    )
    told_time = time.time() - t0

    r1, r2, r3, r4, r5, r6, c1, c2 = told_best
    told_freqs, told_vout = analyze_active_filter(r1, r2, r3, r4, r5, r6, c1, c2)

    results['TabPFN-Old'] = {
        'params': told_best,
        'cost':   told_cost,
        'history': told_history,
        'time':   told_time,
        'evals':  told_evals,
        'freqs':  told_freqs,
        'vout':   told_vout,
    }
    print(f"\n  TabPFN-Old  →  Cost={told_cost:.6g}  Time={told_time:.1f}s  Evals={told_evals}")

    # ────────────────── 4. TabPFN-New (quantile) ──────────────────
    print("\n" + "=" * 75)
    print("[4/4] TabPFN-New (Native Quantile Uncertainty)")
    print("=" * 75)

    random.seed(42)
    np.random.seed(42)

    t0 = time.time()
    tnew_best, tnew_cost, tnew_history, tnew_evals = tabpfn_new_bo(
        max_iters=tabpfn_new_iters,
        n_initial=30,
        n_candidates=500,
        early_stop=40,
        target_cost=0.005,
        refit_every=5,
        n_verify=3,
        max_context=150,
        verbose=True,
    )
    tnew_time = time.time() - t0

    r1, r2, r3, r4, r5, r6, c1, c2 = tnew_best
    tnew_freqs, tnew_vout = analyze_active_filter(r1, r2, r3, r4, r5, r6, c1, c2)

    results['TabPFN-New'] = {
        'params': tnew_best,
        'cost':   tnew_cost,
        'history': tnew_history,
        'time':   tnew_time,
        'evals':  tnew_evals,
        'freqs':  tnew_freqs,
        'vout':   tnew_vout,
    }
    print(f"\n  TabPFN-New  →  Cost={tnew_cost:.6g}  Time={tnew_time:.1f}s  Evals={tnew_evals}")

    # ────────────────── Summary table ──────────────────
    active = {k: v for k, v in results.items()}
    names = list(active.keys())

    print("\n" + "=" * 75)
    print("COMPARISON SUMMARY")
    print("=" * 75)

    header = f"{'Metric':<20}" + "".join(f"{n:<18}" for n in names) + "Winner"
    print(header)
    print("-" * 75)

    # Best cost
    costs = {n: active[n]['cost'] for n in names}
    winner = min(costs, key=costs.get)
    row = f"{'Best Cost':<20}" + "".join(f"{costs[n]:<18.6g}" for n in names) + winner
    print(row)

    # Runtime
    times = {n: active[n]['time'] for n in names}
    winner = min(times, key=times.get)
    row = f"{'Runtime (s)':<20}" + "".join(f"{times[n]:<18.1f}" for n in names) + winner
    print(row)

    # Evaluations
    evals = {n: active[n]['evals'] for n in names}
    winner = min(evals, key=evals.get)
    row = f"{'SPICE Evals':<20}" + "".join(f"{evals[n]:<18}" for n in names) + winner
    print(row)

    # Cost per eval
    cpe = {n: active[n]['cost'] / max(active[n]['evals'], 1) for n in names}
    winner = min(cpe, key=cpe.get)
    row = f"{'Cost/Eval':<20}" + "".join(f"{cpe[n]:<18.6g}" for n in names) + winner
    print(row)

    print("-" * 75)

    # Component values
    print("\n" + "=" * 75)
    print("OPTIMIZED COMPONENTS")
    print("=" * 75)
    comp_header = f"{'Component':<12}" + "".join(f"{n:<18}" for n in names)
    print(comp_header)
    print("-" * 75)

    for i, cname in enumerate(['R1', 'R2', 'R3', 'R4', 'R5', 'R6']):
        row = f"{cname:<12}"
        for n in names:
            val = active[n]['params'][i] / 1e3
            row += f"{val:<18.2f}"
        print(row + " kΩ")

    for j, cname in enumerate(['C1', 'C2']):
        row = f"{cname:<12}"
        for n in names:
            val = active[n]['params'][6 + j] * 1e9
            row += f"{val:<18.2f}"
        print(row + " nF")

    print("=" * 75)

    return results


def print_results_table(results):
    """Print final cost and efficiency values in a tabular format to terminal."""

    names = list(results.keys())

    print("\n" + "=" * 90)
    print("  FINAL COST & EFFICIENCY TABLE")
    print("=" * 90)

    # Header
    col_w = 18
    header = f"{'Metric':<22}" + "".join(f"{n:<{col_w}}" for n in names) + "Winner"
    print(header)
    print("-" * 90)

    # Best cost
    costs = {n: results[n]['cost'] for n in names}
    winner = min(costs, key=costs.get)
    row = f"{'Best Cost':<22}" + "".join(f"{costs[n]:<{col_w}.6g}" for n in names) + winner
    print(row)

    # Runtime
    times = {n: results[n]['time'] for n in names}
    winner = min(times, key=times.get)
    row = f"{'Runtime (s)':<22}" + "".join(f"{times[n]:<{col_w}.1f}" for n in names) + winner
    print(row)

    # Evaluations
    evals = {n: results[n]['evals'] for n in names}
    winner = min(evals, key=evals.get)
    row = f"{'SPICE Evals':<22}" + "".join(f"{evals[n]:<{col_w}}" for n in names) + winner
    print(row)

    # Cost per eval
    cpe = {n: results[n]['cost'] / max(results[n]['evals'], 1) for n in names}
    winner = min(cpe, key=cpe.get)
    row = f"{'Cost/Eval':<22}" + "".join(f"{cpe[n]:<{col_w}.6g}" for n in names) + winner
    print(row)

    # Time per eval
    tpe = {n: results[n]['time'] / max(results[n]['evals'], 1) for n in names}
    winner = min(tpe, key=tpe.get)
    row = f"{'Time/Eval (s)':<22}" + "".join(f"{tpe[n]:<{col_w}.4f}" for n in names) + winner
    print(row)

    print("=" * 90)


def plot_comparison(results):
    """Create 2 separate PDF plots: Convergence and Optimized Filter Responses."""

    from matplotlib.backends.backend_pdf import PdfPages

    names = list(results.keys())
    colors = [OPTIMIZERS[n]['color'] for n in names]

    # ── Plot 1: Convergence curves (separate PDF) ──
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for n, c in zip(names, colors):
        h = results[n]['history']
        lbl = f"{n} ({results[n]['cost']:.4g})"
        ax1.plot(h, color=c, lw=2, label=lbl)
    ax1.set_yscale('log')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Best Cost (log)', fontsize=12)
    ax1.set_title('Convergence — All Optimizers', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig('convergence.pdf', format='pdf')
    print(f"\nPlot saved to: convergence.pdf")
    plt.close(fig1)

    # ── Plot 2: Frequency response overlay (separate PDF) ──
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for n, c in zip(names, colors):
        if results[n]['freqs'] is not None:
            ax2.plot(results[n]['freqs'], results[n]['vout'], color=c, lw=1.8, label=n)
    ax2.axvline(F_TARGET, color='green', ls='--', lw=1.5, label=f'Target ({int(F_TARGET)}Hz)')
    ax2.axvline(F_REJECT_LOW, color='gray', ls=':', lw=1, label='Reject band')
    ax2.axvline(F_REJECT_HIGH, color='gray', ls=':', lw=1)
    ax2.set_xscale('log')
    ax2.set_xlabel('Frequency (Hz)', fontsize=12)
    ax2.set_ylabel('Gain', fontsize=12)
    ax2.set_title('Optimized Filter Responses — All Optimizers', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig('filter_responses.pdf', format='pdf')
    print(f"Plot saved to: filter_responses.pdf")
    plt.close(fig2)


if __name__ == "__main__":
    print("\n" + "#" * 75)
    print("  BANDPASS FILTER: SA vs GP-BO vs TabPFN-Old vs TabPFN-New")
    print("#" * 75 + "\n")

    results = run_all(
        sa_iters=200,
        gp_iters=100,
        tabpfn_old_iters=200,
        tabpfn_new_iters=200,
    )

    print_results_table(results)
    plot_comparison(results)

    print("\n" + "=" * 75)
    print("  COMPARISON COMPLETE!")
    print("=" * 75)
