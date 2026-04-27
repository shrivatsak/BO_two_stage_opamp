#type: ignore
import random
import math         
import numpy as np
import matplotlib.pyplot as plt
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

F_TARGET = 1000.0
F_REJECT_LOW = 200.0
F_REJECT_HIGH = 5000.0

R_values = [100, 120, 150, 180, 220, 270, 330, 390, 470, 560, 680, 820,
            1e3, 1.2e3, 1.5e3, 1.8e3, 2.2e3, 2.7e3, 3.3e3, 3.9e3, 4.7e3, 5.6e3, 6.8e3, 8.2e3,               #E12
            10e3, 12e3, 15e3, 18e3, 22e3, 27e3, 33e3, 39e3, 47e3, 56e3, 68e3, 82e3, 100e3]
C_values = [1e-9, 1.5e-9, 2.2e-9, 3.3e-9, 4.7e-9, 6.8e-9,
            10e-9, 15e-9, 22e-9, 33e-9, 47e-9, 68e-9, 100e-9]

#the circuit
def analyze_active_filter(r1, r2, r3, r4, r5, r6, c1, c2, debug=False):
   
    try:
        circuit = Circuit('Tow-Thomas Biquad')

        circuit.SinusoidalVoltageSource('input', 'vi', circuit.gnd, amplitude=1@u_V)

        
        circuit.R('1', 'vi', 'op1_positive', r1@u_Ohm)    
        circuit.R('2', 'op1_positive', 'op2_output', r2@u_Ohm)       
        circuit.R('3', 'op1_negative', 'vo', r3@u_Ohm)       
        circuit.R('4', 'op1_negative', 'op1_output', r4@u_Ohm)
               
        circuit.R('g1', 'op1_positive', circuit.gnd, 1e9@u_Ohm)
        circuit.R('g2', 'op1_negative', circuit.gnd, 1e9@u_Ohm)
       
        circuit.VCVS('1', 'op1_output', circuit.gnd, 'op1_positive', 'op1_negative', 1e5)
                
        circuit.R('5', 'op1_output', 'op2_negative', r5@u_Ohm)       
        circuit.C('1', 'op2_negative', 'op2_output', c1@u_F)
       
        circuit.R('g3', 'op2_negative', circuit.gnd, 1e9@u_Ohm)
        
        circuit.VCVS('2', 'op2_output', circuit.gnd, circuit.gnd, 'op2_negative', 1e5)
        
        circuit.R('6', 'op2_output', 'op3_negative', r6@u_Ohm)
        circuit.C('2', 'op3_negative', 'vo', c2@u_F)
        
      
        circuit.R('g4', 'op3_negative', circuit.gnd, 1e9@u_Ohm)
        
        circuit.VCVS('3', 'vo', circuit.gnd, circuit.gnd, 'op3_negative', 1e5)

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.ac(start_frequency=50@u_Hz, stop_frequency=20e3@u_Hz,
                                number_of_points=300, variation='dec')

        freqs = np.array(analysis.frequency)
        vout = np.abs(np.array(analysis['vo']))
        
        if debug:
            print(f"Max gain: {np.max(vout):.3f}, Min gain: {np.min(vout):.3f}")
        
        if np.max(vout) < 1e-12 or np.isnan(vout).any() or np.isinf(vout).any():
            if debug:
                print("Warning: Invalid simulation output")
            return None, None
            
        return freqs, vout

    except Exception as e:
        if debug:
            print(f"Simulation error: {e}")
        return None, None


#CostFunction
def eval_cost(r1, r2, r3, r4, r5, r6, c1, c2):
    freq, vout = analyze_active_filter(r1, r2, r3, r4, r5, r6, c1, c2)
    
    if freq is None or vout is None:
        return 1e6
    
    gain_pass = np.interp(F_TARGET, freq, vout)
    gain_low = np.interp(F_REJECT_LOW, freq, vout)
    gain_high = np.interp(F_REJECT_HIGH, freq, vout)
    
    max_gain = np.max(vout)
    if max_gain > 1e-12:
        gain_pass_norm = gain_pass / max_gain
        gain_low_norm = gain_low / max_gain
        gain_high_norm = gain_high / max_gain
    else:
        return 1e6
    
    cost = (1.0 - gain_pass_norm)**2 + 2.5 * (gain_low_norm**2 + gain_high_norm**2)
    return cost


# Neighbour
def get_neighbour(r1, r2, r3, r4, r5, r6, c1, c2):
    if random.random() < 0.15:
        return ([random.choice(R_values) for _ in range(6)] +
                [random.choice(C_values) for _ in range(2)])

    choice = random.choice(['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'c1', 'c2'])
    params = [r1, r2, r3, r4, r5, r6, c1, c2]
    
    if 'r' in choice:
        idx = int(choice[1]) - 1
        i = R_values.index(params[idx])
        j = max(0, min(len(R_values)-1, i + random.choice([-2, -1, 1, 2])))
        params[idx] = R_values[j]
    elif choice == 'c1':
        i = C_values.index(c1)
        j = max(0, min(len(C_values)-1, i + random.choice([-1, 1])))
        params[6] = C_values[j]
    else:
        i = C_values.index(c2)
        j = max(0, min(len(C_values)-1, i + random.choice([-1, 1])))
        params[7] = C_values[j]
        
    return params


#SA
def simulated_annealing(iters=400, temp_initial=2.0, cooling_rate=0.95):
    r1 = random.choice([r for r in R_values if 1e3 <= r <= 10e3])
    r2 = random.choice([r for r in R_values if 1e3 <= r <= 10e3])
    r3 = random.choice([r for r in R_values if 1e3 <= r <= 10e3])
    r4 = random.choice([r for r in R_values if 1e3 <= r <= 10e3])
    r5 = random.choice([r for r in R_values if 1e3 <= r <= 50e3])
    r6 = random.choice([r for r in R_values if 1e3 <= r <= 50e3])
    c1 = random.choice([c for c in C_values if c >= 10e-9])
    c2 = random.choice([c for c in C_values if c >= 10e-9])
                                       
    cost_curr = eval_cost(r1, r2, r3, r4, r5, r6, c1, c2)
    
    best = (r1, r2, r3, r4, r5, r6, c1, c2)
    best_cost = cost_curr
    history = [cost_curr]
    temp = temp_initial

    for i in range(iters):
        params_n = get_neighbour(r1, r2, r3, r4, r5, r6, c1, c2)
        r1_n, r2_n, r3_n, r4_n, r5_n, r6_n, c1_n, c2_n = params_n
        cost_new = eval_cost(r1_n, r2_n, r3_n, r4_n, r5_n, r6_n, c1_n, c2_n)
        
        if cost_new < cost_curr or random.random() < math.exp((cost_curr - cost_new) / temp):
            r1, r2, r3, r4, r5, r6, c1, c2 = params_n
            cost_curr = cost_new
            
        if cost_curr < best_cost:
            best = (r1, r2, r3, r4, r5, r6, c1, c2)
            best_cost = cost_curr
            
        history.append(best_cost)
        temp *= cooling_rate
        
        if (i + 1) % 10 == 0:
            print(f"Iter {i+1}/{iters} | T={temp:.3f} | Current={cost_curr:.4g} | Best={best_cost:.4g}")
        
    print("\nOptimization complete.")
    return best, best_cost, history



if __name__ == "__main__":
    random.seed(42)
    
    print("Starting optimization...")
    print("Target: 1000 Hz bandpass, reject 200 Hz and 5000 Hz\n")
    
    (r1_opt, r2_opt, r3_opt, r4_opt, r5_opt, r6_opt, c1_opt, c2_opt), best_cost, hist = simulated_annealing()

    print("\n--- Optimized Component Values ---")
    print(f"R1 = {r1_opt/1e3:.2f} kΩ")
    print(f"R2 = {r2_opt/1e3:.2f} kΩ")
    print(f"R3 = {r3_opt/1e3:.2f} kΩ")
    print(f"R4 = {r4_opt/1e3:.2f} kΩ")
    print(f"R5 = {r5_opt/1e3:.2f} kΩ")
    print(f"R6 = {r6_opt/1e3:.2f} kΩ")
    print(f"C1 = {c1_opt*1e9:.2f} nF")
    print(f"C2 = {c2_opt*1e9:.2f} nF")
    print(f"Best Cost = {best_cost:.5g}")

    print("\nTesting final solution...")
    freqs, vout = analyze_active_filter(r1_opt, r2_opt, r3_opt, r4_opt, r5_opt, r6_opt, 
                                        c1_opt, c2_opt, debug=True)

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(hist, color='darkorange', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Cost (log scale)')
    plt.title('SA Cost History')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

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
