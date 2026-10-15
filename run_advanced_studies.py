"""
Advanced Studies Script: PBC Topology, Trotter Order, & Noise Sensitivity
Qiskit Fall Fest 2026 Phase 1 Hackathon - Problem S3
"""

import sys
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

from src.problem import create_ising_chain
from src.processors import load_processor
from src.routing import transpile_with_routing_analysis
from src.metrics import predict_analytical_optimal_n

fig_dir = PROJECT_ROOT / 'results' / 'figures'
tab_dir = PROJECT_ROOT / 'results' / 'tables'
fig_dir.mkdir(parents=True, exist_ok=True)
tab_dir.mkdir(parents=True, exist_ok=True)

proc_A = load_processor(PROJECT_ROOT / 'processors' / 'processor_A.json')
proc_B = load_processor(PROJECT_ROOT / 'processors' / 'processor_B.json')


def make_noise(two_q_err=0.01, single_q_err=0.001, readout_err=0.02):
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(single_q_err, 1), ['x', 'sx', 'rz'])
    noise.add_all_qubit_quantum_error(depolarizing_error(two_q_err, 2), ['cx'])
    noise.add_all_qubit_readout_error(ReadoutError([[1.0 - readout_err, readout_err], [readout_err, 1.0 - readout_err]]))
    return noise


# -------------------------------------------------------------------------
# STUDY 1: Periodic Boundary Conditions (Routing Overhead on Trees)
# -------------------------------------------------------------------------
print("[1/3] Running Periodic Boundary Condition (PBC) study with exact SWAP counting...")
NUM_QUBITS = 4
J = 1.0
h = 0.5
TIME = 1.0
SHOTS = 2048
STEPS = [1, 2, 3, 4, 5, 6, 8, 10]

sim_pbc = create_ising_chain(num_qubits=NUM_QUBITS, J=J, h=h, periodic=True)
exact_state_pbc = sim_pbc.exact_evolved_state(None, TIME)
exact_E_pbc = sim_pbc.exact_energy_expectation(exact_state_pbc)

aer_A = AerSimulator(noise_model=make_noise(0.01))
aer_B = AerSimulator(noise_model=make_noise(0.01))

pbc_records = []
for n in STEPS:
    qc_b = sim_pbc.build_trotter_circuit(TIME, n, order=1, basis=None)
    qc_z = sim_pbc.build_trotter_circuit(TIME, n, order=1, basis='Z')
    qc_x = sim_pbc.build_trotter_circuit(TIME, n, order=1, basis='X')

    # Ideal
    ideal_sv = sim_pbc.ideal_statevector(TIME, n, order=1)
    trot_E = sim_pbc.exact_energy_expectation(ideal_sv)
    trot_err = abs(trot_E - exact_E_pbc) / abs(exact_E_pbc)

    # Proc A with exact SWAPs
    tA, rA = transpile_with_routing_analysis(qc_b, proc_A, optimization_level=1, seed_transpiler=42)
    tAz, _ = transpile_with_routing_analysis(qc_z, proc_A, optimization_level=1, seed_transpiler=42)
    tAx, _ = transpile_with_routing_analysis(qc_x, proc_A, optimization_level=1, seed_transpiler=42)
    counts_Az = aer_A.run(tAz, shots=SHOTS, seed_simulator=42).result().get_counts()
    counts_Ax = aer_A.run(tAx, shots=SHOTS, seed_simulator=42).result().get_counts()
    eA, _ = sim_pbc.compute_energy_from_counts(counts_Az, counts_Ax, SHOTS)
    tot_err_A = abs(eA - exact_E_pbc) / abs(exact_E_pbc)

    # Proc B with exact SWAPs
    tB, rB = transpile_with_routing_analysis(qc_b, proc_B, optimization_level=1, seed_transpiler=42)
    tBz, _ = transpile_with_routing_analysis(qc_z, proc_B, optimization_level=1, seed_transpiler=42)
    tBx, _ = transpile_with_routing_analysis(qc_x, proc_B, optimization_level=1, seed_transpiler=42)
    counts_Bz = aer_B.run(tBz, shots=SHOTS, seed_simulator=42).result().get_counts()
    counts_Bx = aer_B.run(tBx, shots=SHOTS, seed_simulator=42).result().get_counts()
    eB, _ = sim_pbc.compute_energy_from_counts(counts_Bz, counts_Bx, SHOTS)
    tot_err_B = abs(eB - exact_E_pbc) / abs(exact_E_pbc)

    pbc_records.append({
        'steps': n,
        'trotter_error': trot_err,
        'depth_A': tA.depth(),
        'cx_A': rA['transpiled_2q_gates'],
        'swaps_A': rA['exact_swaps'],
        'error_A': tot_err_A,
        'depth_B': tB.depth(),
        'cx_B': rB['transpiled_2q_gates'],
        'swaps_B': rB['exact_swaps'],
        'error_B': tot_err_B
    })

df_pbc = pd.DataFrame(pbc_records)
df_pbc.to_csv(tab_dir / 'pbc_comparison.csv', index=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
ax1.plot(df_pbc['steps'], df_pbc['cx_A'], 'o-', color='#D32F2F', linewidth=2, label='Processor A (T-Tree) 2Q Gates')
ax1.plot(df_pbc['steps'], df_pbc['cx_B'], 's-', color='#1976D2', linewidth=2, label='Processor B (Heavy-Hex Tree) 2Q Gates')
ax1.set_xlabel('Trotter Steps ($n$)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Two-Qubit (CX) Gates', fontsize=11, fontweight='bold')
ax1.set_title('Periodic Ring 2Q Gate Scaling (Exact SWAPs)', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', framealpha=0.9)
ax1.grid(True, linestyle=':', alpha=0.6)

ax2.plot(df_pbc['steps'], df_pbc['trotter_error'], 'k--', linewidth=2, label='Algorithmic Trotter Error (Ideal)')
ax2.plot(df_pbc['steps'], df_pbc['error_A'], 'o-', color='#D32F2F', linewidth=2.2, label='Processor A Total Error')
ax2.plot(df_pbc['steps'], df_pbc['error_B'], 's-', color='#1976D2', linewidth=2.2, label='Processor B Total Error')
ax2.set_xlabel('Trotter Steps ($n$)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Relative Energy Error', fontsize=11, fontweight='bold')
ax2.set_title('Periodic TFIM Ring Error Tradeoff on Tree Topologies', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', framealpha=0.9)
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
fig.savefig(fig_dir / 'periodic_boundary_routing.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("      Saved periodic_boundary_routing.png and pbc_comparison.csv")


# -------------------------------------------------------------------------
# STUDY 2: 1st vs 2nd-Order Trotter Convergence Study
# -------------------------------------------------------------------------
print("[2/3] Running 1st vs 2nd-Order Trotter comparison study...")
sim_obc = create_ising_chain(num_qubits=NUM_QUBITS, J=J, h=h, periodic=False)
exact_E_obc = sim_obc.exact_energy_expectation(sim_obc.exact_evolved_state(None, TIME))

order_records = []
for n in STEPS:
    sv_1 = sim_obc.ideal_statevector(TIME, n, order=1)
    e_trot_1 = sim_obc.exact_energy_expectation(sv_1)
    err_trot_1 = abs(e_trot_1 - exact_E_obc) / abs(exact_E_obc)
    qc_b1 = sim_obc.build_trotter_circuit(TIME, n, order=1, basis=None)
    t1, _ = transpile_with_routing_analysis(qc_b1, proc_B, optimization_level=1, seed_transpiler=42)

    qcz_1, _ = transpile_with_routing_analysis(sim_obc.build_trotter_circuit(TIME, n, order=1, basis='Z'), proc_B, optimization_level=1, seed_transpiler=42)
    qcx_1, _ = transpile_with_routing_analysis(sim_obc.build_trotter_circuit(TIME, n, order=1, basis='X'), proc_B, optimization_level=1, seed_transpiler=42)
    e_noisy_1, _ = sim_obc.compute_energy_from_counts(aer_B.run(qcz_1, shots=SHOTS, seed_simulator=42).result().get_counts(),
                                                      aer_B.run(qcx_1, shots=SHOTS, seed_simulator=42).result().get_counts(), SHOTS)
    err_tot_1 = abs(e_noisy_1 - exact_E_obc) / abs(exact_E_obc)

    sv_2 = sim_obc.ideal_statevector(TIME, n, order=2)
    e_trot_2 = sim_obc.exact_energy_expectation(sv_2)
    err_trot_2 = abs(e_trot_2 - exact_E_obc) / abs(exact_E_obc)
    qc_b2 = sim_obc.build_trotter_circuit(TIME, n, order=2, basis=None)
    t2, _ = transpile_with_routing_analysis(qc_b2, proc_B, optimization_level=1, seed_transpiler=42)

    qcz_2, _ = transpile_with_routing_analysis(sim_obc.build_trotter_circuit(TIME, n, order=2, basis='Z'), proc_B, optimization_level=1, seed_transpiler=42)
    qcx_2, _ = transpile_with_routing_analysis(sim_obc.build_trotter_circuit(TIME, n, order=2, basis='X'), proc_B, optimization_level=1, seed_transpiler=42)
    e_noisy_2, _ = sim_obc.compute_energy_from_counts(aer_B.run(qcz_2, shots=SHOTS, seed_simulator=42).result().get_counts(),
                                                      aer_B.run(qcx_2, shots=SHOTS, seed_simulator=42).result().get_counts(), SHOTS)
    err_tot_2 = abs(e_noisy_2 - exact_E_obc) / abs(exact_E_obc)

    order_records.append({
        'steps': n,
        'order1_trotter_error': err_trot_1,
        'order1_total_error': err_tot_1,
        'order1_depth': t1.depth(),
        'order1_cx': t1.count_ops().get('cx', 0),
        'order2_trotter_error': err_trot_2,
        'order2_total_error': err_tot_2,
        'order2_depth': t2.depth(),
        'order2_cx': t2.count_ops().get('cx', 0)
    })

df_order = pd.DataFrame(order_records)
df_order.to_csv(tab_dir / 'trotter_order_comparison.csv', index=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
ax1.loglog(df_order['steps'], df_order['order1_trotter_error'], 'o-', color='#1976D2', linewidth=2.2, label=r'1st Order $\sim \mathcal{O}(\Delta t)$')
ax1.loglog(df_order['steps'], df_order['order2_trotter_error'], 's-', color='#E65100', linewidth=2.2, label=r'2nd Order (Strang) $\sim \mathcal{O}(\Delta t^2)$')
ax1.set_xlabel('Trotter Steps ($n$)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Algorithmic Trotter Error', fontsize=11, fontweight='bold')
ax1.set_title('Trotter Discretization Convergence', fontsize=12, fontweight='bold')
ax1.legend(framealpha=0.9)
ax1.grid(True, which="both", linestyle=':', alpha=0.6)

ax2.plot(df_order['steps'], df_order['order1_total_error'], 'o-', color='#1976D2', linewidth=2.2, label='1st Order Total Error (n*=3)')
ax2.plot(df_order['steps'], df_order['order2_total_error'], 's-', color='#E65100', linewidth=2.2, label='2nd Order Total Error (n*=2)')
ax2.axvline(x=3, color='#1976D2', linestyle=':', label='1st Order Sweet Spot (n*=3)')
ax2.axvline(x=2, color='#E65100', linestyle=':', label='2nd Order Sweet Spot (n*=2)')
ax2.set_xlabel('Trotter Steps ($n$)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Total Relative Error', fontsize=11, fontweight='bold')
ax2.set_title('1st vs 2nd-Order Tradeoff Under Hardware Noise', fontsize=12, fontweight='bold')
ax2.legend(framealpha=0.9)
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
fig.savefig(fig_dir / 'trotter_order_comparison.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("      Saved trotter_order_comparison.png and trotter_order_comparison.csv")


# -------------------------------------------------------------------------
# STUDY 3: Noise Sensitivity Sweep (n* vs 2Q Hardware Error Rate)
# -------------------------------------------------------------------------
print("[3/3] Running noise sensitivity sweep...")
noise_rates = [0.002, 0.005, 0.010, 0.020, 0.040]
noise_records = []

for err_2q in noise_rates:
    sim_noise = AerSimulator(noise_model=make_noise(two_q_err=err_2q))
    step_errs = []
    for n in STEPS:
        qcz, _ = transpile_with_routing_analysis(sim_obc.build_trotter_circuit(TIME, n, order=1, basis='Z'), proc_B, optimization_level=1, seed_transpiler=42)
        qcx, _ = transpile_with_routing_analysis(sim_obc.build_trotter_circuit(TIME, n, order=1, basis='X'), proc_B, optimization_level=1, seed_transpiler=42)
        e_meas, _ = sim_obc.compute_energy_from_counts(sim_noise.run(qcz, shots=SHOTS, seed_simulator=42).result().get_counts(),
                                                      sim_noise.run(qcx, shots=SHOTS, seed_simulator=42).result().get_counts(), SHOTS)
        tot_err = abs(e_meas - exact_E_obc) / abs(exact_E_obc)
        step_errs.append({'n': n, 'err': tot_err})

    best = min(step_errs, key=lambda x: x['err'])
    ana = predict_analytical_optimal_n(J, h, TIME, NUM_QUBITS, err_2q, order=1)
    noise_records.append({
        'two_qubit_error_rate': f"{err_2q * 100:.1f}%",
        'empirical_n_star': best['n'],
        'analytical_n_continuous': round(ana['analytical_n_continuous'], 2),
        'analytical_n_rounded': ana['analytical_n_optimal'],
        'min_achieved_error': f"{best['err'] * 100:.2f}%"
    })

df_noise = pd.DataFrame(noise_records)
df_noise.to_csv(tab_dir / 'noise_sensitivity.csv', index=False)

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
rates_pct = [r * 100 for r in noise_rates]
emp_n = [r['empirical_n_star'] for r in noise_records]
ana_n = [r['analytical_n_continuous'] for r in noise_records]

ax.plot(rates_pct, emp_n, 'o-', color='#D32F2F', linewidth=2.2, label='Empirical Discrete Minimum $n^*$')
ax.plot(rates_pct, ana_n, 's--', color='#1976D2', linewidth=2, label=r'Continuous Analytical Prediction $n^* \approx \sqrt{C_{\mathrm{Trotter}}/C_{\mathrm{Noise}}}$')
ax.set_xlabel('Two-Qubit (CX) Error Rate (%)', fontsize=11, fontweight='bold')
ax.set_ylabel('Optimal Trotter Steps ($n^*$)', fontsize=11, fontweight='bold')
ax.set_title('Optimal Step Count $n^*$ Shift Across Hardware Quality Regimes', fontsize=12, fontweight='bold')
ax.set_xticks(rates_pct)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(framealpha=0.9)

plt.tight_layout()
fig.savefig(fig_dir / 'noise_sensitivity.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("      Saved noise_sensitivity.png and noise_sensitivity.csv")
print("All advanced studies completed!")
