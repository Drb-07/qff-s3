"""
End-to-End Experiment Execution Script - Qiskit Fall Fest 2026 Phase 1: Problem S3
Executes full experimental pipeline:
1. Exact classical baseline diagonalization
2. Multi-step Trotter sweeps (n = 1..10) across ideal, Processor A (T-shape), and Processor B (Heavy-Hex)
3. Exact SWAP gate counting and routing analysis
4. Multi-seed statistical variance tracking (mean +/- std over 5 seeds)
5. Readout error mitigation applied to BOTH processors
6. Noisy state fidelity (density matrix) vs ideal Trotter fidelity
7. Circuit drawings generated and saved to results/figures/
8. 5-qubit chain comparison showcasing Processor A's T-shape routing bottleneck vs Processor B's 0 SWAPs
"""

import os
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

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

from src.problem import create_ising_chain
from src.processors import load_processor
from src.routing import transpile_with_routing_analysis, transpile_circuit
from src.metrics import (
    mitigate_readout_counts,
    find_optimal_trotter_step,
    compute_noisy_state_fidelity,
    predict_analytical_optimal_n
)
from src.visualization import (
    plot_processor_graph,
    plot_trotter_error_tradeoff,
    plot_circuit_scaling,
    plot_architecture_tradeoffs,
    plot_error_mitigation_impact
)


def create_noise_model(proc_cfg: dict) -> NoiseModel:
    """Construct Aer NoiseModel from processor JSON specification."""
    noise = NoiseModel()
    noise_cfg = proc_cfg.get('noise', {})
    single_err = noise_cfg.get('single_qubit_error', 0.001)
    two_err = noise_cfg.get('two_qubit_error', 0.01)
    readout_err = noise_cfg.get('readout_error', 0.02)

    noise.add_all_qubit_quantum_error(depolarizing_error(single_err, 1), ['x', 'sx', 'rz'])
    noise.add_all_qubit_quantum_error(depolarizing_error(two_err, 2), ['cx'])
    noise.add_all_qubit_readout_error(
        ReadoutError([[1.0 - readout_err, readout_err], [readout_err, 1.0 - readout_err]])
    )
    return noise


def main():
    print("=" * 70)
    print("STARTING REPRODUCIBLE EXPERIMENT PIPELINE: PROBLEM S3")
    print("=" * 70)

    fig_dir = PROJECT_ROOT / 'results' / 'figures'
    tab_dir = PROJECT_ROOT / 'results' / 'tables'
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Processors
    proc_A = load_processor(PROJECT_ROOT / 'processors' / 'processor_A.json')
    proc_B = load_processor(PROJECT_ROOT / 'processors' / 'processor_B.json')
    print(f"[1/6] Processors loaded: {proc_A['name']} (5Q T-Shape) and {proc_B['name']} (7Q Heavy-Hex)")

    plot_processor_graph(proc_A, title="Processor A (5 Qubits, T-Shaped Tree)",
                         save_path=fig_dir / 'processor_A_graph.png')
    plot_processor_graph(proc_B, title="Processor B (7 Qubits, Heavy-Hex Tree)",
                         save_path=fig_dir / 'processor_B_graph.png')
    print("      Saved processor graphs to results/figures/")

    # 2. Physics & Problem Setup
    NUM_QUBITS = 4
    J = 1.0
    h = 0.5
    TIME = 1.0
    SHOTS = 2048
    STEPS_RANGE = [1, 2, 3, 4, 5, 6, 7, 8, 10]
    SEEDS = [42, 43, 44, 45, 46]

    sim = create_ising_chain(num_qubits=NUM_QUBITS, J=J, h=h, periodic=False)

    # Exact Classical Baseline
    exact_state = sim.exact_evolved_state(None, TIME)
    exact_energy = sim.exact_energy_expectation(exact_state)
    denom = max(1e-9, abs(exact_energy))
    print(f"[2/6] Exact Classical Baseline: E_exact = {exact_energy:.6f}")

    noise_A = create_noise_model(proc_A)
    noise_B = create_noise_model(proc_B)
    aer_noisy_A = AerSimulator(noise_model=noise_A)
    aer_noisy_B = AerSimulator(noise_model=noise_B)
    ro_A = proc_A['noise']['readout_error']
    ro_B = proc_B['noise']['readout_error']

    # Draw and save sample transpiled circuits (n = 3 steps)
    qc_sample = sim.build_trotter_circuit(TIME, 3, order=1, basis=None)
    tA_sample, _ = transpile_with_routing_analysis(qc_sample, proc_A, optimization_level=1, seed_transpiler=42)
    tB_sample, _ = transpile_with_routing_analysis(qc_sample, proc_B, optimization_level=1, seed_transpiler=42)

    figA = tA_sample.draw(output='mpl', fold=25)
    figA.savefig(fig_dir / 'circuit_processor_A.png', dpi=180, bbox_inches='tight')
    plt.close(figA)

    figB = tB_sample.draw(output='mpl', fold=25)
    figB.savefig(fig_dir / 'circuit_processor_B.png', dpi=180, bbox_inches='tight')
    plt.close(figB)
    print("      Saved transpiled circuit diagrams: circuit_processor_A.png, circuit_processor_B.png")

    # 3. Parametric Trotter Step Sweep (n = 1..10)
    print(f"[3/6] Running parametric Trotter step sweeps (n = {STEPS_RANGE})...")
    records_A = []
    records_B = []
    trotter_errors = []

    for n in STEPS_RANGE:
        ideal_sv = sim.ideal_statevector(TIME, n, order=1)
        fid_ideal = float(np.abs(np.vdot(exact_state, ideal_sv))**2)
        trot_E = sim.exact_energy_expectation(ideal_sv)
        trot_err = abs(trot_E - exact_energy) / denom
        trotter_errors.append(trot_err)

        qc_bare = sim.build_trotter_circuit(TIME, n, order=1, basis=None)
        qc_z = sim.build_trotter_circuit(TIME, n, order=1, basis='Z')
        qc_x = sim.build_trotter_circuit(TIME, n, order=1, basis='X')

        t_A, r_A = transpile_with_routing_analysis(qc_bare, proc_A, optimization_level=1, seed_transpiler=42)
        t_A_z, _ = transpile_with_routing_analysis(qc_z, proc_A, optimization_level=1, seed_transpiler=42)
        t_A_x, _ = transpile_with_routing_analysis(qc_x, proc_A, optimization_level=1, seed_transpiler=42)

        t_B, r_B = transpile_with_routing_analysis(qc_bare, proc_B, optimization_level=1, seed_transpiler=42)
        t_B_z, _ = transpile_with_routing_analysis(qc_z, proc_B, optimization_level=1, seed_transpiler=42)
        t_B_x, _ = transpile_with_routing_analysis(qc_x, proc_B, optimization_level=1, seed_transpiler=42)

        # Multi-seed executions
        energies_A, energies_A_mit = [], []
        energies_B, energies_B_mit = [], []

        for s in SEEDS:
            # Proc A
            cz_A = aer_noisy_A.run(t_A_z, shots=SHOTS, seed_simulator=s).result().get_counts()
            cx_A = aer_noisy_A.run(t_A_x, shots=SHOTS, seed_simulator=s).result().get_counts()
            e_raw_A, se_A = sim.compute_energy_from_counts(cz_A, cx_A, SHOTS)
            energies_A.append(e_raw_A)

            mit_Az = mitigate_readout_counts(cz_A, ro_A, NUM_QUBITS)
            mit_Ax = mitigate_readout_counts(cx_A, ro_A, NUM_QUBITS)
            pzA = {k: int(round(v * SHOTS)) for k, v in mit_Az.items()}
            pxA = {k: int(round(v * SHOTS)) for k, v in mit_Ax.items()}
            e_mit_A, _ = sim.compute_energy_from_counts(pzA, pxA, SHOTS)
            energies_A_mit.append(e_mit_A)

            # Proc B
            cz_B = aer_noisy_B.run(t_B_z, shots=SHOTS, seed_simulator=s).result().get_counts()
            cx_B = aer_noisy_B.run(t_B_x, shots=SHOTS, seed_simulator=s).result().get_counts()
            e_raw_B, se_B = sim.compute_energy_from_counts(cz_B, cx_B, SHOTS)
            energies_B.append(e_raw_B)

            mit_Bz = mitigate_readout_counts(cz_B, ro_B, NUM_QUBITS)
            mit_Bx = mitigate_readout_counts(cx_B, ro_B, NUM_QUBITS)
            pzB = {k: int(round(v * SHOTS)) for k, v in mit_Bz.items()}
            pxB = {k: int(round(v * SHOTS)) for k, v in mit_Bx.items()}
            e_mit_B, _ = sim.compute_energy_from_counts(pzB, pxB, SHOTS)
            energies_B_mit.append(e_mit_B)

        mean_eA = float(np.mean(energies_A))
        std_eA = float(np.std(energies_A))
        mean_eA_mit = float(np.mean(energies_A_mit))
        std_eA_mit = float(np.std(energies_A_mit))

        mean_eB = float(np.mean(energies_B))
        std_eB = float(np.std(energies_B))
        mean_eB_mit = float(np.mean(energies_B_mit))
        std_eB_mit = float(np.std(energies_B_mit))

        # Noisy state fidelity
        fidA_noisy = compute_noisy_state_fidelity(t_A, exact_state, noise_A, NUM_QUBITS)
        fidB_noisy = compute_noisy_state_fidelity(t_B, exact_state, noise_B, NUM_QUBITS)

        records_A.append({
            'trotter_steps': n,
            'dt': TIME / n,
            'depth': t_A.depth(),
            'two_qubit_ops': r_A['transpiled_2q_gates'],
            'swap_count': r_A['exact_swaps'],
            'ideal_fidelity': fid_ideal,
            'noisy_fidelity': fidA_noisy,
            'trotter_error': trot_err,
            'trotter_energy': trot_E,
            'measured_energy': mean_eA,
            'measured_energy_std': std_eA,
            'mitigated_energy': mean_eA_mit,
            'mitigated_energy_std': std_eA_mit,
            'standard_error': se_A,
            'noise_error': abs(mean_eA - trot_E) / denom,
            'total_error': abs(mean_eA - exact_energy) / denom,
            'mitigated_error': abs(mean_eA_mit - exact_energy) / denom
        })

        records_B.append({
            'trotter_steps': n,
            'dt': TIME / n,
            'depth': t_B.depth(),
            'two_qubit_ops': r_B['transpiled_2q_gates'],
            'swap_count': r_B['exact_swaps'],
            'ideal_fidelity': fid_ideal,
            'noisy_fidelity': fidB_noisy,
            'trotter_error': trot_err,
            'trotter_energy': trot_E,
            'measured_energy': mean_eB,
            'measured_energy_std': std_eB,
            'mitigated_energy': mean_eB_mit,
            'mitigated_energy_std': std_eB_mit,
            'standard_error': se_B,
            'noise_error': abs(mean_eB - trot_E) / denom,
            'total_error': abs(mean_eB - exact_energy) / denom,
            'mitigated_error': abs(mean_eB_mit - exact_energy) / denom
        })

    df_A = pd.DataFrame(records_A)
    df_B = pd.DataFrame(records_B)
    df_A.to_csv(tab_dir / 'trotter_sweep_processor_A.csv', index=False)
    df_B.to_csv(tab_dir / 'trotter_sweep_processor_B.csv', index=False)
    print("      Saved trotter_sweep_processor_A.csv and trotter_sweep_processor_B.csv")

    # 4. Optimal Trotter Step Analysis
    opt_A = find_optimal_trotter_step(records_A)
    opt_B = find_optimal_trotter_step(records_B)
    print(f"[4/6] Optimal Trotter Step Sweet Spot:")
    print(f"      Processor A (T-Shape):   n* = {opt_A['optimal_n']}, Min Error = {opt_A['min_total_error']*100:.2f}%")
    print(f"      Processor B (Heavy-Hex): n* = {opt_B['optimal_n']}, Min Error = {opt_B['min_total_error']*100:.2f}%")

    opt_df = pd.DataFrame([
        {
            'Processor': 'Processor A',
            'Topology': 'T-Shaped Tree (5Q)',
            'Optimal_n': opt_A['optimal_n'],
            'Min_Total_Error': opt_A['min_total_error'],
            'Depth_at_Opt': opt_A['optimal_depth'],
            '2Q_Gates_at_Opt': opt_A['optimal_2q_gates'],
            'SWAPs_at_Opt': opt_A['optimal_swaps'],
            'Trotter_Error_at_Opt': opt_A['trotter_error_at_opt'],
            'Noise_Error_at_Opt': opt_A['noise_error_at_opt']
        },
        {
            'Processor': 'Processor B',
            'Topology': 'Heavy-Hex Tree (7Q)',
            'Optimal_n': opt_B['optimal_n'],
            'Min_Total_Error': opt_B['min_total_error'],
            'Depth_at_Opt': opt_B['optimal_depth'],
            '2Q_Gates_at_Opt': opt_B['optimal_2q_gates'],
            'SWAPs_at_Opt': opt_B['optimal_swaps'],
            'Trotter_Error_at_Opt': opt_B['trotter_error_at_opt'],
            'Noise_Error_at_Opt': opt_B['noise_error_at_opt']
        }
    ])
    opt_df.to_csv(tab_dir / 'optimal_trotter_steps.csv', index=False)

    # 5. Core Visualizations
    plot_trotter_error_tradeoff(
        STEPS_RANGE,
        trotter_errors,
        df_A['noise_error'].tolist(),
        df_B['noise_error'].tolist(),
        df_A['total_error'].tolist(),
        df_B['total_error'].tolist(),
        opt_A['optimal_n'],
        opt_B['optimal_n'],
        save_path=fig_dir / 'trotter_error_tradeoff.png'
    )

    plot_circuit_scaling(
        STEPS_RANGE,
        df_A['depth'].tolist(),
        df_B['depth'].tolist(),
        df_A['two_qubit_ops'].tolist(),
        df_B['two_qubit_ops'].tolist(),
        save_path=fig_dir / 'circuit_scaling_comparison.png'
    )

    plot_error_mitigation_impact(
        STEPS_RANGE,
        df_A['total_error'].tolist(),
        df_A['mitigated_error'].tolist(),
        df_B['total_error'].tolist(),
        df_B['mitigated_error'].tolist(),
        save_path=fig_dir / 'error_mitigation_study.png'
    )

    # 6. Benchmark Comparison Table at n=5
    ref_idx = STEPS_RANGE.index(5)
    row_A5 = records_A[ref_idx]
    row_B5 = records_B[ref_idx]

    comp_summary = pd.DataFrame({
        'Metric': [
            'Physical Qubits',
            'Processor Topology',
            'Circuit Depth (n=5)',
            'Two-Qubit (CX) Gates (n=5)',
            'Routing SWAP Gates (n=5)',
            'Measured Energy (Raw, 5 seeds)',
            'Measured Energy (Mitigated, 5 seeds)',
            'Total Error (Raw, n=5)',
            'Total Error (Mitigated, n=5)',
            'Noisy State Fidelity (n=5)',
            'Optimal Trotter Step (n*)',
            'Minimum Achievable Error at n*'
        ],
        'Processor A': [
            proc_A['num_qubits'],
            proc_A['topology'],
            row_A5['depth'],
            row_A5['two_qubit_ops'],
            row_A5['swap_count'],
            f"{row_A5['measured_energy']:.4f} +/- {row_A5['measured_energy_std']:.4f}",
            f"{row_A5['mitigated_energy']:.4f} +/- {row_A5['mitigated_energy_std']:.4f}",
            f"{row_A5['total_error']*100:.2f}%",
            f"{row_A5['mitigated_error']*100:.2f}%",
            f"{row_A5['noisy_fidelity']*100:.2f}%",
            opt_A['optimal_n'],
            f"{opt_A['min_total_error']*100:.2f}%"
        ],
        'Processor B': [
            proc_B['num_qubits'],
            proc_B['topology'],
            row_B5['depth'],
            row_B5['two_qubit_ops'],
            row_B5['swap_count'],
            f"{row_B5['measured_energy']:.4f} +/- {row_B5['measured_energy_std']:.4f}",
            f"{row_B5['mitigated_energy']:.4f} +/- {row_B5['mitigated_energy_std']:.4f}",
            f"{row_B5['total_error']*100:.2f}%",
            f"{row_B5['mitigated_error']*100:.2f}%",
            f"{row_B5['noisy_fidelity']*100:.2f}%",
            opt_B['optimal_n'],
            f"{opt_B['min_total_error']*100:.2f}%"
        ],
        'Difference (B - A)': [
            proc_B['num_qubits'] - proc_A['num_qubits'],
            "Heavy-Hex vs T-Shape",
            row_B5['depth'] - row_A5['depth'],
            row_B5['two_qubit_ops'] - row_A5['two_qubit_ops'],
            row_B5['swap_count'] - row_A5['swap_count'],
            f"{(row_B5['measured_energy'] - row_A5['measured_energy']):.4f}",
            f"{(row_B5['mitigated_energy'] - row_A5['mitigated_energy']):.4f}",
            f"{(row_B5['total_error'] - row_A5['total_error'])*100:.2f}%",
            f"{(row_B5['mitigated_error'] - row_A5['mitigated_error'])*100:.2f}%",
            f"{(row_B5['noisy_fidelity'] - row_A5['noisy_fidelity'])*100:.2f}%",
            opt_B['optimal_n'] - opt_A['optimal_n'],
            f"{(opt_B['min_total_error'] - opt_A['min_total_error'])*100:.2f}%"
        ]
    })
    comp_summary.to_csv(tab_dir / 'AB_comparison_summary.csv', index=False)
    print("      Saved AB_comparison_summary.csv with multi-seed error bars and mitigation")

    # 7. 5-Qubit Chain Study (Architectural Divergence Demonstration with Physical Simulation)
    print("[5/6] Running 5-Qubit Open Chain scaling benchmark with full physical noisy simulation...")
    sim5 = create_ising_chain(num_qubits=5, J=1.0, h=0.5, periodic=False)
    exact_sv5 = sim5.exact_evolved_state(None, TIME)
    exact_E5 = sim5.exact_energy_expectation(exact_sv5)

    n5_steps = 3
    qc5_bare = sim5.build_trotter_circuit(TIME, n5_steps, order=1, basis=None)
    qc5_z = sim5.build_trotter_circuit(TIME, n5_steps, order=1, basis='Z')
    qc5_x = sim5.build_trotter_circuit(TIME, n5_steps, order=1, basis='X')

    tA5_bare, rA5 = transpile_with_routing_analysis(qc5_bare, proc_A, optimization_level=1, seed_transpiler=42)
    tB5_bare, rB5 = transpile_with_routing_analysis(qc5_bare, proc_B, optimization_level=1, seed_transpiler=42)

    tA5_z = transpile_circuit(qc5_z, proc_A, optimization_level=1, seed_transpiler=42)
    tA5_x = transpile_circuit(qc5_x, proc_A, optimization_level=1, seed_transpiler=42)
    tB5_z = transpile_circuit(qc5_z, proc_B, optimization_level=1, seed_transpiler=42)
    tB5_x = transpile_circuit(qc5_x, proc_B, optimization_level=1, seed_transpiler=42)

    # Multi-seed noisy simulation across 5 seeds (seeds 42..46) to provide rigorous error bars
    eA5_list, eA5_mit_list = [], []
    eB5_list, eB5_mit_list = [], []

    for s in SEEDS:
        res_A5_z = aer_noisy_A.run(tA5_z, shots=SHOTS, seed_simulator=s).result().get_counts()
        res_A5_x = aer_noisy_A.run(tA5_x, shots=SHOTS, seed_simulator=s).result().get_counts()
        eA5_val, _ = sim5.compute_energy_from_counts(res_A5_z, res_A5_x, SHOTS)
        eA5_list.append(eA5_val)

        mit_A5_z = mitigate_readout_counts(res_A5_z, 0.02, 5)
        mit_A5_x = mitigate_readout_counts(res_A5_x, 0.02, 5)
        pzA5 = {k: int(round(v * SHOTS)) for k, v in mit_A5_z.items()}
        pxA5 = {k: int(round(v * SHOTS)) for k, v in mit_A5_x.items()}
        eA5_mit_val, _ = sim5.compute_energy_from_counts(pzA5, pxA5, SHOTS)
        eA5_mit_list.append(eA5_mit_val)

        res_B5_z = aer_noisy_B.run(tB5_z, shots=SHOTS, seed_simulator=s).result().get_counts()
        res_B5_x = aer_noisy_B.run(tB5_x, shots=SHOTS, seed_simulator=s).result().get_counts()
        eB5_val, _ = sim5.compute_energy_from_counts(res_B5_z, res_B5_x, SHOTS)
        eB5_list.append(eB5_val)

        mit_B5_z = mitigate_readout_counts(res_B5_z, 0.02, 5)
        mit_B5_x = mitigate_readout_counts(res_B5_x, 0.02, 5)
        pzB5 = {k: int(round(v * SHOTS)) for k, v in mit_B5_z.items()}
        pxB5 = {k: int(round(v * SHOTS)) for k, v in mit_B5_x.items()}
        eB5_mit_val, _ = sim5.compute_energy_from_counts(pzB5, pxB5, SHOTS)
        eB5_mit_list.append(eB5_mit_val)

    mean_eA = float(np.mean(eA5_list))
    std_eA = float(np.std(eA5_list))
    mean_eA_mit = float(np.mean(eA5_mit_list))
    std_eA_mit = float(np.std(eA5_mit_list))

    mean_eB = float(np.mean(eB5_list))
    std_eB = float(np.std(eB5_list))
    mean_eB_mit = float(np.mean(eB5_mit_list))
    std_eB_mit = float(np.std(eB5_mit_list))

    denom5 = abs(exact_E5)
    errA5 = abs(mean_eA - exact_E5) / denom5
    errA5_std = std_eA / denom5
    errA5_mit = abs(mean_eA_mit - exact_E5) / denom5
    errA5_mit_std = std_eA_mit / denom5

    errB5 = abs(mean_eB - exact_E5) / denom5
    errB5_std = std_eB / denom5
    errB5_mit = abs(mean_eB_mit - exact_E5) / denom5
    errB5_mit_std = std_eB_mit / denom5

    # Noisy state fidelities via density matrix
    fid_A5 = compute_noisy_state_fidelity(tA5_bare, exact_sv5, noise_A, 5)
    fid_B5 = compute_noisy_state_fidelity(tB5_bare, exact_sv5, noise_B, 5)

    df_5q = pd.DataFrame([
        {
            'Architecture': 'Processor A (5Q T-Tree)',
            'Transpiled_Depth': rA5['transpiled_depth'],
            '2Q_Gates': rA5['transpiled_2q_gates'],
            'SWAP_Gates': rA5['exact_swaps'],
            'Extra_Routing_CX': rA5['extra_2q_gates'],
            'Measured_Energy': f"{mean_eA:.4f} +/- {std_eA:.4f}",
            'Mitigated_Energy': f"{mean_eA_mit:.4f} +/- {std_eA_mit:.4f}",
            'Raw_Total_Error': f"{errA5*100:.2f}% +/- {errA5_std*100:.2f}%",
            'Mitigated_Total_Error': f"{errA5_mit*100:.2f}% +/- {errA5_mit_std*100:.2f}%",
            'Noisy_State_Fidelity': f"{fid_A5*100:.2f}%"
        },
        {
            'Architecture': 'Processor B (7Q Heavy-Hex Tree)',
            'Transpiled_Depth': rB5['transpiled_depth'],
            '2Q_Gates': rB5['transpiled_2q_gates'],
            'SWAP_Gates': rB5['exact_swaps'],
            'Extra_Routing_CX': rB5['extra_2q_gates'],
            'Measured_Energy': f"{mean_eB:.4f} +/- {std_eB:.4f}",
            'Mitigated_Energy': f"{mean_eB_mit:.4f} +/- {std_eB_mit:.4f}",
            'Raw_Total_Error': f"{errB5*100:.2f}% +/- {errB5_std*100:.2f}%",
            'Mitigated_Total_Error': f"{errB5_mit*100:.2f}% +/- {errB5_mit_std*100:.2f}%",
            'Noisy_State_Fidelity': f"{fid_B5*100:.2f}%"
        },
        {
            'Architecture': 'Advantage (B vs A)',
            'Transpiled_Depth': f"{rB5['transpiled_depth'] - rA5['transpiled_depth']} ({(rB5['transpiled_depth'] - rA5['transpiled_depth'])/rA5['transpiled_depth']*100:.1f}%)",
            '2Q_Gates': f"{rB5['transpiled_2q_gates'] - rA5['transpiled_2q_gates']} ({(rB5['transpiled_2q_gates'] - rA5['transpiled_2q_gates'])/rA5['transpiled_2q_gates']*100:.1f}%)",
            'SWAP_Gates': f"-{rA5['exact_swaps']}",
            'Extra_Routing_CX': f"-{rA5['extra_2q_gates']}",
            'Measured_Energy': f"{mean_eB - mean_eA:.4f}",
            'Mitigated_Energy': f"{mean_eB_mit - mean_eA_mit:.4f}",
            'Raw_Total_Error': f"{(errB5 - errA5)*100:.2f}%",
            'Mitigated_Total_Error': f"{(errB5_mit - errA5_mit)*100:.2f}%",
            'Noisy_State_Fidelity': f"+{(fid_B5 - fid_A5)*100:.2f}%"
        }
    ])
    df_5q.to_csv(tab_dir / '5qubit_chain_comparison.csv', index=False)
    print(f"      5-Qubit Chain: Proc A required {rA5['exact_swaps']} SWAPs (+{rA5['extra_2q_gates']} CX) | Proc B required {rB5['exact_swaps']} SWAPs")
    print(f"      5-Qubit Fidelity: Proc A = {fid_A5*100:.2f}% vs Proc B = {fid_B5*100:.2f}% (+{(fid_B5-fid_A5)*100:.2f}%)")
    print(f"      5-Qubit Energy (5 seeds): Proc A = {mean_eA:.4f} +/- {std_eA:.4f} | Proc B = {mean_eB:.4f} +/- {std_eB:.4f}")

    print("[6/6] Pipeline completed successfully!")


if __name__ == "__main__":
    main()
