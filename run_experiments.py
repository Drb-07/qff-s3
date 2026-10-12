"""
End-to-End Experiment Execution Script
Qiskit Fall Fest 2026 - Phase 1 Hackathon: Problem S3 (Noise in Trotter Simulation)

Executes the complete experimental pipeline:
1. Classical exact baseline diagonalization
2. Multi-step Trotter sweeps (n = 1..10) across ideal, Processor A, and Processor B
3. Routing analysis, 2Q gate overhead, and SWAP decomposition
4. Optimal Trotter step (n*) identification for each hardware architecture
5. Error mitigation study (Section 16)
6. Export of all tables to results/tables/ and figures to results/figures/
"""

import os
import sys
import json
import math
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

from src.problem import TrotterSimulator, create_ising_chain
from src.processors import load_processor
from src.routing import transpile_circuit, analyze_routing_overhead
from src.metrics import (
    circuit_metrics,
    proportion_standard_error,
    mitigate_readout_counts,
    find_optimal_trotter_step
)
from src.visualization import (
    plot_processor_graph,
    plot_trotter_error_tradeoff,
    plot_circuit_scaling,
    plot_architecture_tradeoffs,
    plot_error_mitigation_impact
)


def create_noise_model(processor_config: dict) -> NoiseModel:
    """Construct Aer NoiseModel from processor JSON specification."""
    noise = NoiseModel()
    noise_cfg = processor_config.get('noise', {})

    single_err = noise_cfg.get('single_qubit_error', 0.001)
    two_err = noise_cfg.get('two_qubit_error', 0.01)
    readout_err = noise_cfg.get('readout_error', 0.02)

    # 1Q gate depolarizing errors
    noise.add_all_qubit_quantum_error(
        depolarizing_error(single_err, 1),
        ['x', 'sx', 'rz']
    )
    # 2Q gate depolarizing errors
    noise.add_all_qubit_quantum_error(
        depolarizing_error(two_err, 2),
        ['cx']
    )
    # Readout errors
    noise.add_all_qubit_readout_error(
        ReadoutError([
            [1.0 - readout_err, readout_err],
            [readout_err, 1.0 - readout_err]
        ])
    )
    return noise


def main():
    print("=" * 70)
    print("STARTING QFF 2026 EXPERIMENT PIPELINE: PROBLEM S3")
    print("=" * 70)

    # Directories
    fig_dir = PROJECT_ROOT / 'results' / 'figures'
    tab_dir = PROJECT_ROOT / 'results' / 'tables'
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Processors
    proc_A = load_processor(PROJECT_ROOT / 'processors' / 'processor_A.json')
    proc_B = load_processor(PROJECT_ROOT / 'processors' / 'processor_B.json')
    print(f"[1/6] Processors loaded: {proc_A['name']} (5Q) and {proc_B['name']} (7Q)")

    # Generate processor topology plots
    plot_processor_graph(proc_A, title="Processor A (5 Qubits, Linear Topology)",
                         save_path=fig_dir / 'processor_A_graph.png')
    plot_processor_graph(proc_B, title="Processor B (7 Qubits, Heavy-Hex-Inspired)",
                         save_path=fig_dir / 'processor_B_graph.png')
    print("      Saved processor graphs to results/figures/")

    # 2. Physics & Problem Setup
    NUM_QUBITS = 4
    J = 1.0
    h = 0.5
    TIME = 1.0
    SHOTS = 2048
    STEPS_RANGE = [1, 2, 3, 4, 5, 6, 7, 8, 10]

    sim = create_ising_chain(num_qubits=NUM_QUBITS, J=J, h=h)

    # Exact Classical Baseline
    exact_state = sim.exact_evolved_state(None, TIME)
    exact_energy = sim.exact_energy_expectation(exact_state)
    print(f"[2/6] Exact Classical Baseline at T={TIME}:")
    print(f"      Exact Hamiltonian Energy E_exact = {exact_energy:.6f}")

    # Simulators
    aer_ideal = AerSimulator()
    noise_A = create_noise_model(proc_A)
    noise_B = create_noise_model(proc_B)
    aer_noisy_A = AerSimulator(noise_model=noise_A)
    aer_noisy_B = AerSimulator(noise_model=noise_B)

    # 3. Parametric Trotter Step Sweep
    print(f"[3/6] Running parametric Trotter step sweeps (n = {STEPS_RANGE})...")

    records_A = []
    records_B = []
    trotter_errors = []

    for n in STEPS_RANGE:
        # Algorithmic ideal state fidelity & energy
        fid_ideal = sim.ideal_state_fidelity(TIME, n, order=1)
        trotter_state = sim.ideal_statevector(TIME, n, order=1)
        trotter_energy = sim.exact_energy_expectation(trotter_state)
        trot_err = abs(trotter_energy - exact_energy) / abs(exact_energy)
        trotter_errors.append(trot_err)

        # Circuits for Pauli measurements
        qc_z = sim.build_trotter_circuit(TIME, n, order=1, basis='Z')
        qc_x = sim.build_trotter_circuit(TIME, n, order=1, basis='X')
        qc_bare = sim.build_trotter_circuit(TIME, n, order=1, basis=None)

        # Transpile for A
        t_A_bare = transpile_circuit(qc_bare, proc_A, optimization_level=1, seed_transpiler=42)
        t_A_z = transpile_circuit(qc_z, proc_A, optimization_level=1, seed_transpiler=42)
        t_A_x = transpile_circuit(qc_x, proc_A, optimization_level=1, seed_transpiler=42)
        routing_A = analyze_routing_overhead(qc_bare, t_A_bare)

        # Transpile for B
        t_B_bare = transpile_circuit(qc_bare, proc_B, optimization_level=1, seed_transpiler=42)
        t_B_z = transpile_circuit(qc_z, proc_B, optimization_level=1, seed_transpiler=42)
        t_B_x = transpile_circuit(qc_x, proc_B, optimization_level=1, seed_transpiler=42)
        routing_B = analyze_routing_overhead(qc_bare, t_B_bare)

        # Execute on Noisy Simulator A
        res_A_z = aer_noisy_A.run(t_A_z, shots=SHOTS, seed_simulator=42).result().get_counts()
        res_A_x = aer_noisy_A.run(t_A_x, shots=SHOTS, seed_simulator=42).result().get_counts()
        e_noisy_A, se_A = sim.compute_energy_from_counts(res_A_z, res_A_x, SHOTS)
        noise_err_A = abs(e_noisy_A - trotter_energy) / abs(exact_energy)
        total_err_A = abs(e_noisy_A - exact_energy) / abs(exact_energy)

        # Execute on Noisy Simulator B
        res_B_z = aer_noisy_B.run(t_B_z, shots=SHOTS, seed_simulator=42).result().get_counts()
        res_B_x = aer_noisy_B.run(t_B_x, shots=SHOTS, seed_simulator=42).result().get_counts()
        e_noisy_B, se_B = sim.compute_energy_from_counts(res_B_z, res_B_x, SHOTS)
        noise_err_B = abs(e_noisy_B - trotter_energy) / abs(exact_energy)
        total_err_B = abs(e_noisy_B - exact_energy) / abs(exact_energy)

        # Readout mitigation for B
        readout_err_B = proc_B['noise']['readout_error']
        mit_B_z = mitigate_readout_counts(res_B_z, readout_err_B, NUM_QUBITS)
        mit_B_x = mitigate_readout_counts(res_B_x, readout_err_B, NUM_QUBITS)
        # Scaled pseudo-counts for energy computation
        pseudo_z = {k: int(round(v * SHOTS)) for k, v in mit_B_z.items()}
        pseudo_x = {k: int(round(v * SHOTS)) for k, v in mit_B_x.items()}
        e_mit_B, _ = sim.compute_energy_from_counts(pseudo_z, pseudo_x, SHOTS)
        mit_err_B = abs(e_mit_B - exact_energy) / abs(exact_energy)

        records_A.append({
            'trotter_steps': n,
            'dt': TIME / n,
            'logical_depth': qc_bare.depth(),
            'depth': t_A_bare.depth(),
            'two_qubit_ops': routing_A['transpiled_2q_gates'],
            'swap_count': routing_A['effective_swaps'],
            'total_ops': routing_A['total_ops'],
            'ideal_fidelity': fid_ideal,
            'trotter_energy': trotter_energy,
            'trotter_error': trot_err,
            'measured_energy': e_noisy_A,
            'standard_error': se_A,
            'noise_error': noise_err_A,
            'total_error': total_err_A
        })

        records_B.append({
            'trotter_steps': n,
            'dt': TIME / n,
            'logical_depth': qc_bare.depth(),
            'depth': t_B_bare.depth(),
            'two_qubit_ops': routing_B['transpiled_2q_gates'],
            'swap_count': routing_B['effective_swaps'],
            'total_ops': routing_B['total_ops'],
            'ideal_fidelity': fid_ideal,
            'trotter_energy': trotter_energy,
            'trotter_error': trot_err,
            'measured_energy': e_noisy_B,
            'standard_error': se_B,
            'noise_error': noise_err_B,
            'total_error': total_err_B,
            'mitigated_energy': e_mit_B,
            'mitigated_error': mit_err_B
        })

    df_A = pd.DataFrame(records_A)
    df_B = pd.DataFrame(records_B)

    # Save sweep tables
    df_A.to_csv(tab_dir / 'trotter_sweep_processor_A.csv', index=False)
    df_B.to_csv(tab_dir / 'trotter_sweep_processor_B.csv', index=False)
    print("      Saved trotter_sweep_processor_A.csv and trotter_sweep_processor_B.csv")

    # 4. Find Optimal Trotter Steps n*
    opt_A = find_optimal_trotter_step(records_A)
    opt_B = find_optimal_trotter_step(records_B)
    print(f"[4/6] Optimal Trotter Step Analysis:")
    print(f"      Processor A (Linear):    n* = {opt_A['optimal_n']}, Min Total Error = {opt_A['min_total_error']:.4f}")
    print(f"      Processor B (Heavy-Hex): n* = {opt_B['optimal_n']}, Min Total Error = {opt_B['min_total_error']:.4f}")

    opt_df = pd.DataFrame([
        {
            'Processor': 'Processor A',
            'Topology': 'Linear (5Q)',
            'Optimal_n': opt_A['optimal_n'],
            'Min_Total_Error': opt_A['min_total_error'],
            'Depth_at_Opt': opt_A['optimal_depth'],
            '2Q_Gates_at_Opt': opt_A['optimal_2q_gates'],
            'Trotter_Error_at_Opt': opt_A['trotter_error_at_opt'],
            'Noise_Error_at_Opt': opt_A['noise_error_at_opt']
        },
        {
            'Processor': 'Processor B',
            'Topology': 'Heavy-Hex-Inspired (7Q)',
            'Optimal_n': opt_B['optimal_n'],
            'Min_Total_Error': opt_B['min_total_error'],
            'Depth_at_Opt': opt_B['optimal_depth'],
            '2Q_Gates_at_Opt': opt_B['optimal_2q_gates'],
            'Trotter_Error_at_Opt': opt_B['trotter_error_at_opt'],
            'Noise_Error_at_Opt': opt_B['noise_error_at_opt']
        }
    ])
    opt_df.to_csv(tab_dir / 'optimal_trotter_steps.csv', index=False)

    # 5. Summary A/B Comparison at standard reference point n=5
    ref_idx = STEPS_RANGE.index(5)
    row_A5 = records_A[ref_idx]
    row_B5 = records_B[ref_idx]

    ab_summary = pd.DataFrame({
        'Metric': [
            'Physical Qubits',
            'Circuit Depth (n=5)',
            'Two-Qubit (CX) Gates (n=5)',
            'Routing SWAP Gates (n=5)',
            'Total Operations (n=5)',
            'Hamiltonian Energy (n=5)',
            'Energy Standard Error (sigma)',
            'Energy Relative Error (n=5)',
            'Optimal Trotter Step (n*)',
            'Minimum Achievable Error at n*'
        ],
        'Processor A': [
            proc_A['num_qubits'],
            row_A5['depth'],
            row_A5['two_qubit_ops'],
            row_A5['swap_count'],
            row_A5['total_ops'],
            f"{row_A5['measured_energy']:.4f}",
            f"{row_A5['standard_error']:.4f}",
            f"{row_A5['total_error']:.4f}",
            opt_A['optimal_n'],
            f"{opt_A['min_total_error']:.4f}"
        ],
        'Processor B': [
            proc_B['num_qubits'],
            row_B5['depth'],
            row_B5['two_qubit_ops'],
            row_B5['swap_count'],
            row_B5['total_ops'],
            f"{row_B5['measured_energy']:.4f}",
            f"{row_B5['standard_error']:.4f}",
            f"{row_B5['total_error']:.4f}",
            opt_B['optimal_n'],
            f"{opt_B['min_total_error']:.4f}"
        ],
        'Difference (B - A)': [
            proc_B['num_qubits'] - proc_A['num_qubits'],
            row_B5['depth'] - row_A5['depth'],
            row_B5['two_qubit_ops'] - row_A5['two_qubit_ops'],
            row_B5['swap_count'] - row_A5['swap_count'],
            row_B5['total_ops'] - row_A5['total_ops'],
            f"{row_B5['measured_energy'] - row_A5['measured_energy']:.4f}",
            f"{row_B5['standard_error'] - row_A5['standard_error']:.4f}",
            f"{row_B5['total_error'] - row_A5['total_error']:.4f}",
            opt_B['optimal_n'] - opt_A['optimal_n'],
            f"{opt_B['min_total_error'] - opt_A['min_total_error']:.4f}"
        ]
    })
    ab_summary.to_csv(tab_dir / 'AB_comparison_summary.csv', index=False)
    print("      Saved AB_comparison_summary.csv")

    # Save comprehensive final_results.json
    final_json = {
        'problem_statement': {
            'code': 'S3',
            'name': 'Noise in Trotter Simulation',
            'track': 'Quantum Simulation'
        },
        'parameters': {
            'num_qubits': NUM_QUBITS,
            'J': J,
            'h': h,
            'evolution_time': TIME,
            'shots': SHOTS,
            'exact_energy': exact_energy
        },
        'optimal_trotter_steps': {
            'processor_A': opt_A,
            'processor_B': opt_B
        },
        'processor_A_summary': row_A5,
        'processor_B_summary': row_B5
    }
    with open(tab_dir / 'final_comparison.json', 'w') as f:
        json.dump(final_json, f, indent=2)

    # 6. Generate All Required Figures
    print("[5/6] Generating scientific figures...")

    # The Money Plot: U-curve showing Trotter error vs Noise error
    plot_trotter_error_tradeoff(
        steps=STEPS_RANGE,
        trotter_errors=trotter_errors,
        noisy_errors_A=df_A['noise_error'].tolist(),
        noisy_errors_B=df_B['noise_error'].tolist(),
        total_errors_A=df_A['total_error'].tolist(),
        total_errors_B=df_B['total_error'].tolist(),
        opt_n_A=opt_A['optimal_n'],
        opt_n_B=opt_B['optimal_n'],
        save_path=fig_dir / 'trotter_error_tradeoff.png'
    )

    # Circuit scaling
    plot_circuit_scaling(
        steps=STEPS_RANGE,
        depths_A=df_A['depth'].tolist(),
        depths_B=df_B['depth'].tolist(),
        gates_2q_A=df_A['two_qubit_ops'].tolist(),
        gates_2q_B=df_B['two_qubit_ops'].tolist(),
        save_path=fig_dir / 'circuit_scaling_comparison.png'
    )

    # Architecture tradeoffs
    summary_plot_data = {
        'processor_A': {
            'depth': row_A5['depth'],
            'two_qubit_ops': row_A5['two_qubit_ops'],
            'swap_count': row_A5['swap_count'],
            'energy_error': row_A5['total_error']
        },
        'processor_B': {
            'depth': row_B5['depth'],
            'two_qubit_ops': row_B5['two_qubit_ops'],
            'swap_count': row_B5['swap_count'],
            'energy_error': row_B5['total_error']
        }
    }
    plot_architecture_tradeoffs(summary_plot_data, save_path=fig_dir / 'architecture_tradeoffs.png')

    # Circuit comparison bar chart (for Section 12)
    categories = ['Depth', '2Q Gates', 'Total Ops']
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    x = np.arange(len(categories))
    w = 0.35
    ax.bar(x - w/2, [row_A5['depth'], row_A5['two_qubit_ops'], row_A5['total_ops']], w, label='Processor A (Linear)', color='#D32F2F')
    ax.bar(x + w/2, [row_B5['depth'], row_B5['two_qubit_ops'], row_B5['total_ops']], w, label='Processor B (Heavy-Hex)', color='#1976D2')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title('Circuit Gate Overhead Comparison at n=5', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    fig.savefig(fig_dir / 'circuit_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Error mitigation study plot (Section 16)
    plot_error_mitigation_impact(
        steps=STEPS_RANGE,
        raw_errors_B=df_B['total_error'].tolist(),
        mitigated_errors_B=df_B['mitigated_error'].tolist(),
        save_path=fig_dir / 'error_mitigation_study.png'
    )

    print("[6/6] EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"      All tables saved to: {tab_dir}")
    print(f"      All figures saved to: {fig_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
