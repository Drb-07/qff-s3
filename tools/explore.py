"""
explore.py — Optional Command-Line Interface (CLI) Parameter Explorer
Qiskit Fall Fest 2026 Phase 1: Problem S3 (Noise in Trotter Simulation)

ROLE & PURPOSE:
This script provides an optional command-line exploration interface for judges
who prefer terminal evaluation with pre-configured physical parameter presets.
The primary project deliverables remain:
  1. main.ipynb (End-to-end reproducible notebook with ipywidgets GUI)
  2. report.md & README.md (Comprehensive scientific report and documentation)
  3. interactive_app.py (Streamlit real-time quantum dashboard)
  4. interactive_dashboard.html (Standalone browser-based exploration tool)
"""

import argparse
import sys
import os
import math
import json
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

from src.problem import create_ising_chain
from src.processors import load_processor
from src.routing import transpile_circuit, analyze_routing_overhead
from src.metrics import find_optimal_trotter_step, predict_analytical_optimal_n
from src.visualization import plot_trotter_error_tradeoff

# Curated Presets for Judges
PRESETS = {
    'default': {
        'name': 'Standard TFIM Dynamics (Baseline Benchmark)',
        'num_qubits': 4,
        'J': 1.0,
        'h': 0.5,
        'time': 1.0,
        'order': 1,
        'periodic': False,
        'description': 'Moderate ferromagnetic coupling; standard benchmark instance from starter guide.'
    },
    'critical': {
        'name': 'Quantum Critical Point (h = J)',
        'num_qubits': 4,
        'J': 1.0,
        'h': 1.0,
        'time': 1.0,
        'order': 1,
        'periodic': False,
        'description': 'Phase transition regime with maximal quantum fluctuations and operator non-commutativity.'
    },
    'strong_field': {
        'name': 'Strong-Field Disordered Regime (h > J)',
        'num_qubits': 4,
        'J': 0.5,
        'h': 1.5,
        'time': 1.0,
        'order': 1,
        'periodic': False,
        'description': 'Single-qubit X rotation dynamics dominate over two-qubit ZZ interactions.'
    },
    'long_time': {
        'name': 'Long-Time Evolution Challenge (T = 2.0s)',
        'num_qubits': 4,
        'J': 1.0,
        'h': 0.5,
        'time': 2.0,
        'order': 1,
        'periodic': False,
        'description': 'Tests deep circuit accumulation; Trotter error is quadrupled, pushing n* to deeper circuits.'
    },
    'compact_3q': {
        'name': '3-Qubit Fast System (N = 3)',
        'num_qubits': 3,
        'J': 1.0,
        'h': 0.8,
        'time': 1.2,
        'order': 1,
        'periodic': False,
        'description': 'Ultra-fast 3-spin chain suitable for instantaneous verification.'
    },
    'periodic_ring': {
        'name': 'Periodic Boundary Ring (Topology Benchmark)',
        'num_qubits': 4,
        'J': 1.0,
        'h': 0.5,
        'time': 1.0,
        'order': 1,
        'periodic': True,
        'description': 'Closed 4-spin ring showcasing routing SWAP overhead on tree topologies.'
    },
    'second_order': {
        'name': '2nd-Order Trotter-Suzuki Strang Splitting',
        'num_qubits': 4,
        'J': 1.0,
        'h': 0.5,
        'time': 1.0,
        'order': 2,
        'periodic': False,
        'description': 'Symmetric 2nd-order Trotter with O(dt^2) error convergence and double 2Q gate density.'
    }
}


def build_noise(proc_cfg, override_2q_err=None):
    """Build Aer NoiseModel from processor configuration with optional 2Q override."""
    noise = NoiseModel()
    cfg = proc_cfg.get('noise', {})
    two_err = override_2q_err if override_2q_err is not None else cfg.get('two_qubit_error', 0.01)

    noise.add_all_qubit_quantum_error(
        depolarizing_error(cfg.get('single_qubit_error', 0.001), 1),
        ['x', 'sx', 'rz']
    )
    noise.add_all_qubit_quantum_error(
        depolarizing_error(two_err, 2),
        ['cx']
    )
    r_err = cfg.get('readout_error', 0.02)
    noise.add_all_qubit_readout_error(
        ReadoutError([[1.0 - r_err, r_err], [r_err, 1.0 - r_err]])
    )
    return noise, two_err


def run_benchmark(num_qubits=4, J=1.0, h=0.5, time=1.0, shots=2048,
                  order=1, periodic=False, two_q_error=None,
                  output_file=None, verbose=False, steps_range=None):
    """Run full evaluation on given parameters and find n*."""
    if steps_range is None:
        steps_range = [1, 2, 3, 4, 5, 6, 8, 10]

    proc_A = load_processor(PROJECT_ROOT / 'processors' / 'processor_A.json')
    proc_B = load_processor(PROJECT_ROOT / 'processors' / 'processor_B.json')

    sim = create_ising_chain(num_qubits=num_qubits, J=J, h=h, periodic=periodic)

    # 1. Exact baseline
    exact_state = sim.exact_evolved_state(None, time)
    exact_energy = sim.exact_energy_expectation(exact_state)

    print("\n" + "=" * 75)
    boundary_str = "Periodic (PBC - Closed Ring)" if periodic else "Open (OBC - Linear Chain)"
    print(f"BENCHMARK RUN: N={num_qubits} Qubits | J={J} | h={h} | T={time}s | Order={order}")
    print(f"Boundary: {boundary_str}")
    print(f"Exact Continuous Ground Truth Energy: E_exact = {exact_energy:.6f}")
    print("=" * 75)

    # Simulators
    noise_A, actual_2q_A = build_noise(proc_A, two_q_error)
    noise_B, actual_2q_B = build_noise(proc_B, two_q_error)
    sim_A = AerSimulator(noise_model=noise_A)
    sim_B = AerSimulator(noise_model=noise_B)

    records_A, records_B = [], []
    trotter_errs = []

    for n in steps_range:
        # Ideal math
        fid = sim.ideal_state_fidelity(time, n, order=order)
        ideal_sv = sim.ideal_statevector(time, n, order=order)
        trot_E = sim.exact_energy_expectation(ideal_sv)
        t_err = abs(trot_E - exact_energy) / max(1e-9, abs(exact_energy))
        trotter_errs.append(t_err)

        # Circuits
        qc_z = sim.build_trotter_circuit(time, n, order=order, basis='Z')
        qc_x = sim.build_trotter_circuit(time, n, order=order, basis='X')
        qc_b = sim.build_trotter_circuit(time, n, order=order, basis=None)

        # Proc A transpilation & routing
        tA = transpile_circuit(qc_b, proc_A, optimization_level=1, seed_transpiler=42)
        rA = analyze_routing_overhead(qc_b, tA)
        tA_z = transpile_circuit(qc_z, proc_A, optimization_level=1, seed_transpiler=42)
        tA_x = transpile_circuit(qc_x, proc_A, optimization_level=1, seed_transpiler=42)
        rA_z = sim_A.run(tA_z, shots=shots, seed_simulator=42).result().get_counts()
        rA_x = sim_A.run(tA_x, shots=shots, seed_simulator=42).result().get_counts()
        eA, seA = sim.compute_energy_from_counts(rA_z, rA_x, shots)

        # Proc B transpilation & routing
        tB = transpile_circuit(qc_b, proc_B, optimization_level=1, seed_transpiler=42)
        rB = analyze_routing_overhead(qc_b, tB)
        tB_z = transpile_circuit(qc_z, proc_B, optimization_level=1, seed_transpiler=42)
        tB_x = transpile_circuit(qc_x, proc_B, optimization_level=1, seed_transpiler=42)
        rB_z = sim_B.run(tB_z, shots=shots, seed_simulator=42).result().get_counts()
        rB_x = sim_B.run(tB_x, shots=shots, seed_simulator=42).result().get_counts()
        eB, seB = sim.compute_energy_from_counts(rB_z, rB_x, shots)

        records_A.append({
            'trotter_steps': n, 'depth': tA.depth(),
            'two_qubit_ops': rA['transpiled_2q_gates'],
            'swap_count': rA['effective_swaps'],
            'ideal_fidelity': fid,
            'trotter_error': t_err,
            'measured_energy': eA,
            'total_error': abs(eA - exact_energy) / max(1e-9, abs(exact_energy)),
            'noise_error': abs(eA - trot_E) / max(1e-9, abs(exact_energy))
        })
        records_B.append({
            'trotter_steps': n, 'depth': tB.depth(),
            'two_qubit_ops': rB['transpiled_2q_gates'],
            'swap_count': rB['effective_swaps'],
            'ideal_fidelity': fid,
            'trotter_error': t_err,
            'measured_energy': eB,
            'total_error': abs(eB - exact_energy) / max(1e-9, abs(exact_energy)),
            'noise_error': abs(eB - trot_E) / max(1e-9, abs(exact_energy))
        })

    opt_A = find_optimal_trotter_step(records_A)
    opt_B = find_optimal_trotter_step(records_B)

    # Analytical prediction
    ana_A = predict_analytical_optimal_n(J, h, time, num_qubits, actual_2q_A, order, periodic, 'processor_A')
    ana_B = predict_analytical_optimal_n(J, h, time, num_qubits, actual_2q_B, order, periodic, 'processor_B')

    # Summary table
    summary_df = pd.DataFrame({
        'Steps (n)': steps_range,
        'Trotter Err': [f"{r['trotter_error']*100:.2f}%" for r in records_A],
        'A Depth': [r['depth'] for r in records_A],
        'A 2Q (SWAPs)': [f"{r['two_qubit_ops']} ({r['swap_count']})" for r in records_A],
        'A Tot Err': [f"{r['total_error']*100:.2f}%" for r in records_A],
        'B Depth': [r['depth'] for r in records_B],
        'B 2Q (SWAPs)': [f"{r['two_qubit_ops']} ({r['swap_count']})" for r in records_B],
        'B Tot Err': [f"{r['total_error']*100:.2f}%" for r in records_B]
    })
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 75)
    print("[OPTIMAL SWEET SPOT (n*)]:")
    print(f"   Processor A (T-Tree):    Empirical n* = {opt_A['optimal_n']} | Min Error: {opt_A['min_total_error']*100:.2f}% | Depth: {opt_A['optimal_depth']} | 2Q Gates: {opt_A['optimal_2q_gates']}")
    print(f"                            Analytical n* ~ {ana_A['analytical_n_continuous']:.2f} (rounded: {ana_A['analytical_n_optimal']})")
    print(f"   Processor B (Heavy-Hex): Empirical n* = {opt_B['optimal_n']} | Min Error: {opt_B['min_total_error']*100:.2f}% | Depth: {opt_B['optimal_depth']} | 2Q Gates: {opt_B['optimal_2q_gates']}")
    print(f"                            Analytical n* ~ {ana_B['analytical_n_continuous']:.2f} (rounded: {ana_B['analytical_n_optimal']})")
    print("=" * 75)

    if verbose:
        print("\n[VERBOSE ROUTING & NOISE DETAILS]:")
        print(f"   Processor A 2Q Error Rate: {actual_2q_A*100:.2f}% | Per-step 2Q CX count: {ana_A['cx_per_step']}")
        print(f"   Processor B 2Q Error Rate: {actual_2q_B*100:.2f}% | Per-step 2Q CX count: {ana_B['cx_per_step']}")
        if periodic:
            print("   -> PBC OBSERVATION: Closing the boundary ring on tree topologies requires routing SWAP gates.")
            print("      Processor A routes through its central hub, while Processor B routes across heavy-hex tree branches.")

    # Plot & Save
    fig_dir = PROJECT_ROOT / 'results' / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_path = fig_dir / 'interactive_benchmark.png'
    plot_trotter_error_tradeoff(
        steps=steps_range,
        trotter_errors=trotter_errs,
        noisy_errors_A=[r['noise_error'] for r in records_A],
        noisy_errors_B=[r['noise_error'] for r in records_B],
        total_errors_A=[r['total_error'] for r in records_A],
        total_errors_B=[r['total_error'] for r in records_B],
        opt_n_A=opt_A['optimal_n'],
        opt_n_B=opt_B['optimal_n'],
        save_path=save_path
    )
    print(f"Benchmark plot saved to: {save_path}")

    # Output file export
    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == '.json':
            export_data = {
                'parameters': {
                    'num_qubits': num_qubits, 'J': J, 'h': h, 'time': time,
                    'order': order, 'periodic': periodic, 'shots': shots
                },
                'empirical_optimal': {'processor_A': opt_A, 'processor_B': opt_B},
                'analytical_prediction': {'processor_A': ana_A, 'processor_B': ana_B},
                'records_A': records_A,
                'records_B': records_B
            }
            with open(out_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"Results exported to JSON: {out_path}")
        else:
            summary_df.to_csv(out_path, index=False)
            print(f"Results exported to CSV: {out_path}")

    return opt_A, opt_B


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Explore Trotter sweet spot n* on custom inputs or judge presets.")
    parser.add_argument('--preset', choices=list(PRESETS.keys()), help="Choose a pre-configured physical regime.")
    parser.add_argument('--qubits', type=int, default=4, help="Number of qubits (default: 4)")
    parser.add_argument('--J', type=float, default=1.0, help="Ising coupling strength J (default: 1.0)")
    parser.add_argument('--h', type=float, default=0.5, help="Transverse field h (default: 0.5)")
    parser.add_argument('--time', type=float, default=1.0, help="Total evolution time T (default: 1.0)")
    parser.add_argument('--order', type=int, choices=[1, 2], default=1, help="Trotter order (1 or 2, default: 1)")
    parser.add_argument('--periodic', action='store_true', help="Use periodic boundary conditions (closed ring)")
    parser.add_argument('--noise-2q', type=float, default=None, help="Override two-qubit gate error rate")
    parser.add_argument('--shots', type=int, default=2048, help="Number of shots (default: 2048)")
    parser.add_argument('--output', type=str, default=None, help="File path to export results (.csv or .json)")
    parser.add_argument('--verbose', action='store_true', help="Print verbose circuit and routing breakdown")

    args = parser.parse_args()

    if args.preset:
        p = PRESETS[args.preset]
        print(f"\n[LOADING PRESET]: {p['name']}")
        print(f"Description: {p['description']}")
        run_benchmark(
            num_qubits=p['num_qubits'],
            J=p['J'],
            h=p['h'],
            time=p['time'],
            order=p.get('order', 1),
            periodic=p.get('periodic', False),
            two_q_error=args.noise_2q,
            shots=args.shots,
            output_file=args.output,
            verbose=args.verbose
        )
    else:
        run_benchmark(
            num_qubits=args.qubits,
            J=args.J,
            h=args.h,
            time=args.time,
            order=args.order,
            periodic=args.periodic,
            two_q_error=args.noise_2q,
            shots=args.shots,
            output_file=args.output,
            verbose=args.verbose
        )
