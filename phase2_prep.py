"""
Phase 2 Preparation Module
Qiskit Fall Fest 2026 - Problem S3: Noise in Trotter Simulation

Demonstrates Phase 2 readiness:
1. Extensibility to 8-qubit and 10-qubit TFIM simulation lattices.
2. Architecture benchmarking against 10-qubit extended heavy-hex Processor C.
3. Distant port-to-port Bell-state generation (|Phi+>) between Port 0 and Port 9,
   evaluating routing SWAP overhead and entanglement fidelity under hardware noise.
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

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

from src.problem import create_ising_chain
from src.processors import load_processor
from src.routing import transpile_circuit, analyze_routing_overhead
from src.metrics import circuit_metrics, find_optimal_trotter_step, predict_analytical_optimal_n
from src.visualization import plot_processor_graph


def build_processor_c_noise(proc_cfg):
    """Build Aer NoiseModel for Processor C."""
    noise = NoiseModel()
    cfg = proc_cfg.get('noise', {})
    noise.add_all_qubit_quantum_error(depolarizing_error(cfg.get('single_qubit_error', 0.0008), 1), ['x', 'sx', 'rz'])
    noise.add_all_qubit_quantum_error(depolarizing_error(cfg.get('two_qubit_error', 0.008), 2), ['cx'])
    r_err = cfg.get('readout_error', 0.015)
    noise.add_all_qubit_readout_error(ReadoutError([[1.0 - r_err, r_err], [r_err, 1.0 - r_err]]))
    return noise


def build_port_bell_circuit(port_a: int, port_b: int, num_qubits: int, basis: str = 'Z') -> QuantumCircuit:
    """
    Construct circuit creating Bell state |Phi+> = (|00> + |11>)/sqrt(2) between port_a and port_b.
    Uses logical CNOT and transpiler routing across the topology.
    """
    qc = QuantumCircuit(num_qubits)
    qc.h(port_a)
    qc.cx(port_a, port_b)

    if basis == 'Z':
        # Measure ZZ
        qc.measure_all()
    elif basis == 'X':
        # Measure XX (Hadamard before measurement)
        qc.h(port_a)
        qc.h(port_b)
        qc.measure_all()
    elif basis == 'Y':
        # Measure YY (S_dagger then H before measurement)
        qc.sdg(port_a)
        qc.h(port_a)
        qc.sdg(port_b)
        qc.h(port_b)
        qc.measure_all()

    return qc


def evaluate_bell_fidelity(counts_z, counts_x, counts_y, port_a, port_b, num_qubits, shots):
    """Calculate Bell state fidelity: F = (1 + <ZZ> + <XX> - <YY>) / 4."""
    def get_correlator(counts):
        val_sum = 0
        for bstr, count in counts.items():
            b_a = int(bstr[num_qubits - 1 - port_a])
            b_b = int(bstr[num_qubits - 1 - port_b])
            eigen = 1 if (b_a ^ b_b) == 0 else -1
            val_sum += eigen * count
        return val_sum / shots

    zz = get_correlator(counts_z)
    xx = get_correlator(counts_x)
    yy = get_correlator(counts_y)
    fidelity = (1.0 + zz + xx - yy) / 4.0
    return float(fidelity), float(zz), float(xx), float(yy)


def main():
    print("=" * 70)
    print("PHASE 2 EXTENSION & READINESS BENCHMARK")
    print("=" * 70)

    fig_dir = PROJECT_ROOT / 'results' / 'figures'
    tab_dir = PROJECT_ROOT / 'results' / 'tables'
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Processor C
    proc_C = load_processor(PROJECT_ROOT / 'processors' / 'processor_C.json')
    print(f"[1/3] Loaded Processor C: {proc_C['name']} ({proc_C['num_qubits']} Qubits, Topology: {proc_C['topology']})")
    print(f"      Designated Ports: {proc_C['ports']}")

    # Plot Processor C topology
    plot_processor_graph(proc_C, title="Processor C (10 Qubits, Extended Heavy-Hex Modular)",
                         save_path=fig_dir / 'processor_C_graph.png')
    print("      Saved processor_C_graph.png")

    # 2. Port-to-Port Bell State Distribution
    print("[2/3] Evaluating Port-to-Port Bell State Generation (Port 0 to Port 9)...")
    port_0, port_9 = proc_C['ports'][0], proc_C['ports'][1]
    shots = 4096
    sim_C = AerSimulator(noise_model=build_processor_c_noise(proc_C))

    qc_bare_bell = QuantumCircuit(proc_C['num_qubits'])
    qc_bare_bell.h(port_0)
    qc_bare_bell.cx(port_0, port_9)
    transpiled_bell = transpile_circuit(qc_bare_bell, proc_C, optimization_level=2, seed_transpiler=42)
    routing_bell = analyze_routing_overhead(qc_bare_bell, transpiled_bell)

    qc_z = transpile_circuit(build_port_bell_circuit(port_0, port_9, proc_C['num_qubits'], 'Z'), proc_C, optimization_level=2, seed_transpiler=42)
    qc_x = transpile_circuit(build_port_bell_circuit(port_0, port_9, proc_C['num_qubits'], 'X'), proc_C, optimization_level=2, seed_transpiler=42)
    qc_y = transpile_circuit(build_port_bell_circuit(port_0, port_9, proc_C['num_qubits'], 'Y'), proc_C, optimization_level=2, seed_transpiler=42)

    cz = sim_C.run(qc_z, shots=shots, seed_simulator=42).result().get_counts()
    cx = sim_C.run(qc_x, shots=shots, seed_simulator=42).result().get_counts()
    cy = sim_C.run(qc_y, shots=shots, seed_simulator=42).result().get_counts()

    bell_fid, zz_val, xx_val, yy_val = evaluate_bell_fidelity(cz, cx, cy, port_0, port_9, proc_C['num_qubits'], shots)
    print(f"      Transpiled Depth: {transpiled_bell.depth()} | Total CX: {routing_bell['transpiled_2q_gates']} (SWAPs: {routing_bell['effective_swaps']})")
    print(f"      Measured Correlators: <ZZ>={zz_val:.4f}, <XX>={xx_val:.4f}, <YY>={yy_val:.4f}")
    print(f"      Port-to-Port Bell Fidelity F_bell = {bell_fid * 100:.2f}% (Threshold: > 50% for entanglement)")

    # 3. 8-Qubit TFIM Scaling Preview on Processor C
    print("[3/3] 8-Qubit TFIM Trotter Scaling Preview...")
    N_8 = 8
    sim_8q = create_ising_chain(num_qubits=N_8, J=1.0, h=0.5, periodic=False)
    steps_preview = [1, 2, 3, 4, 5]
    scaling_8q_records = []

    for n in steps_preview:
        qc_b = sim_8q.build_trotter_circuit(time=1.0, num_steps=n, order=1, basis=None)
        t_c = transpile_circuit(qc_b, proc_C, optimization_level=1, seed_transpiler=42)
        r_c = analyze_routing_overhead(qc_b, t_c)
        ana_8q = predict_analytical_optimal_n(1.0, 0.5, 1.0, N_8, 0.008, order=1, periodic=False)
        scaling_8q_records.append({
            'steps': n,
            'logical_depth': qc_b.depth(),
            'transpiled_depth': t_c.depth(),
            'transpiled_2q_gates': r_c['transpiled_2q_gates'],
            'effective_swaps': r_c['effective_swaps']
        })

    df_8q = pd.DataFrame(scaling_8q_records)
    print(df_8q.to_string(index=False))
    print(f"      Analytical Predicted n* for 8-Qubit System on Processor C: n* ~ {ana_8q['analytical_n_continuous']:.2f}")

    # Summary export
    phase2_summary = {
        'processor_C': {
            'num_qubits': proc_C['num_qubits'],
            'topology': proc_C['topology'],
            'ports': proc_C['ports']
        },
        'bell_state_benchmark': {
            'ports': [port_0, port_9],
            'circuit_depth': transpiled_bell.depth(),
            'cx_gates': routing_bell['transpiled_2q_gates'],
            'swaps': routing_bell['effective_swaps'],
            'bell_fidelity': bell_fid,
            'entanglement_verified': bool(bell_fid > 0.5)
        },
        'eight_qubit_scaling': scaling_8q_records,
        'analytical_prediction_8q': ana_8q
    }

    with open(tab_dir / 'phase2_readiness.json', 'w') as f:
        json.dump(phase2_summary, f, indent=2)

    # Plot preview
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(df_8q['steps'], df_8q['transpiled_depth'], 'o-', color='#00897B', linewidth=2.2, label='8-Qubit Transpiled Depth')
    ax.plot(df_8q['steps'], df_8q['transpiled_2q_gates'], 's-', color='#E53935', linewidth=2.2, label='8-Qubit 2Q (CX) Gates')
    ax.set_xlabel('Trotter Steps (n)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Circuit Metric Count', fontsize=11, fontweight='bold')
    ax.set_title('Phase 2 Readiness: 8-Qubit TFIM Circuit Complexity on Processor C', fontsize=12, fontweight='bold')
    ax.legend(framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    fig.savefig(fig_dir / 'phase2_scaling_preview.png', dpi=200, bbox_inches='tight')
    plt.close()

    print(f"      Saved phase2_scaling_preview.png and phase2_readiness.json")
    print("PHASE 2 BENCHMARK COMPLETED SUCCESSFULLY!")


if __name__ == '__main__':
    main()
