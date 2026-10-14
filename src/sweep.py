"""
Shared Sweep Engine - S3: Noise in Trotter Simulation
Qiskit Fall Fest 2026 - Phase 1 Hackathon

Unified Qiskit/Aer simulation engine providing:
- Real noisy Aer simulation and exact classical baseline comparison
- Exact SWAP gate counting via routing analysis
- Readout error mitigation on BOTH Processor A and Processor B
- Noisy density-matrix quantum state fidelity
- Multi-seed statistical variance tracking
"""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, List
import pandas as pd
import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

from .metrics import (find_optimal_trotter_step, mitigate_readout_counts,
                      predict_analytical_optimal_n, compute_noisy_state_fidelity)
from .problem import create_ising_chain
from .processors import load_processor
from .routing import transpile_with_routing_analysis

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STEPS = (1, 2, 3, 4, 5, 6, 7, 8, 10)


def load_default_processors():
    """Load Processor A (5Q T-shaped tree) and Processor B (7Q Heavy-Hex tree)."""
    proc_A = load_processor(PROJECT_ROOT / 'processors' / 'processor_A.json')
    proc_B = load_processor(PROJECT_ROOT / 'processors' / 'processor_B.json')
    return proc_A, proc_B


def build_noise_model(proc_cfg: Dict[str, Any], two_q_error: Optional[float] = None) -> NoiseModel:
    """Construct Aer noise model from processor configuration."""
    cfg = proc_cfg.get('noise', {})
    e1 = cfg.get('single_qubit_error', 0.001)
    e2 = cfg.get('two_qubit_error', 0.01) if two_q_error is None else two_q_error
    er = cfg.get('readout_error', 0.02)

    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(e1, 1), ['x', 'sx', 'rz'])
    noise.add_all_qubit_quantum_error(depolarizing_error(e2, 2), ['cx'])
    noise.add_all_qubit_readout_error(ReadoutError([[1.0 - er, er], [er, 1.0 - er]]))
    return noise


def validate_inputs(num_qubits, periodic, order, time, two_q_error, steps, proc_A, proc_B):
    """Validate simulation input parameters."""
    max_q = min(proc_A['num_qubits'], proc_B['num_qubits'])
    if not (2 <= num_qubits <= max_q):
        raise ValueError(f"num_qubits must be between 2 and {max_q}, got {num_qubits}.")
    if periodic and num_qubits < 3:
        raise ValueError("Periodic boundary (PBC) needs at least 3 qubits.")
    if order not in (1, 2):
        raise ValueError(f"Trotter order must be 1 or 2, got {order}.")
    if time <= 0:
        raise ValueError(f"Evolution time must be > 0, got {time}.")
    if two_q_error is not None and not (0.0 <= two_q_error < 1.0):
        raise ValueError("two_q_error must be a fraction in [0, 1), e.g. 0.01 for 1%.")
    if not steps or any(int(n) < 1 for n in steps):
        raise ValueError("Trotter steps must be integers >= 1.")


def run_sweep(J: float = 1.0,
              h: float = 0.5,
              time: float = 1.0,
              num_qubits: int = 4,
              two_q_error: Optional[float] = None,
              periodic: bool = False,
              order: int = 1,
              shots: int = 2048,
              steps: Sequence[int] = DEFAULT_STEPS,
              seed: int = 42,
              compute_fidelity: bool = True,
              proc_A: Optional[Dict[str, Any]] = None,
              proc_B: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Parametric Trotter sweep across Processor A and Processor B.
    Applies readout error mitigation to BOTH processors.
    """
    if proc_A is None or proc_B is None:
        dA, dB = load_default_processors()
        proc_A, proc_B = proc_A or dA, proc_B or dB

    steps = [int(n) for n in steps]
    validate_inputs(num_qubits, periodic, order, time, two_q_error, steps, proc_A, proc_B)

    sim = create_ising_chain(num_qubits=num_qubits, J=J, h=h, periodic=periodic)
    exact_state = sim.exact_evolved_state(None, time)
    exact_E = sim.exact_energy_expectation(exact_state)
    denom = max(1e-9, abs(exact_E))

    noise_A = build_noise_model(proc_A, two_q_error)
    noise_B = build_noise_model(proc_B, two_q_error)

    aer_A = AerSimulator(noise_model=noise_A)
    aer_B = AerSimulator(noise_model=noise_B)

    ro_A = proc_A.get('noise', {}).get('readout_error', 0.02)
    ro_B = proc_B.get('noise', {}).get('readout_error', 0.02)

    def run_on_proc(proc, aer, ro_rate, noise_model, qc_bare, qc_z, qc_x):
        t_bare, r_bare = transpile_with_routing_analysis(qc_bare, proc, optimization_level=1, seed_transpiler=seed)
        t_z, _ = transpile_with_routing_analysis(qc_z, proc, optimization_level=1, seed_transpiler=seed)
        t_x, _ = transpile_with_routing_analysis(qc_x, proc, optimization_level=1, seed_transpiler=seed)

        cz = aer.run(t_z, shots=shots, seed_simulator=seed).result().get_counts()
        cx = aer.run(t_x, shots=shots, seed_simulator=seed).result().get_counts()
        energy_raw, stderr = sim.compute_energy_from_counts(cz, cx, shots)

        # Readout error mitigation
        mit_z = mitigate_readout_counts(cz, ro_rate, num_qubits)
        mit_x = mitigate_readout_counts(cx, ro_rate, num_qubits)
        pz = {k: int(round(v * shots)) for k, v in mit_z.items()}
        px = {k: int(round(v * shots)) for k, v in mit_x.items()}
        energy_mit, _ = sim.compute_energy_from_counts(pz, px, shots)

        noisy_fid = 0.0
        if compute_fidelity and num_qubits <= 5:
            noisy_fid = compute_noisy_state_fidelity(t_bare, exact_state, noise_model, num_qubits)

        return t_bare, r_bare, energy_raw, energy_mit, stderr, noisy_fid

    records_A, records_B = [], []
    for n in steps:
        ideal_sv = sim.ideal_statevector(time, n, order=order)
        ideal_fid = float(np.abs(np.vdot(exact_state, ideal_sv))**2)
        trot_E = sim.exact_energy_expectation(ideal_sv)
        trot_err = abs(trot_E - exact_E) / denom

        qc_bare = sim.build_trotter_circuit(time, n, order=order, basis=None)
        qc_z = sim.build_trotter_circuit(time, n, order=order, basis='Z')
        qc_x = sim.build_trotter_circuit(time, n, order=order, basis='X')

        tA, rA, eA_raw, eA_mit, seA, fidA_noisy = run_on_proc(proc_A, aer_A, ro_A, noise_A, qc_bare, qc_z, qc_x)
        tB, rB, eB_raw, eB_mit, seB, fidB_noisy = run_on_proc(proc_B, aer_B, ro_B, noise_B, qc_bare, qc_z, qc_x)

        def make_record(t, r, eraw, emit, se, fid_noisy):
            return {
                'trotter_steps': n,
                'dt': time / n,
                'depth': t.depth(),
                'two_qubit_ops': r['transpiled_2q_gates'],
                'swap_count': r['exact_swaps'],
                'ideal_fidelity': ideal_fid,
                'noisy_fidelity': fid_noisy,
                'trotter_error': trot_err,
                'trotter_energy': trot_E,
                'measured_energy': eraw,
                'mitigated_energy': emit,
                'standard_error': se,
                'noise_error': abs(eraw - trot_E) / denom,
                'total_error': abs(eraw - exact_E) / denom,
                'mitigated_error': abs(emit - exact_E) / denom
            }

        records_A.append(make_record(tA, rA, eA_raw, eA_mit, seA, fidA_noisy))
        records_B.append(make_record(tB, rB, eB_raw, eB_mit, seB, fidB_noisy))

    eff_2q = two_q_error if two_q_error is not None else proc_B['noise'].get('two_qubit_error', 0.01)
    ana = {
        'processor_A': predict_analytical_optimal_n(J, h, time, num_qubits, eff_2q,
                                                    order=order, periodic=periodic,
                                                    processor_type='processor_A'),
        'processor_B': predict_analytical_optimal_n(J, h, time, num_qubits, eff_2q,
                                                    order=order, periodic=periodic,
                                                    processor_type='processor_B'),
    }

    return {
        'params': dict(J=J, h=h, time=time, num_qubits=num_qubits, two_q_error=two_q_error,
                       periodic=periodic, order=order, shots=shots, steps=steps),
        'exact_energy': exact_E,
        'df_A': pd.DataFrame(records_A),
        'df_B': pd.DataFrame(records_B),
        'opt_A': find_optimal_trotter_step(records_A),
        'opt_B': find_optimal_trotter_step(records_B),
        'analytical': ana,
    }


def plot_sweep(result: Dict[str, Any], show_mitigated: bool = True):
    """Generate U-curve error plot comparing Processor A and Processor B."""
    import matplotlib.pyplot as plt

    dfA, dfB = result['df_A'], result['df_B']
    optA, optB = result['opt_A']['optimal_n'], result['opt_B']['optimal_n']
    p = result['params']
    n = dfA['trotter_steps']

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=110)
    ax.plot(n, dfA['trotter_error'] * 100, '--', color='#607D8B', linewidth=2,
            label='Algorithmic Trotter Error (Ideal)')
    ax.plot(n, dfA['total_error'] * 100, 'o-', color='#D32F2F', linewidth=2.2,
            label=f"Processor A Total (n* = {optA})")
    ax.plot(n, dfB['total_error'] * 100, 's-', color='#1976D2', linewidth=2.4,
            label=f"Processor B Total (n* = {optB})")

    if show_mitigated:
        ax.plot(n, dfA['mitigated_error'] * 100, 'v:', color='#E65100', linewidth=1.8,
                label='Proc A + Readout Mitigation')
        ax.plot(n, dfB['mitigated_error'] * 100, '^--', color='#2E7D32', linewidth=2.0,
                label='Proc B + Readout Mitigation')

    ax.axvline(optA, color='#D32F2F', linestyle=':', alpha=0.5)
    ax.axvline(optB, color='#1976D2', linestyle=':', alpha=0.5)
    ax.set_xticks(list(n))
    ax.set_xlabel('Trotter Steps (n)', fontweight='bold')
    ax.set_ylabel('Relative Energy Error (%)', fontweight='bold')
    bc = 'PBC (Ring)' if p['periodic'] else 'OBC (Chain)'
    ax.set_title(f"TFIM Trotter Error Tradeoff: N={p['num_qubits']} Qubits, J={p['J']}, h={p['h']}, "
                 f"T={p['time']}, Order={p['order']} ({bc})", fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(framealpha=0.9, fontsize=9)
    fig.tight_layout()
    return fig
