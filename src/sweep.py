"""
Shared Sweep Engine - S3: Noise in Trotter Simulation
Qiskit Fall Fest 2026 - Phase 1 Hackathon

One real Qiskit/Aer sweep implementation used by EVERY front-end
(Jupyter widget, Streamlit app, scripts), so they all give identical numbers
for identical inputs.

    from src.sweep import run_sweep, plot_sweep
    res = run_sweep(J=1.0, h=0.5, time=1.0, num_qubits=4, two_q_error=0.01)
    res['opt_B']['optimal_n'], res['df_B']
"""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

from .metrics import (find_optimal_trotter_step, mitigate_readout_counts,
                      predict_analytical_optimal_n)
from .problem import create_ising_chain
from .processors import load_processor
from .routing import analyze_routing_overhead, transpile_circuit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STEPS = (1, 2, 3, 4, 5, 6, 7, 8, 10)


def load_default_processors():
    """Load Processor A (5Q linear) and Processor B (7Q heavy-hex)."""
    proc_A = load_processor(PROJECT_ROOT / 'processors' / 'processor_A.json')
    proc_B = load_processor(PROJECT_ROOT / 'processors' / 'processor_B.json')
    return proc_A, proc_B


def build_noise_model(proc_cfg: Dict[str, Any], two_q_error: Optional[float] = None) -> NoiseModel:
    """Aer noise model from a processor config. `two_q_error` (fraction) overrides the JSON value."""
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
    """Fail early with a readable message instead of a cryptic transpiler error."""
    max_q = min(proc_A['num_qubits'], proc_B['num_qubits'])
    if not (2 <= num_qubits <= max_q):
        raise ValueError(f"num_qubits must be between 2 and {max_q} "
                         f"(Processor A has {proc_A['num_qubits']} qubits), got {num_qubits}.")
    if periodic and num_qubits < 3:
        raise ValueError("Periodic boundary (PBC) needs at least 3 qubits.")
    if order not in (1, 2):
        raise ValueError(f"Trotter order must be 1 or 2, got {order}.")
    if time <= 0:
        raise ValueError(f"Evolution time must be > 0, got {time}.")
    if two_q_error is not None and not (0.0 <= two_q_error < 1.0):
        raise ValueError("two_q_error is a fraction in [0, 1), e.g. 0.01 for 1%.")
    if not steps or any(int(n) < 1 for n in steps):
        raise ValueError("Trotter step list must contain integers >= 1.")


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
              proc_A: Optional[Dict[str, Any]] = None,
              proc_B: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Full Trotter-step sweep on Processor A and B with real Aer noisy simulation.

    two_q_error: 2Q depolarizing error as a FRACTION (0.01 = 1%). None -> use the processor JSON values.
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

    aer_A = AerSimulator(noise_model=build_noise_model(proc_A, two_q_error))
    aer_B = AerSimulator(noise_model=build_noise_model(proc_B, two_q_error))
    ro_B = proc_B.get('noise', {}).get('readout_error', 0.02)

    def run_on(proc, aer, qc_bare, qc_z, qc_x):
        t_bare = transpile_circuit(qc_bare, proc, optimization_level=1, seed_transpiler=seed)
        t_z = transpile_circuit(qc_z, proc, optimization_level=1, seed_transpiler=seed)
        t_x = transpile_circuit(qc_x, proc, optimization_level=1, seed_transpiler=seed)
        rout = analyze_routing_overhead(qc_bare, t_bare)
        cz = aer.run(t_z, shots=shots, seed_simulator=seed).result().get_counts()
        cx = aer.run(t_x, shots=shots, seed_simulator=seed).result().get_counts()
        energy, stderr = sim.compute_energy_from_counts(cz, cx, shots)
        return t_bare, rout, cz, cx, energy, stderr

    records_A, records_B = [], []
    for n in steps:
        ideal_sv = sim.ideal_statevector(time, n, order=order)
        trot_E = sim.exact_energy_expectation(ideal_sv)
        trot_err = abs(trot_E - exact_E) / denom

        qc_bare = sim.build_trotter_circuit(time, n, order=order, basis=None)
        qc_z = sim.build_trotter_circuit(time, n, order=order, basis='Z')
        qc_x = sim.build_trotter_circuit(time, n, order=order, basis='X')

        tA, rA, _, _, eA, seA = run_on(proc_A, aer_A, qc_bare, qc_z, qc_x)
        tB, rB, czB, cxB, eB, seB = run_on(proc_B, aer_B, qc_bare, qc_z, qc_x)

        # Readout-error mitigation (Processor B)
        mit_z = mitigate_readout_counts(czB, ro_B, num_qubits)
        mit_x = mitigate_readout_counts(cxB, ro_B, num_qubits)
        pz = {k: int(round(v * shots)) for k, v in mit_z.items()}
        px = {k: int(round(v * shots)) for k, v in mit_x.items()}
        e_mit, _ = sim.compute_energy_from_counts(pz, px, shots)

        def rec(t, r, e, se):
            return {
                'trotter_steps': n, 'dt': time / n, 'depth': t.depth(),
                'two_qubit_ops': r['transpiled_2q_gates'], 'swap_count': r['effective_swaps'],
                'trotter_error': trot_err, 'measured_energy': e, 'standard_error': se,
                'noise_error': abs(e - trot_E) / denom,
                'total_error': abs(e - exact_E) / denom,
            }

        records_A.append(rec(tA, rA, eA, seA))
        rb = rec(tB, rB, eB, seB)
        rb['mitigated_error'] = abs(e_mit - exact_E) / denom
        records_B.append(rb)

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
    """U-curve figure for a run_sweep() result. Returns a matplotlib Figure (caller shows/closes it)."""
    import matplotlib.pyplot as plt

    dfA, dfB = result['df_A'], result['df_B']
    optA, optB = result['opt_A']['optimal_n'], result['opt_B']['optimal_n']
    p = result['params']
    n = dfA['trotter_steps']

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=110)
    ax.plot(n, dfA['trotter_error'] * 100, '--', color='#607D8B', linewidth=2,
            label='Algorithmic Trotter error')
    ax.plot(n, dfB['noise_error'] * 100, ':', color='#FFA000', linewidth=2,
            label='Hardware noise (Processor B)')
    ax.plot(n, dfA['total_error'] * 100, 'o-', color='#D32F2F', linewidth=2.2,
            label=f'Processor A total (n* = {optA})')
    ax.plot(n, dfB['total_error'] * 100, 's-', color='#1976D2', linewidth=2.4,
            label=f'Processor B total (n* = {optB})')
    if show_mitigated and 'mitigated_error' in dfB:
        ax.plot(n, dfB['mitigated_error'] * 100, '^--', color='#2E7D32', linewidth=1.8,
                label='Processor B + readout mitigation')
    ax.axvline(optB, color='#1976D2', linestyle=':', alpha=0.6)
    ax.set_xticks(list(n))
    ax.set_xlabel('Trotter steps (n)', fontweight='bold')
    ax.set_ylabel('Relative energy error (%)', fontweight='bold')
    bc = 'PBC' if p['periodic'] else 'OBC'
    ax.set_title(f"N={p['num_qubits']}, J={p['J']}, h={p['h']}, T={p['time']}, "
                 f"order={p['order']}, {bc}", fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    return fig
