"""
Metrics Module - Qiskit Fall Fest 2026 Phase 1
Calculates problem-specific metrics (energy error, algorithmic Trotter error,
noisy state fidelity, total variation distance) and architecture metrics (depth, 2Q gates, exact SWAPs),
including analytical uncertainty quantification (sigma_p) and readout error mitigation.
"""

import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from qiskit import QuantumCircuit


def circuit_metrics(qc: QuantumCircuit) -> Dict[str, Any]:
    """Extract comprehensive circuit metrics."""
    ops = qc.count_ops()
    two_q_names = {'cx', 'cz', 'ecr', 'swap', 'rxx', 'ryy', 'rzz'}

    two_q_ops = sum(n for g, n in ops.items() if g.lower() in two_q_names)
    single_q_ops = sum(n for g, n in ops.items() if g.lower() not in two_q_names and g.lower() not in {'barrier', 'measure'})
    swap_count = ops.get('swap', 0)

    return {
        'depth': qc.depth(),
        'num_qubits': qc.num_qubits,
        'total_ops': sum(ops.values()),
        'two_qubit_ops': two_q_ops,
        'swap_count': swap_count,
        'cx_count': ops.get('cx', 0),
        'single_qubit_ops': single_q_ops,
        'operations': dict(ops)
    }


def proportion_standard_error(p: float, n: int) -> float:
    """Calculate standard error for a proportion: sigma = sqrt(p*(1-p)/n)."""
    if n <= 0 or p < 0 or p > 1:
        return 0.0
    return math.sqrt(max(0.0, p * (1.0 - p)) / n)


def total_variation_distance(p_dist: Dict[str, float], q_dist: Dict[str, float]) -> float:
    """Calculate Total Variation Distance TVD = 0.5 * sum |p_x - q_x|."""
    all_keys = set(p_dist.keys()).union(set(q_dist.keys()))
    tvd = 0.5 * sum(abs(p_dist.get(k, 0.0) - q_dist.get(k, 0.0)) for k in all_keys)
    return float(tvd)


def mitigate_readout_counts(raw_counts: Dict[str, int],
                            readout_error: float,
                            num_qubits: int) -> Dict[str, float]:
    """
    Apply single-qubit tensor-product matrix inversion for readout error mitigation.
    M_single = [[1 - e, e], [e, 1 - e]].
    """
    if readout_error <= 0.0:
        total = sum(raw_counts.values())
        return {k: v / total for k, v in raw_counts.items()}

    total_shots = sum(raw_counts.values())
    raw_vec = np.zeros(2 ** num_qubits)
    for bstr, count in raw_counts.items():
        idx = int(bstr, 2)
        raw_vec[idx] = count / total_shots

    # 1-qubit inverse response matrix
    det = (1.0 - readout_error)**2 - readout_error**2
    if det == 0:
        return {k: v / total_shots for k, v in raw_counts.items()}
    M_inv_1q = (1.0 / det) * np.array([[1.0 - readout_error, -readout_error],
                                       [-readout_error, 1.0 - readout_error]])

    # Build full tensor inverse
    M_inv = M_inv_1q
    for _ in range(num_qubits - 1):
        M_inv = np.kron(M_inv, M_inv_1q)

    mitigated_vec = M_inv @ raw_vec
    mitigated_vec = np.clip(mitigated_vec, 0.0, None)
    if np.sum(mitigated_vec) > 0:
        mitigated_vec /= np.sum(mitigated_vec)

    mitigated_counts = {}
    for idx, prob in enumerate(mitigated_vec):
        if prob > 1e-6:
            bstr = format(idx, f'0{num_qubits}b')
            mitigated_counts[bstr] = float(prob)

    return mitigated_counts


def compute_noisy_state_fidelity(qc: QuantumCircuit,
                                 ideal_state: np.ndarray,
                                 noise_model: Any,
                                 num_qubits: int) -> float:
    """
    Compute noisy quantum state fidelity F_noisy = <psi_exact | rho_noisy | psi_exact>
    using Qiskit Aer density matrix simulation, properly accounting for transpiler layout
    and active vs ancilla qubit mapping.
    """
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import state_fidelity, partial_trace, Statevector, DensityMatrix

    sim = AerSimulator(noise_model=noise_model, method='density_matrix')
    qc_dm = qc.copy()
    qc_dm.remove_final_measurements(inplace=True)
    qc_dm.save_density_matrix()

    try:
        res = sim.run(qc_dm).result()
        rho = res.data(0)['density_matrix']
        total_qubits = qc_dm.num_qubits

        # Extract active physical qubit indices from transpiler layout
        tl = getattr(qc, 'layout', None)
        if tl and tl.final_index_layout() is not None:
            final_layout = tl.final_index_layout()
        elif tl and tl.initial_index_layout() is not None:
            final_layout = tl.initial_index_layout()[:num_qubits]
        else:
            final_layout = list(range(num_qubits))

        phys_indices = final_layout[:num_qubits]
        ancillas = [p for p in range(total_qubits) if p not in phys_indices]

        if ancillas:
            rho_reduced = partial_trace(rho, ancillas)
        else:
            rho_reduced = rho

        # Re-order subsystems to match original virtual qubit ordering (q_0, ..., q_{k-1})
        remaining_phys = sorted(phys_indices)
        k = num_qubits
        pos_in_reduced = [remaining_phys.index(phys_indices[v]) for v in range(k)]

        ket_axes = [k - 1 - pos_in_reduced[v] for v in range(k)]
        bra_axes = [k - 1 - pos_in_reduced[v] + k for v in range(k)]

        arr = np.array(rho_reduced.data).reshape([2] * (2 * k))
        arr_perm = np.transpose(arr, ket_axes + bra_axes).reshape((2**k, 2**k))
        rho_final = DensityMatrix(arr_perm)

        return float(state_fidelity(Statevector(ideal_state), rho_final))
    except Exception:
        return 0.0


def find_optimal_trotter_step(sweep_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Find optimal Trotter step n* where total energy error is minimized."""
    if not sweep_records:
        return {}

    best_record = min(sweep_records, key=lambda r: r.get('total_error', float('inf')))
    return {
        'optimal_n': best_record['trotter_steps'],
        'min_total_error': best_record.get('total_error'),
        'optimal_depth': best_record.get('depth'),
        'optimal_2q_gates': best_record.get('two_qubit_ops'),
        'optimal_swaps': best_record.get('swap_count', 0),
        'trotter_error_at_opt': best_record.get('trotter_error'),
        'noise_error_at_opt': best_record.get('noise_error')
    }


def predict_analytical_optimal_n(J: float = 1.0,
                                 h: float = 0.5,
                                 time: float = 1.0,
                                 num_qubits: int = 4,
                                 two_q_error: float = 0.005,
                                 order: int = 1,
                                 periodic: bool = False,
                                 processor_type: str = 'processor_A') -> Dict[str, Any]:
    """
    Continuous analytical estimate of n* from commutator norms and 2Q gate noise scaling:
    eps_total(n) ~ C_trotter / n^order + C_noise * n.

    Note: Continuous analytical estimates capture qualitative scaling, but differ from
    empirical discrete minima at extreme noise limits (e.g. <=0.2% or >=4.0%) due to
    integer step quantization and non-linear multi-qubit error accumulation.
    """
    num_terms = num_qubits if periodic else (num_qubits - 1)
    nominal_2q_per_step = 2 * num_terms
    if order == 2:
        nominal_2q_per_step *= 2

    swaps_per_step = 0
    if periodic and processor_type == 'processor_A':
        swaps_per_step = 2 * (2 if order == 2 else 1)

    cx_per_step = nominal_2q_per_step + (3 * swaps_per_step)
    c_noise = cx_per_step * two_q_error

    energy_scale = max(1.0, J * num_terms + h * num_qubits)
    commutator_norm = 2.0 * abs(J) * abs(h) * num_terms

    if order == 1:
        c_trotter = (0.5 * commutator_norm * (time ** 2)) / energy_scale
        n_continuous = math.sqrt(c_trotter / c_noise) if c_noise > 0 else 10.0
    else:
        c_trotter = (0.0833 * commutator_norm * (abs(J) + abs(h)) * (time ** 3)) / energy_scale
        n_continuous = ((2.0 * c_trotter / c_noise) ** (1.0 / 3.0)) if c_noise > 0 else 10.0

    n_optimal = max(1, int(round(n_continuous)))
    predicted_min_error = (c_trotter / (n_optimal ** order)) + (c_noise * n_optimal)

    return {
        'analytical_n_continuous': float(n_continuous),
        'analytical_n_optimal': int(n_optimal),
        'predicted_min_error': float(predicted_min_error),
        'c_trotter': float(c_trotter),
        'c_noise': float(c_noise),
        'cx_per_step': int(cx_per_step)
    }
