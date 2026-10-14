"""
cyclic_study.py — Optional Exploration: Cyclic Heavy-Hex Variant Study
Qiskit Fall Fest 2026 Phase 1: Problem S3 (Noise in Trotter Simulation)

ROLE & PURPOSE:
This optional sensitivity study addresses the topological ambiguity present in
the organizer presentation slide (where the illustration of Processor B schematically
shows closed cycles, whereas the text specifies a 7-qubit heavy-hex-inspired graph).

The baseline deliverables rigorously treat Processor B as the canonical acyclic 7-qubit
heavy-hex tree (girth infinity). This script defines an optional cyclic heavy-hex variant
by adding the cycle-closing coupler [2, 6] (forming a 5-cycle 1-2-6-5-3-1), and evaluates
both Open Boundary Conditions (OBC) and Periodic Boundary Conditions (PBC) to prove
why hardware cycle topology natively eliminates routing SWAPs for closed rings.
"""

import sys
import copy
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import numpy as np
import pandas as pd
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

from src.problem import create_ising_chain
from src.processors import load_processor
from src.routing import transpile_circuit, transpile_with_routing_analysis
from src.metrics import mitigate_readout_counts, compute_noisy_state_fidelity


def make_noise(two_q_err=0.01, single_q_err=0.001, readout_err=0.02):
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(single_q_err, 1), ['x', 'sx', 'rz'])
    noise.add_all_qubit_quantum_error(depolarizing_error(two_q_err, 2), ['cx'])
    noise.add_all_qubit_readout_error(ReadoutError([[1.0 - readout_err, readout_err], [readout_err, 1.0 - readout_err]]))
    return noise


def run_cyclic_comparison():
    tab_dir = PROJECT_ROOT / 'results' / 'tables'
    tab_dir.mkdir(parents=True, exist_ok=True)

    proc_A = load_processor(PROJECT_ROOT / 'processors' / 'processor_A.json')
    proc_B = load_processor(PROJECT_ROOT / 'processors' / 'processor_B.json')

    # Optional Cyclic Heavy-Hex Variant
    proc_B_cyclic = copy.deepcopy(proc_B)
    proc_B_cyclic['name'] = 'processor_B_cyclic'
    proc_B_cyclic['description'] = 'Optional 7-qubit cyclic heavy-hex variant with coupler [2, 6] forming cycle 1-2-6-5-3-1'
    proc_B_cyclic['coupling_map'].extend([[2, 6], [6, 2]])

    noise = make_noise(0.01, 0.001, 0.02)
    aer = AerSimulator(noise_model=noise)

    SHOTS = 2048
    SEEDS = [42, 43, 44, 45, 46]
    TIME = 1.0
    STEPS = 3

    benchmarks = [
        ('5-Spin Open Chain (OBC N=5)', False),
        ('5-Spin Periodic Ring (PBC N=5)', True)
    ]

    records = []

    for bench_name, is_pbc in benchmarks:
        sim = create_ising_chain(num_qubits=5, J=1.0, h=0.5, periodic=is_pbc)
        sv_exact = sim.exact_evolved_state(None, TIME)
        E_exact = sim.exact_energy_expectation(sv_exact)

        qc_bare = sim.build_trotter_circuit(TIME, STEPS, order=1, basis=None)
        qc_z = sim.build_trotter_circuit(TIME, STEPS, order=1, basis='Z')
        qc_x = sim.build_trotter_circuit(TIME, STEPS, order=1, basis='X')

        procs = [
            ('Processor A (5Q T-Tree)', proc_A),
            ('Processor B (7Q Tree Baseline)', proc_B),
            ('Processor B (7Q Cyclic Heavy-Hex Variant)', proc_B_cyclic)
        ]

        for p_label, p_cfg in procs:
            t_bare, r_info = transpile_with_routing_analysis(qc_bare, p_cfg, optimization_level=1, seed_transpiler=42)
            tz = transpile_circuit(qc_z, p_cfg, optimization_level=1, seed_transpiler=42)
            tx = transpile_circuit(qc_x, p_cfg, optimization_level=1, seed_transpiler=42)

            raw_energies, mit_energies = [], []
            for s in SEEDS:
                cz = aer.run(tz, shots=SHOTS, seed_simulator=s).result().get_counts()
                cx = aer.run(tx, shots=SHOTS, seed_simulator=s).result().get_counts()
                e_raw, _ = sim.compute_energy_from_counts(cz, cx, SHOTS)
                raw_energies.append(e_raw)

                mz = mitigate_readout_counts(cz, 0.02, 5)
                mx = mitigate_readout_counts(cx, 0.02, 5)
                pz = {k: int(round(v * SHOTS)) for k, v in mz.items()}
                px = {k: int(round(v * SHOTS)) for k, v in mx.items()}
                e_mit, _ = sim.compute_energy_from_counts(pz, px, SHOTS)
                mit_energies.append(e_mit)

            m_raw = float(np.mean(raw_energies))
            s_raw = float(np.std(raw_energies))
            m_mit = float(np.mean(mit_energies))
            s_mit = float(np.std(mit_energies))

            denom = abs(E_exact)
            err_raw = abs(m_raw - E_exact) / denom
            err_raw_std = s_raw / denom
            err_mit = abs(m_mit - E_exact) / denom
            err_mit_std = s_mit / denom

            fid = compute_noisy_state_fidelity(t_bare, sv_exact, noise, 5)

            records.append({
                'Benchmark': bench_name,
                'Architecture': p_label,
                'Transpiled_Depth': t_bare.depth(),
                '2Q_Gates': r_info['transpiled_2q_gates'],
                'SWAP_Gates': r_info['exact_swaps'],
                'Measured_Energy': f"{m_raw:.4f} +/- {s_raw:.4f}",
                'Mitigated_Energy': f"{m_mit:.4f} +/- {s_mit:.4f}",
                'Raw_Total_Error': f"{err_raw*100:.2f}% +/- {err_raw_std*100:.2f}%",
                'Mitigated_Total_Error': f"{err_mit*100:.2f}% +/- {err_mit_std*100:.2f}%",
                'Noisy_State_Fidelity': f"{fid*100:.2f}%"
            })

    df = pd.DataFrame(records)
    out_csv = tab_dir / 'cyclic_variant_comparison.csv'
    df.to_csv(out_csv, index=False)
    print(f"Saved cyclic variant comparison to {out_csv}")
    print(df.to_string())
    return df


if __name__ == '__main__':
    run_cyclic_comparison()
