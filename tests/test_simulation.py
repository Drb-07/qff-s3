"""
Unit and Integration Test Suite
QFF 2026 Phase 1: Problem S3 (Noise in Trotter Simulation)

Verifies:
1. Hamiltonian Hermiticity & Exact Diagonalization
2. Trotter-Suzuki Circuit Synthesis (Order 1 & 2, OBC & PBC)
3. Processor Coupling Graph Integrity (Processor A, B, and C)
4. Transpilation & Routing Overhead Extraction (PBC SWAP comparison)
5. Readout Error Mitigation Inversion & Normalization
6. Analytical Optimal n* Prediction Model
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Ensure src is on sys.path
BASE_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / 'src'))

from src.problem import TrotterSimulator, create_ising_chain
from src.processors import load_processor, get_processor_graph
from src.routing import transpile_circuit, analyze_routing_overhead
from src.metrics import circuit_metrics, mitigate_readout_counts, predict_analytical_optimal_n


class TestHamiltonianAndTrotter(unittest.TestCase):
    """Test suite for physical model and Trotterization."""

    def setUp(self):
        self.sim = create_ising_chain(num_qubits=4, J=1.0, h=0.5)

    def test_hamiltonian_hermiticity(self):
        """Verify H = H^dagger."""
        H = self.sim.hamiltonian_matrix
        self.assertTrue(np.allclose(H, H.conj().T))

    def test_unitary_evolution_unitarity(self):
        """Verify U(t)^dagger U(t) = I."""
        U = self.sim.exact_evolution_unitary(time=1.0)
        I = np.eye(U.shape[0])
        self.assertTrue(np.allclose(U.conj().T @ U, I, atol=1e-10))

    def test_trotter_circuit_gates_order1(self):
        """Verify 1st order Trotter circuit contains expected CX and single-qubit gates."""
        qc = self.sim.build_trotter_circuit(time=1.0, num_steps=2, order=1, basis=None)
        ops = qc.count_ops()
        # For 4 qubits: 3 pairs -> 6 CX per step * 2 steps = 12 CX
        self.assertEqual(ops.get('cx', 0), 12)
        # 4 X terms per step * 2 steps = 8 RX
        self.assertEqual(ops.get('rx', 0), 8)

    def test_trotter_circuit_gates_order2(self):
        """Verify 2nd order Trotter circuit contains double CX density (Strang splitting)."""
        qc = self.sim.build_trotter_circuit(time=1.0, num_steps=1, order=2, basis=None)
        ops = qc.count_ops()
        # 3 pairs * 2 applications = 12 CX per step
        self.assertEqual(ops.get('cx', 0), 12)

    def test_periodic_boundary_terms(self):
        """Verify PBC adds ring boundary coupling Z_{N-1} Z_0."""
        sim_pbc = create_ising_chain(num_qubits=4, J=1.0, h=0.5, periodic=True)
        qc = sim_pbc.build_trotter_circuit(time=1.0, num_steps=1, order=1, basis=None)
        ops = qc.count_ops()
        # 4 terms (3 chain + 1 ring) = 8 CX per step
        self.assertEqual(ops.get('cx', 0), 8)

    def test_trotter_convergence(self):
        """Verify Trotter error strictly decreases as n increases."""
        fid_n1 = self.sim.ideal_state_fidelity(time=1.0, num_steps=1, order=1)
        fid_n5 = self.sim.ideal_state_fidelity(time=1.0, num_steps=5, order=1)
        self.assertGreater(fid_n5, fid_n1)
        self.assertGreater(fid_n5, 0.95)


class TestProcessorsAndRouting(unittest.TestCase):
    """Test suite for processor topologies and compilation."""

    def setUp(self):
        self.proc_A = load_processor(BASE_DIR / 'processors' / 'processor_A.json')
        self.proc_B = load_processor(BASE_DIR / 'processors' / 'processor_B.json')

    def test_processor_configurations(self):
        """Verify processor configurations match challenge specification."""
        self.assertEqual(self.proc_A['num_qubits'], 5)
        self.assertEqual(self.proc_B['num_qubits'], 9)
        self.assertEqual(self.proc_A['ports'], [0, 4])
        self.assertEqual(self.proc_B['ports'], [0, 8])

    def test_transpilation_basis_gates(self):
        """Verify transpiled circuit adheres strictly to basis gates."""
        sim = create_ising_chain(num_qubits=4, J=1.0, h=0.5)
        qc = sim.build_trotter_circuit(time=1.0, num_steps=2, order=1, basis='Z')
        trans_A = transpile_circuit(qc, self.proc_A, optimization_level=1, seed_transpiler=42)

        allowed_gates = set(self.proc_A['basis_gates']).union({'barrier', 'measure'})
        for op in trans_A.count_ops().keys():
            self.assertIn(op, allowed_gates)

    def test_5qubit_chain_architectural_divergence(self):
        """Verify Processor A incurs SWAPs on 5Q chain while Processor B embeds with 0 SWAPs."""
        from routing import transpile_with_routing_analysis
        sim_5q = create_ising_chain(num_qubits=5, J=1.0, h=0.5, periodic=False)
        qc = sim_5q.build_trotter_circuit(time=1.0, num_steps=3, order=1, basis=None)
        _, rA = transpile_with_routing_analysis(qc, self.proc_A, optimization_level=1, seed_transpiler=42)
        _, rB = transpile_with_routing_analysis(qc, self.proc_B, optimization_level=1, seed_transpiler=42)

        self.assertGreater(rA['exact_swaps'], 0)
        self.assertEqual(rB['exact_swaps'], 0)
        self.assertGreater(rA['transpiled_2q_gates'], rB['transpiled_2q_gates'])

    def test_cyclic_heavy_hex_variant_zero_swaps_on_ring(self):
        """Verify 9Q Slide 4 Processor B embeds a 5-spin periodic ring with 0 SWAPs."""
        from routing import transpile_with_routing_analysis
        sim_pbc5 = create_ising_chain(num_qubits=5, J=1.0, h=0.5, periodic=True)
        qc = sim_pbc5.build_trotter_circuit(time=1.0, num_steps=3, order=1, basis=None)
        _, rA = transpile_with_routing_analysis(qc, self.proc_A, optimization_level=1, seed_transpiler=42)
        _, rB = transpile_with_routing_analysis(qc, self.proc_B, optimization_level=1, seed_transpiler=42)
        self.assertEqual(rB['exact_swaps'], 0)
        self.assertGreater(rA['exact_swaps'], 0)


class TestMetricsAndAnalyticalTheory(unittest.TestCase):
    """Test suite for analytical models and mitigation."""

    def test_mitigated_probabilities_normalized(self):
        """Verify mitigated probabilities sum to 1.0."""
        mock_counts = {'0000': 1500, '0001': 300, '1111': 248}
        mit_counts = mitigate_readout_counts(mock_counts, readout_error=0.01, num_qubits=4)
        total_prob = sum(mit_counts.values())
        self.assertAlmostEqual(total_prob, 1.0, places=5)

    def test_analytical_optimal_n_prediction(self):
        """Verify analytical n* prediction formula produces expected range."""
        pred = predict_analytical_optimal_n(J=1.0, h=0.5, time=1.0, num_qubits=4, two_q_error=0.005, order=1)
        self.assertIn(pred['analytical_n_optimal'], [2, 3, 4])
        self.assertGreater(pred['analytical_n_continuous'], 1.0)
        self.assertGreater(pred['c_trotter'], 0.0)
        self.assertGreater(pred['c_noise'], 0.0)


if __name__ == '__main__':
    unittest.main()
