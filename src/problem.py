"""
Problem Module - S3: Noise in Trotter Simulation
Qiskit Fall Fest 2026 - Phase 1 Hackathon

Implements Trotter-Suzuki decomposition for Hamiltonian time evolution simulation,
exact classical diagonalized baselines, Pauli expectation measurements, and error analysis.

Hamiltonian: Transverse-Field Ising Model (TFIM)
Open Boundary:     H = -J * sum_{i=0}^{N-2} (Z_i Z_{i+1}) - h * sum_{i=0}^{N-1} X_i
Periodic Boundary: H = -J * [ sum_{i=0}^{N-2} (Z_i Z_{i+1}) + Z_{N-1} Z_0 ] - h * sum_{i=0}^{N-1} X_i
"""

import numpy as np
import scipy.linalg
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from typing import List, Tuple, Dict, Any, Optional
import math


class TrotterSimulator:
    """
    Trotter-Suzuki simulator for time evolution under the Transverse-Field Ising Model (TFIM).
    Supports 1st-order and symmetric 2nd-order Trotterizations under open (OBC) and periodic (PBC) boundaries.
    """

    def __init__(self, num_qubits: int, J: float = 1.0, h: float = 0.5, periodic: bool = False):
        """
        Initialize the simulator.

        Args:
            num_qubits: Number of spins / qubits in the chain
            J: Nearest-neighbor Ising coupling strength (ZZ interaction)
            h: Transverse magnetic field strength (X term)
            periodic: If True, adds boundary coupling Z_{N-1} Z_0 (closed ring topology)

        Raises:
            ValueError: If num_qubits < 2 or if periodic=True with num_qubits < 3
        """
        if num_qubits < 2:
            raise ValueError(f"Simulation requires at least 2 qubits, got {num_qubits}.")
        if periodic and num_qubits < 3:
            raise ValueError(f"Periodic boundary conditions (closed ring) require at least 3 qubits, got {num_qubits}.")

        self.num_qubits = int(num_qubits)
        self.J = float(J)
        self.h = float(h)
        self.periodic = bool(periodic)
        self.sparse_pauli_op = self._construct_sparse_pauli_op()
        self.hamiltonian_matrix = self._construct_exact_hamiltonian()

    def _construct_sparse_pauli_op(self) -> SparsePauliOp:
        """Construct Qiskit SparsePauliOp for Hamiltonian."""
        pauli_list = []

        # Nearest-neighbor ZZ terms
        for i in range(self.num_qubits - 1):
            chars = ['I'] * self.num_qubits
            chars[i] = 'Z'
            chars[i + 1] = 'Z'
            # Little-endian indexing in Qiskit
            pauli_list.append((''.join(reversed(chars)), -self.J))

        # Periodic boundary coupling: Z_{N-1} Z_0
        if self.periodic:
            chars = ['I'] * self.num_qubits
            chars[0] = 'Z'
            chars[self.num_qubits - 1] = 'Z'
            pauli_list.append((''.join(reversed(chars)), -self.J))

        # Transverse field X terms: -h * sum X_i
        for i in range(self.num_qubits):
            chars = ['I'] * self.num_qubits
            chars[i] = 'X'
            pauli_list.append((''.join(reversed(chars)), -self.h))

        return SparsePauliOp.from_list(pauli_list)

    def _construct_exact_hamiltonian(self) -> np.ndarray:
        """Construct exact 2^N x 2^N Hamiltonian matrix from SparsePauliOp."""
        return np.array(self.sparse_pauli_op.to_matrix(sparse=False), dtype=complex)

    def exact_evolution_unitary(self, time: float) -> np.ndarray:
        """Calculate exact continuous time-evolution unitary U_exact = exp(-i * H * t)."""
        return scipy.linalg.expm(-1j * self.hamiltonian_matrix * time)

    def exact_evolved_state(self, initial_state: Optional[np.ndarray], time: float) -> np.ndarray:
        """
        Calculate exact evolved statevector |psi(t)> = U_exact(t) |psi(0)>.
        Default initial state: |0...0>.
        """
        if initial_state is None:
            initial_state = np.zeros(2 ** self.num_qubits, dtype=complex)
            initial_state[0] = 1.0  # |00...0>
        U = self.exact_evolution_unitary(time)
        return U @ initial_state

    def exact_energy_expectation(self, state: np.ndarray) -> float:
        """Calculate <psi| H |psi> for a statevector."""
        return float(np.real(np.vdot(state, self.hamiltonian_matrix @ state)))

    def build_trotter_circuit(self,
                              time: float,
                              num_steps: int,
                              order: int = 1,
                              basis: Optional[str] = None) -> QuantumCircuit:
        """
        Build Trotterized time evolution circuit.

        Args:
            time: Total evolution time
            num_steps: Number of Trotter steps
            order: 1 for first-order Trotter, 2 for second-order Trotter-Suzuki (symmetric Strang)
            basis: Measurement basis: None (pure unitary), 'Z' (measure in Z), 'X' (measure in X)

        Returns:
            QuantumCircuit implementing Trotter evolution
        """
        dt = time / num_steps
        qc = QuantumCircuit(self.num_qubits)

        for step in range(num_steps):
            if order == 1:
                # 1st order Trotter: exp(-i H_ZZ dt) * exp(-i H_X dt)
                self._apply_zz_terms(qc, dt)
                self._apply_x_terms(qc, dt)
            elif order == 2:
                # Symmetric 2nd order Trotter-Suzuki:
                # exp(-i H_ZZ dt/2) * exp(-i H_X dt) * exp(-i H_ZZ dt/2)
                self._apply_zz_terms(qc, dt / 2.0)
                self._apply_x_terms(qc, dt)
                self._apply_zz_terms(qc, dt / 2.0)
            else:
                raise ValueError(f"Unsupported Trotter order: {order}. Supported: 1, 2")

        # Add basis measurement if requested
        if basis == 'Z':
            qc.measure_all()
        elif basis == 'X':
            # Rotate to X basis by applying Hadamard before measurement
            for q in range(self.num_qubits):
                qc.h(q)
            qc.measure_all()

        return qc

    def _apply_zz_terms(self, qc: QuantumCircuit, dt: float):
        """
        Apply exp(-i * (-J) * Z_i Z_j * dt) = exp(i * J * dt * Z_i Z_j).
        Decomposition: CNOT(q1, q2) -> RZ(-2 * J * dt, q2) -> CNOT(q1, q2).
        """
        angle = -2.0 * self.J * dt

        # Nearest neighbors in chain
        for i in range(self.num_qubits - 1):
            qc.cx(i, i + 1)
            qc.rz(angle, i + 1)
            qc.cx(i, i + 1)

        # Periodic boundary condition: connects N-1 and 0 (creates closed ring)
        # Note on CNOT direction & routing:
        # On a linear processor (Processor A: 0-1-2-3-4), logical qubit (N-1) is non-adjacent to qubit 0.
        # Calling cx(self.num_qubits - 1, 0) intentionally triggers compiler routing SWAPs
        # (e.g. 2 SWAPs = 6 additional CX gates for N=4), demonstrating the physical topology
        # overhead of linear vs cyclic architectures (where Processor B embeds the 4-cycle with 0 SWAPs).
        if self.periodic:
            qc.cx(self.num_qubits - 1, 0)
            qc.rz(angle, 0)
            qc.cx(self.num_qubits - 1, 0)

    def _apply_x_terms(self, qc: QuantumCircuit, dt: float):
        """
        Apply exp(-i * (-h) * X_i * dt) = exp(i * h * dt * X_i).
        Using RX(theta) = exp(-i * theta/2 * X) -> RX(-2 * h * dt).
        """
        angle = -2.0 * self.h * dt
        for i in range(self.num_qubits):
            qc.rx(angle, i)

    def compute_energy_from_counts(self,
                                   z_counts: Dict[str, int],
                                   x_counts: Dict[str, int],
                                   shots: int) -> Tuple[float, float]:
        """
        Compute total Hamiltonian expectation <H> from Z-basis and X-basis measurement counts.
        Also returns analytical standard error sigma_<H>.
        """
        zz_expectations = []
        zz_variances = []

        # Pairs to measure for ZZ
        zz_pairs = [(i, i + 1) for i in range(self.num_qubits - 1)]
        if self.periodic:
            zz_pairs.append((self.num_qubits - 1, 0))

        for q1, q2 in zz_pairs:
            val_sum = 0
            for bitstring, count in z_counts.items():
                # Qiskit bitstring order is little-endian: bitstring[num_qubits - 1 - q]
                b1 = int(bitstring[self.num_qubits - 1 - q1])
                b2 = int(bitstring[self.num_qubits - 1 - q2])
                eigenval = 1 if (b1 ^ b2) == 0 else -1
                val_sum += eigenval * count
            exp_val = val_sum / shots
            var_val = max(0.0, 1.0 - exp_val ** 2)
            zz_expectations.append(exp_val)
            zz_variances.append(var_val)

        # X Expectation values
        x_expectations = []
        x_variances = []
        for i in range(self.num_qubits):
            val_sum = 0
            for bitstring, count in x_counts.items():
                b = int(bitstring[self.num_qubits - 1 - i])
                eigenval = 1 if b == 0 else -1
                val_sum += eigenval * count
            exp_val = val_sum / shots
            var_val = max(0.0, 1.0 - exp_val ** 2)
            x_expectations.append(exp_val)
            x_variances.append(var_val)

        # Total energy: <H> = -J sum <ZZ> - h sum <X>
        total_energy = -self.J * sum(zz_expectations) - self.h * sum(x_expectations)

        total_var = (self.J ** 2 * sum(zz_variances) + self.h ** 2 * sum(x_variances)) / shots
        stderr = math.sqrt(total_var)

        return float(total_energy), float(stderr)

    def ideal_statevector(self, time: float, num_steps: int, order: int = 1) -> np.ndarray:
        """Compute the ideal statevector produced by noiseless Trotter evolution."""
        qc = self.build_trotter_circuit(time=time, num_steps=num_steps, order=order, basis=None)
        sv = Statevector.from_instruction(qc)
        return sv.data

    def ideal_state_fidelity(self, time: float, num_steps: int, order: int = 1) -> float:
        """
        Calculate algorithmic fidelity F = |<psi_exact(t) | psi_trotter(t)>|^2.
        Measures purely algorithmic Trotter discretization error.
        """
        exact_state = self.exact_evolved_state(None, time)
        trotter_state = self.ideal_statevector(time, num_steps, order)
        overlap = np.vdot(exact_state, trotter_state)
        return float(np.abs(overlap) ** 2)

    def exact_z_expectations(self, state: np.ndarray) -> Dict[str, Any]:
        """
        Calculate exact single-qubit Z expectations <Z_i> and average magnetization <M_z>
        for a given statevector.
        """
        z_vals = []
        for q in range(self.num_qubits):
            val = 0.0
            for idx, amp in enumerate(state):
                prob = float(np.abs(amp) ** 2)
                bit = (idx >> q) & 1
                val += prob if bit == 0 else -prob
            z_vals.append(val)

        return {
            'z_values': z_vals,
            'z0': float(z_vals[0]),
            'mean_z': float(np.mean(z_vals))
        }

    def compute_z_from_counts(self, z_counts: Dict[str, int], shots: int) -> Dict[str, Any]:
        """
        Compute single-qubit Z expectations <Z_i> and average magnetization <M_z>
        from Z-basis measurement counts.
        """
        z_vals = []
        for i in range(self.num_qubits):
            val_sum = 0
            for bitstring, count in z_counts.items():
                b = int(bitstring[self.num_qubits - 1 - i])
                eigenval = 1 if b == 0 else -1
                val_sum += eigenval * count
            z_vals.append(val_sum / shots)

        return {
            'z_values': [float(v) for v in z_vals],
            'z0': float(z_vals[0]),
            'mean_z': float(np.mean(z_vals))
        }


def create_ising_chain(num_qubits: int = 4, J: float = 1.0, h: float = 0.5, periodic: bool = False) -> TrotterSimulator:
    """Convenience factory function for TrotterSimulator."""
    return TrotterSimulator(num_qubits=num_qubits, J=J, h=h, periodic=periodic)
