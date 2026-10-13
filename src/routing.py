"""
Routing Module
Qiskit Fall Fest 2026 - Phase 1 Hackathon

Handles transpilation, processor routing, and SWAP / 2Q gate overhead analysis.
Decomposes SWAP gates into basis CX gates (1 SWAP = 3 CX) and records hardware metrics.
"""

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from typing import Dict, Any, Tuple, Optional
import numpy as np


def create_coupling_map(processor: Dict[str, Any]) -> CouplingMap:
    """Create Qiskit CouplingMap from processor configuration."""
    return CouplingMap(processor['coupling_map'])


def transpile_circuit(qc: QuantumCircuit,
                      processor: Dict[str, Any],
                      optimization_level: int = 1,
                      seed_transpiler: Optional[int] = 42) -> QuantumCircuit:
    """
    Transpile circuit to target processor coupling map and basis gates.
    """
    coupling_map = create_coupling_map(processor)
    transpiled = transpile(
        qc,
        coupling_map=coupling_map,
        basis_gates=processor['basis_gates'],
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler
    )
    return transpiled


def analyze_routing_overhead(logical_qc: QuantumCircuit,
                             transpiled_qc: QuantumCircuit) -> Dict[str, Any]:
    """
    Analyze routing overhead introduced by transpilation.
    Accounts for decomposition of SWAP gates into 3 CX gates.
    """
    logical_ops = logical_qc.count_ops()
    transpiled_ops = transpiled_qc.count_ops()

    two_q_names = {'cx', 'cz', 'ecr', 'swap', 'rxx', 'ryy', 'rzz'}

    logical_2q = sum(logical_ops.get(g, 0) for g in two_q_names)
    transpiled_2q = sum(transpiled_ops.get(g, 0) for g in two_q_names)

    explicit_swaps = transpiled_ops.get('swap', 0)
    extra_2q = max(0, transpiled_2q - logical_2q)
    # Each routed interaction requiring a SWAP adds 3 CX gates in the cx basis
    effective_swaps = explicit_swaps + (extra_2q // 3)

    return {
        'logical_depth': logical_qc.depth(),
        'transpiled_depth': transpiled_qc.depth(),
        'depth_increase': transpiled_qc.depth() - logical_qc.depth(),
        'logical_2q_gates': logical_2q,
        'transpiled_2q_gates': transpiled_2q,
        'explicit_swaps': explicit_swaps,
        'effective_swaps': effective_swaps,
        'extra_2q_gates': extra_2q,
        'total_ops': sum(transpiled_ops.values()),
        'logical_ops': dict(logical_ops),
        'transpiled_ops': dict(transpiled_ops)
    }


def compare_processors(logical_qc: QuantumCircuit,
                       processor_A: Dict[str, Any],
                       processor_B: Dict[str, Any],
                       optimization_level: int = 1,
                       seed_transpiler: Optional[int] = 42) -> Dict[str, Any]:
    """
    Compare circuit transpilation and routing across Processor A and Processor B.
    """
    transpiled_A = transpile_circuit(logical_qc, processor_A, optimization_level, seed_transpiler)
    transpiled_B = transpile_circuit(logical_qc, processor_B, optimization_level, seed_transpiler)

    overhead_A = analyze_routing_overhead(logical_qc, transpiled_A)
    overhead_B = analyze_routing_overhead(logical_qc, transpiled_B)

    return {
        'processor_A': {
            'name': processor_A['name'],
            'transpiled_depth': transpiled_A.depth(),
            'swap_count': overhead_A['effective_swaps'],
            'two_qubit_gates': overhead_A['transpiled_2q_gates'],
            'total_ops': overhead_A['total_ops'],
            'operations': overhead_A['transpiled_ops'],
            'transpiled_circuit': transpiled_A
        },
        'processor_B': {
            'name': processor_B['name'],
            'transpiled_depth': transpiled_B.depth(),
            'swap_count': overhead_B['effective_swaps'],
            'two_qubit_gates': overhead_B['transpiled_2q_gates'],
            'total_ops': overhead_B['total_ops'],
            'operations': overhead_B['transpiled_ops'],
            'transpiled_circuit': transpiled_B
        },
        'comparison': {
            'depth_diff': overhead_B['transpiled_depth'] - overhead_A['transpiled_depth'],
            'swap_diff': overhead_B['effective_swaps'] - overhead_A['effective_swaps'],
            'gate_2q_diff': overhead_B['transpiled_2q_gates'] - overhead_A['transpiled_2q_gates'],
            'total_ops_diff': overhead_B['total_ops'] - overhead_A['total_ops']
        }
    }
