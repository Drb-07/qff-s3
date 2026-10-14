"""
Routing Module - Qiskit Fall Fest 2026 Phase 1
Handles transpilation, processor routing, and exact SWAP / 2Q gate overhead analysis.
"""

from typing import Dict, Any, Optional, Tuple
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap


def create_coupling_map(processor: Dict[str, Any]) -> CouplingMap:
    """Create Qiskit CouplingMap from processor configuration."""
    return CouplingMap(processor['coupling_map'])


def transpile_circuit(qc: QuantumCircuit,
                      processor: Dict[str, Any],
                      optimization_level: int = 1,
                      seed_transpiler: Optional[int] = 42) -> QuantumCircuit:
    """
    Transpile circuit to target processor coupling map and strictly decomposed basis gates:
    ['rz', 'sx', 'x', 'cx'].
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


def transpile_with_routing_analysis(qc: QuantumCircuit,
                                    processor: Dict[str, Any],
                                    optimization_level: int = 1,
                                    seed_transpiler: Optional[int] = 42) -> Tuple[QuantumCircuit, Dict[str, Any]]:
    """
    Transpile circuit and accurately measure real routing SWAP gates.

    Step 1: Transpile to coupling map with basis containing 'swap' to identify real SWAPs inserted by routing.
    Step 2: Decompose SWAP gates into 3 CX gates to produce physical hardware basis ['rz', 'sx', 'x', 'cx'].
    """
    coupling_map = create_coupling_map(processor)

    # Transpile preserving SWAP gates to directly count routed SWAPs
    intermediate_basis = ['rz', 'sx', 'x', 'cx', 'swap']
    t_routed = transpile(
        qc,
        coupling_map=coupling_map,
        basis_gates=intermediate_basis,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler
    )

    exact_swaps = t_routed.count_ops().get('swap', 0)

    # Decompose SWAPs into basis CX gates for hardware execution
    if exact_swaps > 0:
        t_hardware = transpile(
            t_routed,
            coupling_map=coupling_map,
            basis_gates=processor['basis_gates'],
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler
        )
    else:
        t_hardware = t_routed

    logical_ops = qc.count_ops()
    hw_ops = t_hardware.count_ops()
    two_q_names = {'cx', 'cz', 'ecr', 'swap', 'rxx', 'ryy', 'rzz'}

    logical_2q = sum(logical_ops.get(g, 0) for g in two_q_names)
    hw_2q = sum(hw_ops.get(g, 0) for g in two_q_names)

    routing_info = {
        'logical_depth': qc.depth(),
        'transpiled_depth': t_hardware.depth(),
        'depth_increase': t_hardware.depth() - qc.depth(),
        'logical_2q_gates': logical_2q,
        'transpiled_2q_gates': hw_2q,
        'exact_swaps': exact_swaps,
        'effective_swaps': exact_swaps,
        'swap_count': exact_swaps,
        'extra_2q_gates': max(0, hw_2q - logical_2q),
        'total_ops': sum(hw_ops.values()),
        'logical_ops': dict(logical_ops),
        'transpiled_ops': dict(hw_ops)
    }

    return t_hardware, routing_info


def analyze_routing_overhead(logical_qc: QuantumCircuit,
                             transpiled_qc: QuantumCircuit) -> Dict[str, Any]:
    """Analyze routing overhead between logical and transpiled circuits."""
    logical_ops = logical_qc.count_ops()
    transpiled_ops = transpiled_qc.count_ops()
    two_q_names = {'cx', 'cz', 'ecr', 'swap', 'rxx', 'ryy', 'rzz'}

    logical_2q = sum(logical_ops.get(g, 0) for g in two_q_names)
    transpiled_2q = sum(transpiled_ops.get(g, 0) for g in two_q_names)
    explicit_swaps = transpiled_ops.get('swap', 0)

    return {
        'logical_depth': logical_qc.depth(),
        'transpiled_depth': transpiled_qc.depth(),
        'depth_increase': transpiled_qc.depth() - logical_qc.depth(),
        'logical_2q_gates': logical_2q,
        'transpiled_2q_gates': transpiled_2q,
        'explicit_swaps': explicit_swaps,
        'exact_swaps': explicit_swaps,
        'effective_swaps': explicit_swaps,
        'swap_count': explicit_swaps,
        'extra_2q_gates': max(0, transpiled_2q - logical_2q),
        'total_ops': sum(transpiled_ops.values()),
        'logical_ops': dict(logical_ops),
        'transpiled_ops': dict(transpiled_ops)
    }


def compare_processors(qc: QuantumCircuit,
                       processor_A: Dict[str, Any],
                       processor_B: Dict[str, Any],
                       optimization_level: int = 1,
                       seed: Optional[int] = 42) -> Dict[str, Any]:
    """Compare transpilation of a logical circuit on two processors."""
    tA, routA = transpile_with_routing_analysis(qc, processor_A, optimization_level, seed)
    tB, routB = transpile_with_routing_analysis(qc, processor_B, optimization_level, seed)
    return {
        'processor_A': routA,
        'processor_B': routB,
        'circuit_A': tA,
        'circuit_B': tB
    }
