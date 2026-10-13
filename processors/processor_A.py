"""
Processor A Definition Script
Generates processor_A.json for the QFF 2026 Phase 1 Hackathon

Processor A: 5-qubit linear topology with synthetic noise model
"""

import json
from pathlib import Path

def create_processor_A():
    """
    Create Processor A configuration
    - 5 qubits in linear topology: 0—1—2—3—4
    - Synthetic noise model
    - Basis gates: rz, sx, x, cx
    - Ports at qubits 0 and 4
    """

    processor_A = {
        "name": "processor_A",
        "description": "5-qubit linear processor with synthetic noise model",
        "num_qubits": 5,
        "coupling_map": [
            [0, 1],
            [1, 0],
            [1, 2],
            [2, 1],
            [2, 3],
            [3, 2],
            [3, 4],
            [4, 3]
        ],
        "basis_gates": ["rz", "sx", "x", "cx"],
        "ports": [0, 4],
        "topology": "linear",
        "noise": {
            "type": "synthetic",
            "model": "depolarizing",
            "single_qubit_error": 0.001,
            "two_qubit_error": 0.01,
            "readout_error": 0.02,
            "description": "Synthetic noise model for baseline comparison"
        },
        "metadata": {
            "phase": 1,
            "processor_type": "baseline",
            "graph_type": "fixed_linear",
            "created_for": "QFF_2026_Phase_1"
        }
    }

    return processor_A


def save_processor_json(processor_config, filename="processor_A.json"):
    """Save processor configuration to JSON file"""
    output_path = Path(__file__).parent / filename

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processor_config, f, indent=4, ensure_ascii=False)

    print(f"[OK] Processor A configuration saved to: {output_path}")
    print(f"  - Qubits: {processor_config['num_qubits']}")
    print(f"  - Topology: {processor_config['topology']}")
    print(f"  - Coupling map edges: {len(processor_config['coupling_map'])}")
    print(f"  - Basis gates: {', '.join(processor_config['basis_gates'])}")
    print(f"  - Ports: {processor_config['ports']}")

    return output_path


if __name__ == "__main__":
    # Generate Processor A configuration
    processor_A_config = create_processor_A()

    # Save to JSON
    json_path = save_processor_json(processor_A_config)

    # Verify the file was created
    if json_path.exists():
        print(f"\n[SUCCESS] processor_A.json is ready for Phase 1 benchmarking.")
    else:
        print(f"\n[ERROR] Failed to create processor_A.json")
