"""
Processor B Definition Script
Generates processor_B.json for the QFF 2026 Phase 1 Hackathon

Processor B: 7-qubit heavy-hex-inspired topology
"""

import json
from pathlib import Path

def create_processor_B():
    """
    Create Processor B configuration
    - 7 qubits in heavy-hex-inspired topology
    - Contemporary compilation baseline
    - More connected than Processor A
    - Basis gates: rz, sx, x, cx
    """

    # Heavy-hex inspired coupling map for 7 qubits
    # More connected topology than linear
    processor_B = {
        "name": "processor_B",
        "description": "7-qubit heavy-hex-inspired processor with contemporary compilation baseline",
        "num_qubits": 7,
        "coupling_map": [
            # Center qubit (3) connects to surrounding qubits
            [0, 1],
            [1, 0],
            [1, 2],
            [2, 1],
            [1, 3],  # Center connections
            [3, 1],
            [2, 3],
            [3, 2],
            [3, 4],
            [4, 3],
            [3, 5],
            [5, 3],
            [4, 6],
            [6, 4],
            [5, 6],
            [6, 5]
        ],
        "basis_gates": ["rz", "sx", "x", "cx"],
        "ports": [0, 6],
        "topology": "heavy_hex_inspired",
        "noise": {
            "type": "contemporary",
            "model": "depolarizing",
            "single_qubit_error": 0.001,
            "two_qubit_error": 0.01,
            "readout_error": 0.02,
            "description": "Contemporary noise model for compilation baseline"
        },
        "metadata": {
            "phase": 1,
            "processor_type": "contemporary_baseline",
            "graph_type": "heavy_hex_inspired",
            "connectivity": "higher_than_processor_A",
            "created_for": "QFF_2026_Phase_1"
        }
    }

    return processor_B


def save_processor_json(processor_config, filename="processor_B.json"):
    """Save processor configuration to JSON file"""
    output_path = Path(__file__).parent / filename

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processor_config, f, indent=4, ensure_ascii=False)

    print(f"[OK] Processor B configuration saved to: {output_path}")
    print(f"  - Qubits: {processor_config['num_qubits']}")
    print(f"  - Topology: {processor_config['topology']}")
    print(f"  - Coupling map edges: {len(processor_config['coupling_map'])}")
    print(f"  - Basis gates: {', '.join(processor_config['basis_gates'])}")
    print(f"  - Ports: {processor_config['ports']}")

    return output_path


if __name__ == "__main__":
    # Generate Processor B configuration
    processor_B_config = create_processor_B()

    # Save to JSON
    json_path = save_processor_json(processor_B_config)

    # Verify the file was created
    if json_path.exists():
        print(f"\n[SUCCESS] processor_B.json is ready for Phase 1 benchmarking.")
    else:
        print(f"\n[ERROR] Failed to create processor_B.json")
