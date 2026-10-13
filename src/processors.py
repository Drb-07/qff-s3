"""
Processors Module
Load and configure Processor A and Processor B definitions
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import networkx as nx
import matplotlib.pyplot as plt


def load_processor(path: str) -> Dict[str, Any]:
    """
    Load processor configuration from JSON file.

    Args:
        path: Path to processor JSON file

    Returns:
        Dictionary with processor configuration
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_processor_graph(processor: Dict[str, Any]) -> nx.Graph:
    """
    Create a NetworkX graph from processor coupling map.

    Args:
        processor: Processor configuration dictionary

    Returns:
        NetworkX Graph representing the processor topology
    """
    G = nx.Graph()
    G.add_nodes_from(range(processor['num_qubits']))

    # Add edges from coupling map (as undirected graph for visualization)
    edges = set()
    for edge in processor['coupling_map']:
        edges.add(tuple(sorted(edge)))
    G.add_edges_from(edges)

    return G


def visualize_processor(processor: Dict[str, Any], save_path: Optional[str] = None):
    """
    Visualize processor topology.

    Args:
        processor: Processor configuration dictionary
        save_path: Optional path to save the figure
    """
    G = get_processor_graph(processor)

    plt.figure(figsize=(8, 6))

    # Color ports differently
    port_nodes = processor.get('ports', [])
    node_colors = ['#4CAF50' if n in port_nodes else '#2196F3' for n in G.nodes()]

    # Use spring layout for better visualization
    pos = nx.spring_layout(G, seed=42)

    nx.draw(G, pos, with_labels=True, node_color=node_colors,
            node_size=800, font_size=12, font_weight='bold',
            edge_color='gray', width=2)

    plt.title(f"{processor['name']} - {processor['topology']} topology\n"
              f"({processor['num_qubits']} qubits)", fontsize=14)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4CAF50', label='Port qubits'),
        Patch(facecolor='#2196F3', label='Non-port qubits')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved processor graph to: {save_path}")

    plt.show()


def print_processor_info(processor: Dict[str, Any]):
    """Print formatted processor information."""
    print(f"\n{'='*60}")
    print(f"Processor: {processor['name']}")
    print(f"{'='*60}")
    print(f"  Description: {processor.get('description', 'N/A')}")
    print(f"  Number of qubits: {processor['num_qubits']}")
    print(f"  Topology: {processor.get('topology', 'N/A')}")
    print(f"  Basis gates: {', '.join(processor['basis_gates'])}")
    print(f"  Ports: {processor['ports']}")
    print(f"  Coupling map edges: {len(processor['coupling_map'])}")

    if 'noise' in processor:
        noise = processor['noise']
        print(f"\n  Noise Model:")
        print(f"    Type: {noise.get('type', 'N/A')}")
        print(f"    Single-qubit error: {noise.get('single_qubit_error', 'N/A')}")
        print(f"    Two-qubit error: {noise.get('two_qubit_error', 'N/A')}")
        print(f"    Readout error: {noise.get('readout_error', 'N/A')}")


# Example usage
if __name__ == "__main__":
    # Load processors using absolute path relative to this file
    base_dir = Path(__file__).parent.parent
    processor_A = load_processor(base_dir / "processors" / "processor_A.json")
    processor_B = load_processor(base_dir / "processors" / "processor_B.json")

    print_processor_info(processor_A)
    print_processor_info(processor_B)
