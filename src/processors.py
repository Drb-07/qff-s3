"""
Processors Module - Qiskit Fall Fest 2026 Phase 1
Load and configure Processor A (5Q T-shaped tree) and Processor B (9Q Slide 4 Graph)
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def load_processor(path: Any) -> Dict[str, Any]:
    """Load processor configuration from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_processor_graph(processor: Dict[str, Any]) -> nx.Graph:
    """Create a NetworkX graph from processor coupling map."""
    G = nx.Graph()
    G.add_nodes_from(range(processor['num_qubits']))
    edges = set()
    for edge in processor['coupling_map']:
        edges.add(tuple(sorted(edge)))
    G.add_edges_from(edges)
    return G


def visualize_processor(processor: Dict[str, Any], save_path: Optional[str] = None):
    """
    Visualize processor topology using its defined coordinate layout.
    """
    G = get_processor_graph(processor)
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=130)

    port_nodes = set(processor.get('ports', []))
    node_colors = ['#2E7D32' if n in port_nodes else '#1976D2' for n in G.nodes()]

    # Use geometric layout if defined in JSON, else spring_layout
    layout_data = processor.get('layout')
    if layout_data:
        pos = {int(k): tuple(v) for k, v in layout_data.items()}
    else:
        pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_edges(G, pos, ax=ax, width=2.5, edge_color='#64748B')
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=900,
                           edgecolors='#0F172A', linewidths=1.8)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=13, font_color='white',
                            font_weight='bold')

    ax.set_title(f"{processor['name']} — {processor.get('topology', '').replace('_', ' ').title()}\n"
                 f"({processor['num_qubits']} Physical Qubits | Designated Ports: {processor.get('ports')})",
                 fontsize=12, fontweight='bold', pad=12)
    ax.axis('off')

    legend_elements = [
        Patch(facecolor='#2E7D32', edgecolor='#0F172A', label='Designated Communication Port'),
        Patch(facecolor='#1976D2', edgecolor='#0F172A', label='Internal Processor Qubit')
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9, fontsize=9)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved processor graph to: {save_path}")

    return fig


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
