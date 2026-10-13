"""
Visualization Module
Qiskit Fall Fest 2026 - Phase 1 Hackathon

Generates publication-quality scientific figures:
- Processor coupling topology graphs with designated ports
- The Trotter Error vs Noise Error U-curve (Optimal n* identification)
- Transpiled depth & 2Q gate scaling comparison
- Multi-panel architecture tradeoff analysis
- Error mitigation impact plot
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path


def plot_processor_graph(processor: Dict[str, Any],
                         title: Optional[str] = None,
                         save_path: Optional[str] = None) -> plt.Figure:
    """Plot processor topology graph highlighting ports and coupling links."""
    G = nx.Graph()
    G.add_nodes_from(range(processor['num_qubits']))

    edges = set()
    for edge in processor['coupling_map']:
        edges.add(tuple(sorted(edge)))
    G.add_edges_from(edges)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

    port_nodes = set(processor.get('ports', []))
    node_colors = ['#4CAF50' if n in port_nodes else '#1976D2' for n in G.nodes()]

    topology = processor.get('topology', 'default')
    if topology == 'linear':
        pos = {i: (i * 1.5, 0.0) for i in range(processor['num_qubits'])}
    elif 'heavy_hex' in topology:
        # Custom clean coordinate layout for heavy-hex-inspired 7-qubit
        pos = {
            0: (0.0, 1.0),
            1: (1.0, 1.5),
            2: (1.0, 0.5),
            3: (2.0, 1.0),
            4: (3.0, 1.5),
            5: (3.0, 0.5),
            6: (4.0, 1.0)
        }
        # Fallback if nodes don't match
        if len(set(pos.keys()).intersection(set(G.nodes()))) != len(G.nodes()):
            pos = nx.spring_layout(G, seed=42)
    else:
        pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=900, edgecolors='black', linewidths=1.5)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#555555', width=2.2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_color='white', font_weight='bold')

    if title is None:
        title = f"{processor['name'].upper()} ({processor['num_qubits']} Qubits, {topology.replace('_', ' ').title()})"
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4CAF50', edgecolor='black', label='Designated Ports'),
        Patch(facecolor='#1976D2', edgecolor='black', label='Internal Qubits')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    ax.axis('off')

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


def plot_trotter_error_tradeoff(steps: List[int],
                                trotter_errors: List[float],
                                noisy_errors_A: List[float],
                                noisy_errors_B: List[float],
                                total_errors_A: List[float],
                                total_errors_B: List[float],
                                opt_n_A: int,
                                opt_n_B: int,
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    The centerpiece scientific plot:
    Demonstrates the competition between Trotter algorithmic error and hardware noise error,
    revealing optimal Trotter steps n*_A and n*_B.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)

    # 1. Trotter algorithmic error (black dashed)
    ax.plot(steps, trotter_errors, 'k--', linewidth=2.2, label=r'Algorithmic Trotter Error $\epsilon_{\mathrm{Trotter}} \sim \mathcal{O}(\Delta t)$', zorder=3)

    # 2. Total error Processor A
    ax.plot(steps, total_errors_A, color='#D32F2F', marker='o', linewidth=2.5, label='Total Error: Processor A (Linear)', zorder=4)

    # 3. Total error Processor B
    ax.plot(steps, total_errors_B, color='#1976D2', marker='s', linewidth=2.5, label='Total Error: Processor B (Heavy-Hex)', zorder=4)

    # Annotate optimal points
    idx_A = steps.index(opt_n_A) if opt_n_A in steps else 0
    idx_B = steps.index(opt_n_B) if opt_n_B in steps else 0

    ax.scatter([opt_n_A], [total_errors_A[idx_A]], color='#D32F2F', s=200, edgecolor='black', linewidth=2, zorder=5)
    ax.annotate(f'$n_A^* = {opt_n_A}$\nErr: {total_errors_A[idx_A]:.3f}',
                xy=(opt_n_A, total_errors_A[idx_A]),
                xytext=(opt_n_A - 0.5, total_errors_A[idx_A] + 0.08),
                arrowprops=dict(facecolor='#D32F2F', shrink=0.08, width=1.5, headwidth=6),
                fontweight='bold', color='#B71C1C', fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#D32F2F', alpha=0.9))

    ax.scatter([opt_n_B], [total_errors_B[idx_B]], color='#1976D2', s=200, edgecolor='black', linewidth=2, zorder=5)
    ax.annotate(f'$n_B^* = {opt_n_B}$\nErr: {total_errors_B[idx_B]:.3f}',
                xy=(opt_n_B, total_errors_B[idx_B]),
                xytext=(opt_n_B + 0.8, total_errors_B[idx_B] - 0.08),
                arrowprops=dict(facecolor='#1976D2', shrink=0.08, width=1.5, headwidth=6),
                fontweight='bold', color='#0D47A1', fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor='#1976D2', alpha=0.9))

    ax.set_xlabel('Trotter Step Count ($n$)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Simulation Error $|E - E_{\mathrm{exact}}| / |E_{\mathrm{exact}}|$', fontsize=12, fontweight='bold')
    ax.set_title('Trotter Error vs Hardware Noise Tradeoff: Finding Optimal Step Count $n^*$', fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(steps)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10, loc='upper right')

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


def plot_circuit_scaling(steps: List[int],
                         depths_A: List[int],
                         depths_B: List[int],
                         gates_2q_A: List[int],
                         gates_2q_B: List[int],
                         save_path: Optional[str] = None) -> plt.Figure:
    """Plot circuit depth and two-qubit gate scaling with Trotter steps."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # Depth scaling
    ax1.plot(steps, depths_A, 'o-', color='#D32F2F', linewidth=2, label='Processor A (Linear)')
    ax1.plot(steps, depths_B, 's-', color='#1976D2', linewidth=2, label='Processor B (Heavy-Hex)')
    ax1.set_xlabel('Trotter Steps ($n$)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Transpiled Circuit Depth', fontsize=11, fontweight='bold')
    ax1.set_title('Circuit Depth Scaling', fontsize=12, fontweight='bold')
    ax1.set_xticks(steps)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()

    # 2Q gates scaling
    ax2.plot(steps, gates_2q_A, 'o-', color='#D32F2F', linewidth=2, label='Processor A (Linear)')
    ax2.plot(steps, gates_2q_B, 's-', color='#1976D2', linewidth=2, label='Processor B (Heavy-Hex)')
    ax2.set_xlabel('Trotter Steps ($n$)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Two-Qubit (CX) Gates', fontsize=11, fontweight='bold')
    ax2.set_title('2Q Gate Count Scaling', fontsize=12, fontweight='bold')
    ax2.set_xticks(steps)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


def plot_architecture_tradeoffs(summary_data: Dict[str, Any],
                                save_path: Optional[str] = None) -> plt.Figure:
    """Multi-panel bar chart comparing metrics across architectures."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), dpi=150)

    metrics_A = summary_data['processor_A']
    metrics_B = summary_data['processor_B']

    categories = ['Processor A (Linear)', 'Processor B (Heavy-Hex)']
    colors = ['#D32F2F', '#1976D2']

    # 1. Depth
    ax1 = axes[0, 0]
    ax1.bar(categories, [metrics_A['depth'], metrics_B['depth']], color=colors, width=0.55)
    ax1.set_title('Circuit Depth', fontweight='bold')
    ax1.grid(axis='y', linestyle=':', alpha=0.6)

    # 2. 2Q Gates
    ax2 = axes[0, 1]
    ax2.bar(categories, [metrics_A['two_qubit_ops'], metrics_B['two_qubit_ops']], color=colors, width=0.55)
    ax2.set_title('Two-Qubit (CX) Gates', fontweight='bold')
    ax2.grid(axis='y', linestyle=':', alpha=0.6)

    # 3. Effective SWAPs
    ax3 = axes[1, 0]
    ax3.bar(categories, [metrics_A.get('swap_count', 0), metrics_B.get('swap_count', 0)], color=colors, width=0.55)
    ax3.set_title('Routing SWAP Overhead', fontweight='bold')
    ax3.grid(axis='y', linestyle=':', alpha=0.6)

    # 4. Energy Error
    ax4 = axes[1, 1]
    err_A = metrics_A.get('energy_error', 0.0)
    err_B = metrics_B.get('energy_error', 0.0)
    ax4.bar(categories, [err_A, err_B], color=colors, width=0.55)
    ax4.set_title('Energy Relative Error', fontweight='bold')
    ax4.grid(axis='y', linestyle=':', alpha=0.6)

    plt.suptitle('Hardware Architecture Benchmark: Processor A vs Processor B', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


def plot_error_mitigation_impact(steps: List[int],
                                 raw_errors_B: List[float],
                                 mitigated_errors_B: List[float],
                                 save_path: Optional[str] = None) -> plt.Figure:
    """Plot comparing unmitigated vs readout-mitigated Trotter simulation."""
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
    ax.plot(steps, raw_errors_B, 'o--', color='#E65100', linewidth=2, label='Unmitigated Noisy (Processor B)')
    ax.plot(steps, mitigated_errors_B, 's-', color='#2E7D32', linewidth=2.2, label='Readout Mitigated (Processor B)')

    ax.set_xlabel('Trotter Steps ($n$)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Total Simulation Error', fontsize=11, fontweight='bold')
    ax.set_title('Impact of Error Mitigation on Simulation Accuracy (Section 16)', fontsize=12, fontweight='bold')
    ax.set_xticks(steps)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig
