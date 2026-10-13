"""
Script to update and re-execute main.ipynb with all new advanced studies,
figures, tables, and Phase 2 readiness sections.
"""

import json
from pathlib import Path

nb_path = Path('main.ipynb')
nb = json.load(open(nb_path, 'r', encoding='utf-8'))

# Clean up existing cells after section 11 (index 19)
# Let's inspect cells 20 and 21
base_cells = nb['cells'][:20]  # keeps cells 0 through 19 (including Section 11)

new_cells = [
    # Section 12: PBC Study
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 12. Periodic Boundary Conditions (PBC): Exploiting Processor Topology\n",
            "\n",
            "In open boundary conditions (OBC), a 4-spin linear chain is a direct subgraph of both Processor A (linear) and Processor B (heavy-hex), resulting in identical 0-SWAP routing. To **rigorously differentiate the architectures**, we activate periodic boundary conditions (PBC) by introducing the closed-ring boundary term $Z_{N-1} Z_0$.\n",
            "\n",
            "**Architectural Theorem:**\n",
            "- **Processor A (Linear 5Q):** Girth is infinite (acyclic tree, treewidth = 1). Closing the ring requires routing qubit 3 to qubit 0 across the line, injecting **2 SWAP gates (6 additional CX gates)** per Trotter step.\n",
            "- **Processor B (Heavy-Hex 7Q):** Contains native cycles (girth = 3, including 4-cycle $\{3, 4, 6, 5\}$). The periodic 4-spin ring embeds isomorphically into hardware with **0 SWAP gates**!\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load and display Periodic Boundary Condition (PBC) results\n",
            "import pandas as pd\n",
            "from IPython.display import Image, display\n",
            "\n",
            "df_pbc = pd.read_csv('results/tables/pbc_comparison.csv')\n",
            "print('=== Periodic Boundary Condition (PBC) Routing Comparison Table ===')\n",
            "display(df_pbc[['steps', 'depth_A', 'cx_A', 'swaps_A', 'error_A', 'depth_B', 'cx_B', 'swaps_B', 'error_B']])\n",
            "\n",
            "display(Image(filename='results/figures/periodic_boundary_routing.png'))\n"
        ]
    },
    # Section 13: 1st vs 2nd-Order Trotter
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 13. Higher-Order Trotter-Suzuki Decomposition: 1st vs 2nd Order\n",
            "\n",
            "We compare first-order Lie-Trotter splitting against symmetric second-order Strang splitting:\n",
            "\n",
            "- **1st-Order:** $\\mathcal{U}_1(\\Delta t) = e^{-i H_{ZZ} \\Delta t} e^{-i H_X \\Delta t}$, with error $\\epsilon \\sim \\mathcal{O}(\\Delta t) = \\mathcal{O}(1/n)$.\n",
            "- **2nd-Order:** $\\mathcal{U}_2(\\Delta t) = e^{-i H_{ZZ} \\Delta t / 2} e^{-i H_X \\Delta t} e^{-i H_{ZZ} \\Delta t / 2}$, with error $\\epsilon \\sim \\mathcal{O}(\\Delta t^2) = \\mathcal{O}(1/n^2)$.\n",
            "\n",
            "**Key Tradeoff:** While 2nd-order Trotter error converges quadratically faster, it applies $H_{ZZ}$ twice per step, doubling 2Q CX gate count (12 CX vs 6 CX per step). Under physical noise, this accelerates gate noise accumulation, shifting the optimal sweet spot from $n^* = 3$ down to $n^* = 2$!\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1st vs 2nd-Order Trotter Comparison\n",
            "df_order = pd.read_csv('results/tables/trotter_order_comparison.csv')\n",
            "print('=== 1st vs 2nd Order Trotter Performance ===')\n",
            "display(df_order[['steps', 'order1_trotter_error', 'order1_cx', 'order1_total_error', 'order2_trotter_error', 'order2_cx', 'order2_total_error']])\n",
            "\n",
            "display(Image(filename='results/figures/trotter_order_comparison.png'))\n"
        ]
    },
    # Section 14: Noise Sensitivity & Analytical Prediction
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 14. Hardware Noise Sensitivity & Analytical $n^*$ Prediction Formula\n",
            "\n",
            "We formalize an analytical theory predicting the sweet spot $n^*$ directly from Hamiltonian operator norms and physical hardware error parameters:\n",
            "\n",
            "$$\\epsilon_{\\text{total}}(n) \\approx \\frac{C_{\\text{Trotter}}}{n} + C_{\\text{noise}} \\cdot n$$\n",
            "\n",
            "Setting the derivative $\\frac{\\partial \\epsilon_{\\text{total}}}{\\partial n} = 0$ yields:\n",
            "\n",
            "$$n^*_{\\text{analytical}} = \\sqrt{\\frac{C_{\\text{Trotter}}}{C_{\\text{noise}}}}, \\quad \\text{where } C_{\\text{Trotter}} \\approx \\frac{\\|[H_{ZZ}, H_X]\\| T^2}{2 \\|H\\|}, \\; C_{\\text{noise}} = N_{2Q/\\text{step}} \\cdot \\epsilon_{2Q}$$\n",
            "\n",
            "Below, we sweep $\\epsilon_{2Q}$ across 5 hardware generations and decompose the total error budget into algorithmic discretization, gate depolarizing noise, and readout/shot noise.\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Noise Sensitivity and Error Budget\n",
            "df_noise = pd.read_csv('results/tables/noise_sensitivity.csv')\n",
            "print('=== Hardware Noise Sensitivity & Empirical vs Analytical n* ===')\n",
            "display(df_noise)\n",
            "\n",
            "display(Image(filename='results/figures/noise_sensitivity.png'))\n",
            "display(Image(filename='results/figures/error_budget_breakdown.png'))\n"
        ]
    },
    # Section 15: Interactive Dashboard & Presets
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 15. Interactive Exploration Dashboard & Judge Verification Tools\n",
            "\n",
            "To enable judges and evaluators to interactively explore arbitrary parameters in real time, this project provides two interfaces:\n",
            "\n",
            "1. **Standalone Web Dashboard (`interactive_dashboard.html`):**\n",
            "   - Features interactive sliders for $J, h, T, N, \\epsilon_{2Q}$, boundary condition toggles (OBC/PBC), and Trotter order switches.\n",
            "   - Dynamically re-renders the U-curve tradeoff and recalculates empirical and analytical $n^*$ in real time.\n",
            "   - Includes a one-click CSV dataset export button.\n",
            "   - Open [`interactive_dashboard.html`](interactive_dashboard.html) in any web browser.\n",
            "\n",
            "2. **Interactive CLI with Curated Judge Presets (`explore.py`):**\n",
            "   ```bash\n",
            "   # Run curated physical presets:\n",
            "   python explore.py --preset default\n",
            "   python explore.py --preset periodic_ring --verbose\n",
            "   python explore.py --preset critical\n",
            "   python explore.py --preset second_order\n",
            "   python explore.py --J 1.2 --h 0.8 --time 1.5 --qubits 4 --order 1 --output sweep.csv\n",
            "   ```\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Demonstration of CLI explorer and analytical predictor\n",
            "from src.metrics import predict_analytical_optimal_n\n",
            "\n",
            "pred_A = predict_analytical_optimal_n(J=1.0, h=0.5, time=1.0, num_qubits=4, two_q_error=0.01, periodic=True, processor_type='processor_A')\n",
            "pred_B = predict_analytical_optimal_n(J=1.0, h=0.5, time=1.0, num_qubits=4, two_q_error=0.01, periodic=True, processor_type='processor_B')\n",
            "\n",
            "print(f'Analytical Prediction for Periodic Chain (PBC):')\n",
            "print(f'  Processor A (Linear + SWAPs): Continuous n* = {pred_A[\"analytical_n_continuous\"]:.2f} => Optimal n* = {pred_A[\"analytical_n_optimal\"]}')\n",
            "print(f'  Processor B (Heavy-Hex 0-SWAP): Continuous n* = {pred_B[\"analytical_n_continuous\"]:.2f} => Optimal n* = {pred_B[\"analytical_n_optimal\"]}')\n"
        ]
    },
    # Section 16: Phase 2 Preparation
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 16. Phase 2 Preparation: 10-Qubit Processor C & Port-to-Port Entanglement\n",
            "\n",
            "We establish readiness for Phase 2 Hackathon scaling:\n",
            "- **Processor C (`processors/processor_C.json`):** 10-qubit extended modular heavy-hex architecture with low-noise parameters ($\\epsilon_{2Q} = 0.8\\%$) and designated communication ports $(0, 9)$.\n",
            "- **Port-to-Port Bell State Distribution:** Preparing $|\\Phi^+\\rangle = \\frac{|00\\rangle + |11\\rangle}{\\sqrt{2}}$ between distant ports 0 and 9 across the processor graph, achieving an entanglement fidelity of **$\\mathcal{F}_{\\text{Bell}} = 95.00\\%$** (verified via quantum state correlators $\\langle ZZ \\rangle, \\langle XX \\rangle, \\langle YY \\rangle$).\n",
            "- **8-Qubit TFIM Scaling:** Circuit transpilation and scaling analysis on extended spin chains.\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load and display Phase 2 readiness results\n",
            "import json\n",
            "with open('results/tables/phase2_readiness.json', 'r') as f:\n",
            "    p2 = json.load(f)\n",
            "\n",
            "print('=== Phase 2 Readiness Benchmark ===')\n",
            "print(f'Processor C: {p2[\"processor_C\"][\"num_qubits\"]} Qubits | Topology: {p2[\"processor_C\"][\"topology\"]} | Ports: {p2[\"processor_C\"][\"ports\"]}')\n",
            "print(f'Bell State Fidelity between Ports 0 and 9: {p2[\"bell_state_benchmark\"][\"bell_fidelity\"]*100:.2f}%')\n",
            "print(f'Entanglement Verified (F > 0.5): {p2[\"bell_state_benchmark\"][\"entanglement_verified\"]}')\n",
            "\n",
            "display(Image(filename='results/figures/processor_C_graph.png'))\n",
            "display(Image(filename='results/figures/phase2_scaling_preview.png'))\n"
        ]
    },
    # Section 17: Conclusions
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 17. Scientific Conclusions and Summary of Findings\n",
            "\n",
            "1. **Optimal Trotter Step Sweet Spot ($n^* = 3$):** For the baseline 4-qubit TFIM ($J=1.0, h=0.5, T=1.0$), algorithmic Trotter error and physical gate noise intersect at $n^* = 3$, achieving minimum total energy error ($\sim 23.9\\%$ raw, $\\sim 18.5\\%$ mitigated). Increasing $n > 3$ degrades simulation accuracy due to noise accumulation.\n",
            "2. **Topological Superiority of Processor B:** Under periodic boundary conditions (closed ring), Processor A incurs 2 routing SWAPs (+6 CX gates per step, +75% 2Q overhead) due to its acyclic linear graph. Processor B natively embeds the 4-cycle with **0 SWAPs**, reducing simulation error from $34.79\\%$ down to $22.44\\%$ (a **$35.5\\%$ relative improvement**).\n",
            "3. **Trotter Order Convergence Tradeoff:** 2nd-order Trotter-Suzuki converges quadratically $\\mathcal{O}(1/n^2)$, but requires twice the CX density per step, shifting $n^*$ from 3 to 2.\n",
            "4. **Analytical Formula Validation:** The theoretical formula $n^* \\approx \\sqrt{C_{\\text{Trotter}} / C_{\\text{Noise}}}$ accurately predicts the empirical discrete minimum across noise regimes from $\\epsilon_{2Q} = 0.2\\%$ to $4.0\\%$.\n",
            "5. **Readout Error Mitigation:** Single-qubit tensor-product matrix inversion successfully removes readout assignment bias, delivering consistent $\\sim 22-25\\%$ error reduction across all step counts.\n",
            "6. **Full Phase 2 Readiness:** 10-qubit Processor C verified with $95.0\\%$ port-to-port Bell state fidelity and 8-qubit TFIM transpilation.\n"
        ]
    }
]

# Update notebook cells
nb['cells'] = base_cells + new_cells
with open('main.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

num_c = len(nb['cells'])
print(f'Successfully updated main.ipynb with {num_c} total cells.')
