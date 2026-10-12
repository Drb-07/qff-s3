# S3 — Noise in Trotter Simulation

**Qiskit Fall Fest 2026 | SRM University - AP & quanTA / University of Saskatchewan**  
**Phase 1 Online Hackathon Submission (Quantum Simulation Track)**

[![Qiskit 2.4](https://img.shields.io/badge/Qiskit-2.4.1-613394.svg)](https://qiskit.org/)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Tests Passing](https://img.shields.io/badge/Tests-100%25%20Passing-brightgreen.svg)](tests/test_simulation.py)
[![Interactive Dashboard](https://img.shields.io/badge/Dashboard-Interactive%20HTML-blueviolet.svg)](interactive_dashboard.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Executive Summary

To simulate continuous Hamiltonian time evolution $U(t) = \exp(-iHt)$ on digital quantum hardware, time is sliced into $n$ discrete Trotter steps (Trotter-Suzuki product formula). This establishes a fundamental scientific competition between two opposing error mechanisms:
1. **Algorithmic Discretization Error ($\epsilon_{\mathrm{Trotter}} \sim \mathcal{O}(t^2/n)$):** Decreases as step count $n$ rises.
2. **Hardware Noise Error ($\epsilon_{\mathrm{Noise}}$):** Rises monotonically with $n$ as circuit depth and two-qubit gate count accumulate gate infidelities and environmental decoherence.

The net error forms a convex U-curve with a global minimum at an **optimal step count $n^*$**. This project discovers $n^*$, formulates an analytical predictive theory $n^* \approx \sqrt{C_{\mathrm{Trotter}}/C_{\mathrm{Noise}}}$, demonstrates the graph-theoretical routing advantage of heavy-hex topology over linear chains under periodic boundary conditions, and provides interactive tools for judges.

---

## 🎮 Interactive Judge Interfaces

### 1. Standalone Web Dashboard (`interactive_dashboard.html`)
Open [`interactive_dashboard.html`](interactive_dashboard.html) directly in any web browser (Chrome, Edge, Firefox, Safari):
- **Live Sliders:** Adjust $J, h, T, N, \epsilon_{2Q}$ and observe the U-curve dynamically recalculate in real time.
- **Topology & Order Toggles:** Switch between Open Chain (OBC) and Periodic Ring (PBC), or 1st vs 2nd-order Trotter-Suzuki.
- **Dynamic Sweet Spot ($n^*$):** Automatically highlights discrete $n^*$ and displays analytical theoretical predictions.
- **One-Click Export:** Download active sweep tables directly as CSV.

### 2. Interactive CLI with Curated Physical Presets (`explore.py`)
Run the CLI explorer with instant physical presets:
```bash
# Periodic boundary ring: shows Processor A SWAP overhead vs Processor B 0 SWAPs
python explore.py --preset periodic_ring --verbose

# Quantum critical point (maximal non-commutativity h = J)
python explore.py --preset critical

# 2nd-Order Trotter Strang splitting (quadratic error convergence)
python explore.py --preset second_order

# Custom parameters with CSV export
python explore.py --J 1.2 --h 0.8 --time 1.5 --qubits 4 --order 1 --output custom_sweep.csv
```

### 3. Real-Time Streamlit Simulator (`interactive_app.py`)
Launch the real-time Python simulation dashboard powered by Qiskit Aer:
```bash
streamlit run interactive_app.py
```
- Real-time Qiskit noisy simulation on every parameter change.
- Interactive parameter sliders, processor comparison toggles, and animated U-curve updates.
- One-click CSV export of live simulation sweeps.

---

## 🔬 Problem Specification & Key Discoveries

| Attribute | Value |
| :--- | :--- |
| **Track** | Quantum Simulation |
| **Problem Code** | `S3` |
| **Physical Model** | Transverse-Field Ising Model: $H = -J \sum Z_i Z_{i+1} - h \sum X_i$ |
| **Baseline Parameters** | $N = 4$ qubits, $J = 1.0$, $h = 0.5$, $T = 1.0$, $2048$ shots |
| **Exact Ground Truth** | Continuous unitary matrix exponential $U(t) = \exp(-iHt)$, $E_{\mathrm{exact}} = -3.0000$ |
| **Processors** | Processor A (5Q Linear) and Processor B (7Q Heavy-Hex) |
| **Core Sweet Spot** | Optimal step count identified at **$n^* = 3$** (Total error: $23.9\%$, mitigated: $18.5\%$) |
| **Topology Discovery** | Under Periodic Boundary Conditions (PBC), Processor B contains a native 4-cycle and routes with **0 SWAPs**, reducing error by **$35.5\%$** over Processor A! |
| **Analytical Formula** | Verified $n^*_{\mathrm{analytical}} = \sqrt{C_{\mathrm{Trotter}} / C_{\mathrm{Noise}}}$ matching empirical simulation. |
| **Phase 2 Readiness** | 10-qubit Processor C verified with **$95.00\%$** port-to-port Bell state fidelity. |

---

## 📂 Repository Structure

```
S3_Trotter_Simulation/
├── README.md                           # Project documentation & quickstart guide
├── report.md                           # Comprehensive Phase 1 technical scientific report
├── main.ipynb                          # Fully executed end-to-end reproducible notebook
├── interactive_dashboard.html          # Standalone client-side interactive web dashboard
├── explore.py                          # Interactive CLI explorer with judge presets
├── run_experiments.py                 # Core experimental execution pipeline
├── run_advanced_studies.py             # Advanced studies (PBC, 2nd-order, noise sweep, budget)
├── phase2_prep.py                      # Phase 2 scaling & Bell state benchmark
├── requirements.txt                    # Project dependencies
├── processors/
│   ├── processor_A.json                # Processor A definition (5Q Linear)
│   ├── processor_B.json                # Processor B definition (7Q Heavy-Hex)
│   └── processor_C.json                # Processor C definition (10Q Extended Heavy-Hex for Phase 2)
├── src/
│   ├── problem.py                      # TFIM Hamiltonian, exact baseline, Trotter circuits (OBC/PBC, 1st/2nd)
│   ├── processors.py                   # Processor loading and graph analysis
│   ├── routing.py                      # Transpilation, SWAP decomposition, routing overhead
│   ├── metrics.py                      # Circuit metrics, analytical n* prediction, readout mitigation
│   └── visualization.py                # Publication-quality figure generation
├── results/
│   ├── figures/                        # Generated scientific figures (11 figures)
│   │   ├── processor_A_graph.png       # Processor A linear topology
│   │   ├── processor_B_graph.png       # Processor B heavy-hex topology
│   │   ├── processor_C_graph.png       # Processor C 10Q topology
│   │   ├── trotter_error_tradeoff.png  # Core U-curve showing n* = 3
│   │   ├── circuit_scaling_comparison.png # Depth & 2Q gate scaling
│   │   ├── circuit_comparison.png      # Gate overhead comparison at n=5
│   │   ├── architecture_tradeoffs.png  # Multi-panel architectural trade-off
│   │   ├── error_mitigation_study.png  # Readout error mitigation impact (>22% recovery)
│   │   ├── periodic_boundary_routing.png # PBC routing SWAP overhead (35.5% B advantage)
│   │   ├── trotter_order_comparison.png # 1st vs 2nd-order Trotter convergence
│   │   ├── noise_sensitivity.png       # n* vs physical hardware error rate
│   │   ├── error_budget_breakdown.png  # Stacked error decomposition
│   │   └── phase2_scaling_preview.png  # 8-qubit TFIM scaling on Processor C
│   └── tables/                         # Generated CSV and JSON benchmark tables
│       ├── AB_comparison_summary.csv   # Side-by-side architecture summary
│       ├── trotter_sweep_processor_A.csv # Full sweep data for Processor A
│       ├── trotter_sweep_processor_B.csv # Full sweep data for Processor B
│       ├── optimal_trotter_steps.csv   # Summary of optimal n* points
│       ├── pbc_comparison.csv          # Periodic boundary condition comparison table
│       ├── trotter_order_comparison.csv# 1st vs 2nd-order comparison table
│       ├── noise_sensitivity.csv       # Noise sweep table
│       ├── error_budget.csv            # Error budget decomposition table
│       ├── final_comparison.json       # Machine-readable output summary
│       └── phase2_readiness.json       # Phase 2 Bell fidelity and 8Q benchmark data
├── tests/
│   └── test_simulation.py              # Pytest unit tests (100% passing)
└── assets/
    └── qff_phase1_latex_images/        # Formatted equation PNGs for submission
```

---

## ⚡ Quickstart & Reproducibility

### 1. Installation
```bash
git clone <repository-url>
cd S3_Trotter_Simulation
pip install -r requirements.txt
```

### 2. Run Headless Experiments
```bash
# Run core baseline sweep
python run_experiments.py

# Run advanced studies (PBC, 2nd-order, noise sensitivity, error budget)
python run_advanced_studies.py

# Run Phase 2 readiness benchmark
python phase2_prep.py
```

### 3. Run Unit Tests
```bash
python -m unittest tests/test_simulation.py -v
```
All 7 unit tests pass in ~1.1 seconds.

---

## 📊 Summary of Scientific Discoveries

### 1. Optimal Trotter Step Sweet Spot ($n^* = 3$)
Algorithmic discretization error ($76.5\%$ at $n=1$) and physical gate noise intersect at $n^* = 3$, achieving minimum total error of $23.96\%$. Readout error mitigation further reduces this to **$18.46\%$** (a $22.9\%$ relative recovery).

<p align="center">
  <img src="results/figures/trotter_error_tradeoff.png" width="85%" alt="Trotter Error vs Noise Error Tradeoff">
</p>

### 2. Architectural Superiority under Periodic Boundaries (PBC)
When the closed-ring interaction $Z_3 Z_0$ is activated:
- **Processor A (Linear Graph $P_5$):** Must route across the line via 2 SWAP chains (**6 extra CX gates per step**).
- **Processor B (Heavy-Hex Graph):** Directly embeds the 4-cycle with **0 SWAP gates**!
- **Impact:** Processor B achieves a **$35.5\%$ error reduction** ($22.44\%$ vs $34.79\%$) at $n^* = 2$.

<p align="center">
  <img src="results/figures/periodic_boundary_routing.png" width="85%" alt="Periodic Boundary Routing Comparison">
</p>

### 3. 1st vs 2nd-Order Trotter Convergence Tradeoff
- 1st-Order converges as $\mathcal{O}(\Delta t) = \mathcal{O}(1/n)$.
- 2nd-Order Strang splitting converges as $\mathcal{O}(\Delta t^2) = \mathcal{O}(1/n^2)$.
- Because 2nd-order applies $H_{ZZ}$ twice per step (12 CX vs 6 CX), physical noise accumulates twice as fast, shifting the sweet spot from $n^* = 3$ to $n^* = 2$.

<p align="center">
  <img src="results/figures/trotter_order_comparison.png" width="85%" alt="Trotter Order Comparison">
</p>

### 4. Hardware Noise Sensitivity & Analytical Theory
We derived and validated the closed-form analytical prediction:
$$n^*_{\mathrm{analytical}} = \sqrt{\frac{C_{\mathrm{Trotter}}}{C_{\mathrm{Noise}}}} = \sqrt{\frac{\|[H_{ZZ}, H_X]\| T^2 / (2\|H\|)}{N_{2Q/\mathrm{step}} \cdot \epsilon_{2Q}}}$$
Across error rates $\epsilon_{2Q} \in [0.2\%, 4.0\%]$, the analytical prediction closely tracks empirical simulations.

<p align="center">
  <img src="results/figures/noise_sensitivity.png" width="85%" alt="Noise Sensitivity Study">
</p>

### 5. Complete Error Budget Breakdown
Stacked decomposition demonstrates how algorithmic Trotter error dominates at low $n$, while 2Q gate depolarizing noise dominates at high $n$.

<p align="center">
  <img src="results/figures/error_budget_breakdown.png" width="85%" alt="Error Budget Breakdown">
</p>

### 6. Phase 2 Scaling & Port-to-Port Bell Entanglement
- Prepared 10-qubit extended modular heavy-hex **Processor C** (`processors/processor_C.json`).
- Demonstrated distributed Bell state preparation $|\Phi^+\rangle$ between distant communication ports $\{0, 9\}$, achieving **$95.00\%$ entanglement fidelity**.

---

## 🤖 AI Assistance Disclosure

In compliance with the official Hackathon AI Policy (Section 11):
- **Tool used:** Anthropic Claude Code / Google Antigravity pair-programming assistant.
- **Scope of assistance:** Codebase scaffolding, syntax optimization for Qiskit 2.4, LaTeX equation rendering, and markdown report structure formatting.
- **Verification:** All quantum circuits, exact diagonalization baselines, matrix operations, and error analyses were independently mathematically verified and validated through end-to-end code execution.

---

## 📜 References

1. **Nielsen, M. A., & Chuang, I. L.** (2010). *Quantum Computation and Quantum Information*. Cambridge University Press.
2. **Lloyd, S.** (1996). Universal quantum simulators. *Science*, 273(5278), 1073-1078.
3. **Suzuki, M.** (1991). General theory of fractal path integrals with applications to many-body theories and statistical physics. *Journal of Mathematical Physics*, 32(2), 400-407.
4. **Qiskit SDK Documentation** (v2.4.1): [https://qiskit.org/documentation/](https://qiskit.org/documentation/)
5. **QFF 2026 Phase 1 Starter Guide & Guidelines**, SRM University - AP & quanTA / University of Saskatchewan.
