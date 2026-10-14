# S3 — Noise in Trotter Simulation

**Qiskit Fall Fest 2026 | SRM University - AP & quanTA / University of Saskatchewan**  
**Phase 1 Online Hackathon Submission (Quantum Simulation Track)**

[![Qiskit 2.4](https://img.shields.io/badge/Qiskit-2.4.1-613394.svg)](https://qiskit.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests Passing](https://img.shields.io/badge/Tests-100%25%20Passing-brightgreen.svg)](tests/test_simulation.py)
[![Interactive Dashboard](https://img.shields.io/badge/Dashboard-Interactive%20HTML-blueviolet.svg)](interactive_dashboard.html)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

---

## 📌 Executive Summary

To simulate continuous Hamiltonian time evolution $U(t) = \exp(-iHt)$ on digital quantum hardware, time is sliced into $n$ discrete Trotter steps (Trotter-Suzuki product formula). This establishes a fundamental scientific competition between two opposing error mechanisms:
1. **Algorithmic Discretization Error ($\epsilon_{\mathrm{Trotter}} \sim \mathcal{O}(t^2/n)$):** Decreases as step count $n$ rises.
2. **Hardware Noise Error ($\epsilon_{\mathrm{Noise}}$):** Rises monotonically with $n$ as circuit depth and two-qubit gate count accumulate gate infidelities and environmental decoherence.

The net error forms a convex U-curve with a global minimum at an **optimal step count $n^* = 3$**. This project discovers $n^*$, formulates an analytical predictive theory $n^* \approx \sqrt{C_{\mathrm{Trotter}}/C_{\mathrm{Noise}}}$, demonstrates the physical architectural advantage of heavy-hex topology over T-shaped tree hardware on 5-qubit spin chains, applies symmetric readout error mitigation to both processors, and provides interactive tools for judges.

### AI Assistance Disclosure
> In compliance with the Qiskit Fall Fest 2026 guidelines, generative AI tools (Antigravity / Google Gemini) were utilized during this project for code structure refinement, unit test authoring, LaTeX formula verification, and visualization layout. All quantum simulation algorithms, Hamiltonian operator definitions, Trotter-Suzuki product formulas, exact diagonalization baselines, error analysis, and architectural interpretations were designed, reviewed, and mathematically verified by the authors.

---

## 🎮 Interactive Judge Interfaces

### 1. Standalone Web Dashboard (`interactive_dashboard.html`)
Open [`interactive_dashboard.html`](interactive_dashboard.html) directly in any web browser (Chrome, Edge, Firefox, Safari):
- **Live Sliders:** Adjust $J, h, T, N, \epsilon_{2Q}$ and observe the U-curve dynamically recalculate in real time.
- **Topology & Order Toggles:** Switch between Open Chain (OBC) and Periodic Ring (PBC), or 1st vs 2nd-order Trotter-Suzuki.
- **Dynamic Sweet Spot ($n^*$):** Automatically highlights discrete $n^*$ and displays analytical theoretical predictions.
- **One-Click Export:** Download active sweep tables directly as CSV.

### 2. Real-Time Streamlit Simulator (`interactive_app.py`)
Launch the real-time Python simulation dashboard powered by Qiskit Aer:
```bash
streamlit run interactive_app.py
```
- Real-time Qiskit noisy simulation on every parameter change.
- Interactive parameter sliders, processor comparison toggles, and animated U-curve updates.
- One-click CSV export of live simulation sweeps.

### 3. Optional CLI Parameter Explorer (`tools/explore.py`)
For judges preferring terminal evaluation, an optional CLI explorer is provided with curated physical presets:
```bash
# Standard baseline benchmark
python tools/explore.py --preset default

# Periodic boundary ring routing benchmark
python tools/explore.py --preset periodic_ring --verbose

# Quantum critical point (maximal non-commutativity h = J)
python tools/explore.py --preset critical

# 2nd-Order Trotter Strang splitting (quadratic error convergence)
python tools/explore.py --preset second_order

# Custom parameters with CSV export
python tools/explore.py --J 1.2 --h 0.8 --time 1.5 --qubits 4 --order 1 --output custom_sweep.csv
```

---

## 🔬 Problem Specification & Key Discoveries

| Attribute | Value |
| :--- | :--- |
| **Track** | Quantum Simulation |
| **Problem Code** | `S3` |
| **Physical Model** | Transverse-Field Ising Model: $H = -J \sum Z_i Z_{i+1} - h \sum X_i$ |
| **Baseline Parameters** | $N = 4$ qubits, $J = 1.0$, $h = 0.5$, $T = 1.0$, $2048$ shots |
| **Exact Ground Truth** | Continuous unitary matrix exponential $U(t) = \exp(-iHt)$, $E_{\mathrm{exact}} = -3.0000$ |
| **Target Processors** | Processor A (5Q T-Shaped Tree) & Processor B (7Q Heavy-Hex Tree Baseline) |
| **Core Sweet Spot** | Optimal step count identified at **$n^* = 3$** (Total error: $\sim 23.8\%$ raw, $\sim 18.5\%$ mitigated) |
| **$N=4$ Error Equivalence** | At $N=4$, both chips compile to identical circuits. The $23.79\%$ vs $23.95\%$ difference is **purely statistical shot noise** from finite sampling ($N_{\text{shots}} = 2048$), NOT architecture. |
| **5-Qubit Chain Divergence** | In this model, Processor B avoids 3 SWAPs (+9 CX) and achieves **lower energy error**: **$23.70\% \pm 1.33\%$ vs $28.72\% \pm 1.12\%$** raw (**-5.02 percentage points**, $17.5\%$ rel) and **$18.16\% \pm 1.39\%$ vs $23.53\% \pm 1.19\%$** mitigated (**-5.37 percentage points**, $22.8\%$ rel), where $\pm$ denotes sample standard deviation over 5 seeds (SEM $\approx 0.5\%$). Secondary density-matrix fidelity reflects $74.46\%$ vs $57.10\%$ (+17.36 percentage points). |
| **Cyclic Variant Study** | Optional exploration (`src/cyclic_study.py`) evaluates a cyclic heavy-hex variant with 0 SWAPs on periodic rings vs 9 SWAPs on trees. |
| **Transpiled Circuits** | Full high-resolution circuit diagrams generated for both processors (`circuit_processor_A.png`, `circuit_processor_B.png`). |
| **Analytical Formula** | Continuous prediction $n^*_{\mathrm{analytical}} = \sqrt{C_{\mathrm{Trotter}} / C_{\mathrm{Noise}}}$ matches discrete simulation within $\pm 1\text{--}2$ steps in the mid-range. |
| **Topological Assumption** | Processor B is modeled on the canonical IBM 7-qubit heavy-hex subtree architecture (girth $\infty$), stating this assumption explicitly. |
| **Controlled Noise** | Both processors use identical noise values ($0.1\%$ 1Q, $1.0\%$ 2Q, $2.0\%$ readout) as a deliberate controlled scientific experiment. |
| **Statistical Rigor** | Multi-seed evaluation across seeds 42..46 reporting sample standard deviations ($s$) and standard error of the mean ($\text{SEM} \approx 0.5\%$). |

---

## 📂 Repository Structure

```
S3_Trotter_Simulation/
├── README.md                           # Project documentation & quickstart guide
├── report.md                           # Comprehensive Phase 1 technical scientific report
├── main.ipynb                          # Fully executed end-to-end reproducible notebook
├── interactive_dashboard.html          # Standalone client-side interactive web dashboard
├── interactive_app.py                  # Real-time Streamlit Qiskit Aer dashboard
├── run_experiments.py                 # Core experimental execution pipeline
├── run_advanced_studies.py             # Advanced studies (PBC, 2nd-order, noise sweep, budget)
├── requirements.txt                    # Pinned project dependencies (==)
├── LICENSE                             # Apache-2.0 License
├── tools/
│   └── explore.py                      # Optional CLI parameter explorer with presets
├── processors/
│   ├── processor_A.json                # Authoritative Processor A definition (5Q T-Tree, Slide 4)
│   └── processor_B.json                # Authoritative Processor B definition (7Q Heavy-Hex Tree, Slide 4)
├── src/
│   ├── problem.py                      # TFIM Hamiltonian, exact baseline, Trotter circuits (OBC/PBC, 1st/2nd)
│   ├── processors.py                   # Processor loading and geometric graph visualization
│   ├── routing.py                      # Transpilation, exact SWAP decomposition, routing overhead
│   ├── sweep.py                        # Parametric sweep engine, noisy simulation, readout mitigation
│   ├── metrics.py                      # Circuit metrics, noisy state fidelity, analytical n*, mitigation
│   ├── cyclic_study.py                 # Optional cyclic heavy-hex variant study (OBC vs PBC)
│   └── visualization.py                # Publication-quality figure generation
├── tests/
│   └── test_simulation.py              # Pytest test suite (100% passing)
└── results/
    ├── figures/                        # Generated scientific figures
    │   ├── processor_A_graph.png       # Processor A 5Q T-shaped tree graph
    │   ├── processor_B_graph.png       # Processor B 7Q heavy-hex tree graph
    │   ├── circuit_processor_A.png     # Transpiled circuit on Processor A (required)
    │   ├── circuit_processor_B.png     # Transpiled circuit on Processor B (required)
    │   ├── trotter_error_tradeoff.png  # Core U-curve showing n* = 3
    │   ├── circuit_scaling_comparison.png # Depth & 2Q CX gate scaling (with output)
    │   ├── error_mitigation_study.png  # Dual-processor readout mitigation impact
    │   ├── periodic_boundary_routing.png # PBC routing SWAP overhead on trees
    │   ├── trotter_order_comparison.png # 1st vs 2nd-order Trotter convergence
    │   └── noise_sensitivity.png       # n* vs physical hardware error rate
    └── tables/                         # Generated CSV benchmark tables
        ├── AB_comparison_summary.csv   # Side-by-side architecture summary (multi-seed stats)
        ├── 5qubit_chain_comparison.csv # Full physical 5Q chain simulation comparison table (5 seeds)
        ├── cyclic_variant_comparison.csv # Optional cyclic heavy-hex variant comparison table
        ├── trotter_sweep_processor_A.csv # Full sweep data for Processor A
        ├── trotter_sweep_processor_B.csv # Full sweep data for Processor B
        ├── pbc_comparison.csv          # Periodic boundary condition comparison table
        ├── trotter_order_comparison.csv # 1st vs 2nd-order comparison table
        └── noise_sensitivity.csv       # Noise sensitivity sweep data
```

---

## ⚡ Quickstart & Reproducibility Guide

### 1. Installation
Install pinned dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run All Experiments & Generate Figures
Execute the complete experimental pipeline:
```bash
python run_experiments.py
python run_advanced_studies.py
```

### 3. Run Test Suite
```bash
pytest tests/ -v
```

### 4. Launch Interactive Tools
```bash
# Browser dashboard
# Open interactive_dashboard.html in any browser

# Streamlit interactive app
streamlit run interactive_app.py

# Optional CLI explorer
python tools/explore.py --preset default
```
