# S3 — Noise in Trotter Simulation

**Qiskit Fall Fest 2026 | SRM University - AP & quanTA / University of Saskatchewan**  
**Phase 1 Online Hackathon Submission (Quantum Simulation Track)**

[![Qiskit 2.4](https://img.shields.io/badge/Qiskit-2.4.1-613394.svg)](https://qiskit.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests Passing](https://img.shields.io/badge/Tests-12%2F12%20Passing-brightgreen.svg)](tests/test_simulation.py)
[![Interactive Dashboard](https://img.shields.io/badge/Dashboard-Interactive%20HTML-blueviolet.svg)](interactive_dashboard.html)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

---

## 📌 Executive Summary

Simulating continuous Hamiltonian dynamics $U(t) = \exp(-iHt)$ on digital quantum hardware requires discretizing time into $n$ finite intervals via Trotter-Suzuki product formulas. This creates a fundamental physical competition between two opposing error mechanisms:
1. **Algorithmic Discretization Error ($\epsilon_{\mathrm{Trotter}} \sim \mathcal{O}(t^2/n)$):** Decreases monotonically as step count $n$ rises.
2. **Physical Hardware Noise Error ($\epsilon_{\mathrm{Noise}}$):** Increases monotonically as $n$ rises due to circuit depth accumulation, two-qubit gate infidelities, and environmental decoherence.

The composite error forms a convex U-curve with a global minimum at the **optimal step count $n^* = 3$**.

This project provides:
- Complete implementation of the mandatory **Problem S3 benchmark** on a 4-spin Transverse-Field Ising Model (TFIM) under controlled, reproducible conditions.
- Strict architectural isolation between **Processor A (5Q T-shaped baseline)** and **Processor B (7Q heavy-hex baseline)**.
- Rigorous statistical error bars ($\pm$ sample standard deviation over 5 seeds, $\text{SEM} \approx 0.5\%$).
- Application of symmetric readout error mitigation to both processors.
- Clear separation of **mandatory core benchmarks** from **optional architectural sensitivity studies** (5-qubit chain divergence, periodic boundary routing, and cyclic heavy-hex variants).
- Complete self-containment for clean evaluation (`main.ipynb` executes end-to-end even in an uninitialized workspace).
- Interactive evaluation tools (standalone HTML dashboard, Streamlit real-time simulator, and CLI explorer).

### AI Assistance Disclosure
> In compliance with the Qiskit Fall Fest 2026 guidelines, generative AI tools (Antigravity / Google Gemini) were utilized during this project for code structure refinement, unit test authoring, LaTeX formula verification, and visualization layout. All quantum simulation algorithms, Hamiltonian operator definitions, Trotter-Suzuki product formulas, exact diagonalization baselines, error analysis, and architectural interpretations were designed, reviewed, and mathematically verified by the authors.

---

## 🔬 Part I: Core Common Benchmark (Problem S3)

The mandatory Phase 1 benchmark evaluates real-time spin dynamics under identical controlled conditions on Processor A and Processor B.

### 1. Problem Specification Sheet
| Parameter | Value / Description | Notes |
| :--- | :--- | :--- |
| **Problem Code** | `S3` | Quantum Simulation Track |
| **Physical Model** | Transverse-Field Ising Model (TFIM) | $H = -J \sum_{i=0}^{N-2} Z_i Z_{i+1} - h \sum_{i=0}^{N-1} X_i$ |
| **System Size ($N$)** | $4$ Spins / Qubits | Open Boundary Conditions (OBC) |
| **Coupling Constants** | $J = 1.0$, $h = 0.5$ | Ferromagnetic regime with transverse field |
| **Simulation Time ($T$)** | $T = 1.0$ | Time units ($\hbar = 1$) |
| **Measurement Shots** | $2048$ shots | Pauli $Z$ and $X$ basis measurements |
| **Classical Reference** | Exact Matrix Exponential ($2^4 = 16$) | $U_{\mathrm{exact}}(t) = \exp(-i H t)$, $E_{\mathrm{exact}} = -3.0000$ |
| **Decomposition** | 1st & 2nd-Order Trotter-Suzuki | Strang splitting for 2nd-order |

### 2. Processor Architecture Definitions & Assumptions
- **Processor A (`processors/processor_A.json`):** 5 physical qubits, T-shaped tree graph (`edges: [[0, 1], [1, 2], [1, 3], [3, 4]]`) with communication ports at $\{0, 4\}$. Modeled after the canonical 5-qubit baseline graph.
- **Processor B (`processors/processor_B.json`):** 7 physical qubits, heavy-hex tree graph (`edges: [[0, 1], [1, 2], [1, 3], [3, 5], [4, 5], [5, 6]]`) with communication ports at $\{0, 6\}$. Modeled after the canonical IBM 7-qubit heavy-hex subtree (Jakarta/Nairobi unit cell, girth $\infty$).
- **Controlled Noise Model:** Both processors are evaluated under identical synthetic noise: single-qubit gate error $= 0.1\%$, two-qubit CX error $= 1.0\%$, and readout assignment error $= 2.0\%$. This deliberate choice isolates topological routing effects from calibration disparities.
- **Basis Gate Set:** Standardized across both architectures: $\mathcal{B} = \{rz, sx, x, cx\}$.

### 3. Core Benchmark Findings ($N=4$ Spins)
| Metric | Processor A (5Q T-Tree) | Processor B (7Q Heavy-Hex Tree) | Difference (B - A) | Notes / Explanation |
| :--- | :--- | :--- | :--- | :--- |
| **Optimal Step Count ($n^*$)** | **$3$** | **$3$** | $0$ | Identical U-curve minimum |
| **Min Total Error at $n^*$** | **$23.79\%$** | **$23.95\%$** | $+0.16\%$ | Difference is purely statistical shot noise |
| **Transpiled Depth ($n=5$)** | $58$ | $58$ | $0$ | Identical circuit structure |
| **Two-Qubit (CX) Gates ($n=5$)**| $30$ | $30$ | $0$ | $6$ CX per Trotter step |
| **Routing SWAP Gates ($n=5$)** | **$0$** | **$0$** | $0$ | 4-qubit line embeds isomorphically |
| **Measured Energy (Raw, 5 seeds)** | $-2.1127 \pm 0.0360$ | $-2.1127 \pm 0.0360$ | $0.0000$ | $s = 0.0360$, $\text{SEM} \approx 0.016$ |
| **Total Error (Raw, $n=5$)** | $29.58\%$ | $29.58\%$ | $0.00\%$ | Dominated by physical noise |
| **Measured Energy (Mitigated)** | $-2.2644 \pm 0.0386$ | $-2.2644 \pm 0.0386$ | $0.0000$ | Readout mitigation applied symmetrically |
| **Total Error (Mitigated, $n=5$)**| **$24.52\%$** | **$24.52\%$** | $0.00\%$ | **$+5.06$ percentage points improvement** |
| **Noisy State Fidelity ($n=5$)** | $74.31\%$ | $74.31\%$ | $0.00\%$ | Evaluated via full density matrix |

> [!NOTE]
> **Key Scientific Conclusion for Core Benchmark:** At $N=4$ qubits, the open TFIM chain embeds as an isomorphic line subgraph into both Processor A (e.g., path $0-1-3-4$) and Processor B (path $0-1-3-5$) without requiring routing SWAP gates. Consequently, both processors generate structurally identical transpiled circuits, confirming that under identical noise channels, hardware performance is equivalent. The observed $0.16\%$ variation in minimum error is entirely attributable to finite-shot sampling variance ($2048$ shots).

---

## 🔍 Part II: Extended Architectural Sensitivity Studies (Optional)

In accordance with hackathon instructions, optional investigations beyond the mandatory common benchmark are strictly separated below.

### Study A: 5-Qubit Chain Divergence ($N=5$)
When the physical problem size expands to 5 spins, the chain can no longer be embedded linearly into Processor A without traversing its T-branch hub (Qubit 1):
- **Processor A Bottleneck:** The compiler is forced to insert **3 SWAP gates (+9 CX)** per Trotter step, inflating circuit depth by $+64.1\%$ (64 vs 39).
- **Processor B Advantage:** The 7-qubit heavy-hex topology contains a 5-vertex simple path ($0-1-3-5-6$), accommodating the entire chain with **0 SWAP gates**.
- **Physical Impact:**
  - **Raw Energy Error:** Processor B achieves **$23.70\% \pm 1.33\%$** vs Processor A's **$28.72\% \pm 1.12\%$** (**$-5.02$ percentage points**, $17.5\%$ relative error reduction).
  - **Mitigated Energy Error:** **$18.16\% \pm 1.39\%$** vs **$23.53\% \pm 1.19\%$** (**$-5.37$ percentage points**, $22.8\%$ relative error reduction).
  - **Noisy State Fidelity:** $74.46\%$ on Processor B vs $57.10\%$ on Processor A (**$+17.36$ percentage points**).
  - *Statistical Note:* All errors reported as mean $\pm$ sample standard deviation ($s$) over 5 independent seeds (seeds 42..46), with standard error of the mean $\text{SEM} \approx 0.5\%$.

### Study B: Periodic Boundary Condition (PBC) Routing Overhead
When closed boundary conditions are simulated (ring coupling $Z_{N-1}Z_0$), neither tree architecture natively supports the cyclic connection:
- Both processors require routing SWAPs ($1$ SWAP at $n=1$, scaling to $10$ SWAPs at $n=5$).
- Circuit depth increases from 58 (open chain) to 140 (periodic ring), accelerating noise accumulation and shifting the sweet spot to smaller step counts.

### Study C: Optional Cyclic Heavy-Hex Variant (`src/cyclic_study.py`)
To address the topological visual ambiguity in presentation slides (which illustrate cycles while specifying a 7-qubit graph), we implemented an optional cyclic variant with coupler `[2, 6]` forming cycle $1-2-6-5-3-1$:
- **Open Chain (OBC $N=5$):** Yields identical performance ($74.46\%$ fidelity, $23.70\%$ error) across both tree and cyclic Processor B, proving our baseline benchmark is invariant to cycle presence.
- **Periodic Ring (PBC $N=5$):** The cyclic variant eliminates all 9 routing SWAPs, reducing circuit depth from 96 to 60 and boosting fidelity from $55.73\%$ to $69.78\%$ ($+14.05$ percentage points).

### Study D: 1st vs 2nd-Order Trotter Decomposition
- 2nd-order Strang splitting $\exp(-iA\Delta t/2)\exp(-iB\Delta t)\exp(-iA\Delta t/2)$ provides quadratic error convergence $\mathcal{O}(\Delta t^2)$ but doubles the two-qubit gate depth per step.
- Under $1.0\%$ CX gate error, 2nd-order reaches its noise-limited minimum earlier, shifting optimal steps to $n^* = 2$ (vs $n^* = 3$ for 1st-order).

### Study E: Noise Sensitivity & Analytical $n^*$ Formula
We derive an analytical prediction for the optimal step count:
$$n^*_{\mathrm{analytical}} = \sqrt{\frac{C_{\mathrm{Trotter}}}{C_{\mathrm{Noise}}}} = \sqrt{\frac{\frac{1}{2} \| [A, B] \| t^2}{N_{\mathrm{CX\_per\_step}} \cdot \epsilon_{\mathrm{2Q}}}}$$
- Across hardware error rates from $0.2\%$ to $4.0\%$, the continuous formula predicts $n^* \in [5.00, 1.12]$, matching empirical discrete integer minima within $\pm 1\text{--}2$ steps.

---

## 🎮 Interactive Judge Interfaces

### 1. Standalone Web Dashboard (`interactive_dashboard.html`)
Open [`interactive_dashboard.html`](interactive_dashboard.html) directly in any web browser:
- **Live Physical Sliders:** Adjust $J, h, T, N, \epsilon_{2Q}$ and observe real-time U-curve recalculation.
- **Topology & Order Toggles:** Switch between Open Chain (OBC) and Periodic Ring (PBC), or 1st vs 2nd-order Trotter-Suzuki.
- **Analytical Overlay:** Displays real-time continuous theoretical curve against discrete simulation points.
- **One-Click Export:** Download active sweep tables directly as CSV.

### 2. Real-Time Streamlit Simulator (`interactive_app.py`)
Launch the interactive Python app powered by Qiskit Aer:
```bash
streamlit run interactive_app.py
```
- Real-time Qiskit simulation execution on every slider adjustment.
- Side-by-side Processor A vs B interactive plots and data export.

### 3. Optional CLI Parameter Explorer (`tools/explore.py`)
```bash
# Standard baseline benchmark
python tools/explore.py --preset default

# Periodic boundary ring routing benchmark
python tools/explore.py --preset periodic_ring --verbose

# Quantum critical point (h = J)
python tools/explore.py --preset critical

# Custom sweep with CSV export
python tools/explore.py --J 1.2 --h 0.8 --time 1.5 --qubits 4 --output custom_sweep.csv
```

---

## 📂 Complete Repository Structure

```
S3_Trotter_Simulation/
├── .gitignore                          # Git ignore configuration (caches, pyc, checkpoints)
├── LICENSE                             # Apache-2.0 License
├── requirements.txt                    # Pinned Python dependencies (==)
├── README.md                           # Main documentation & benchmark overview
├── report.md                           # Comprehensive scientific & architectural report
├── main.ipynb                          # Self-contained executable Jupyter notebook
├── interactive_dashboard.html          # Standalone client-side interactive dashboard
├── interactive_app.py                  # Real-time Streamlit Qiskit simulation dashboard
├── run_experiments.py                 # Core experimental execution pipeline
├── run_advanced_studies.py             # Advanced studies (PBC, 2nd-order, noise sweep)
├── data/                               # Data storage per starter guide layout
│   ├── raw/
│   │   ├── full_raw_counts.json        # Full raw shot measurement counts across all n and seeds (77 KB)
│   │   └── raw_counts_sample.json      # Representative sample extract (n=3) for quick inspection
│   └── processed/
│       ├── AB_comparison.csv           # Starter guide alias for AB comparison
│       ├── final_comparison.csv        # Tabular export of final parameters and metrics
│       ├── processor_A_metrics.csv     # Starter guide alias for Processor A sweep
│       └── processor_B_metrics.csv     # Starter guide alias for Processor B sweep
├── processors/
│   ├── processor_A.json                # Processor A definition (5Q T-shaped baseline graph)
│   └── processor_B.json                # Processor B definition (7Q heavy-hex unit cell baseline)
├── src/
│   ├── __init__.py                     # Modular package initialization
│   ├── problem.py                      # TFIM Hamiltonian, exact baseline, Trotter circuits
│   ├── processors.py                   # Processor graph parsing and layout visualization
│   ├── routing.py                      # Transpilation, exact SWAP counting, routing analysis
│   ├── sweep.py                        # Parametric sweep engine and noisy Aer simulation
│   ├── metrics.py                      # Circuit metrics, density-matrix fidelity, readout mitigation
│   ├── cyclic_study.py                 # Optional cyclic heavy-hex variant study
│   └── visualization.py                # Publication-quality matplotlib figure generation
├── tests/
│   └── test_simulation.py              # Pytest test suite (12/12 passing)
├── tools/
│   ├── explore.py                      # Optional CLI parameter explorer with presets
│   ├── setup_aliases.py                # Generates starter guide folder layout & aliases
│   └── update_notebook.py              # Utility to refresh notebook markdown and tables
└── results/
    ├── figures/                        # Publication figures (including starter guide aliases)
    │   ├── AB_performance.png          # Starter guide alias for error tradeoff plot
    │   ├── architecture_tradeoffs.png  # Multi-panel architectural tradeoff summary
    │   ├── circuit_processor_A.png     # High-resolution transpiled circuit on Processor A
    │   ├── circuit_processor_B.png     # High-resolution transpiled circuit on Processor B
    │   ├── circuit_scaling_comparison.png # Depth & 2Q CX gate scaling vs steps
    │   ├── error_mitigation_study.png  # Dual-processor readout mitigation impact
    │   ├── logical_vs_transpiled.png   # Starter guide alias for circuit scaling
    │   ├── noise_sensitivity.png       # Optimal step n* vs physical hardware error rate
    │   ├── periodic_boundary_routing.png # PBC routing SWAP overhead comparison
    │   ├── processor_A_graph.png       # Processor A coupling topology diagram
    │   ├── processor_B_graph.png       # Processor B coupling topology diagram
    │   ├── trotter_error_tradeoff.png  # Core U-curve establishing n* = 3 sweet spot
    │   └── trotter_order_comparison.png # 1st vs 2nd-order Trotter convergence
    └── tables/                         # Benchmark result tables (CSVs & JSON)
        ├── 5qubit_chain_comparison.csv # 5Q physical chain simulation metrics (5 seeds)
        ├── AB_comparison.csv           # Starter guide alias for side-by-side summary
        ├── AB_comparison_summary.csv   # Side-by-side architecture summary table
        ├── cyclic_variant_comparison.csv # Cyclic vs acyclic heavy-hex benchmark
        ├── error_budget.csv            # Error budget breakdown across steps
        ├── final_comparison.csv        # Tabular benchmark parameters & metrics
        ├── final_comparison.json       # JSON benchmark parameters & metrics
        ├── noise_sensitivity.csv       # Noise sweep empirical vs analytical n* data
        ├── optimal_trotter_steps.csv   # Optimal step metrics at minimum error
        ├── pbc_comparison.csv          # Periodic boundary condition comparison data
        ├── processor_A_metrics.csv     # Starter guide alias for Processor A sweep
        ├── processor_B_metrics.csv     # Starter guide alias for Processor B sweep
        ├── trotter_order_comparison.csv # 1st vs 2nd-order Trotter comparison data
        ├── trotter_sweep_processor_A.csv # Parametric sweep data for Processor A
        └── trotter_sweep_processor_B.csv # Parametric sweep data for Processor B
```

---

## ⚡ Quickstart & Reproducibility Guide

### 1. Environment Setup
Clone the repository and install dependencies in Python 3.10+:
```bash
git clone <repository-url>
cd S3_Trotter_Simulation
pip install -r requirements.txt
```

### 2. Run Test Suite
Verify that all unit tests pass:
```bash
pytest tests/ -v
```
*Expected: 12 passed in ~5 seconds.*

### 3. Run Experimental Pipelines
Execute the core benchmark and all advanced sensitivity studies:
```bash
python run_experiments.py
python run_advanced_studies.py
python -c "import src.cyclic_study as cs; cs.run_cyclic_comparison()"
python tools/setup_aliases.py
```
*Total runtime is ~30 seconds on standard consumer hardware.*

### 4. Run the Jupyter Notebook
Open `main.ipynb` in Jupyter Notebook, JupyterLab, or VS Code:
```bash
jupyter notebook main.ipynb
```
Select **Kernel -> Restart & Run All**. The notebook is fully self-contained and executes end-to-end with zero errors.

### 5. Repository Hygiene & Git Push Instructions
To ensure no cache files or bytecode are committed when submitting to GitHub:
```bash
# If pycache files were previously tracked, untrack them:
git rm -r --cached src/__pycache__ tests/__pycache__ tools/__pycache__ 2>$null

# Commit clean repository
git add .
git commit -m "Submit Phase 1 Problem S3 Solution"
git push origin main
```
> [!TIP]
> If uploading repository files through the GitHub web UI, do not drag and drop `__pycache__` or `.pytest_cache` folders. The included `.gitignore` automatically prevents them from being tracked in local git CLI workflows.
