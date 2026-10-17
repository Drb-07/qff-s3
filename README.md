# S3 — Noise in Trotter Simulation

**Qiskit Fall Fest 2026 | SRM University - AP & quanTA / University of Saskatchewan**  
**Phase 1 Online Hackathon Submission (Quantum Simulation Track)**

[![Qiskit 2.4](https://img.shields.io/badge/Qiskit-2.4.1-613394.svg)](https://qiskit.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests Passing](https://img.shields.io/badge/Tests-12%2F12%20Passing-brightgreen.svg)](tests/test_simulation.py)
[![Interactive Dashboard](https://img.shields.io/badge/Dashboard-Interactive%20HTML-blueviolet.svg)](interactive_dashboard.html)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

---

## 👥 Team Information
- **Team Name:** Organic Particles
- **Institution / Affiliation:** SRM University - AP 
- **Track:** Quantum Simulation (Problem Code: `S3`)

---

## 🤖 AI Assistance Disclosure
> **In compliance with the Qiskit Fall Fest 2026 Guidelines:**  
> Generative AI tools (Antigravity / Google Gemini) were utilized during this project for code structure refinement, unit test authoring, LaTeX formula verification, and visualization layout. All quantum simulation algorithms, Hamiltonian operator definitions, Trotter-Suzuki product formulas, exact diagonalization baselines, error analysis, and architectural interpretations were designed, reviewed, and mathematically verified by the authors.

---

## 📌 Executive Summary

Simulating continuous Hamiltonian dynamics $U(t) = \exp(-iHt)$ on digital quantum processors requires discretizing time into $n$ finite intervals via Trotter-Suzuki product formulas. This induces a fundamental physical competition between two opposing error mechanisms:
1. **Algorithmic Discretization Error ($\epsilon_{\mathrm{Trotter}} \sim \mathcal{O}(t^2/n)$):** Decreases monotonically as step count $n$ increases.
2. **Physical Hardware Noise Error ($\epsilon_{\mathrm{Noise}}$):** Increases monotonically as $n$ rises due to gate depth accumulation, two-qubit CX infidelities, and environmental decoherence.

The composite error forms a convex U-curve with a global minimum at the **optimal step count $n^* = 3$** under baseline depolarizing noise.

### Key Contributions & Deliverables:
- **Core Benchmark ($N=4$ Spins):** Rigorous, reproducible evaluation on a 4-spin Transverse-Field Ising Model (TFIM) across **Processor A** (5Q T-tree) and **Processor B** (9Q heavy-hex graph).
- **Multiple Observables:** Classical exact baselines and noisy simulation results evaluated for total energy $E(t)$, single-qubit observable $\langle Z_0 \rangle$, and mean magnetization $\langle M_z \rangle = \frac{1}{N}\sum \langle Z_i \rangle$.
- **Architectural Reconciliation:** Follows the definitive **9-qubit, 12-edge coupling map** (Slide 4 and Master Guide) as primary Processor B, while evaluating the Kit's 8-edge tree as a labeled variant.
- **Statistical Rigor:** 5 independent simulation seeds (seeds 42..46) with 4096 measurement shots, providing sample mean and standard deviation ($\pm$) error bars.
- **Symmetric Error Mitigation:** Tensor-product readout error mitigation ($1.0\%$ assignment error) applied symmetrically across both processors.
- **Optional Architectural Studies:** Demonstrates topological divergence on 5-qubit chains ($N=5$) and periodic rings (PBC), where Processor B's native cycles eliminate routing SWAPs.
- **Processor D Proposal:** Evaluates next-generation hyperbolic / cyclic connectivity for scalable simulation.
- **Self-Contained Notebook & Interactive Tools:** `main.ipynb` executes cleanly end-to-end, supplemented by a browser-based HTML dashboard, Streamlit simulator, and CLI explorer.

---

## 🔬 Part I: Core Common Benchmark (Problem S3)

### 1. Problem Specification Sheet
| Parameter | Value / Description | Notes / Justification |
| :--- | :--- | :--- |
| **Problem Code** | `S3` | Quantum Simulation Track |
| **Physical Model** | Transverse-Field Ising Model (TFIM) | $H = -J \sum_{i=0}^{N-2} Z_i Z_{i+1} - h \sum_{i=0}^{N-1} X_i$ |
| **System Size ($N$)** | $4$ Spins / Qubits | Open Boundary Conditions (OBC) |
| **Coupling Strengths** | $J = 1.0$, $h = 0.5$ | Ferromagnetic regime with transverse magnetic field |
| **Evolution Time ($T$)** | $T = 1.0$ | Time units ($\hbar = 1$), satisfies challenge bound $T \le 2.0$ |
| **Measurement Shots** | $4096$ shots | Pauli $Z$ and $X$ basis measurements per seed |
| **Transpiler Config** | `optimization_level=1`, `seed=42` | SABRE layout and SABRE routing |
| **Basis Gate Set** | $\mathcal{B} = \{rz, sx, x, cx\}$ | Native superconducting hardware basis |
| **Noise Model** | Synthetic depolarizing + readout | 1Q: $0.001$ ($0.1\%$), 2Q CX: $0.005$ ($0.5\%$), Readout: $0.01$ ($1.0\%$) |
| **Classical References** | Exact Diagonalization ($2^4 = 16$) | $E_{\mathrm{exact}} = -3.000000$, $\langle Z_0 \rangle \approx 0.672315$, $\langle M_z \rangle \approx 0.767129$ |

### 2. Processor Topology Spec Decision & Documentation

| Processor | Primary Specification (Used in Core Benchmark) | Variant Specification (Labeled Study) | Rationale & Guide Alignment |
| :--- | :--- | :--- | :--- |
| **Processor B** | **9 Qubits, 12 Edges (24 directed couplers)** (`processors/processor_B.json`) | **9 Qubits, 8 Edges (16 directed couplers)** (`processors/processor_B_tree8.json`) | Every document specifies 9 qubits. The 12-edge graph matches Slide 4, the Config doc, and the Master Guide. The Kit's 8-edge tree is run as an additional variant. |
| **Processor A** | **5 Qubits, 4 Edges (T-shaped tree)** (`processors/processor_A.json`) | **5 Qubits, 4 Edges (Linear Chain)** (`processors/processor_A_chain.json`) | Slide 4, the Config doc, and Master Guide specify the T-shaped tree with hub at qubit 1. The Kit's linear chain is run as an additional variant. |

- **Designated Ports:** Processor A ports are $\{0, 4\}$; Processor B ports are $\{0, 8\}$.
- **Controlled Noise Comparison:** Both processors are evaluated under identical synthetic noise rates ($0.1\%$ 1Q, $0.5\%$ 2Q CX, $1.0\%$ readout) to rigorously isolate graph topology and compiler routing from hardware calibration variances.

### 3. Core Benchmark Results ($N=4$ Spins, $T=1.0$)

Below is the summary of the core benchmark across Trotter steps $n=1, 2, 4$ and the optimal step $n^*=3$:

| Architecture | Steps ($n$) | Depth | 2Q CX | SWAPs | Energy (Raw, 5 seeds) | Energy (Mitigated) | Energy Err (Raw) | Energy Err (Mit) | $\langle Z_0 \rangle$ (Raw / Mit) | $\langle M_z \rangle$ (Raw / Mit) | Fidelity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Processor A (5Q T-Tree)** | $1$ | $14$ | $6$ | $0$ | $-0.8010 \pm 0.0350$ | $-0.8340 \pm 0.0358$ | $73.30\%$ | $72.20\%$ | $0.5209$ / $0.5317$ | $0.5188$ / $0.5295$ | $32.20\%$ |
| **Processor B (9Q Slide 4)** | $1$ | $14$ | $6$ | $0$ | $-0.8010 \pm 0.0350$ | $-0.8340 \pm 0.0358$ | $73.30\%$ | $72.20\%$ | $0.5209$ / $0.5317$ | $0.5188$ / $0.5295$ | $32.20\%$ |
| **Processor A (5Q T-Tree)** | $2$ | $25$ | $12$ | $0$ | $-2.4301 \pm 0.0328$ | $-2.5168 \pm 0.0339$ | $19.00\%$ | $16.11\%$ | $0.6164$ / $0.6290$ | $0.6908$ / $0.7050$ | $81.33\%$ |
| **Processor B (9Q Slide 4)** | $2$ | $25$ | $12$ | $0$ | $-2.4301 \pm 0.0328$ | $-2.5168 \pm 0.0339$ | $19.00\%$ | $16.11\%$ | $0.6164$ / $0.6290$ | $0.6908$ / $0.7050$ | $81.33\%$ |
| **Processor A (5Q T-Tree)** | **$3$ ($n^*$)** | **$36$** | **$18$** | **$0$** | **$-2.5132 \pm 0.0285$** | **$-2.6017 \pm 0.0289$** | **$16.23\%$** | **$13.28\%$** | **$0.6209$ / $0.6336$** | **$0.6906$ / $0.7047$** | **$85.48\%$** |
| **Processor B (9Q Slide 4)** | **$3$ ($n^*$)** | **$36$** | **$18$** | **$0$** | **$-2.5132 \pm 0.0285$** | **$-2.6017 \pm 0.0289$** | **$16.23\%$** | **$13.28\%$** | **$0.6209$ / $0.6336$** | **$0.6906$ / $0.7047$** | **$85.48\%$** |
| **Processor A (5Q T-Tree)** | $4$ | $47$ | $24$ | $0$ | $-2.4805 \pm 0.0300$ | $-2.5674 \pm 0.0306$ | $17.32\%$ | $14.42\%$ | $0.6158$ / $0.6283$ | $0.6801$ / $0.6939$ | $84.77\%$ |
| **Processor B (9Q Slide 4)** | $4$ | $47$ | $24$ | $0$ | $-2.4805 \pm 0.0300$ | $-2.5674 \pm 0.0306$ | $17.32\%$ | $14.42\%$ | $0.6158$ / $0.6283$ | $0.6801$ / $0.6939$ | $84.77\%$ |
| **Processor A (5Q T-Tree)** | $5$ | $58$ | $30$ | $0$ | $-2.4188 \pm 0.0312$ | $-2.5035 \pm 0.0321$ | $19.37\%$ | $16.55\%$ | $0.6073$ / $0.6197$ | $0.6682$ / $0.6818$ | $82.88\%$ |
| **Processor B (9Q Slide 4)** | $5$ | $58$ | $30$ | $0$ | $-2.4188 \pm 0.0312$ | $-2.5035 \pm 0.0321$ | $19.37\%$ | $16.55\%$ | $0.6073$ / $0.6197$ | $0.6682$ / $0.6818$ | $82.88\%$ |

> [!NOTE]
> **Key Scientific Takeaway for $N=4$ Core Benchmark:**  
> On the standard 4-spin open chain, SABRE layout maps the circuit to physical qubits $[0, 1, 3, 4]$ on Processor A and $[0, 1, 5, 7]$ on Processor B. Both physical embeddings are isomorphic 4-qubit linear paths. Consequently, both compile to **0 SWAP gates**, identical depth (58 at $n=5$), and identical 2Q gate counts (30 CX). Any small numerical variations are **purely statistical sampling noise** from finite quantum measurement sampling ($N_{\mathrm{shots}} = 4096$). Neither processor possesses an architectural advantage on the 4-qubit open chain.

---

## 🔍 Part II: Extended Architectural Studies (Optional)

Hackathon rules emphasize keeping optional studies clearly separated from the common benchmark. Below are four targeted architectural sensitivity investigations.

### Study A: 5-Qubit Chain Divergence ($N=5$ Open Chain)
When the chain expands to $N=5$ spins, Processor A encounters a topological obstruction (maximum simple path length on its T-graph is 4 vertices):
- **Processor A Bottleneck:** Forces SABRE to insert **3 SWAP gates (+9 CX gates)**, increasing circuit depth from 39 to 64 ($+64.1\%$).
- **Processor B Advantage:** The 9-qubit Slide 4 graph contains simple paths of length 5 (e.g. $[0, 1, 5, 7, 8]$) and compiles with **0 SWAP gates**.
- **Results Over 5 Seeds (mean $\pm$ std):**
  - **Transpiled Depth:** Processor A $= 64$ vs Processor B $= 39$ ($-39.1\%$).
  - **Two-Qubit (CX) Gates:** Processor A $= 33$ vs Processor B $= 24$ ($-27.3\%$).
  - **Raw Energy Error:** Processor A $= 19.09\% \pm 1.29\%$ vs Processor B $= 16.24\% \pm 1.19\%$ ($-2.85$ percentage points).
  - **Mitigated Energy Error:** Processor A $= 16.22\% \pm 1.34\%$ vs Processor B $= 13.27\% \pm 1.24\%$ ($-2.95$ percentage points).
  - **Noisy State Fidelity:** Processor A $= 64.16\%$ vs Processor B $= 81.32\%$ (**$+17.16$ percentage points boost**).

### Study B: Periodic Boundary Conditions (PBC Ring, $N=5$)
Under closed periodic boundaries ($Z_{N-1}Z_0$), graph cycle topology becomes the decisive bottleneck:
- **Processor A (5Q T-Tree, acyclic):** Incurs **9 SWAPs (57 CX)**, depth $= 96$, fidelity $= 65.91\%$.
- **Processor B Tree Variant (9Q Tree, acyclic):** Incurs **9 SWAPs (57 CX)**, depth $= 96$, fidelity $= 65.91\%$.
- **Processor B Primary (9Q Slide 4 Graph, cyclic):** Contains native 5-cycles (e.g. $0-1-5-3-2-0$), compiling with **0 SWAPs (30 CX)**, depth $= 60$, and fidelity $= 77.86\%$ (**$+11.95$ percentage points advantage**).

### Study C: 1st vs 2nd-Order Trotter Discretization
- 1st-order Trotter error scales as $\mathcal{O}(\Delta t)$, achieving minimum total error at $n^* = 3$ ($16.23\%$).
- 2nd-order Strang splitting achieves $\mathcal{O}(\Delta t^2)$ convergence but doubles CX gate density per step. Under $0.5\%$ CX error, noise accumulates faster, shifting the sweet spot to $n^* = 2$ ($17.41\%$).

### Study D: Noise Sensitivity & Analytical $n^*$ Scaling
We derive the continuous optimal step count from commutator norms and two-qubit noise scaling:
$$n^*_{\mathrm{analytical}} \approx \sqrt{\frac{C_{\mathrm{Trotter}}}{C_{\mathrm{Noise}}}} = \sqrt{\frac{\frac{1}{2} \| [A, B] \| t^2}{N_{\mathrm{CX/step}} \cdot \epsilon_{\mathrm{2Q}}}}$$
- At $\epsilon_{\mathrm{2Q}} = 0.2\%$: Analytical continuous $= 5.00$, discrete rounded $= 5$, empirical minimum $= 4$.
- At $\epsilon_{\mathrm{2Q}} = 0.5\%$: Analytical continuous $= 3.16$, discrete rounded $= 3$, empirical minimum $= 3$.
- At $\epsilon_{\mathrm{2Q}} = 1.0\%$: Analytical continuous $= 2.24$, discrete rounded $= 2$, empirical minimum $= 3$.
- At $\epsilon_{\mathrm{2Q}} = 4.0\%$: Analytical continuous $= 1.12$, discrete rounded $= 1$, empirical minimum $= 2$.
Matches empirical discrete simulation within $\pm 1\text{--}2$ steps across all noise regimes.

---

## 🏛️ Part III: Processor D Proposal (Rubric Evaluation Item)

The Hackathon rubric allocates 30 points for hardware architecture exploration and proposals. Drawing on the organizer's published research in hyperbolic graph embeddings and non-Euclidean quantum circuits, we propose **Processor D**:

### Architectural Concept: Hyperbolic $\{5, 4\}$ Tessellation Subgraph
- **Topological Motivation:** While Processor A (T-tree) suffers from simple path bottlenecks ($L_{\max}=4$) and tree graphs incur $\mathcal{O}(N)$ SWAP overhead for periodic rings, planar heavy-hex architectures have finite girth and constant vertex degree.
- **Processor D Graph:** A 16-qubit hyperbolic $\{5, 4\}$ pentagonal cell network with uniform degree 3 or 4 and negative spatial curvature.
- **Key Advantages for Hamiltonian Simulation:**
  1. **Logarithmic Graph Diameter:** Diameter scales as $\mathcal{O}(\log N)$ rather than $\mathcal{O}(N)$, drastically reducing multi-qubit routing distance.
  2. **Native Closed Rings:** Any $k$-spin periodic boundary condition ($k=5, 6, 8$) embeds into hardware with 0 SWAPs.
  3. **High Cut-Width:** Enhances entanglement distribution efficiency for 2D quantum spin liquid simulations.

---

## ⚠️ Known Limitations & Assumptions

1. **Noise Channel Simplification:** The simulation utilizes synthetic Markovian depolarizing noise and single-qubit tensor-product readout errors. Real superconducting processors exhibit coherent rotation errors, spectator crosstalk, and temporal $T_1/T_2$ drift.
2. **Fixed Transpiler Seed:** All compilation runs utilize `seed_transpiler=42` and SABRE heuristic routing to ensure byte-level deterministic reproducibility.
3. **Finite Step Quantization:** Trotter steps are discrete integers ($n \in \mathbb{N}^+$), causing slight divergence from smooth continuous analytical minima at extreme noise thresholds.

---

## 🎮 Interactive Evaluation Tools

### 1. Standalone Web Dashboard (`interactive_dashboard.html`)
Open [`interactive_dashboard.html`](interactive_dashboard.html) directly in any browser:
- Live physical parameter sliders ($J, h, T, N, \epsilon_{\mathrm{2Q}}$).
- Instant U-curve recalculation with analytical overlay.
- Real-time comparison between Processor A, Processor B, and cyclic variants.

### 2. Streamlit Application (`interactive_app.py`)
```bash
streamlit run interactive_app.py
```
- Real-time Aer simulation and live Qiskit circuit transpilation viewer.

### 3. CLI Parameter Explorer (`tools/explore.py`)
```bash
python tools/explore.py --preset default
python tools/explore.py --preset periodic_ring --verbose
```

---

## 📂 Repository Structure

```
S3_Trotter_Simulation/
├── LICENSE                             # Apache-2.0 License
├── README.md                           # Master documentation & benchmark summary
├── report.md                           # Comprehensive scientific paper & architectural report
├── requirements.txt                    # Pinned Python dependencies
├── main.ipynb                          # Self-contained reproducible Jupyter Notebook
├── interactive_dashboard.html          # Client-side standalone interactive UI
├── interactive_app.py                  # Streamlit real-time quantum dashboard
├── run_experiments.py                 # Core simulation pipeline & master CSV generator
├── run_advanced_studies.py             # Advanced studies (PBC, 2nd-order, noise sensitivity)
├── data/
│   ├── raw/
│   │   ├── full_raw_counts.json        # Complete raw shot counts across sweeps & seeds
│   │   └── raw_counts_sample.json      # Representative counts sample for quick inspection
│   └── processed/
│       ├── AB_comparison.csv           # Starter guide alias for AB summary
│       ├── final_comparison.csv        # Rubric-compliant master comparison table
│       ├── processor_A_metrics.csv     # Starter guide alias for Proc A metrics
│       └── processor_B_metrics.csv     # Starter guide alias for Proc B metrics
├── processors/
│   ├── processor_A.json                # Primary: 5Q T-shaped tree graph
│   ├── processor_A_chain.json          # Variant: 5Q linear chain (Kit spec)
│   ├── processor_B.json                # Primary: 9Q Slide 4 heavy-hex graph (12 edges)
│   └── processor_B_tree8.json          # Variant: 9Q acyclic tree (Kit spec, 8 edges)
├── src/
│   ├── problem.py                      # TFIM Hamiltonian, Trotter decomposition, observables
│   ├── processors.py                   # Topology parser & coordinate mapping
│   ├── routing.py                      # SABRE layout, exact SWAP extraction, routing analysis
│   ├── sweep.py                        # Parametric sweep engine & multi-observable tracking
│   ├── metrics.py                      # Error calculation, density matrix fidelity, mitigation
│   ├── cyclic_study.py                 # Optional cyclic heavy-hex variant study
│   └── visualization.py                # Scientific publication-quality figure generation
├── tests/
│   └── test_simulation.py              # Pytest unit and integration test suite (12/12 passing)
├── tools/
│   └── explore.py                      # CLI parameter explorer with presets
└── results/
    ├── figures/                        # High-resolution scientific figures (PNG)
    │   ├── processor_A_graph.png
    │   ├── processor_B_graph.png
    │   ├── circuit_processor_A.png
    │   ├── circuit_processor_B.png
    │   ├── trotter_error_tradeoff.png
    │   ├── circuit_scaling_comparison.png
    │   ├── error_mitigation_study.png
    │   ├── periodic_boundary_routing.png
    │   ├── trotter_order_comparison.png
    │   └── noise_sensitivity.png
    └── tables/                         # Generated data tables (CSV)
        ├── AB_comparison_summary.csv   # N=4 side-by-side comparison table
        ├── final_comparison.csv        # Comprehensive rubric comparison table (n=1,2,4, n*)
        ├── 5qubit_chain_comparison.csv # 5Q chain architectural divergence data
        ├── cyclic_variant_comparison.csv # Cyclic vs tree comparison on OBC and PBC
        ├── pbc_comparison.csv          # Periodic boundary condition comparison
        ├── trotter_order_comparison.csv# 1st vs 2nd-order Strang comparison
        ├── noise_sensitivity.csv       # Optimal step vs 2Q hardware error rate
        ├── trotter_sweep_processor_A.csv
        └── trotter_sweep_processor_B.csv
```

---

## ⚡ Quickstart & Verification Guide

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run unit & integration tests
pytest tests/ -v

# 3. Re-run reproducible pipelines
python run_experiments.py
python run_advanced_studies.py
python -c "import src.cyclic_study as cs; cs.run_cyclic_comparison()"

# 4. Open and execute Jupyter Notebook
jupyter notebook main.ipynb
```
