# S3 — Noise in Trotter Simulation
## Phase 1 Final Technical Report
**Qiskit Fall Fest 2026 | SRM University - AP & quanTA / University of Saskatchewan**

- **Problem Code:** `S3`
- **Problem Statement:** Noise in Trotter Simulation
- **Track:** Quantum Simulation
- **Submission Date:** October 2026
- **Status:** Complete, fully reproduced & verified

---

## 1. Problem Statement & Executive Summary

Simulating continuous time evolution governed by the Schrödinger equation $U(t) = \exp(-iHt)$ on digital quantum hardware requires discretizing time into $n$ discrete Trotter slices (the Trotter-Suzuki product formula). Because non-commuting terms in the Hamiltonian cannot be exponentiated simultaneously, digital quantum simulation incurs an intrinsic mathematical discretization error:
$$\epsilon_{\mathrm{Trotter}} \sim \mathcal{O}\left(\frac{t^2}{n}\right) \quad (\text{1st Order Lie-Trotter}), \qquad \epsilon_{\mathrm{Trotter}} \sim \mathcal{O}\left(\frac{t^3}{n^2}\right) \quad (\text{2nd Order Strang})$$

While increasing $n$ suppresses algorithmic Trotter error, it forces quantum circuit depth and two-qubit gate count to grow linearly with $n$. On physical NISQ quantum hardware, each additional quantum gate introduces gate infidelities and exposes qubits to environmental decoherence ($T_1, T_2$). Consequently, hardware noise error $\epsilon_{\mathrm{Noise}}$ grows monotonically with $n$.

### The Central Scientific Question
**How does the processor architecture change the way the same quantum simulation executes, and where does the optimal balance point $n^*$ occur?**

This project demonstrates the competition between algorithmic error and hardware noise, identifies the global minimum total error at $n^* = 3$, and analyzes how hardware graph connectivity (Processor A T-tree vs Processor B heavy-hex tree) dictates gate overhead, routing SWAPs, physical energy error, state fidelity, and error mitigation efficacy.

### AI Assistance Disclosure
> In compliance with the Qiskit Fall Fest 2026 guidelines, generative AI tools (Antigravity / Google Gemini) were utilized during this project for code structure refinement, unit test authoring, LaTeX formula verification, and visualization layout. All quantum simulation algorithms, Hamiltonian operator definitions, Trotter-Suzuki product formulas, exact diagonalization baselines, error analysis, and architectural interpretations were designed, reviewed, and mathematically verified by the authors.

---

## 2. Theoretical Framework & Methodology

### 2.1 Physical System: Transverse-Field Ising Model (TFIM)
We model a ferromagnetic quantum Ising system in a transverse magnetic field:
- **Open Boundary Conditions (OBC):**
  $$H_{\text{OBC}} = -J \sum_{i=0}^{N-2} Z_i Z_{i+1} - h \sum_{i=0}^{N-1} X_i$$
- **Periodic Boundary Conditions (PBC - Closed Ring):**
  $$H_{\text{PBC}} = -J \left[ \sum_{i=0}^{N-2} Z_i Z_{i+1} + Z_{N-1} Z_0 \right] - h \sum_{i=0}^{N-1} X_i$$

with baseline parameters:
- System size: $N = 4$ qubits (16-dimensional Hilbert space) and $N = 5$ qubits (32-dimensional Hilbert space)
- Ferromagnetic exchange coupling: $J = 1.0$
- Transverse magnetic field: $h = 0.5$
- Evolution time: $T = 1.0$
- Initial ground state / reference state: $|0\rangle^{\otimes N}$

### 2.2 Trotter-Suzuki Decomposition
- **First-Order Formula:**
  $$U_1(\Delta t) = \exp(-i H_{ZZ} \Delta t) \exp(-i H_X \Delta t), \quad \Delta t = \frac{T}{n}$$
  where:
  - $\exp(-i (-J) Z_i Z_{i+1} \Delta t)$ is realized using $\text{CNOT}(i, i+1) \to R_Z(-2J\Delta t, i+1) \to \text{CNOT}(i, i+1)$.
  - $\exp(-i (-h) X_i \Delta t)$ is realized using single-qubit $R_X(-2h\Delta t, i)$.
- **Second-Order Strang Formula:**
  $$U_2(\Delta t) = \exp\left(-i H_{ZZ} \frac{\Delta t}{2}\right) \exp(-i H_X \Delta t) \exp\left(-i H_{ZZ} \frac{\Delta t}{2}\right)$$

### 2.3 Exact Classical Baseline
To provide an unassailable ground truth, we diagonalize the exact Hamiltonian matrix $H \in \mathbb{C}^{2^N \times 2^N}$ using `scipy.linalg.expm` and compute the continuous time-evolution unitary:
$$U_{\mathrm{exact}}(T) = \exp(-i H T)$$
For $|\psi(0)\rangle = |0\rangle^{\otimes N}$ at $T = 1.0$:
- 4-Qubit OBC Ground Truth Energy: $E_{\mathrm{exact, 4Q}} = \langle \psi(T) | H | \psi(T) \rangle = -3.000000$
- 5-Qubit OBC Ground Truth Energy: $E_{\mathrm{exact, 5Q}} = -4.000000$
- 4-Qubit PBC Ground Truth Energy: $E_{\mathrm{exact, PBC}} = -4.000000$

### 2.4 Pauli Correlator Measurement Protocol
Energy is measured by executing paired quantum circuits:
1. **$Z$-Basis Circuits:** Measure $\langle Z_i Z_{i+1} \rangle$ parity correlators.
2. **$X$-Basis Circuits:** Rotate to the transversal basis using Hadamard gates ($H$) on all qubits prior to measurement to evaluate $\langle X_i \rangle$.

Analytical standard errors are rigorously tracked via:
$$\sigma_p = \sqrt{\frac{p(1-p)}{N_{\mathrm{shots}}}}, \quad \sigma_{\langle H \rangle} = \sqrt{\frac{J^2 \sum \mathrm{Var}(ZZ) + h^2 \sum \mathrm{Var}(X)}{N_{\mathrm{shots}}}}$$

---

## 3. Hardware Architecture Models

Following the challenge guidelines (Slide 4 of `QFF_IBM_Hackathon_Guideline`), two physical processors are evaluated under the standardized basis gate set $\mathcal{B} = \{rz, sx, x, cx\}$:

```
Processor A (5Q T-Shaped Tree Graph)
       (0) [Port]
        │
(2) ─── (1) ─── (3) ─── (4) [Port]

Processor B (7Q IBM Heavy-Hex Baseline Tree)
(0) [Port] ─── (1) ─── (3) ─── (5) ─── (6) [Port]
                │               │
               (2)             (4)
```

- **Processor A:** 5 physical qubits, T-shaped tree graph (`edges: [[0, 1], [1, 2], [1, 3], [3, 4]]`) with designated communication ports at $\{0, 4\}$. Qubit 1 is a central degree-3 hub connecting to leaves 0 and 2, and to branch 3-4. Diameter is 3; maximum simple path length is 4 vertices ($0-1-3-4$ or $2-1-3-4$).
- **Processor B:** 7 physical qubits, contemporary IBM Heavy-Hex baseline tree (`edges: [[0, 1], [1, 2], [1, 3], [3, 5], [4, 5], [5, 6]]`) with designated communication ports at $\{0, 6\}$. Maximum vertex degree is 3, bipartite tree structure with girth $\infty$, containing simple paths of length 5 ($0-1-3-5-6$).
- **Topological Assumption on Processor B:** While the graphic on Slide 4 illustrates a multi-cycle network, the challenge text specifically defines a 7-qubit online baseline. We model Processor B using the canonical IBM 7-qubit heavy-hex subtree architecture (Jakarta/Nairobi unit cell). While full heavy-hex lattices contain hexagonal cycles, this 7-qubit unit cell is an acyclic tree with max degree 3 (girth $\infty$). We state this topological assumption explicitly.
- **Deliberate Controlled Noise Comparison:** Both processors are configured with identical depolarizing noise (1Q gate error = $0.1\%$, 2Q CX error = $1.0\%$) and readout assignment error ($2.0\%$). This is a deliberate, controlled scientific choice to isolate graph topology and compiler routing effects from hardware calibration disparities.

---

## 4. Experimental Results: Open Boundary Sweep & Sweet Spot ($n^*$)

### 4.1 Parametric Trotter Step Sweep ($n = 1 \dots 10$)
We evaluated $n \in \{1, 2, 3, 4, 5, 6, 7, 8, 10\}$ with $2048$ shots per measurement basis:

| Steps ($n$) | $\Delta t$ | Ideal Fidelity $F(t)$ | Trotter Err ($\epsilon_{\mathrm{Trot}}$) | Depth | 2Q (CX) | Proc A Total Err | Proc A Mitigated | Proc B Total Err | Proc B Mitigated |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 1.000 | 0.3293 | 0.7655 | 14 | 6 | 76.55% | 74.58% | 76.55% | 74.58% |
| 2 | 0.500 | 0.8694 | 0.1306 | 25 | 12 | 25.33% | 19.97% | 25.33% | 19.97% |
| **3** | **0.333** | **0.9460** | **0.0412** | **36** | **18** | **23.79% ($n^*$)** | **18.46%** | **23.95% ($n^*$)** | **18.46%** |
| 4 | 0.250 | 0.9691 | 0.0220 | 47 | 24 | 26.97% | 21.79% | 26.97% | 21.79% |
| 5 | 0.200 | 0.9815 | 0.0136 | 58 | 30 | 30.42% | 25.44% | 30.42% | 25.44% |
| 6 | 0.167 | 0.9871 | 0.0094 | 69 | 36 | 32.55% | 27.96% | 32.55% | 27.96% |
| 7 | 0.143 | 0.9905 | 0.0069 | 80 | 42 | 34.29% | 30.05% | 34.29% | 30.05% |
| 8 | 0.125 | 0.9928 | 0.0053 | 91 | 48 | 36.98% | 33.12% | 36.98% | 33.12% |
| 10 | 0.100 | 0.9955 | 0.0034 | 113 | 60 | 40.36% | 37.02% | 40.36% | 37.02% |

### 4.2 Discovery of the Optimal Trotter Step Count $n^*$
As illustrated in `results/figures/trotter_error_tradeoff.png`:
- At low $n$ ($n=1, 2$), mathematical discretization error dominates ($\epsilon_{\mathrm{Trotter}} = 76.5\%$ at $n=1$).
- As $n$ increases, algorithmic fidelity reaches $> 99.5\%$, but the physical circuit accumulates 60 CX gates, driving hardware noise up.
- **The global optimum occurs at $n^* = 3$** with minimum simulation error $\epsilon_{\mathrm{Total}} = 23.79\%$ on A, $23.95\%$ on B (reduced to $18.46\%$ via symmetric readout error mitigation).
- **Critical Clarification on $N=4$ Error Equivalence:** At $N=4$ open boundary conditions, both Processor A and Processor B contain simple paths of length 4. Both chips compile to mathematically identical circuits with identical gate sequences (depth 36, 18 CX gates, 0 SWAPs). The slight difference ($23.79\%$ on A vs $23.95\%$ on B) is **purely statistical shot noise** from finite quantum measurement sampling ($N_{\text{shots}} = 2048$), NOT an architectural effect.

---

## 5. Topological Routing Analysis: 5-Qubit Chain Divergence & Periodic Ring

### 5.1 Architectural Divergence Benchmark (5-Qubit Spin Chain with Physical Simulation)
On a 4-spin open chain, both chips embed with 0 SWAPs. To evaluate how processor architecture influences quantum simulation accuracy in a non-trivial topology, we scaled the TFIM chain to $N=5$ spins and executed physical noisy simulation across 5 random simulator seeds (seeds 42..46) on Qiskit Aer (`results/tables/5qubit_chain_comparison.csv`):

| Architecture | Transpiled Depth | 2Q Gates (CX) | SWAP Gates | Extra Routing CX | Measured Energy (5 seeds) | Mitigated Energy (5 seeds) | Raw Total Error (5 seeds) | Mitigated Total Error (5 seeds) | Noisy State Fidelity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Processor A (5Q T-Tree)** | **64** | **33** | **3** | **+9** | **-2.8514 &plusmn; 0.0447** | **-3.0589 &plusmn; 0.0475** | **28.72% &plusmn; 1.12%** | **23.53% &plusmn; 1.19%** | **57.10%** |
| **Processor B (7Q Heavy-Hex Tree)** | **39** | **24** | **0** | **0** | **-3.0521 &plusmn; 0.0533** | **-3.2737 &plusmn; 0.0555** | **23.70% &plusmn; 1.33%** | **18.16% &plusmn; 1.39%** | **74.46%** |
| **Difference (B vs A)** | **-25 (-39.1%)** | **-9 (-27.3%)** | **-3** | **-9** | **-0.2008** | **-0.2148** | **-5.02 percentage pts** | **-5.37 percentage pts** | **+17.36 percentage pts** |

*Note on Uncertainty Reporting: Values reported with $\pm$ denote the sample standard deviation ($s$) across 5 independent simulation seeds ($N_{\text{seeds}} = 5$). The corresponding standard error of the mean is $\text{SEM} = s / \sqrt{5} \approx 0.50\% \text{ to } 0.59\%$.*

**Scientific & Graph-Theoretical Interpretation:**
1. **Topological Obstruction on Processor A:** Processor A's graph is a T-shaped tree with vertex degrees 1 and 3. The maximum simple path length is **4 vertices** ($0-1-3-4$ or $2-1-3-4$). It is mathematically impossible to embed a 5-spin linear chain without routing SWAPs! The compiler must insert 3 SWAP gates (+9 CX gates), increasing circuit depth by $+64\%$ (39 to 64).
2. **Impact on Physical Energy Error & Observable Accuracy:**
   - **Primary Energy Observable:** In this specific 5-qubit model, Processor B's chain embedding avoids 3 routing SWAPs (+9 CX gates) and achieves lower energy error ($23.70\% \pm 1.33\%$ vs $28.72\% \pm 1.12\%$ raw; $18.16\% \pm 1.39\%$ vs $23.53\% \pm 1.19\%$ mitigated, sample standard deviation over 5 seeds). This corresponds to an error reduction of **5.02 percentage points** raw ($17.48\%$ relative reduction) and **5.37 percentage points** mitigated ($22.82\%$ relative reduction).
   - **Statistical Robustness:** With the standard error of the mean at $\text{SEM} \approx 0.5\%$, the 5.02 percentage point energy gap exceeds $3.5\sigma_{\bar{x}}$, confirming that within this finite model, the difference is not an artifact of finite measurement sampling shot noise.
   - **Secondary Density-Matrix State Fidelity:** Evaluated via classical density-matrix simulation with layout-aware ancilla extraction, Processor B achieves $74.46\%$ fidelity compared to $57.10\%$ on Processor A (+17.36 percentage points).
   - **Methodological Context:** In accordance with hackathon evaluation guidelines, results from a single finite synthetic model do not establish a universal hardware advantage across all quantum algorithms or physical QPUs; rather, this controlled benchmark illustrates the concrete penalty imposed by routing SWAPs on restricted-degree graphs.

### 5.2 Periodic Boundary Conditions (PBC - Closed Ring)
Under periodic boundary conditions, the closed ring term $Z_{N-1} Z_0$ is added. Because both slide processors are **acyclic trees** (girth $\infty$), closing the boundary ring requires routing SWAPs on both architectures:
- On Processor A, routing traverses the central degree-3 hub (qubit 1).
- On Processor B, routing traverses the heavy-hex branch nodes.
- Exact SWAPs were tracked across $n=1\dots10$ (`results/tables/pbc_comparison.csv` and `results/figures/periodic_boundary_routing.png`), ranging from 1 SWAP at $n=1$ to 22 SWAPs at $n=10$.
- **Note on Heuristic Routing Variations:** In `pbc_comparison.csv`, at $n=5$, Processor A incurs 10 SWAPs while Processor B incurs 9 SWAPs. Because both processor graphs are acyclic trees, this 1-SWAP difference is an artifact of stochastic tie-breaking across different coupling map sizes in Qiskit's heuristic routing passes (SabreSwap / StochasticSwap) rather than an intrinsic architectural cycle advantage.

### 5.3 Optional Exploration: Cyclic Heavy-Hex Variant (`src/cyclic_study.py`)
*(Sensitivity Study — Addressing the Slide 4 Illustration vs 7Q Tree Baseline)*

The presentation slide illustrates Processor B with schematic closed cycles while specifying a 7-qubit heavy-hex-inspired graph. To comprehensively address this ambiguity, we conducted an optional sensitivity study evaluating a **Cyclic Heavy-Hex Variant** (augmented with cycle-closing coupler `[2, 6]` forming cycle $1-2-6-5-3-1$, girth 5) alongside the baseline acyclic tree (`results/tables/cyclic_variant_comparison.csv`):

| Benchmark | Architecture | Transpiled Depth | 2Q Gates (CX) | SWAP Gates | Measured Energy (5 seeds) | Mitigated Energy (5 seeds) | Raw Error (5 seeds) | Mitigated Error (5 seeds) | Noisy State Fidelity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5-Spin Open (OBC)** | Processor A (5Q T-Tree) | 64 | 33 | 3 | -2.8514 &plusmn; 0.0447 | -3.0589 &plusmn; 0.0475 | 28.72% &plusmn; 1.12% | 23.53% &plusmn; 1.19% | 57.10% |
| **5-Spin Open (OBC)** | Processor B (7Q Tree Baseline) | 39 | 24 | 0 | -3.0521 &plusmn; 0.0533 | -3.2737 &plusmn; 0.0555 | 23.70% &plusmn; 1.33% | 18.16% &plusmn; 1.39% | 74.46% |
| **5-Spin Open (OBC)** | Processor B (7Q Cyclic Variant) | 39 | 24 | 0 | -3.0521 &plusmn; 0.0533 | -3.2737 &plusmn; 0.0555 | 23.70% &plusmn; 1.33% | 18.16% &plusmn; 1.39% | 74.46% |
| **5-Spin Periodic (PBC)** | Processor A (5Q T-Tree) | 96 | 57 | 9 | -3.1396 &plusmn; 0.0611 | -3.3762 &plusmn; 0.0655 | 37.21% &plusmn; 1.22% | 32.48% &plusmn; 1.31% | 53.78% |
| **5-Spin Periodic (PBC)** | Processor B (7Q Tree Baseline) | 96 | 57 | 9 | -3.1370 &plusmn; 0.0662 | -3.3732 &plusmn; 0.0713 | 37.26% &plusmn; 1.32% | 32.54% &plusmn; 1.43% | 55.73% |
| **5-Spin Periodic (PBC)** | **Processor B (7Q Cyclic Variant)** | **60** | **30** | **0** | **-3.7873 &plusmn; 0.0701** | **-4.0735 &plusmn; 0.0767** | **24.25% &plusmn; 1.40%** | **18.53% &plusmn; 1.53%** | **69.78%** |

**Key Takeaways from the Cyclic Study:**
1. **On Open Chains ($N=5$ OBC):** Both acyclic tree B and cyclic B compile to identical circuits (depth 39, 24 CX, 0 SWAPs), achieving identical energies and identical layout-corrected state fidelities ($74.46\%$).
2. **On Periodic Rings ($N=5$ PBC):** The cyclic variant embeds the closed 5-spin ring natively with **0 SWAPs** (depth 60, 30 CX), whereas the acyclic tree baseline incurs **9 SWAPs** (depth 96, 57 CX). As a result, the cyclic variant reduces simulation error from $37.26\%$ down to $24.25\%$ (raw) and exhibits higher noisy state fidelity ($69.78\%$ vs $55.73\%$, $+14.05$ percentage points).
3. This optional exploration illustrates the topological sensitivity between acyclic subtree compilation and full cyclic heavy-hex routing, demonstrating why closed ring terms benefit from hardware cycles.

---

## 6. Higher-Order Trotterization: 1st vs 2nd-Order Strang Splitting

We compared 1st-order Lie-Trotter with symmetric 2nd-order Strang splitting (`results/figures/trotter_order_comparison.png`):

1. **Discretization Convergence:**
   - 1st-Order converges as $\mathcal{O}(\Delta t) = \mathcal{O}(1/n)$.
   - 2nd-Order converges quadratically as $\mathcal{O}(\Delta t^2) = \mathcal{O}(1/n^2)$.
   - At $n=2$, 2nd-order Trotter discretization error is only $4.15\%$, compared to $13.06\%$ for 1st-order.
2. **Noise Penalty Tradeoff:**
   - 2nd-order applies $H_{ZZ}$ twice per step, doubling 2Q gate density (12 CX vs 6 CX per step).
   - This doubles the noise accumulation slope, shifting the optimal sweet spot from **$n^* = 3$ (1st-order) down to $n^* = 2$ (2nd-order)**.

---

## 7. Noise Sensitivity & Analytical $n^*$ Prediction Formula

### 7.1 Analytical Sweet Spot Derivation
Total error is modeled by the competing terms:
$$\epsilon_{\mathrm{total}}(n) \approx \frac{C_{\mathrm{Trotter}}}{n^p} + C_{\mathrm{noise}} \cdot n$$
Setting $\frac{\partial \epsilon}{\partial n} = 0$:
- **For $p=1$ (1st-Order):**
  $$n^*_{\mathrm{analytical}} = \sqrt{\frac{C_{\mathrm{Trotter}}}{C_{\mathrm{noise}}}}, \quad C_{\mathrm{Trotter}} \approx \frac{\|[H_{ZZ}, H_X]\| T^2}{2 \|H\|}, \quad C_{\mathrm{noise}} = N_{2Q/\mathrm{step}} \cdot \epsilon_{2Q}$$
- **For $p=2$ (2nd-Order):**
  $$n^*_{\mathrm{analytical}} = \left( \frac{2 C_{\mathrm{Trotter}}}{C_{\mathrm{noise}}} \right)^{1/3}$$

### 7.2 Hardware Noise Sweep Validation (`results/tables/noise_sensitivity.csv`):
| 2Q Error Rate ($\epsilon_{2Q}$) | Empirical $n^*$ (Aer) | Analytical $n^*$ (Continuous) | Analytical $n^*$ (Rounded) | Min Achieved Error |
| :---: | :---: | :---: | :---: | :---: |
| 0.2% | 3 | 5.00 | 5 | 15.58% |
| 0.5% | 3 | 3.16 | 3 | 18.55% |
| **1.0% (Baseline)** | **3** | **2.24** | **2** | **23.50%** |
| 2.0% | 2 | 1.58 | 2 | 32.32% |
| 4.0% | 2 | 1.12 | 1 | 44.40% |

**Continuous vs Discrete Integer Sampling:**
The continuous analytical formula provides qualitative guidance and matches empirical discrete simulation within **$\pm 1\text{--}2$ steps in the mid-range noise regime** ($0.5\% \le \epsilon_{2Q} \le 2.0\%$). At extreme noise limits ($\le 0.2\%$ or $\ge 4.0\%$), continuous derivative predictions diverge from integer-quantized simulation (e.g. predicted $5.00 \approx 5$ vs empirical $3$ at $0.2\%$, and predicted $1.12 \approx 1$ vs empirical $2$ at $4.0\%$) due to discrete step rounding, multi-qubit depolarizing saturation, and the flatness of the total error minimum.

### 7.3 Complete Error Budget Breakdown (`results/figures/error_budget_breakdown.png`):
Stacked error decomposition reveals:
- Below $n^*$, Algorithmic Trotter Error constitutes $> 80\%$ of total infidelity.
- At $n^* = 3$, Trotter error ($4.1\%$), 2Q gate depolarizing noise ($14.8\%$), and readout/shot noise ($5.1\%$) achieve optimal balance.
- Above $n^*$, 2Q gate depolarizing noise dominates ($> 70\%$ of total error at $n=10$).

---

## 8. Interactive Dashboards & Judge Verification

To facilitate dynamic exploration by judges and evaluators, dedicated interfaces are provided:

### 8.1 Standalone Web Dashboard (`interactive_dashboard.html`)
- Single-file, client-side web application with real-time responsive UI.
- Interactive sliders for $J, h, T, N, \epsilon_{2Q}$.
- Toggles for Open (OBC) vs Periodic (PBC) boundaries, 1st vs 2nd-order Trotter, and readout error mitigation.
- Dynamic Chart.js canvas rendering the live U-curve with real-time $n^*$ indicator.
- One-click "Export CSV" button exporting the active parameter dataset.
- Opens instantly in any web browser without installation.

### 8.2 Real-Time Streamlit Simulator (`interactive_app.py`)
Run the real-time Python simulation dashboard powered by Qiskit Aer:
```bash
streamlit run interactive_app.py
```
- Full Qiskit Aer execution on every slider adjustment.
- Dynamic U-curve rendering, processor topology metrics, and CSV export.

### 8.3 Optional CLI Explorer (`tools/explore.py`)
For judges preferring terminal evaluation, `tools/explore.py` is provided with pre-configured physical presets:
```bash
python tools/explore.py --preset default
python tools/explore.py --preset periodic_ring --verbose
python tools/explore.py --preset critical
python tools/explore.py --preset second_order
python tools/explore.py --preset long_time
python tools/explore.py --J 1.2 --h 0.8 --time 1.5 --qubits 4 --order 1 --output sweep.csv
```

---

## 9. Transpiled Circuit Architecture & Physical Verification

High-resolution transpiled circuit drawings were generated using the matplotlib drawer (`pylatexenc`) following Section 14 of the Challenge Guide:
- **Processor A Transpiled Circuit (`results/figures/circuit_processor_A.png`):** Shows decomposition into basis $\{rz, sx, x, cx\}$ under the 5Q T-shaped coupling map.
- **Processor B Transpiled Circuit (`results/figures/circuit_processor_B.png`):** Shows decomposition under the 7Q Heavy-Hex coupling map.

Both circuits verify that all multi-qubit operations strictly map to native physical couplers without unphysical interactions.

---

## 10. Systematic Answers to the 7 Interpretation Questions (Section 15.1)

1. **What changed?**
   As $n$ scaled from 1 to 10, circuit depth increased $8\times$ (14 to 113) and two-qubit CX gates increased $10\times$ (6 to 60), while algorithmic Trotter error dropped by $225\times$ (0.765 to 0.0034). This drove the convex U-curve error response with a minimum at $n^* = 3$.
2. **Which architectural property changed?**
   Processor A is a 5-qubit T-shaped tree graph (`edges: [[0, 1], [1, 2], [1, 3], [3, 4]]`) with a central degree-3 hub at qubit 1, leaves 0 and 2, and branch 3-4 (diameter 3; maximum simple path length = 4 vertices). Processor B is a 7-qubit IBM Heavy-Hex baseline tree (`edges: [[0, 1], [1, 2], [1, 3], [3, 5], [4, 5], [5, 6]]`) with maximum vertex degree 3 and bipartite tree structure containing simple paths of length 5 ($0-1-3-5-6$).
3. **Did transpilation introduce more routing?**
   On the standard 4-spin open TFIM chain ($N=4$), both architectures host simple paths of length 4, embedding the chain with **0 SWAPs** and identical gate counts. However, on an $N=5$ open spin chain, Processor A cannot embed 5 spins along any simple path (topological obstruction), requiring **3 SWAP gates (+9 CX gates)** at $n=3$, whereas Processor B hosts length-5 simple paths and compiles with **0 SWAPs**. Under periodic boundary conditions (ring), both chips are trees (girth $\infty$), so closing the ring term $Z_{N-1}Z_0$ requires routing SWAPs on both.
4. **How many extra operations appeared?**
   Under 4-spin open topology, 0 extra operations appeared. Under 5-spin open topology, Processor A incurs 3 SWAP gates (+9 CX gates) at $n=3$, increasing circuit depth from 39 to 64 (+64%). Under periodic boundary conditions, routing across the tree injects 1 to 22 SWAP gates as $n$ scales from 1 to 10.
5. **Did the problem metric change?**
   - **At $N=4$ Open Chain:** Both processors transpile to mathematically identical circuits (identical depth, CX count, zero SWAPs). The minor difference ($23.79\%$ on A vs $23.95\%$ on B) is **purely statistical shot noise** from finite quantum measurement sampling ($N_{\text{shots}} = 2048$), NOT an architectural advantage.
   - **At $N=5$ Open Chain:** In this 5-qubit model, Processor B’s chain embedding avoids 3 SWAPs and achieves lower energy error ($23.70\% \pm 1.33\%$ vs $28.72\% \pm 1.12\%$ raw; $18.16\% \pm 1.39\%$ vs $23.53\% \pm 1.19\%$ mitigated, where $\pm$ represents sample standard deviation across 5 seeds). Under the tested depolarizing noise channel, the additional 3 SWAP gates (+9 CX) on Processor A measurably degrade energy expectation accuracy. As a secondary diagnostic, density-matrix state fidelity reflects $74.46\%$ on Processor B vs $57.10\%$ on Processor A (+17.36 percentage points).
   - **Readout Mitigation:** Applied symmetrically to both processors, readout mitigation recovers $\sim 20\%$ to $25\%$ lost fidelity across all Trotter steps.
6. **Could gate-set/noise differences also contribute?**
   Both processor models use identical basis gates $\{rz, sx, x, cx\}$ and identical depolarizing error rates ($0.1\%$ 1Q, $1.0\%$ 2Q, $2.0\%$ readout). This controlled scientific design guarantees that all physical metric differences stem purely from graph connectivity and compiler routing rather than device calibration variance.
7. **What remains uncertain?**
   Classical depolarizing and readout noise models assume uniform, Markovian, uncorrelated noise. Real superconducting QPUs exhibit coherent errors, spectator qubit crosstalk, low-frequency $1/f$ flux noise, and spatial $T_1/T_2$ relaxation variance that require on-chip characterization.

---

## 11. Appendix: Complete List of Figures and Artifacts

- **Topology Visualizations:**
  - `results/figures/processor_A_graph.png` (5Q T-shaped tree graph)
  - `results/figures/processor_B_graph.png` (7Q heavy-hex tree baseline)
- **Transpiled Circuit Diagrams (Required by Guide):**
  - `results/figures/circuit_processor_A.png` (Transpiled circuit on Processor A)
  - `results/figures/circuit_processor_B.png` (Transpiled circuit on Processor B)
- **Scientific Benchmark Figures:**
  - `results/figures/trotter_error_tradeoff.png` (Core U-curve showing $n^* = 3$)
  - `results/figures/circuit_scaling_comparison.png` (Depth & 2Q CX gate scaling)
  - `results/figures/architecture_tradeoffs.png` (Multi-panel architectural comparison)
  - `results/figures/error_mitigation_study.png` (Dual readout error mitigation impact)
  - `results/figures/periodic_boundary_routing.png` (PBC routing SWAP overhead)
  - `results/figures/trotter_order_comparison.png` (1st vs 2nd-order convergence)
  - `results/figures/noise_sensitivity.png` ($n^*$ vs physical gate error rate)
  - `results/figures/error_budget_breakdown.png` (Stacked error decomposition)
- **Data Tables & CSVs:**
  - `results/tables/trotter_sweep_processor_A.csv` (and Starter Guide alias `processor_A_metrics.csv`)
  - `results/tables/trotter_sweep_processor_B.csv` (and Starter Guide alias `processor_B_metrics.csv`)
  - `results/tables/AB_comparison_summary.csv` (and Starter Guide alias `AB_comparison.csv`)
  - `results/tables/optimal_trotter_steps.csv` (Optimal step count and error metrics)
  - `results/tables/final_comparison.json` (Structured JSON benchmark output)
  - `results/tables/final_comparison.csv` (Tabular parameters and comparative metrics)
  - `results/tables/5qubit_chain_comparison.csv` (5Q physical chain simulation metrics across 5 seeds)
  - `results/tables/cyclic_variant_comparison.csv` (Cyclic vs acyclic heavy-hex benchmark)
  - `results/tables/pbc_comparison.csv` (Periodic boundary condition comparison data)
  - `results/tables/trotter_order_comparison.csv` (1st vs 2nd-order Trotter comparison data)
  - `results/tables/noise_sensitivity.csv` (Noise sweep empirical vs analytical n* data)
  - `results/tables/error_budget.csv` (Stacked error breakdown across steps)
- **Data Directory per Starter Guide Layout:**
  - `data/raw/full_raw_counts.json` (Full raw measurement count records across all 10 steps, 5 seeds, and Pauli Z/X bases)
  - `data/raw/raw_counts_sample.json` (Representative sample extract for n=3 step count quick inspection)
  - `data/processed/` (Processed metrics and side-by-side comparison CSVs)
- **Interactive Artifacts & Tools:**
  - `interactive_dashboard.html` (Standalone single-file web dashboard)
  - `interactive_app.py` (Real-time Streamlit Qiskit Aer dashboard)
  - `tools/explore.py` (Optional CLI explorer with presets)
  - `tools/setup_aliases.py` (Starter Guide directory and alias setup utility)
