"""
Script to update main.ipynb with:
1. Multi-seed 5-qubit statistics and error bars (+/-)
2. Exact noise sensitivity numbers (matching noise_sensitivity.csv: 5.00, 3.16, 2.24, 1.58, 1.12)
3. Inclusion and display of the optional cyclic heavy-hex variant comparison table (results/tables/cyclic_variant_comparison.csv)
4. Leading with energy error and toning down overclaiming language per hackathon guidelines
"""

import json
from pathlib import Path

notebook_path = Path("main.ipynb")

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))

    # Cell 18: Question 5 update
    if "Did the problem metric change?" in src and "At $N=5$ Open Chain:" in src:
        # Replace the N=5 sub-bullet with toned-down wording leading with energy error
        old_bullet_starts = [
            "   - **At $N=5$ Open Chain:** Processor B demonstrates genuine architectural superiority",
            "   - **At $N=5$ Open Chain:** In this 5-qubit model"
        ]
        new_bullet = (
            "   - **At $N=5$ Open Chain:** In this 5-qubit model, Processor B’s chain embedding avoids 3 SWAPs "
            "and achieves lower energy error ($23.70\\% \\pm 1.33\\%$ vs $28.72\\% \\pm 1.12\\%$ raw; "
            "$18.16\\% \\pm 1.39\\%$ vs $23.53\\% \\pm 1.19\\%$ mitigated, where $\\pm$ represents sample standard deviation "
            "across 5 seeds, with standard error of the mean $\\text{SEM} \\approx 0.5\\%$). Under the tested depolarizing noise channel, "
            "the additional 3 SWAP gates (+9 CX) on Processor A measurably degrade energy expectation accuracy. As a secondary diagnostic, "
            "density-matrix state fidelity reflects $74.46\\%$ on Processor B vs $57.10\\%$ on Processor A (+17.36 percentage points).\n"
        )
        lines = cell["source"]
        new_lines = []
        skip = False
        for line in lines:
            if any(line.strip().startswith(prefix.strip()) for prefix in old_bullet_starts):
                new_lines.append(new_bullet)
                skip = True
            elif skip and line.strip().startswith("- **Readout Mitigation:**"):
                skip = False
                new_lines.append(line)
            elif not skip:
                new_lines.append(line)
        cell["source"] = new_lines
        print(f"Updated Question 5 in Cell {idx}")

    # Cell 25: Noise sensitivity text update
    if "Continuous vs Discrete Integer Sampling" in src and "predicted $5.17" in src:
        src = src.replace(
            "predicted $5.17 \\approx 5$ vs empirical $3$ at $0.2\\%$, and predicted $1.16 \\approx 1$ vs empirical $2$ at $4.0\\%",
            "predicted $5.00 \\approx 5$ vs empirical $3$ at $0.2\\%$, and predicted $1.12 \\approx 1$ vs empirical $2$ at $4.0\\%"
        )
        cell["source"] = [line + "\n" for line in src.split("\n")[:-1]] + ([src.split("\n")[-1]] if src.split("\n")[-1] else [])
        print(f"Updated Noise Sensitivity text in Cell {idx}")

    # Cell 29: Section 16 Markdown description
    if "16. Architectural Divergence Benchmark (5-Qubit Chain)" in src:
        src_text = (
            "## 16. Architectural Divergence Benchmark (5-Qubit Chain) & Transpiled Circuit Verification\n\n"
            "On a 4-spin open chain, both Processor A and Processor B embed the problem graph with 0 SWAPs because both chips contain simple paths of length $\\ge 4$. "
            "To **evaluate architectural sensitivity on open spin chains**, we scale the TFIM chain to $N=5$ spins and perform physical noisy simulation across 5 seeds:\n\n"
            "- **Processor A Topological Obstruction:** Processor A's T-shaped graph has a central hub of degree 3 and leaves at qubits 0 and 2. The maximum simple path length is **4 vertices** ($0-1-3-4$ or $2-1-3-4$). "
            "Embedding a 5-spin chain is topologically impossible without routing, forcing the transpiler to inject **3 SWAP gates (+9 CX gates)** at $n=3$, driving depth from 39 to 64 (+64% overhead).\n"
            "- **Primary Physical Metric (Energy Error):** In physical noisy simulation across 5 random seeds (depolarizing + readout noise), Processor B achieves lower energy error: "
            "**$23.70\\% \\pm 1.33\\%$ vs $28.72\\% \\pm 1.12\\%$** raw (-5.02 percentage points) and **$18.16\\% \\pm 1.39\\%$ vs $23.53\\% \\pm 1.19\\%$** mitigated (-5.37 percentage points), "
            "where $\\pm$ denotes sample standard deviation ($s$) and $\\text{SEM} \\approx 0.5\\%$.\n"
            "- **Secondary Metric (State Fidelity):** In density-matrix simulation with layout-aware ancilla extraction, Processor B preserves **$74.46\\%$** state fidelity compared to Processor A's **$57.10\\%$** (+17.36 percentage points).\n"
            "- **Optional Cyclic Variant Study:** To address the presentation slide schematic displaying cycles, we also include an **Optional Exploration: Cyclic Heavy-Hex Variant Study** (`results/tables/cyclic_variant_comparison.csv`) "
            "comparing the acyclic tree baseline against a cyclic heavy-hex variant on open chain and periodic ring geometries.\n\n"
            "Below, we display the complete quantitative physical benchmark tables and the **required transpiled circuit diagrams** (Section 14 of Challenge Guide) for both processors.\n"
        )
        cell["source"] = [line + "\n" for line in src_text.split("\n")[:-1]] + ([src_text.split("\n")[-1]] if src_text.split("\n")[-1] else [])
        print(f"Updated Section 16 description in Cell {idx}")

    # Cell 30: Code cell displaying 5Q table
    if "df_5q = pd.read_csv('results/tables/5qubit_chain_comparison.csv')" in src:
        if "results/tables/cyclic_variant_comparison.csv" not in src:
            new_code = src + """\n# Optional Exploration: Cyclic Heavy-Hex Variant Study (Slide 4 Sensitivity)
try:
    df_cyc = pd.read_csv('results/tables/cyclic_variant_comparison.csv')
    print('\\n=== Optional Exploration: Cyclic Heavy-Hex Variant Comparison Table ===')
    display(df_cyc)
except Exception as e:
    print('Cyclic table note:', e)
"""
            cell["source"] = [line + "\n" for line in new_code.split("\n")[:-1]] + ([new_code.split("\n")[-1]] if new_code.split("\n")[-1] else [])
            print(f"Added cyclic table display to Code Cell {idx}")

    # Cell 31: Scientific Conclusions
    if "2. **Physical Architectural Divergence on 5-Qubit Chain:**" in src:
        conclusion_text = (
            "## 17. Scientific Conclusions and Summary of Findings\n\n"
            "1. **Optimal Trotter Step Sweet Spot ($n^* = 3$):** For the baseline 4-qubit TFIM ($J=1.0, h=0.5, T=1.0$), "
            "algorithmic Trotter error ($\\sim 1/n$) and physical gate noise ($\\sim n$) intersect at $n^* = 3$. "
            "This achieves minimum total energy error ($23.79\\%$ on A, $23.95\\%$ on B raw; $18.46\\%$ mitigated). Increasing $n > 3$ "
            "causes hardware noise accumulation to overwhelm discretization benefits.\n\n"
            "2. **Physical Architectural Divergence on 5-Qubit Chain:** On a 4-qubit chain, both chips embed with 0 SWAPs, yielding identical circuits "
            "where differences are purely statistical shot noise. Scaling to a 5-qubit chain exposes Processor A's T-tree topological bottleneck "
            "(maximum simple path length = 4), requiring 3 SWAP gates (+9 CX gates, depth 64 vs 39). In this model, Processor B avoids these SWAPs "
            "and achieves lower energy error ($23.70\\% \\pm 1.33\\%$ vs $28.72\\% \\pm 1.12\\%$ raw; $18.16\\% \\pm 1.39\\%$ vs $23.53\\% \\pm 1.19\\%$ mitigated, "
            "where $\\pm$ denotes sample standard deviation over 5 seeds; secondary state fidelity: $74.46\\%$ vs $57.10\\%$, +17.36 percentage points).\n\n"
            "3. **Trotter Order Convergence Tradeoff:** 2nd-order Trotter-Suzuki converges quadratically $\\mathcal{O}(\\Delta t^2)$, "
            "reducing discretization error at small $n$ at the cost of double the two-qubit gate density per step, shifting the optimal step count $n^*$ from 3 to 2.\n\n"
            "4. **Analytical Formula Boundaries:** The analytical formula $n^* \\approx \\sqrt{C_{\\text{Trotter}} / C_{\\text{noise}}}$ provides qualitative predictions "
            "matching discrete simulation within $\\pm 1\\text{--}2$ steps in the mid-range ($0.5\\% \\le \\epsilon_{2Q} \\le 2.0\\%$) while diverging at extreme noise limits "
            "due to integer step quantization.\n\n"
            "5. **Dual Error Mitigation:** Readout error mitigation was implemented via symmetric calibration matrix inversion and verified on both processors, "
            "recovering $20\\%\\text{--}25\\%$ lost fidelity and confirming mitigation is processor-agnostic.\n"
        )
        cell["source"] = [line + "\n" for line in conclusion_text.split("\n")[:-1]] + ([conclusion_text.split("\n")[-1]] if conclusion_text.split("\n")[-1] else [])
        print(f"Updated Conclusion in Cell {idx}")

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Finished updating main.ipynb successfully!")
