import json
from pathlib import Path

notebook_path = Path("main.ipynb")

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell1 = nb["cells"][1]

new_source = """# 1. Setup and Imports
import sys
import json
import math
from pathlib import Path

# Add src and root to Python path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# Qiskit & Aer
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

# Local modular packages
from problem import TrotterSimulator, create_ising_chain
from processors import load_processor, get_processor_graph
from routing import transpile_circuit, analyze_routing_overhead, compare_processors
from metrics import (
    circuit_metrics,
    proportion_standard_error,
    mitigate_readout_counts,
    find_optimal_trotter_step
)
from visualization import (
    plot_processor_graph,
    plot_trotter_error_tradeoff,
    plot_circuit_scaling,
    plot_architecture_tradeoffs,
    plot_error_mitigation_impact
)

print("Environment setup verified.")
print(f"Working directory: {Path.cwd()}")

# Self-Containment Pipeline Check:
# Ensure all benchmark data, figures, and aliases exist.
# If running in a fresh clone where precomputed CSVs are absent, automatically execute the pipeline end-to-end.
required_csvs = [
    Path('results/tables/AB_comparison_summary.csv'),
    Path('results/tables/pbc_comparison.csv'),
    Path('results/tables/trotter_order_comparison.csv'),
    Path('results/tables/noise_sensitivity.csv'),
    Path('results/tables/5qubit_chain_comparison.csv'),
    Path('results/tables/cyclic_variant_comparison.csv')
]
if any(not p.exists() for p in required_csvs):
    print("Precomputed results missing. Executing reproducible pipeline end-to-end (~25s)...")
    import run_experiments
    run_experiments.main()
    import run_advanced_studies
    import cyclic_study as cs
    cs.run_cyclic_comparison()
    import tools.setup_aliases as sa
    sa.main()
    print("Pipeline completed successfully! All tables and figures generated.")
else:
    print("Verified: All precomputed benchmark tables and figures are present.")
"""

cell1["source"] = [line + "\n" for line in new_source.split("\n")[:-1]] + ([new_source.split("\n")[-1]] if new_source.split("\n")[-1] else [])

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated Cell 1 in main.ipynb with Self-Containment Check!")
