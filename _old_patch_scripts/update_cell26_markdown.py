"""
Update markdown in Cell 26 of main.ipynb to explain all 3 interactive options clearly.
"""

import json
from pathlib import Path

nb_path = Path('main.ipynb')
nb = json.load(open(nb_path, 'r', encoding='utf-8'))

cell_26_text = [
    "## 15. Interactive Exploration Dashboard & Judge Verification Tools\n",
    "\n",
    "To enable judges and evaluators to interactively explore arbitrary parameters in real time, this project provides three interfaces:\n",
    "\n",
    "### Option 1: Live Streamlit Simulator Embedded in this Notebook\n",
    "The code cell below embeds the real-time **Streamlit Simulator** (`interactive_app.py`) directly inside this notebook via an `IFrame`:\n",
    "- If you run `streamlit run interactive_app.py` in your terminal, the cell below displays the live interactive dashboard right here inside the notebook.\n",
    "- Features real Qiskit Aer noisy execution, interactive sliders, processor comparison toggles, and CSV export.\n",
    "\n",
    "### Option 2: Native Python Jupyter Widgets (`@widgets.interact`)\n",
    "The cell also includes native Python sliders for $N, J, h, T, \\text{order}, \\text{pbc}, \\epsilon_{2Q}$ that execute directly in the Python kernel without requiring external servers.\n",
    "\n",
    "### Option 3: Standalone Zero-Dependency HTML Dashboard (`interactive_dashboard.html`)\n",
    "- Open [`interactive_dashboard.html`](interactive_dashboard.html) in any browser (Chrome, Edge, Firefox) for an instant, offline, client-side exploration dashboard.\n",
    "\n",
    "### Option 4: Interactive CLI with Curated Presets (`explore.py`)\n",
    "```bash\n",
    "# Run curated physical presets from terminal:\n",
    "python explore.py --preset default\n",
    "python explore.py --preset periodic_ring --verbose\n",
    "python explore.py --preset critical\n",
    "python explore.py --preset second_order\n",
    "python explore.py --J 1.2 --h 0.8 --time 1.5 --qubits 4 --order 1 --output sweep.csv\n",
    "```\n"
]

nb['cells'][26]['source'] = cell_26_text

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Updated cell 26 markdown successfully!')
