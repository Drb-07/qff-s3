"""
S3 - Noise in Trotter Simulation
QFF 2026 Phase 1 Hackathon

This package contains modules for:
- Trotter simulation implementation (src.problem)
- Processor configuration loading (src.processors)
- Routing and transpilation (src.routing)
- Metrics calculation and analytical prediction (src.metrics)
- Visualization utilities (src.visualization)
"""

import sys
import importlib

__version__ = "1.0.0"
__author__ = "QFF 2026 Participant"
__problem_code__ = "S3"
__problem_name__ = "Noise in Trotter Simulation"
__track__ = "Quantum Simulation"


def validate_environment():
    """Validate required dependencies and minimum versions."""
    required = {
        'qiskit': '1.0.0',
        'qiskit_aer': '0.14.0',
        'numpy': '1.20.0',
        'scipy': '1.8.0',
        'matplotlib': '3.5.0'
    }
    missing = []
    outdated = []

    for mod_name, min_ver in required.items():
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, '__version__', 'unknown')
        except ImportError:
            missing.append(mod_name)

    if missing:
        raise ImportError(
            f"Missing required dependencies: {', '.join(missing)}. "
            f"Please run: pip install -r requirements.txt"
        )


# Validate dependencies on package import
try:
    validate_environment()
except ImportError as err:
    # Print warning rather than hard crash on partial imports
    import warnings
    warnings.warn(str(err), RuntimeWarning)
