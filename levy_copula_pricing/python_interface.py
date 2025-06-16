import os
import sys

# This will be replaced by the actual compiled module name
# when the package is installed.
try:
    from . import _copula_calculations_cpp_module as _cpp_module
except ImportError:
    # Fallback for development or direct execution if not installed as a package
    # This part might need adjustment based on the exact build output
    # For now, we'll assume the module is directly importable if not packaged.
    # In a proper package, this fallback might not be necessary.
    try:
        import _copula_calculations_cpp_module as _cpp_module
    except ImportError:
        raise ImportError("Could not import the C++ copula calculations module. "
                          "Ensure the package is installed correctly or the module is in PYTHONPATH.")

# Expose the C++ functions
calculate_maxima_cpp = _cpp_module.calculate_maxima_cpp
Q_cpp = _cpp_module.Q_cpp
F1_inv = _cpp_module.F1_inv
F2_inv = _cpp_module.F2_inv
n1 = _cpp_module.n1
n2 = _cpp_module.n2
alpha1 = _cpp_module.alpha1
alpha2 = _cpp_module.alpha2
