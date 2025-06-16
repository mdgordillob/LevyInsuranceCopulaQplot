# Demand-based pricing in multi-line insurance: a Lévy-copula approach

This repository contains the Python library and associated scripts for the research project on demand-based pricing in multi-line insurance using a Lévy-copula approach. The core numerical calculations are optimized using C++ with OpenMP and Boost Math, exposed to Python via `pybind11`.

## Features

*   **Optimized Q-function Calculation:** Efficient 2D numerical integration using Boost's Gauss-Kronrod quadrature.
*   **Local Maxima Detection:** Identifies and extracts optimal pricing points (p1, p2) for the Q-surface.
*   **Plotting Utilities:** Generates contour plots of the Q-surface and summary plots of optimal pricing trajectories.
*   **Python Package:** Designed to be easily installable and usable in Python environments, including Google Colab.

## Installation

To install this package, you will need `Python 3.8+`, `pip`, `cmake`, and a C++ compiler (like `g++` or `clang++`).

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:mdgordillob/LevyInsuranceCopulaQplot.git
    cd LevyInsuranceCopulaQplot
    ```

2.  **Install build dependencies:**
    ```bash
    pip install setuptools wheel cmake pybind11
    ```
    You might also need to install Boost development libraries on your system. For Debian/Ubuntu:
    ```bash
    sudo apt-get update
    sudo apt-get install libboost-all-dev
    ```
    For Fedora:
    ```bash
    sudo dnf install boost-devel
    ```

3.  **Install the package:**
    ```bash
    pip install .
    ```

## Usage

Once installed, you can import the core functionalities:

```python
from levy_copula_pricing.python_interface import calculate_maxima_cpp, F1_inv, F2_inv
import numpy as np

# Example: Calculate maxima for specific parameters
theta = 1.0
eta = 2.0
p1_range = np.linspace(0.4, 21.3, 75)
p2_range = np.linspace(0.4, 21.3, 75)

maxima_results = calculate_maxima_cpp(theta, eta, p1_range, p2_range, return_q_matrix=False)

if maxima_results:
    print(f"Global Max Q: {maxima_results[0]['max_Q']} at (p1={maxima_results[0]['max_p1']}, p2={maxima_results[0]['max_p2']})")
else:
    print("No maxima found.")

# Example: Use inverse CDFs
x = 0.5
f1_inv_val = F1_inv(x)
f2_inv_val = F2_inv(x)
print(f"F1_inv({x}): {f1_inv_val}")
print(f"F2_inv({x}): {f2_inv_val}")
```

You can also run the plotting scripts directly after installation (they will use the installed library):

```bash
python plot_q_normalized.py
python Copulas_graphs.py
python debug_q_surface.py
python test_q_value.py
```

## Project Structure

*   `levy_copula_pricing/`: The main Python package.
    *   `__init__.py`: Package initializer.
    *   `_copula_calculations.cpp`: C++ source code for numerical calculations.
    *   `python_interface.py`: Python module to expose C++ functions.
*   `CMakeLists.txt`: CMake build configuration for the C++ extension.
*   `pyproject.toml`: Python package metadata and build configuration.
*   `plot_q_normalized.py`: Script to generate normalized Q-surface contour plots.
*   `Copulas_graphs.py`: Script to test Corollary 6.2 and generate summary plots.
*   `debug_q_surface.py`: Script for debugging Q-surface calculations.
*   `test_q_value.py`: Script to test individual Q values and inverse CDFs.
*   `plots/`: Directory for generated plots.
*   `eta1_theta2_p2.csv`, `eta1_theta15_p1.csv`: Data files (if used by scripts).
*   `clayton_model`, `clayton_optimized`, `optimizer`: Other related files/directories.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
