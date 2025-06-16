import os
import sys
import numpy as np

# Add the build directory to the Python path to find the C++ module
script_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(script_dir, 'build')
sys.path.insert(0, build_dir) # Changed from append to insert(0)

try:
    import copula_calculations
except ImportError as e:
    print(f"Error importing C++ module: {e}")
    print(f"Ensure the module exists in the '{build_dir}' directory and has been built and installed.")
    print(f"sys.path: {sys.path}") # Added for debugging
    sys.exit(1)

print(f"Loaded copula_calculations from: {copula_calculations.__file__}") # Added for debugging

# Parameters for the test
p1_test = 4.0
p2_test = 5.0
eta_test = 2.0
theta_test = 3.0

# Create dummy P1_plot and P2_plot arrays for the C++ function
# The C++ function expects these, even if we only care about one point's Q value
P1_plot_single = np.array([p1_test])
P2_plot_single = np.array([p2_test])

print(f"Calculating Q for p1={p1_test}, p2={p2_test}, eta={eta_test}, theta={theta_test}")

# Call the C++ function
# The C++ function returns a list of dictionaries, with the first being the global max
results = copula_calculations.calculate_maxima_cpp(
    theta_test, eta_test, P1_plot_single, P2_plot_single, return_q_matrix=False
)

if results:
    q_value_cpp = results[0].get('max_Q', np.nan)
    print(f"Q value from C++ module: {q_value_cpp}")
else:
    print("No results returned from C++ module.")

print("\n--- Testing F_inv functions ---")
test_x = 0.5
f1_inv_cpp_val = copula_calculations.F1_inv_cpp(test_x)
f2_inv_cpp_val = copula_calculations.F2_inv_cpp(test_x)
print(f"F1_inv_cpp({test_x}) from C++ module: {f1_inv_cpp_val}")
print(f"F2_inv_cpp({test_x}) from C++ module: {f2_inv_cpp_val}")
