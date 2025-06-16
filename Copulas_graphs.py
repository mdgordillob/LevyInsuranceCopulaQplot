# -*- coding: utf-8 -*-
"""
Numerically tests Corollary 6.2 using PARALLEL execution over theta/eta.
Finds the global maximum of Q for each pair using a C++ extension,
and plots summary trajectory scatter plots and the Corollary test plot.
Uses eta = [0.5, 1.0, 1.5, 2.0].
"""

import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import time
import sys
import os
import pandas as pd # For easier data handling
from joblib import Parallel, delayed # Re-import joblib

# Add the build directory to the Python path to find the C++ module
script_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(script_dir, 'build')
sys.path.append(build_dir)

# Import the C++ extension
try:
    import copula_calculations
except ImportError as e:
    print(f"Error importing C++ module: {e}")
    print(f"Ensure the module exists in the '{build_dir}' directory.")
    sys.exit(1)

# --- Helper Function for Parallel Execution ---
def process_params(i, j, current_theta, current_eta, P1_plot, P2_plot):
    """Calls C++ function for one (theta, eta) pair."""
    try:
        # Call C++ function to get maxima list (only need global max)
        maxima_list = copula_calculations.calculate_maxima_cpp(
            current_theta, current_eta, P1_plot, P2_plot, return_q_matrix=False
        )

        if not maxima_list: # Handle empty list return
             global_max_info = {'max_Q': np.nan, 'max_p1': np.nan, 'max_p2': np.nan}
        else:
            # Use the global maximum (first in the list)
            global_max_info = maxima_list[0]

        # Return results along with original indices
        return {
            'i': i, 'j': j, # Grid indices for theta, eta
            'theta': current_theta, 'eta': current_eta,
            'max_Q': global_max_info.get('max_Q', np.nan),
            'max_p1': global_max_info.get('max_p1', np.nan),
            'max_p2': global_max_info.get('max_p2', np.nan)
        }

    except Exception as e:
        print(f"\nError processing theta={current_theta:.2f}, eta={current_eta:.2f}: {e}")
        # Return NaN result with indices
        return {
            'i': i, 'j': j, 'theta': current_theta, 'eta': current_eta,
            'max_Q': np.nan, 'max_p1': np.nan, 'max_p2': np.nan
        }

# --- Parameters for Test ---
eta_values_to_test = [1.0,2.0] # Updated eta values
# Create a denser range for theta between 0.1 and 1.0, and sparser above 1.0
theta_range = np.linspace(0.5,2,4)
num_theta_steps = len(theta_range) # Now 34 steps
num_eta_values = len(eta_values_to_test)

# Price range for p1, p2 (passed to C++) - Keep consistent
P1_plot = np.linspace(0.4, 21.3, 75)
P2_plot = np.linspace(0.4, 21.3, 75)

# --- Create Output Directory ---
plot_dir = os.path.join(script_dir, "plots") # General plot directory
summary_plot_dir = os.path.join(plot_dir, "summary") # Specific subdir
os.makedirs(summary_plot_dir, exist_ok=True)

# --- Parallel Calculation ---
start_time = time.time()
total_calcs = num_eta_values * num_theta_steps
print(f"Starting PARALLEL C++ calculations for {total_calcs} (theta, eta) combinations...")

# Create list of tasks for joblib
tasks = [
    delayed(process_params)(i, j, theta, eta, P1_plot, P2_plot)
    for j, eta in enumerate(eta_values_to_test) # Outer loop eta
    for i, theta in enumerate(theta_range)      # Inner loop theta
]

# Run tasks in parallel
results_list = Parallel(n_jobs=-1, backend="threading", verbose=10)(tasks)

end_time = time.time()
print(f"\nTotal calculation time: {end_time - start_time:.2f} seconds.")

# --- Process Results into DataFrame ---
df = pd.DataFrame(results_list)
# Filter strictly for finite max_Q values
df_filtered = df[np.isfinite(df['max_Q'])].copy()

# --- Create Plot for Corollary Test ---
if not df_filtered.empty:
    print(f"Generating plot for Corollary 6.2 test ({len(df_filtered)} valid points)...")
    fig, ax = plt.subplots(figsize=(10, 6))

    for eta_val in eta_values_to_test:
        # Filter the VALID data for the current eta
        df_subset = df_filtered[df_filtered['eta'] == eta_val]
        if not df_subset.empty:
            # Sort by theta before plotting lines
            df_subset = df_subset.sort_values(by='theta')
            ax.plot(df_subset['theta'], df_subset['max_Q'], marker='o', linestyle='-', label=f'eta = {eta_val:.1f}')

    ax.set_xlabel('Theta (θ)')
    ax.set_ylabel('Maximum Q Value (Q̂)')
    ax.set_title('Test of Corollary 6.2: Maximum Q vs. Theta for Fixed Eta (Parallel Run)')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    # Use a distinct filename for this parallel run's plot
    plot_filename = os.path.join(plot_dir, 'corollary_6_2_test_maxQ_vs_theta_parallel_4eta.png') # Updated filename
    plt.savefig(plot_filename)
    plt.close(fig)
    print(f"Saved plot: {plot_filename}")

else:
    print("No valid results were generated to plot for Corollary test.")

# --- Also regenerate the summary scatter plots ---
if not df_filtered.empty:
    print(f"Generating summary trajectory plots ({len(df_filtered)} valid points)...")
    thetas_valid = df_filtered['theta'].values
    etas_valid = df_filtered['eta'].values
    max_p1s_valid = df_filtered['max_p1'].values
    max_p2s_valid = df_filtered['max_p2'].values

    # Plot 1: Max p1 vs Theta (colored by Eta)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    scatter1 = ax1.scatter(thetas_valid, max_p1s_valid, c=etas_valid, cmap='viridis', alpha=0.7, edgecolors='k', s=50)
    ax1.set_xlabel('Theta')
    ax1.set_ylabel('Optimal p1')
    ax1.set_title('Optimal p1 vs. Theta (Color indicates Eta)')
    ax1.grid(True)
    cbar1 = fig1.colorbar(scatter1)
    cbar1.set_label('Eta')
    plt.tight_layout()
    plot1_filename = os.path.join(summary_plot_dir, 'summary_scatter_max_p1_vs_theta_parallel_4eta.png') # Updated filename
    plt.savefig(plot1_filename)
    plt.close(fig1)
    print(f"Saved plot: {plot1_filename}")

    # Plot 2: Max p2 vs Theta (colored by Eta)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    scatter2 = ax2.scatter(thetas_valid, max_p2s_valid, c=etas_valid, cmap='viridis', alpha=0.7, edgecolors='k', s=50)
    ax2.set_xlabel('Theta')
    ax2.set_ylabel('Optimal p2')
    ax2.set_title('Optimal p2 vs. Theta (Color indicates Eta)')
    ax2.grid(True)
    cbar2 = fig2.colorbar(scatter2)
    cbar2.set_label('Eta')
    plt.tight_layout()
    plot2_filename = os.path.join(summary_plot_dir, 'summary_scatter_max_p2_vs_theta_parallel_4eta.png') # Updated filename
    plt.savefig(plot2_filename)
    plt.close(fig2)
    print(f"Saved plot: {plot2_filename}")

    # Note: Plots vs Eta are less informative with only 4 fixed values, but can still be generated
    # Plot 3: Max p1 vs Eta (colored by Theta)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    scatter3 = ax3.scatter(etas_valid, max_p1s_valid, c=thetas_valid, cmap='plasma', alpha=0.7, edgecolors='k', s=50)
    ax3.set_xlabel('Eta')
    ax3.set_ylabel('Optimal p1')
    ax3.set_title('Optimal p1 vs. Eta (Color indicates Theta)')
    ax3.grid(True)
    cbar3 = fig3.colorbar(scatter3)
    cbar3.set_label('Theta')
    plt.tight_layout()
    plot3_filename = os.path.join(summary_plot_dir, 'summary_scatter_max_p1_vs_eta_parallel_4eta.png') # Updated filename
    plt.savefig(plot3_filename)
    plt.close(fig3)
    print(f"Saved plot: {plot3_filename}")

    # Plot 4: Max p2 vs Eta (colored by Theta)
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    scatter4 = ax4.scatter(etas_valid, max_p2s_valid, c=thetas_valid, cmap='plasma', alpha=0.7, edgecolors='k', s=50)
    ax4.set_xlabel('Eta')
    ax4.set_ylabel('Optimal p2')
    ax4.set_title('Optimal p2 vs. Eta (Color indicates Theta)')
    ax4.grid(True)
    cbar4 = fig4.colorbar(scatter4)
    cbar4.set_label('Theta')
    plt.tight_layout()
    plot4_filename = os.path.join(summary_plot_dir, 'summary_scatter_max_p2_vs_eta_parallel_4eta.png') # Updated filename
    plt.savefig(plot4_filename)
    plt.close(fig4)
    print(f"Saved plot: {plot4_filename}")

else:
    print("No valid results were generated for summary plots.")


print("Script finished.")
