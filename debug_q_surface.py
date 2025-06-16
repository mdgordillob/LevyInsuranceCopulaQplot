# -*- coding: utf-8 -*-
"""
Generates detailed Q(p1,p2) contour plots for specific problematic theta values
to help diagnose numerical instability issues.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import time
import sys
import os

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

# --- Plotting Function for Detailed Q Contour ---
# (Copied from main script, includes NaN handling)
def plot_q_contour(P1_mesh, P2_mesh, Q_matrix, theta_val, eta_val, maxima_list, filename):
    """Generates and saves a contour plot of the Q matrix, marking top maxima."""
    if Q_matrix is None or Q_matrix.size == 0:
        print(f"  Plotting Q surface for theta={theta_val:.2f}, eta={eta_val:.2f} (Matrix might be all NaN).")
        # Still try to plot even if all NaN, might show empty plot
    elif not np.any(np.isfinite(Q_matrix)):
         print(f"  Plotting Q surface for theta={theta_val:.2f}, eta={eta_val:.2f} (Matrix contains only NaN/Inf).")
         # Proceed to plot, masked array will handle it

    fig, ax = plt.subplots(figsize=(10, 7))
    # Mask invalid values in Q_matrix for contourf
    Q_matrix_masked = np.ma.masked_invalid(Q_matrix if Q_matrix is not None else np.full(P1_mesh.shape, np.nan))

    if not Q_matrix_masked.mask.all(): # Only plot contours if there is valid data
        min_q = np.nanmin(Q_matrix)
        max_q = np.nanmax(Q_matrix)
        levels = np.linspace(min_q, max_q, 50) if min_q != max_q else np.array([min_q])
        contour = ax.contourf(P1_mesh, P2_mesh, Q_matrix_masked.T, levels=levels, cmap='jet', extend='both') # Transpose Q_matrix if needed
        cbar = fig.colorbar(contour)
        cbar.set_label('Q Value')
    else:
        ax.text(0.5, 0.5, 'All Q values are NaN or Inf', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


    ax.set_xlabel('p1')
    ax.set_ylabel('p2')
    ax.set_title(f'Q(p1, p2) Surface for theta={theta_val:.3f}, eta={eta_val:.1f}')

    # Mark the maxima if found
    if maxima_list:
        colors = ['red', 'yellow']
        markers = ['o', 's']
        labels = ['Max 1', 'Max 2']
        plotted_labels = []
        for k, max_info in enumerate(maxima_list):
            p1_opt = max_info.get('max_p1', np.nan)
            p2_opt = max_info.get('max_p2', np.nan)
            q_val = max_info.get('max_Q', np.nan)
            if np.isfinite(p1_opt) and np.isfinite(p2_opt):
                label = f'{labels[k]} Q={q_val:.2f} at ({p1_opt:.2f}, {p2_opt:.2f})'
                ax.plot(p1_opt, p2_opt, color=colors[k], marker=markers[k], markersize=8 + (2 * (1 - k)), markeredgecolor='k', label=label, linestyle='None')
                plotted_labels.append(label)
        if plotted_labels:
            ax.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    print(f"  Saved debug plot: {filename}")


# --- Parameters for Debugging ---
thetas_to_debug = [0.5,1.5,2] # Values in/around the NaN range
eta_fixed = [1,2] # Fixed eta values for debugging

# Price range for p1, p2 (passed to C++)
P1_plot = np.linspace(0.4, 21.3, 46)
P2_plot = np.linspace(0.4, 21.3, 46)
P1_mesh, P2_mesh = np.meshgrid(P1_plot, P2_plot) # For plotting

# --- Create Output Directory ---
debug_plot_dir = os.path.join(script_dir, "plots", "debug_nan_region")
os.makedirs(debug_plot_dir, exist_ok=True)

# --- Main Loop ---
start_time = time.time()
print(f"Starting debug calculations for {len(thetas_to_debug)} theta values and {len(eta_fixed)} eta values...")
total_calculations = len(thetas_to_debug) * len(eta_fixed)
count = 0

for j, current_eta in enumerate(eta_fixed): # Outer loop for eta
    print(f"\n--- Processing for eta = {current_eta:.1f} ---")
    for i, current_theta in enumerate(thetas_to_debug): # Inner loop for theta
        count += 1
        iter_start_time = time.time()
        print(f"Calculating for theta={current_theta:.3f}, eta={current_eta:.1f} ({count}/{total_calculations})...", end='', flush=True)

        q_matrix_for_plot = None
        maxima_data = []

        try:
            # Call C++ function requesting the full matrix with the current eta
            maxima_list = copula_calculations.calculate_maxima_cpp(
                current_theta, current_eta, P1_plot, P2_plot, return_q_matrix=True
            )

            if not maxima_list:
                 print(f" No maxima found.")
                 # Attempt to get matrix even if no maxima found (might be all NaN)
                 # This assumes the C++ function still calculates and includes it
                 # Need to adjust C++ if it doesn't return matrix when no maxima found
                 # Re-calling just to get matrix if first call didn't have return_q_matrix=True
                 # (Adjusting logic: assume first call gets matrix if needed)
                 # Check if the return is a list and has the matrix key
                 if isinstance(maxima_list, list) and maxima_list and 'q_matrix' in maxima_list[0]:
                     q_matrix_for_plot = maxima_list[0].get('q_matrix')
                 else: # If no maxima or matrix wasn't returned, create NaN matrix
                     q_matrix_for_plot = np.full((len(P1_plot), len(P2_plot)), np.nan)

            else:
                global_max_info = maxima_list[0]
                q_matrix_for_plot = global_max_info.get('q_matrix')
                maxima_data = maxima_list # Keep the list of maxima (could be 1 or 2)

            iter_end_time = time.time();
            print(f" Done in {iter_end_time - iter_start_time:.3f}s.");

            # Generate and save the detail plot using the current eta
            detail_plot_filename = os.path.join(debug_plot_dir, f"debug_q_contour_theta{current_theta:.3f}_eta{current_eta:.1f}.png")
            # Pass the potentially empty maxima_data list and current eta
            plot_q_contour(P1_mesh, P2_mesh, q_matrix_for_plot, current_theta, current_eta, maxima_data, detail_plot_filename)

        except Exception as e:
            iter_end_time = time.time()
            print(f" Error in C++ call for theta={current_theta:.3f}, eta={current_eta:.1f}: {e}. Took {iter_end_time - iter_start_time:.3f}s.")


end_time = time.time()
print(f"\nTotal debug calculation time: {end_time - start_time:.2f} seconds.")
print("Debug script finished.")
