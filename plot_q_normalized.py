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
sys.path.insert(0, build_dir)

try:
    import copula_calculations
except ImportError as e:
    print(f"Error importing C++ module: {e}")
    print(f"Ensure the module exists in the '{build_dir}' directory and has been built.")
    sys.exit(1)

# --- Global Model Parameters (Constants) ---
# These should match the constants in copula_calculations.cpp
R_const = 0.04 

# --- Plotting Function for Q / (R * eta) Contour ---
def plot_normalized_q_contour(P1_mesh, P2_mesh, Q_matrix, theta_val, eta_val, maxima_list, filename):
    """
    Generates and saves a contour plot of the Q matrix normalized by (R * eta),
    marking top maxima.
    """
    # Calculate Q_divReta
    Q_divReta_matrix = Q_matrix / (R_const * eta_val) if (R_const * eta_val) != 0 else np.full(Q_matrix.shape, np.nan)

    if Q_divReta_matrix is None or Q_divReta_matrix.size == 0:
        print(f"  Plotting Q/(R*eta) surface for theta={theta_val:.2f}, eta={eta_val:.2f} (Matrix might be all NaN).")
    elif not np.any(np.isfinite(Q_divReta_matrix)):
         print(f"  Plotting Q/(R*eta) surface for theta={theta_val:.2f}, eta={eta_val:.2f} (Matrix contains only NaN/Inf).")

    fig, ax = plt.subplots(figsize=(10, 7))
    # Mask invalid values for contourf
    Q_matrix_masked = np.ma.masked_invalid(Q_divReta_matrix)

    if not Q_matrix_masked.mask.all(): # Only plot contours if there is valid data
        min_q = np.nanmin(Q_divReta_matrix)
        max_q = np.nanmax(Q_divReta_matrix)
        levels = np.linspace(min_q, max_q, 50) if min_q != max_q else np.array([min_q])
        contour = ax.contourf(P1_mesh, P2_mesh, Q_matrix_masked.T, levels=levels, cmap='jet', extend='both')
        cbar = fig.colorbar(contour)
        cbar.set_label('Q / (R * eta) Value')
    else:
        ax.text(0.5, 0.5, 'All Q / (R * eta) values are NaN or Inf', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    ax.set_xlabel('p1')
    ax.set_ylabel('p2')
    ax.set_title(f'Q / (R * eta) Surface for theta={theta_val:.3f}, eta={eta_val:.1f}')

    # Mark the maxima if found (adjust Q value for display)
    if maxima_list:
        colors = ['red', 'yellow']
        markers = ['o', 's']
        labels = ['Max 1', 'Max 2']
        plotted_labels = []
        for k, max_info in enumerate(maxima_list):
            p1_opt = max_info.get('max_p1', np.nan)
            p2_opt = max_info.get('max_p2', np.nan)
            q_val = max_info.get('max_Q', np.nan)
            q_div_reta_val = q_val / (R_const * eta_val) if (R_const * eta_val) != 0 else np.nan
            if np.isfinite(p1_opt) and np.isfinite(p2_opt):
                label = f'{labels[k]} Q/(R*eta)={q_div_reta_val:.2f} at ({p1_opt:.2f}, {p2_opt:.2f})'
                ax.plot(p1_opt, p2_opt, color=colors[k], marker=markers[k], markersize=8 + (2 * (1 - k)), markeredgecolor='k', label=label, linestyle='None')
                plotted_labels.append(label)
        if plotted_labels:
            ax.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    print(f"  Saved plot: {filename}")


# --- Parameters for Plotting ---
thetas_to_plot = [0.5, 1.0, 1.5, 2.0] # Theta values for graphs
etas_to_plot = [1.0, 2.0, 3.0, 4.0,5.0,6.0] # Eta values for graphs

# Price range for p1, p2 (passed to C++)
P1_plot = np.linspace(0.4, 21.3, 75)
P2_plot = np.linspace(0.4, 21.3, 75)
P1_mesh, P2_mesh = np.meshgrid(P1_plot, P2_plot) # For plotting

# --- Create Output Directory ---
output_plot_dir = os.path.join(script_dir, "plots", "normalized_q_surfaces")
os.makedirs(output_plot_dir, exist_ok=True)

# --- Main Loop ---
start_time = time.time()
print(f"Starting Q/(R*eta) surface calculations for {len(thetas_to_plot)} theta values and {len(etas_to_plot)} eta values...")
total_calculations = len(thetas_to_plot) * len(etas_to_plot)
count = 0

for j, current_eta in enumerate(etas_to_plot):
    print(f"\n--- Processing for eta = {current_eta:.1f} ---")
    for i, current_theta in enumerate(thetas_to_plot):
        count += 1
        iter_start_time = time.time()
        print(f"Calculating for theta={current_theta:.3f}, eta={current_eta:.1f} ({count}/{total_calculations})...", end='', flush=True)

        q_matrix_from_cpp = None
        maxima_data = []

        try:
            # Call C++ function requesting the full matrix
            maxima_list = copula_calculations.calculate_maxima_cpp(
                current_theta, current_eta, P1_plot, P2_plot, return_q_matrix=True
            )

            if maxima_list:
                global_max_info = maxima_list[0]
                q_matrix_from_cpp = global_max_info.get('q_matrix')
                maxima_data = maxima_list
            else:
                q_matrix_from_cpp = np.full((len(P1_plot), len(P2_plot)), np.nan)
                print(f" No maxima or Q matrix found for theta={current_theta:.3f}, eta={current_eta:.1f}.")

            iter_end_time = time.time()
            print(f" Done in {iter_end_time - iter_start_time:.3f}s.")

            # Generate and save the normalized Q plot
            plot_filename = os.path.join(output_plot_dir, f"normalized_q_contour_theta{current_theta:.3f}_eta{current_eta:.1f}.png")
            plot_normalized_q_contour(P1_mesh, P2_mesh, q_matrix_from_cpp, current_theta, current_eta, maxima_data, plot_filename)

        except Exception as e:
            iter_end_time = time.time()
            print(f" Error in C++ call for theta={current_theta:.3f}, eta={current_eta:.1f}: {e}. Took {iter_end_time - iter_start_time:.3f}s.")


end_time = time.time()
print(f"\nTotal calculation and plotting time: {end_time - start_time:.2f} seconds.")
print("Script finished.")
