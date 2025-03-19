import numpy as np
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation

# --------------------------------------------
# BUILDING THE INTERPOLATORS / DENSITY VS TIME
# --------------------------------------------

# List of energy levels and corresponding .bin files
energy_levels = [100, 125.893, 158.489, 199.526, 251.189, 316.228, 398.107, 501.187, 794.328, 1000, 1258.93, 1584.89, 1995.26, 2511.89, 3162.28]  # Example energy levels in GeV
bin_files = ["new_project/bin/bigSolution3-1.bin", "new_project/bin/bigSolution3-2.bin", "new_project/bin/bigSolution3-3.bin", "new_project/bin/bigSolution3-4.bin", "new_project/bin/bigSolution3-5.bin", "new_project/bin/bigSolution3-6.bin", "new_project/bin/bigSolution3-7.bin", "new_project/bin/bigSolution3-8.bin", "new_project/bin/bigSolution3-10.bin", "new_project/bin/bigSolution3-11.bin", "new_project/bin/bigSolution3-12.bin", "new_project/bin/bigSolution3-13.bin", "new_project/bin/bigSolution3-14.bin", "new_project/bin/bigSolution3-15.bin", "new_project/bin/bigSolution3-16.bin"]

# Dictionary to store interpolators for each energy
interpolators = {}

# Define the grid
t_values = np.arange(-10000, 10001, 2)  # t from -10,000 to 10,000 in steps of 10
x_values = np.arange(-10, 11, 2)         # x from -10 to 10 in steps of 2
y_values = np.arange(-10, 11, 2)         # y from -10 to 10 in steps of 2
z_values = np.arange(-10, 11, 2)         # z from -10 to 10 in steps of 2

# Prepare a figure
# plt.figure(figsize=(10, 6))

# Iterate over energy levels and construct interpolators with a progress bar
for E, file in tqdm(zip(energy_levels, bin_files), total=len(energy_levels), desc="Building Interpolators"):
    # Read data from .bin file
    data = np.fromfile(file, dtype=np.float64)
    
    # Validate data length
    expected_length = len(t_values) * len(x_values) * len(y_values) * len(z_values)
    if len(data) != expected_length:
        raise ValueError(f"Mismatch for {file}: Expected {expected_length} values, got {len(data)}")

    # Reshape data to match the grid dimensions
    data = data.reshape(len(t_values), len(x_values), len(y_values), len(z_values))
    # plt.plot(t_values, data[:, 5, 5, 6], label=f"E = {E:.3g} GeV")

    # Create and store 4D interpolator
    interpolators[E] = RegularGridInterpolator((t_values, x_values, y_values, z_values), data)

# plt.show()
# exit()

fig, ax = plt.subplots(figsize=(12, 8))

# Evaluate and plot for each energy
for E in energy_levels:
    # Interpolate values at (t, x, y, z)
    values = interpolators[E]([[t, 0, 0, 0.01] for t in t_values])

    # Plot
    ax.plot(t_values, values, label=f"E = {E:.3g} GeV")

ax.set_title("Density vs. Time for each Energy (x=0, y=0, z=0.01)", fontsize=26)
ax.set_xlabel("Time", fontsize=24)
ax.set_ylabel("Density", fontsize=24)

ax.tick_params(axis='both', which='major', labelsize=18)
ax.grid(True)
ax.legend(fontsize=20, loc = "upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.5)
ax.yaxis.get_offset_text().set_fontsize(18)
plt.tight_layout()
plt.show()

# ---------------
# ENERGY SPECTRUM
# ---------------

# integrated_spectrum = []

# for E in energy_levels:
#     # Retrieve the interpolator for this energy
#     interp = interpolators[E]
    
#     # Evaluate over the desired time range at fixed spatial coords
#     vals = [interp((t, 0, 1, 0.01)) for t in t_values]
    
#     # Compute the average value over time
#     avg_value = np.mean(vals)
    
#     integrated_spectrum.append(avg_value)

# # Plot the averaged spectrum
# plt.figure(figsize=(7,5))
# plt.plot(energy_levels, integrated_spectrum, marker="o")
# plt.xscale("log")
# plt.yscale("log")
# plt.xlabel("Energy [GeV]")
# plt.ylabel("Average Value")
# plt.title("Average Energy Spectrum over t ∈ [-10000, +10000], (x=0, y=0, z=0.01)")
# plt.grid(True)
# plt.show()

# -----------------
# PLOT ALONG X-AXIS
# -----------------

new_x_values = np.arange(-10, 10.1, 0.1)
new_y_values = np.arange(-10, 10.1, 0.1)

fig, ax = plt.subplots(figsize=(12, 8))

# Evaluate and plot for each energy
for E in energy_levels:
    # Interpolate values at (t, x, y, z)
    values = interpolators[E]([[0, x, 0, 0] for x in new_x_values])
    # Plot
    ax.plot(new_x_values, values, label=f"E = {E:.3g} GeV")
    
ax.set_title("Density vs. X for each Energy (t=0, y=0, z=0)", fontsize=26)
ax.set_xlabel("X Coordinate", fontsize=24)
ax.set_ylabel("Density", fontsize=24)

ax.tick_params(axis='both', which='major', labelsize=18)
ax.grid(True)
ax.legend(fontsize=20, loc = "upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.5)
ax.yaxis.get_offset_text().set_fontsize(18)
plt.tight_layout()
plt.show()

# -----------
# CONTOUR MAP
# -----------

colors = ['indigo', 'blue', 'green', 'yellow', 'orange', 'red']
custom_cmap = LinearSegmentedColormap.from_list("indigo_to_red", colors, N=256)

specific_energies = [100, 316.228, 1000]
# specific_t_values = [0, 1000, 3000, 5000]
specific_t_values = np.arange(-200, 1201, 100)
 
X, Y = np.meshgrid(new_x_values, new_y_values)
 
# for E in specific_energies:
#     for t in specific_t_values:
#         fig, ax = plt.subplots(figsize=(12, 8))
#         # Evaluate interpolator at fixed z = 0 for all (x, y)
#         u_values = np.array([
#             [interpolators[E]([t, x, y, 0])[0] if isinstance(interpolators[E]([t, x, y, 0]), np.ndarray) 
#             else interpolators[E]([t, x, y, 0])  # Ensure a scalar value
#             for x in new_x_values] for y in new_y_values
#         ])
 
#         # Create contour plot
#         contour = ax.contourf(X, Y, u_values, cmap=custom_cmap)
#         cbar = fig.colorbar(contour)
#         cbar.ax.tick_params(labelsize=18)  
#         cbar.ax.yaxis.get_offset_text().set_fontsize(20)
#         ax.grid(False)
#         ax.set_title(f"Contour Map (E = {E:.3g} GeV, t = {t}, z = 0)", fontsize=26)
#         ax.set_xlabel("X Coordinate", fontsize=24)
#         ax.set_ylabel("Y Coordinate", fontsize=24)
#         ax.tick_params(axis='both', which='major', labelsize=18)
#         plt.tight_layout()
#         plt.show()
 
# ------------
# GRADIENT MAP
# ------------

X, Y = np.meshgrid(x_values, y_values)

def gradient_of_interpolator_safe(interpolator, t, x, y, z, delta=1e-3):
    """
    Computes the numerical gradient of the interpolated function at (t, x, y, z),
    ensuring that all perturbed points remain within the interpolation grid bounds.

    Returns a tuple of (∂u/∂t, ∂u/∂x, ∂u/∂y, ∂u/∂z).
    """
    def safe_query(var, grid):
         """Ensures the queried value stays within the valid grid range."""
         return np.clip(var, grid[0] + delta, grid[-1] - delta)
 
    # Ensure points stay in bounds
    x_p, x_m = safe_query(x + delta, x_values), safe_query(x - delta, x_values)
    y_p, y_m = safe_query(y + delta, y_values), safe_query(y - delta, y_values) 
    
    # Compute numerical gradients
    dx = (interpolator([[t, x_p, y, z]]) - interpolator([[t, x_m, y, z]])) / (2 * delta)
    dy = (interpolator([[t, x, y_p, z]]) - interpolator([[t, x, y_m, z]])) / (2 * delta)

    return (dx[0], dy[0])
 
#  # Generate 3D gradient vector plots for each energy level and time value
# for E in specific_energies:
#      interpolator = interpolators[E]
 
#      for t in specific_t_values:
#         fig, ax = plt.subplots(figsize=(10, 6))

#         U = np.zeros_like(X, dtype=float)
#         V = np.zeros_like(Y, dtype=float)
 
#         for i in range(X.shape[0]):
#             for j in range(X.shape[1]):
#                 x, y, = X[i, j], Y[i, j]
#                 dx, dy = gradient_of_interpolator_safe(interpolator, t, x, y, 0)

#                 U[i, j], V[i, j] = dx, dy 
        
#         # Normalize arrows for better visualization
#         magnitude = np.sqrt(U**2 + V**2)
#         U /= (magnitude + 1e-9)
#         V /= (magnitude + 1e-9)
        
#         # Plot the arrows
#         ax.quiver(X, Y, U, V, scale=20, color='blue')
        
#         # Labels and title for the specific energy and time
#         ax.set_xlabel("X Coordinate", fontsize=24)
#         ax.set_ylabel("Y Coordinate", fontsize=24)
#         ax.set_title(f"2D Gradient Field at E={E:.3g} GeV, t={t}, z=0", fontsize=26)
#         ax.tick_params(axis='both', which='major', labelsize=18)
#         plt.tight_layout()
#         plt.show()

# --------------
# GRADIENT MOVIE
# --------------

for E in specific_energies:
    fig, ax = plt.subplots(figsize=(10, 6))
    quiver = ax.quiver(X, Y, np.zeros_like(X), np.zeros_like(Y), scale=20, color='blue')
    ax.set_xlabel("X Coordinate", fontsize=24)
    ax.set_ylabel("Y Coordinate", fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    interpolator = interpolators[E]
    
    # Animation function
    def update(t):
        ax.clear()
        ax.set_title(f"2D Gradient Field at E={E:.3g} GeV, t={t}, z=0", fontsize=26)
        
        U = np.zeros_like(X, dtype=float)
        V = np.zeros_like(Y, dtype=float)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x, y = X[i, j], Y[i, j]
                dx, dy = gradient_of_interpolator_safe(interpolator, t, x, y, 0)
                U[i, j], V[i, j] = dx, dy
        
        magnitude = np.sqrt(U**2 + V**2)
        U /= (magnitude + 1e-9)
        V /= (magnitude + 1e-9)
        
        ax.quiver(X, Y, U, V, scale=20, color='blue')
        ax.set_xlabel("X Coordinate", fontsize=24)
        ax.set_ylabel("Y Coordinate", fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=18)
        return ax
    
    # Generate animation
    ani = animation.FuncAnimation(fig, update, frames=specific_t_values, repeat=False)
    
    # Save animation as video
    ani.save(f"new_project/grad_E{E:.3g}.mp4", writer="ffmpeg", fps=5)
    
    plt.close(fig)
