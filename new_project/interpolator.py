import numpy as np
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import matplotlib.pyplot as plt

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

# Evaluate and plot for each energy
for E in energy_levels:
    # Interpolate values at (t, x, y, z)
    values = interpolators[E]([[0, x, 0, 0] for x in x_values])

    # Plot
    plt.plot(x_values, values, label=f"E = {E:.3g} GeV")
    
plt.title("Density vs. X for each Energy (t=0, y=0, z=0)")
plt.xlabel("X Coordinate")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

# -----------
# CONTOUR MAP
# -----------

specific_energies = [100, 316.228, 1000]
specific_t_values = [0, 1000, 3000, 5000]
 
X, Y = np.meshgrid(x_values, y_values)
 
for E in specific_energies:
    for t in specific_t_values:
         # Evaluate interpolator at fixed z = 0 for all (x, y)
         u_values = np.array([
             [interpolators[E]([t, x, y, 0])[0] if isinstance(interpolators[E]([t, x, y, 0]), np.ndarray) 
             else interpolators[E]([t, x, y, 0])  # Ensure a scalar value
             for x in x_values] for y in y_values
     ])
 
         # Create contour plot
         plt.figure(figsize=(6, 5))
         contour = plt.contourf(X, Y, u_values, cmap='viridis')
         plt.colorbar(contour)
         plt.title(f"Contour Map (E = {E:.3g} GeV, t = {t}, z = 0)")
         plt.xlabel("X Coordinate")
         plt.ylabel("Y Coordinate")
         plt.show()
 
# ------------
# GRADIENT MAP
# ------------

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
    t_p, t_m = safe_query(t + delta, t_values), safe_query(t - delta, t_values)
    x_p, x_m = safe_query(x + delta, x_values), safe_query(x - delta, x_values)
    y_p, y_m = safe_query(y + delta, y_values), safe_query(y - delta, y_values)
    z_p, z_m = safe_query(z + delta, z_values), safe_query(z - delta, z_values)
 
     # Compute numerical gradients
    dt = (interpolator([[t_p, x, y, z]]) - interpolator([[t_m, x, y, z]])) / (2 * delta)
    dx = (interpolator([[t, x_p, y, z]]) - interpolator([[t, x_m, y, z]])) / (2 * delta)
    dy = (interpolator([[t, x, y_p, z]]) - interpolator([[t, x, y_m, z]])) / (2 * delta)
    dz = (interpolator([[t, x, y, z_p]]) - interpolator([[t, x, y, z_m]])) / (2 * delta)
 
    return (dt[0], dx[0], dy[0], dz[0])
 
 # Generate 3D gradient vector plots for each energy level and time value
for E in specific_energies:
     interpolator = interpolators[E]
 
     for t in specific_t_values:
         fig = plt.figure(figsize=(10, 6))
         ax = fig.add_subplot(111, projection='3d')
 
         for x in x_values:
             for y in y_values:
                 for z in z_values:
                     # Compute gradient safely
                     grad = gradient_of_interpolator_safe(interpolator, t, x, y, z)
                     _, dx, dy, dz = grad
 
                     # Compute magnitude of gradient
                     grad_magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
 
                     # Adaptive scaling: Adjust length based on the gradient magnitude
                     scale_factor = 2 / (grad_magnitude + 1e-6)  # Prevent division by zero
 
                     # Plot arrows with adaptive scaling
                     ax.quiver(x, y, z, dx, dy, dz, length=scale_factor, color="blue")
 
         # Labels and title for the specific energy and time
         ax.set_xlabel("X Coordinate")
         ax.set_ylabel("Y Coordinate")
         ax.set_zlabel("Z Coordinate")
         ax.set_title(f"3D Gradient Field at E={E} GeV, t={t}")
 
         # Show plot for each time step separately
         plt.show()