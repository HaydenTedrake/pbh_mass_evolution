import numpy as np
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------------------------------
# BUILDING THE INTERPOLATORS / FIRST PLOT
# ---------------------------------------

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
plt.figure(figsize=(10, 6))

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

# Evaluate and plot for each energy
for E in energy_levels:
    # Interpolate values at (t, x, y, z)
    values = interpolators[E]([[t, 0, 0, 0.01] for t in t_values])

    # Plot
    plt.plot(t_values, values, label=f"E = {E:.3g} GeV")
    
plt.title("Density vs. Time for each Energy (x=0, y=0, z=0.01)")
plt.xlabel("Time")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

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
specific_t_values = [0, 1000, 10000]

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