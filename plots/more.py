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
bin_files = ["plots/bin(1)/bigSolution11-1.bin", "plots/bin(1)/bigSolution11-2.bin", "plots/bin(1)/bigSolution11-3.bin", "plots/bin(1)/bigSolution11-4.bin", "plots/bin(1)/bigSolution11-5.bin", "plots/bin(1)/bigSolution11-6.bin", "plots/bin(1)/bigSolution11-7.bin", "plots/bin(1)/bigSolution11-8.bin", "plots/bin(1)/bigSolution11-10.bin", "plots/bin(1)/bigSolution11-11.bin", "plots/bin(1)/bigSolution11-12.bin", "plots/bin(1)/bigSolution11-13.bin", "plots/bin(1)/bigSolution11-14.bin", "plots/bin(1)/bigSolution11-15.bin", "plots/bin(1)/bigSolution11-16.bin"]

# Dictionary to store interpolators for each energy
interpolators = {}

# Define the grid
t_values = np.arange(-10000, 10001, 2)  # t from -10,000 to 10,000 in steps of 2
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
    interpolators[E] = RegularGridInterpolator((t_values, x_values, y_values, z_values), data, method='linear', bounds_error=False, fill_value=None)

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

specific_t_values = np.arange(0, 10500, 500)

fig, ax = plt.subplots(figsize=(12, 8))  # Same size as the time plot

for t in specific_t_values:
    spectrum = []
    for E in energy_levels:
        interp = interpolators[E]
        spectrum.append(interp((t, 0, 1, 0.01)))  # Change point if needed

    ax.plot(energy_levels, spectrum, marker="o", label=f"t = {t} days")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Energy [GeV]", fontsize=24)
ax.set_ylabel("Interpolated Density", fontsize=24)
ax.set_title("Energy Spectra at Various Times (x=0, y=1, z=0.01)", fontsize=26)

ax.tick_params(axis='both', which='major', labelsize=18)
ax.grid(True)
ax.legend(fontsize=16, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.5)
ax.yaxis.get_offset_text().set_fontsize(18)
plt.tight_layout()
plt.show()

# -----------------
# PLOT ALONG X-AXIS
# -----------------

specific_energies = [100, 316.228, 1000]
specific_x_values = np.arange(-10, 11, 2)
specific_t_values1 = np.arange(0, 5001, 1000)
print('hi')

for E in specific_energies:
    interp = interpolators[E]
    
    for t in specific_t_values1:
        # Get density along x-axis at fixed y=0, z=0.01
        values = [interp((t, x, 0, 0.01)) for x in specific_x_values]
        
        # Plot this single time slice
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(specific_x_values, values, marker="o")
        
        ax.set_title(f"Density vs. x (E = {E} GeV, t = {t} days, y=0, z=0.01)", fontsize=26)
        ax.set_xlabel("x", fontsize=24)
        ax.set_ylabel("Density u", fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.grid(True)
        ax.yaxis.get_offset_text().set_fontsize(18)
        plt.tight_layout()
        plt.show()