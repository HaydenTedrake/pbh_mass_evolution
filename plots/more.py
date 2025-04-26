import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation

# --------------------------------------------
# BUILDING THE INTERPOLATORS / DENSITY VS TIME
# --------------------------------------------

# List of energy levels and corresponding .bin files
energy_levels = [100, 125.893, 158.489, 199.526, 251.189, 316.228, 398.107, 501.187, 794.328, 1000, 1258.93, 1584.89, 1995.26, 2511.89, 3162.28]  # Example energy levels in GeV
bin_files = ["plots/bin(1)/bigSolution10-1.bin", "plots/bin(1)/bigSolution10-2.bin", "plots/bin(1)/bigSolution10-3.bin", "plots/bin(1)/bigSolution10-4.bin", "plots/bin(1)/bigSolution10-5.bin", "plots/bin(1)/bigSolution10-6.bin", "plots/bin(1)/bigSolution10-7.bin", "plots/bin(1)/bigSolution10-8.bin", "plots/bin(1)/bigSolution10-10.bin", "plots/bin(1)/bigSolution10-11.bin", "plots/bin(1)/bigSolution10-12.bin", "plots/bin(1)/bigSolution10-13.bin", "plots/bin(1)/bigSolution10-14.bin", "plots/bin(1)/bigSolution10-15.bin", "plots/bin(1)/bigSolution10-16.bin"]

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
ax.set_yscale("log")
ax.set_xlabel("Time", fontsize=24)
ax.set_ylabel("Density", fontsize=24)

ax.tick_params(axis='both', which='major', labelsize=18)
ax.grid(True)
ax.legend(fontsize=20, loc = "upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.5)
ax.yaxis.get_offset_text().set_fontsize(18)
plt.tight_layout()
plt.show()

# # -----
# # SLOPE
# # -----

alpha_values = []
for E in energy_levels:
    val_9000 = interpolators[E]([[9000, 0, 0, 0.01]])[0]
    val_100 = interpolators[E]([[100, 0, 0, 0.01]])[0]
    alpha = (val_9000 - val_100) / (9000 - 100)
    alpha_values.append(alpha)

alpha_interp = interp1d(energy_levels, alpha_values, kind='linear', bounds_error=False, fill_value='extrapolate')

E_dense = np.linspace(min(energy_levels), max(energy_levels), 300)
plt.plot(energy_levels, alpha_values, 'o', label='Sampled')
plt.plot(E_dense, alpha_interp(E_dense), '-', label='Interpolated α(E)')
plt.xlabel('Energy (GeV)')
plt.ylabel('Alpha')
plt.title('Alpha as a function of Energy')
plt.legend()
plt.grid(True)
plt.show()

# # -----------------
# # PLOT ALONG X-AXIS
# # -----------------

# specific_energies = [100, 316.228, 1000]
# specific_x_values = np.arange(-10, 11, 2)
# specific_t_values1 = np.arange(0, 5001, 1000)

# for E in specific_energies:
#     interp = interpolators[E]
    
#     for t in specific_t_values1:
#         # Get density along x-axis at fixed y=0, z=0.01
#         values = [interp((t, x, 0, 0.01)) for x in specific_x_values]
        
#         # Plot this single time slice
#         fig, ax = plt.subplots(figsize=(12, 8))
#         ax.plot(specific_x_values, values, marker="o")
        
#         ax.set_title(f"Density vs. x (E = {E} GeV, t = {t} days, y=0, z=0.01)", fontsize=26)
#         ax.set_xlabel("x", fontsize=24)
#         ax.set_ylabel("Density u", fontsize=24)
#         ax.tick_params(axis='both', which='major', labelsize=18)
#         ax.grid(True)
#         ax.yaxis.get_offset_text().set_fontsize(18)
#         plt.tight_layout()
#         plt.show()