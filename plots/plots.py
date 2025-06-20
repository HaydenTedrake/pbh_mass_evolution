import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product

# ------------------------------
# LOAD BIN FILES AND INTERPOLATE
# ------------------------------

# energy_levels = [100, 125.893, 158.489, 199.526, 251.189, 316.228, 398.107, 501.187, 794.328, 1000, 1258.93, 1584.89, 1995.26, 2511.89, 3162.28]
energy_levels = [130, 200, 320, 500, 790, 1300, 2000, 3200]     

number = 21
bin_files = [f"plots/bin(2)/bigSolution{number}-{i}.bin" for i in [22, 24, 26, 28, 30, 32, 34, 36]]

t_values = np.arange(-10000, 10001, 2)
x_values = np.arange(-10, 11, 2)
y_values = np.arange(-10, 11, 2)
z_values = np.arange(-10, 11, 2)

interpolators = {}

for E, file in tqdm(zip(energy_levels, bin_files), total=len(energy_levels), desc="Building Interpolators"):
    data = np.fromfile(file, dtype=np.float64)
    expected_length = len(x_values) * len(y_values) * len(z_values) * len(t_values)
    if len(data) != expected_length:
        raise ValueError(f"File {file}: Expected {expected_length}, got {len(data)}")
    data = data.reshape(len(t_values), len(x_values), len(y_values), len(z_values))
    interpolators[E] = RegularGridInterpolator((t_values, x_values, y_values, z_values), data, bounds_error=False)

# -------------------
# No. Density vs Time
# -------------------

fig1, ax1 = plt.subplots(figsize=(6, 4))

for E in energy_levels:
    values = interpolators[E]([[t, 0, 0, 0] for t in t_values])
    ax1.plot(t_values, values, label=f"{E} GeV")

ax1.set_xlabel("Time (days)", fontsize=14)
ax1.set_xlim([-6000, -4000])
ax1.set_ylabel("u(0,0,0)", fontsize=14)
ax1.set_ylim([0, 300000])
ax1.set_title(f"bigSolution{number}", fontsize=16)

# Match color-coded legend style
ax1.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), title="")
plt.tight_layout()
plt.show()

# --------------------
# Peak Density vs Time
# --------------------

E = 130
interpolator = interpolators[E]

# Prepare list of all (x, y, z) combinations
spatial_points = list(product(x_values, y_values, z_values))  # all 3D grid points

# Now loop over time values and compute peak density at each t
peak_densities = []

for t in tqdm(t_values, desc=f"Evaluating peak density for E = {E} GeV"):
    points = [[t, x, y, z] for (x, y, z) in spatial_points]
    values = interpolator(points)
    peak_densities.append(np.nanmax(values))  # nanmax in case of out-of-bound issues

peak_densities = np.array(peak_densities)

fig1, ax1 = plt.subplots(figsize=(6, 4))

ax1.plot(t_values, peak_densities, label=f"{E} GeV")
ax1.set_xlabel("Time (days)", fontsize=14)
ax1.set_ylabel("Peak Density", fontsize=14)
ax1.set_ylim([0, 300000])
ax1.set_title(f"bigSolution{number}", fontsize=16)

# Match color-coded legend style
ax1.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), title="")
plt.tight_layout()
plt.show()