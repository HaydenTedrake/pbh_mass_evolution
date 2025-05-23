import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from tqdm import tqdm

# ------------------------------
# LOAD BIN FILES AND INTERPOLATE
# ------------------------------

energy_levels = [100, 125.893, 158.489, 199.526, 251.189, 316.228, 398.107, 501.187,
                 794.328, 1000, 1258.93, 1584.89, 1995.26, 2511.89, 3162.28]

number = 10
bin_files = [f"plots/bin(1)/bigSolution{number}-{i}.bin" for i in [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]]

t_values = np.arange(-10000, 10001, 2)
x_values = np.arange(-10, 11, 2)
y_values = np.arange(-10, 11, 2)
z_values = np.arange(-10, 11, 2)

interpolators = {}

for E, file in tqdm(zip(energy_levels, bin_files), total=len(energy_levels), desc="Building Interpolators"):
    data = np.fromfile(file, dtype=np.float64)
    expected_length = len(t_values) * len(x_values) * len(y_values) * len(z_values)
    if len(data) != expected_length:
        raise ValueError(f"File {file}: Expected {expected_length}, got {len(data)}")
    data = data.reshape(len(t_values), len(x_values), len(y_values), len(z_values))
    interpolators[E] = RegularGridInterpolator((t_values, x_values, y_values, z_values), data, bounds_error=False)

# ----------------------------
# PANEL 1: No. Density vs Time
# ----------------------------

fig1, ax1 = plt.subplots(figsize=(6, 4))

for E in energy_levels:
    values = interpolators[E]([[t, 0, 0, 0.01] for t in t_values])
    ax1.plot(t_values, values, label=f"{E:.3g}")

ax1.set_yscale("log")
ax1.set_xlim([-10000, 10000])
ax1.set_ylim([1e18, 1e28])
ax1.set_xlabel("Time (days)", fontsize=14)
ax1.set_ylabel(r"No. density ($\mathrm{m^{-3}\,GeV^{-1}}$)", fontsize=14)
ax1.set_title(f"bigSolution{number}", fontsize=16)

# Match color-coded legend style
ax1.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), title="")
plt.tight_layout()
plt.show()

# --------------------------
# PANEL 2: Density vs Energy
# --------------------------

fig2, ax2 = plt.subplots(figsize=(6, 4))
t_plot = [1000, 2000, 4000, 8000]

for t in t_plot:
    spectrum = []
    for E in energy_levels:
        val = interpolators[E]((t, 0, 1, 0.01))
        if val is None or val < 1e0:
            val = 1e0
        spectrum.append(val)

    # Skip the first N energies to avoid the rising slope
    ax2.plot(energy_levels[1:], spectrum[1:], label=f"{t} days")

ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_ylim(1e7, 1e26)
ax2.set_xlabel("Energy (GeV)", fontsize=14)
ax2.set_ylabel(r"No. density ($\mathrm{m^{-3}\,GeV^{-1}}$)", fontsize=14)
ax2.set_title("No. Density vs Energy", fontsize=16)
ax2.legend(fontsize=10)
plt.tight_layout()
plt.show()