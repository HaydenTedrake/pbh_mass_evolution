import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from tqdm import tqdm

# ------------------------------
# LOAD BIN FILES AND INTERPOLATE
# ------------------------------

energy_levels = [100, 125.893, 158.489, 199.526, 251.189, 316.228, 398.107, 501.187,
                 794.328, 1000, 1258.93, 1584.89, 1995.26, 2511.89, 3162.28]

bin_files = [f"plots/bin(1)/bigSolution8-{i}.bin" for i in [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]]

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
ax1.set_title("bigSolution8", fontsize=16)

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

# ------------------------
# PANEL 3: Flux vs Energy
# -----------------------

c = 3e8  # speed of light in m/s
times = [1000, 2000, 4000, 8000]
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

fig, ax = plt.subplots(figsize=(6, 4))

dE = np.gradient(energy_levels)
scaling_factor = 1e-38  # Try 1e-38, 1e-42 if you need to tune it

for t_idx, t in enumerate(times):
    flux_values = []

    for i, E in enumerate(energy_levels):
        if i == 0:  # skip the 100 GeV bin
            flux_values.append(np.nan)
            continue

        u = interpolators[E]([[t, 0, 0, 0.01]])[0]
        if abs(u) < 1e-30:
            u = interpolators[E]([[t, 0, 0, 1.0]])[0]

        phi = scaling_factor * c * u / (4 * np.pi * dE[i])
        flux_values.append(phi)

    ax.plot(energy_levels, flux_values, label=f"{t} days", color=colors[t_idx])

# Formatting
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Energy (GeV)")
ax.set_ylabel(r"$\Phi\ (\mathrm{m}^{-2}\ \mathrm{s}^{-1}\ \mathrm{GeV}^{-1})$")
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()

# ---------------------------
# PANEL 4: Instantaneous Flux
# ---------------------------

times = [500, 1000, 2000, 4000, 8000]
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

fig, ax = plt.subplots(figsize=(6, 4))

dE = np.gradient(energy_levels)
scaling_factor = 1.5e-39  # Same as before, or slightly adjusted

for t_idx, t in enumerate(times):
    flux_values = []

    for i, E in enumerate(energy_levels):
        try:
            u = interpolators[E]([[t, 0, 0, 0.01]])[0]
            if abs(u) < 1e-30:
                u = interpolators[E]([[t, 0, 0, 1.0]])[0]

            # Start with the same Φ(E)
            phi = scaling_factor * c * u / dE[i]  # Remove / (4π) to keep per sr

            # Multiply by E^3 for this plot
            flux_e3 = (E**3) * phi
            flux_values.append(flux_e3)

        except Exception as e:
            flux_values.append(np.nan)

    ax.plot(energy_levels, flux_values, label=f"{t} days", color=colors[t_idx])

# Formatting
ax.set_xscale('log')
ax.set_xlabel("Energy (GeV)")
ax.set_ylabel(r"$E^3 \Phi\ (\mathrm{s}^{-1}\ \mathrm{GeV}^{-2}\ \mathrm{sr}^{-1})$")
ax.set_title("Instantaneous Flux")
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()

# ------------------------------
# PANEL 5: Averaged Spectrum
# ------------------------------

time_windows = [
    (0, 4380),
    (1000, 5380),
    (2000, 6380),
    (3000, 7380),
    (4000, 8380),
    (5000, 9380),
]

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'saddlebrown']
scaling_factor = 1.7e-39  # same one used for instantaneous plot
dE = np.gradient(energy_levels)

fig, ax = plt.subplots(figsize=(6, 4))

for (t_start, t_end), color in zip(time_windows, colors):
    flux_avg = []

    for i, E in enumerate(energy_levels):
        flux_time_series = []

        for t in range(t_start, t_end + 1, 10):  # Sample every 10 days
            try:
                u = interpolators[E]([[t, 0, 0, 1.0]])[0]  # z = 1.0 or whatever depth matched best
                phi = scaling_factor * c * u / dE[i]
                flux_time_series.append(phi)
            except:
                continue

        # Average over time
        if flux_time_series:
            avg_phi = np.mean(flux_time_series)
            flux_avg.append(E**3 * avg_phi)
        else:
            flux_avg.append(np.nan)

    ax.plot(energy_levels, flux_avg, label=f"{t_start}–{t_end} days", color=color)

# Formatting
ax.set_xscale("log")
ax.set_xlabel("Energy (GeV)")
ax.set_ylabel(r"$E^3 \Phi\ (\mathrm{s}^{-1}\ \mathrm{GeV}^{-2}\ \mathrm{sr}^{-1})$")
ax.set_title("Averaged Spectrum")
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()