import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RegularGridInterpolator
from tqdm import tqdm

# ------------------------
# Load Gaussian Data Files
# ------------------------

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
    expected_length = len(t_values) * len(x_values) * len(y_values) * len(z_values)
    if len(data) != expected_length:
        raise ValueError(f"File {file}: Expected {expected_length}, got {len(data)}")
    data = data.reshape(len(t_values), len(x_values), len(y_values), len(z_values))
    interpolators[E] = RegularGridInterpolator((t_values, x_values, y_values, z_values), data, bounds_error=False)

# -----------------------
# Response Function Setup
# -----------------------

n = 1000
time_range = np.arange(-500, 501, 1)  # length = n + 1

decay = (400 / 20000) * n

def M_func(t, t_prime):
    delta = t - t_prime
    if delta < 0 or delta > 50:
        return 0
    return (delta ** 2 * np.exp(-delta / decay)) / (2 * decay ** 3)

# Build M matrix
M = np.array([[M_func(ti, tj) for tj in time_range] for ti in time_range])

# --------------------------------
# Use File-Based Gaussian as Input
# --------------------------------

selected_energy = 500

# Get u(t, 0, 0, 0)
a_data = interpolators[selected_energy]([[t, 0, 0, 0] for t in t_values])
interp_a = interp1d(t_values, a_data, bounds_error=False, fill_value=0)
a = interp_a(time_range)
a = a / np.max(a)  # normalize

# Compute response
g = M @ a

# --------------------
# Plot 1: Contour of M
# --------------------

plt.figure()
plt.contourf(M, levels=100)
plt.colorbar()
plt.title("Response Matrix M")
plt.xlabel("j")
plt.ylabel("i")
plt.tight_layout()
plt.show()

# --------------------------
# Plot 2: Input and Response
# --------------------------

plt.figure()
plt.plot(time_range, a, label="a(t) (input)")
plt.plot(time_range, g, label="g(t) (response)")
plt.xlabel("Time (days)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.title(f"Input and Response at E = {selected_energy} GeV")
plt.tight_layout()
plt.show()

# ----------------------------
# Extracted M: g a^T / (a^T a)
# ----------------------------

norm = a.T @ a
M_ext = np.outer(g, a) / norm

# ------------------------------
# Plot 3: Contour of Extracted M
# ------------------------------

plt.figure()
plt.contourf(M_ext, levels=100)
plt.colorbar()
plt.title("Extracted M")
plt.xlabel("j")
plt.ylabel("i")
plt.tight_layout()
plt.show()

# ------------------------
# Plot 4: Slice Comparison
# ------------------------

slice_index = 250
plt.figure()

# Normalize M slice
m_slice = M[:, slice_index]
m_slice = m_slice / np.linalg.norm(m_slice)
plt.plot(m_slice, label=f"M_input({slice_index})")

# Compare with extracted slices
for s in [230, 240, 250]:
    m_ext_slice = M_ext[:, s] / np.linalg.norm(M_ext[:, s])
    plt.plot(m_ext_slice, label=f"M_extracted({s})")

plt.legend()
plt.xlabel("i")
plt.ylabel("M / |M|")
plt.grid(True)
plt.title("Comparison of M Slices")
plt.tight_layout()
plt.show()