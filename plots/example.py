import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.linalg import toeplitz 
from tqdm import tqdm

# ------
# Solver
# ------

def extract_toeplitz_first_col(g, a):
    n = len(a)
    k = n

    # columns are shifted versions of a
    A = np.column_stack([np.roll(a, i) for i in range(k)])
    for i in range(k):
        A[:i, i] = 0

    c, residuals, rank, s = np.linalg.lstsq(A, g, rcond=None)

    if k < n:
        c = np.pad(c, (0, n - k), constant_values=0)

    return c

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

n = 2000
time_range = np.arange(-6000, -3999, 1)

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
peak_index = np.argmax(a)

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

# --------------------------------
# Extracted Toeplitz approximation
# --------------------------------

toeplitz_col = extract_toeplitz_first_col(g, a)
M_ext = toeplitz(toeplitz_col)
print("M_ext[:5, :5] =\n", M_ext[:5, :5])
print("M_ext norm =", np.linalg.norm(M_ext))

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

plt.figure()

# Input M slice
for s in [peak_index - 10, peak_index, peak_index + 10]:
    m_input_slice = M[:, s]
    norm = np.linalg.norm(m_input_slice)
    m_input_slice = m_input_slice / norm
    plt.plot(m_input_slice, label=f"M_input({s})")

# Compare with extracted slices
for s in [peak_index - 10, peak_index, peak_index + 10]:
    col = M_ext[:, s]
    norm = np.linalg.norm(col)
    print(f"a[{s}] = {a[s]}, norm = {norm}")
    if norm > 1e-10:
        m_ext_slice = col / norm
        plt.plot(m_ext_slice, label=f"M_extracted({s})")
    else:
        print(f"Skipped M_ext column {s}: norm too small ({norm})")

plt.legend()
plt.xlabel("i")
plt.ylabel("M / |M|")
plt.grid(True)
plt.title("Comparison of M Slices")
plt.tight_layout()
plt.show()
