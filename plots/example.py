import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.linalg import toeplitz 
from tqdm import tqdm
import pandas as pd
from mpmath import mp, mpf

mp.dps = 50

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
# Use Analytical Gaussian as Input
# --------------------------------

sigma = 200
mu = -5000
norm_factor = np.sqrt(2 * np.pi * sigma**2)
a = 1000 * np.exp(-((time_range - mu)**2) / (2 * sigma**2)) / norm_factor
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
plt.title(f"Input and Response")
plt.tight_layout()
plt.show()

# --------------------------------
# Extracted Toeplitz approximation
# --------------------------------

toeplitz_col = extract_toeplitz_first_col(g, a)
M_ext = toeplitz(toeplitz_col)

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
plt.xlim(750,1250)
plt.show()