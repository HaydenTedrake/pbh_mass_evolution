import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd

# ------
# Solver
# ------

def solve_toeplitz(g, a, bandwidth=50):
    n = len(a)
    
    # Parameterization: Only store non-zero band elements
    def vec_to_M(v):
        M = np.zeros((n,n))
        for i in range(n):
            for j in range(max(0,i-bandwidth), i+1):
                M[i,j] = v[i-j]
        return M
    
    # Optimization objective
    def loss(v):
        return np.linalg.norm(vec_to_M(v) @ a - g)**2 + 1e-8*np.linalg.norm(v)**2
    
    # Solve
    v0 = np.zeros(bandwidth+1)
    res = minimize(loss, v0, method='L-BFGS-B')
    
    return vec_to_M(res.x)

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

a_max = np.max(np.abs(a))
M_ext = solve_toeplitz(g/(a_max+1e-300), a/(a_max+1e-300))

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