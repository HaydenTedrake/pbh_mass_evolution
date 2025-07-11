import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd

# Parameters
n = 1000
time = np.arange(-n//2, n//2 + 1)  # time indices
dk = 400 / 20000 * n
mean = (-5000 / 20000) * n
sigma = (200 / 20000) * n

# Define M and a analytically
def M_func(t, t_prime):
    delta = t - t_prime
    if delta < 0 or delta > 50:  # Strictly lower-triangular
        return 0
    return (delta ** 2 * np.exp(-delta / dk)) / (2 * dk ** 3)

def a_func(t):
    return np.exp(-((t - mean) ** 2) / (2 * sigma ** 2)) / np.sqrt(2 * np.pi * sigma ** 2)

# Build arrays
a = np.array([a_func(t) for t in time])
M = np.array([[M_func(ti, tj) for tj in time] for ti in time])

# Compute g = M a
g = M @ a

# Plot M contour
plt.figure()
plt.contourf(M, levels=100)
plt.colorbar()
plt.title("True M matrix")
plt.xlabel("j")
plt.ylabel("i")
plt.show()

# Plot a and g
plt.figure()
plt.plot(time, a, label='a')
plt.plot(time, g, label='g')
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.title("Input and Response")
plt.show()

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

# Compare extracted M vs true M
a_max = np.max(np.abs(a))
M_ext = solve_toeplitz(g/(a_max+1e-300), a/(a_max+1e-300))

plt.figure()
plt.contourf(M_ext, levels=100)
plt.colorbar()
plt.title("Extracted M matrix")
plt.xlabel("j")
plt.ylabel("i")
plt.show()

peak_index = np.argmax(a)
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
    if norm > 1e-10 and np.all(np.isfinite(col)):
        m_ext_slice = col / norm
        plt.plot(m_ext_slice, label=f"M_extracted({s})")
    else:
        print(f"Skipped M_ext column {s}: invalid data (norm={norm})")

plt.legend()
plt.xlabel("i")
plt.xlim(200, 400)
plt.ylabel("M / |M|")
plt.grid(True)
plt.title("Comparison of M Slices")
plt.tight_layout()
plt.show()

df_M_ext = pd.DataFrame(M_ext)
df_M_ext.to_csv("M_ext.csv", index=False)