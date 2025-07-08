import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz

# Parameters
n = 1000
time = np.arange(-n//2, n//2 + 1)  # time indices
dk = 400 / 20000 * n
mean = (-5000 / 20000) * n
sigma = (200 / 20000) * n

# Define M and a analytically
def M_func(t, t_prime):
    delta = t - t_prime
    if delta < 0 or delta > 50:
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

toeplitz_col = extract_toeplitz_first_col(g, a)

# Compare extracted M vs true M
M_ext = toeplitz(toeplitz_col)

plt.figure()
plt.contourf(M_ext, levels=100)
plt.colorbar()
plt.title("Extracted M matrix")
plt.xlabel("j")
plt.ylabel("i")
plt.show()

print("Norm difference between true M and extracted M:", np.linalg.norm(M - M_ext))
