
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
n = 1000

def time_to_index(t):
    return int(np.floor(t + n / 2) + 1)

def index_to_time(i):
    return i - n / 2 - 1

i_start = 1
i_stop = n + 1

t_start = index_to_time(i_start)
t_stop = index_to_time(i_stop)

# Response function normalization
# ∫ x^2 exp(-x / dk) dx from 0 to ∞ = 2 * dk^3 (when dk > 0)

decay = (400 / 20000) * n
mean = (-5000 / 20000) * n
sigma = (200 / 20000) * n

def M_func(t, t_prime):
    delta = t - t_prime
    if delta < 0 or delta > 50:
        return 0
    return (delta ** 2 * np.exp(-delta / decay)) / (2 * decay ** 3)

def a_func(t):
    return np.exp(-(t - mean) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)

# Construct time array and compute a and M
p = np.array([index_to_time(i) for i in range(i_start, i_stop + 1)])
a = np.array([a_func(index_to_time(i)) for i in range(i_start, i_stop + 1)])
M = np.array([[M_func(index_to_time(i), index_to_time(j)) 
               for j in range(i_start, i_stop + 1)] 
               for i in range(i_start, i_stop + 1)])

# Contour plot of M
plt.figure()
plt.contourf(M, levels=100)
plt.colorbar()
plt.title("Response Matrix M")
plt.xlabel("j")
plt.ylabel("i")
plt.show()

# Compute response vector g = M @ a
g = M @ a

# Plot input and response
plt.figure()
plt.plot(p, a, label="a (input)")
plt.plot(p, g, label="g (response)")
plt.xlabel("Time (days)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.title("Input and Response")
plt.show()

# Check normalization
norm = (a.T @ a)

# Compute extracted M
M_ext = np.outer(g, a) / norm

# Contour plot of extracted M
plt.figure()
plt.contourf(M_ext, levels=100)
plt.colorbar()
plt.title("Extracted M")
plt.xlabel("j")
plt.ylabel("i")
plt.show()

# Compare slices
slice_index = 250
plt.figure()
plt.plot(M[:, slice_index] / np.linalg.norm(M[:, slice_index]), label="M_input(250)")
for s in [230, 240, 250]:
    plt.plot(M_ext[:, s] / np.linalg.norm(M_ext[:, s]), label=f"M_extracted({s})")
plt.legend()
plt.xlabel("i")
plt.ylabel("M / |M|")
plt.grid(True)
plt.title("Comparison of M Slices")
plt.show()
