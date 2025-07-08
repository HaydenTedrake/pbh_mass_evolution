
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


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d, RegularGridInterpolator
# from scipy.linalg import toeplitz 
# from tqdm import tqdm
# import pandas as pd

# # ------
# # Solver
# # ------

# def extract_toeplitz_first_col(g, a):
#     n = len(a)
#     k = n

#     # columns are shifted versions of a
#     A = np.column_stack([np.roll(a, i) for i in range(k)])
#     for i in range(k):
#         A[:i, i] = 0

#     c, residuals, rank, s = np.linalg.lstsq(A, g, rcond=None)

#     if k < n:
#         c = np.pad(c, (0, n - k), constant_values=0)

#     return c

# a = np.array([1, 0, 0])
# g = np.array([1, 0, 0])

# col = extract_toeplitz_first_col(g, a)
# print("Test 1 result:", col)

# a = np.array([1, 1, 1])
# g = np.array([1, 2, 3])

# col = extract_toeplitz_first_col(g, a)
# print("Test 2 result:", col)

# # ------------------------
# # Load Gaussian Data Files
# # ------------------------

# energy_levels = [130, 200, 320, 500, 790, 1300, 2000, 3200]     
# number = 21
# bin_files = [f"plots/bin(2)/bigSolution{number}-{i}.bin" for i in [22, 24, 26, 28, 30, 32, 34, 36]]

# t_values = np.arange(-10000, 10001, 2)
# x_values = np.arange(-10, 11, 2)
# y_values = np.arange(-10, 11, 2)
# z_values = np.arange(-10, 11, 2)

# interpolators = {}

# for E, file in tqdm(zip(energy_levels, bin_files), total=len(energy_levels), desc="Building Interpolators"):
#     data = np.fromfile(file, dtype=np.float64)
#     expected_length = len(t_values) * len(x_values) * len(y_values) * len(z_values)
#     if len(data) != expected_length:
#         raise ValueError(f"File {file}: Expected {expected_length}, got {len(data)}")
#     data = data.reshape(len(t_values), len(x_values), len(y_values), len(z_values))
#     interpolators[E] = RegularGridInterpolator((t_values, x_values, y_values, z_values), data, bounds_error=False)

# # -----------------------
# # Response Function Setup
# # -----------------------

# n = 2000
# time_range = np.arange(-6000, -3999, 1)

# decay = (400 / 20000) * n

# def M_func(t, t_prime):
#     delta = t - t_prime
#     if delta < 0 or delta > 50:
#         return 0
#     return (delta ** 2 * np.exp(-delta / decay)) / (2 * decay ** 3)

# # Build M matrix
# M = np.array([[M_func(ti, tj) for tj in time_range] for ti in time_range])

# # --------------------------------
# # Use File-Based Gaussian as Input
# # --------------------------------

# selected_energy = 500

# # Get u(t, 0, 0, 0)
# a_data = interpolators[selected_energy]([[t, 0, 0, 0] for t in t_values])
# interp_a = interp1d(t_values, a_data, bounds_error=False, fill_value=0)
# a = interp_a(time_range)
# peak_index = np.argmax(a)

# # Compute response
# g = M @ a


# # --------------------
# # Plot 1: Contour of M
# # --------------------

# plt.figure()
# plt.contourf(M, levels=100)
# plt.colorbar()
# plt.title("Response Matrix M")
# plt.xlabel("j")
# plt.ylabel("i")
# plt.tight_layout()
# plt.show()

# # --------------------------
# # Plot 2: Input and Response
# # --------------------------

# plt.figure()
# plt.plot(time_range, a, label="a(t) (input)")
# plt.plot(time_range, g, label="g(t) (response)")
# plt.xlabel("Time (days)")
# plt.ylabel("Amplitude")
# plt.legend()
# plt.grid(True)
# plt.title(f"Input and Response at E = {selected_energy} GeV")
# plt.tight_layout()
# plt.show()

# # --------------------------------
# # Extracted Toeplitz approximation
# # --------------------------------

# df = pd.DataFrame({
#     'time (days)': time_range,
#     'a(t)': a,
#     'g(t)': g
# })

# df.to_csv("time_a_g.csv", index=False)
# print("Saved numerical values to time_a_g.csv")

# plt.figure()
# plt.semilogy(time_range, a, label="a (input)")
# plt.semilogy(time_range, g, label="g (response)")
# plt.xlabel("Time (days)")
# plt.ylabel("Amplitude (log scale)")
# plt.legend()
# plt.grid(True, which="both", ls="--")
# plt.title(f"Logarithmic Plot of Input and Response at E = {selected_energy} GeV")
# plt.tight_layout()
# plt.show()


# toeplitz_col = extract_toeplitz_first_col(g, a)
# M_ext = toeplitz(toeplitz_col)
# print("M_ext[:5, :5] =\n", M_ext[:5, :5])
# print("M_ext norm =", np.linalg.norm(M_ext))

# # ------------------------------
# # Plot 3: Contour of Extracted M
# # ------------------------------

# plt.figure()
# plt.contourf(M_ext, levels=100)
# plt.colorbar()
# plt.title("Extracted M")
# plt.xlabel("j")
# plt.ylabel("i")
# plt.tight_layout()
# plt.show()

# # ------------------------
# # Plot 4: Slice Comparison
# # ------------------------

# plt.figure()

# # Input M slice
# for s in [peak_index - 10, peak_index, peak_index + 10]:
#     m_input_slice = M[:, s]
#     norm = np.linalg.norm(m_input_slice)
#     m_input_slice = m_input_slice / norm
#     plt.plot(m_input_slice, label=f"M_input({s})")

# # Compare with extracted slices
# for s in [peak_index - 10, peak_index, peak_index + 10]:
#     col = M_ext[:, s]
#     norm = np.linalg.norm(col)
#     print(f"a[{s}] = {a[s]}, norm = {norm}")
#     if norm > 1e-10:
#         m_ext_slice = col / norm
#         plt.plot(m_ext_slice, label=f"M_extracted({s})")
#     else:
#         print(f"Skipped M_ext column {s}: norm too small ({norm})")

# plt.legend()
# plt.xlabel("i")
# plt.ylabel("M / |M|")
# plt.grid(True)
# plt.title("Comparison of M Slices")
# plt.tight_layout()
# plt.xlim(750,1250)
# plt.show()