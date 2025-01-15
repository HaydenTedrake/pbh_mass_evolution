# import numpy as np
# import matplotlib.pyplot as plt

# # Parameters for the lognormal distribution
# mu = 10**15  # Mean mass in grams
# sigma = 2  # Standard deviation in log space

# # Generate masses over a logarithmic range
# log_mass = np.linspace(12, 19, 1000)  # log10(mass) range from 10^12 g to 10^19 g
# mass = 10**log_mass  # Convert to linear scale

# # Lognormal probability density function (PDF)
# pdf = (1 / (mass * sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(mass / mu) / sigma) ** 2))

# # Plotting the initial mass distribution
# plt.figure(figsize=(10, 6))
# plt.plot(log_mass, pdf, color='salmon')
# plt.fill_between(log_mass, pdf, color='salmon', alpha=0.5)
# plt.title("Initial Mass Distribution")
# plt.xlabel("log10(Mass) [g]")
# plt.ylabel("Probability Density")
# plt.grid(True)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Parameters for the lognormal distribution
mu = 10**15  # Mean mass in grams
sigma = 2  # Standard deviation in log space
N = 10000  # Number of samples

# Generate samples from the lognormal distribution
samples = np.random.lognormal(mean=np.log(mu), sigma=sigma, size=N)

# Convert to log10 scale for plotting
log_samples = np.log10(samples)

# Plotting the histogram of the samples
plt.figure(figsize=(10, 6))
plt.hist(log_samples, bins=50, color='salmon', alpha=0.7)
plt.title("Initial Mass Distribution (N=10000)")
plt.xlabel("log10(Mass) [g]")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
