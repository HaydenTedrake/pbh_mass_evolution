import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 10000  # Number of samples
sigma = 2  # Standard deviation
mu = 10**15  # Mean of the lognormal distribution

# Define the lognormal PDF
def pdf(mass, mu, sigma):
    return (1 / (mass * sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(mass / mu) / sigma) ** 2))

# Generate mass range
masses = np.logspace(11, 19, 10000)  # Masses from 10^11 to 10^19

# Compute PDF values
pdf_values = pdf(masses, mu, sigma)

# Normalize PDF
pdf_normalized = pdf_values / np.sum(pdf_values)

# Compute the CDF
cdf_values = np.cumsum(pdf_normalized)

# Ensure the CDF is normalized
cdf_values /= cdf_values[-1]

# Inverse transform sampling
random_values = np.random.rand(N)
sampled_masses = np.interp(random_values, cdf_values, masses)

# Create a histogram of the sampled masses
plt.figure(figsize=(10, 6))
plt.hist(np.log10(sampled_masses), bins=100, color="skyblue", edgecolor="black")
plt.xlabel('log10(mass) [g]')
plt.ylabel('Density')
plt.xlim(11, 19)
plt.grid(True, alpha=0.3)
plt.title('Mass Distribution')
plt.show()
