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
