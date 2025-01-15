import numpy as np
import matplotlib.pyplot as plt

# Define the lognormal mass function as given
def lognormal_mass_function(M, mu, sigma):
    """
    Lognormal PBH mass function:
    psi(M) = (1 / (sqrt(2*pi) * sigma * M)) 
             * exp(-(ln(M/mu))^2 / (2*sigma^2))
    """
    return (1.0 / (np.sqrt(2.0 * np.pi) * sigma * M)) * \
           np.exp(- (np.log(M / mu))**2 / (2.0 * sigma**2))

# Parameters
mu = 10.0     # Example value for mu
sigma = 1.0   # Example value for sigma
n_samples = 10_000

# Monte Carlo sampling using NumPy’s lognormal:
#   M ~ LogNormal(mean=log(mu), sigma=sigma)
samples = np.random.lognormal(mean=np.log(mu), sigma=sigma, size=n_samples)

# --- (Optional) Compare histogram to the theoretical PDF ---

# 1) Create bins and histogram from the samples
num_bins = 50
counts, bin_edges = np.histogram(samples, bins=num_bins, density=True)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# 2) Evaluate the lognormal_mass_function at bin centers
pdf_values = lognormal_mass_function(bin_centers, mu, sigma)

# 3) Plot
plt.figure(figsize=(8, 5))

# Histogram of the Monte Carlo samples
plt.bar(bin_centers, counts, width=(bin_edges[1] - bin_edges[0]),
        alpha=0.6, label='Monte Carlo Samples')

# Theoretical PDF
plt.plot(bin_centers, pdf_values, 'r-', lw=2, label='Theoretical PDF')

plt.xlabel('M')
plt.ylabel('Probability Density')
plt.title(f'Lognormal Distribution (mu={mu}, sigma={sigma})')
plt.legend()
plt.grid(True)
plt.show()
