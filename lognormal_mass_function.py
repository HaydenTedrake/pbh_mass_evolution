import numpy as np
import matplotlib.pyplot as plt

# Parameters for the lognormal distribution
mu = 10**15  # Mean mass in grams
sigma = 2  # Standard deviation in log space

# Generate masses over a logarithmic range
log_mass = np.linspace(10, 17, 1000)  # log10(mass) range from 10^12 g to 10^19 g
mass = 10**log_mass  # Convert to linear scale

def pdf(mass, mu, sigma):
    # Lognormal probability density function (PDF)
    pdf = (1 / (mass * sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(mass / mu) / sigma) ** 2))
    return pdf

# Plotting the initial mass distribution
plt.figure(figsize=(10, 6))
plt.plot(log_mass, pdf, color='salmon')
plt.fill_between(log_mass, pdf, color='salmon', alpha=0.5)
plt.title("Initial Mass Distribution")
plt.xlabel("log10(Mass) [g]")
plt.ylabel("Probability Density")
plt.grid(True)
plt.show()
