import numpy as np
import matplotlib.pyplot as plt

def pdf(mass, mu, sigma):
    # Lognormal probability density function (PDF)
    pdf = (1 / (mass * sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(mass / mu) / sigma) ** 2))
    return pdf

def run_pdf(Mdict, N):
    rng = np.random.default_rng()
    
    # Create a range of masses to sample from
    masses = np.logspace(11, 19, 10000)
    
    # Calculate PDF values
    pdf_values = pdf(masses, Mdict['mean'], Mdict['sigma'])
    
    # Normalize PDF to make it a proper probability distribution
    pdf_normalized = pdf_values / np.sum(pdf_values)
    
    # Sample from the distribution
    sampled_indices = rng.choice(len(masses), size=N, p=pdf_normalized)
    sampled_masses = masses[sampled_indices]
    
    return sampled_masses

# Set up parameters
Mdict = {
    'type': 'normal',
    'mean': 1e15,
    'sigma': 2.0
}

N = 10000

# Sample masses
sampled_masses = run_pdf(Mdict, N)

# Create histogram of sampled masses
plt.figure(figsize=(10, 6))
plt.hist(np.log10(sampled_masses), bins=100, density=True)
plt.xlabel('log10(mass) [g]')
plt.ylabel('Number')
plt.xlim(11, 19)
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.title('Mass Distribution')
plt.show()
