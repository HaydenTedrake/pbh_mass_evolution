import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Step 1: Define the lognormal mass function
def lognormal_mass_function(M, mu, sigma):
    """Lognormal PBH mass function"""
    psi = (1 / (np.sqrt(2 * np.pi) * sigma * M)) * \
          np.exp(-(np.log(M / mu)) ** 2 / (2 * sigma ** 2))
    return psi

def sample_lognormal_masses(mu, sigma, n_samples):
    """Sample masses from the lognormal distribution"""
    log_mu = np.log(mu)
    masses = np.random.lognormal(mean=log_mu, sigma=sigma, size=n_samples)
    return masses

# Step 2: Define the function f(M) based on provided particle data
def f(M):
    """Calculate f(M) with M in grams"""
    beta_masses = {
        'mu': 4.53e14,     # muon
        'd': 1.6e14,       # down quark
        's': 9.6e13,       # strange quark
        'c': 2.56e13,      # charm quark
        'T': 2.68e13,      # tau
        'b': 9.07e12,      # bottom quark
        't': 0.24e12,      # top quark (unobserved at time of paper)
        'g': 1.1e14,       # gluon (effective mass)
        'e': 9.42e16,      # electron
        'w': 7.97e11,      # W boson
        'z': 7.01e11,      # Z boson
        'h': 2.25e11       # Higgs boson
    }
    
    base = 1.569
    result = base + 0.569 * (
        np.exp(-M / beta_masses['mu']) +
        3 * np.exp(-M / beta_masses['d']) +
        3 * np.exp(-M / beta_masses['s']) +
        3 * np.exp(-M / beta_masses['c']) +
        np.exp(-M / beta_masses['T']) +
        3 * np.exp(-M / beta_masses['b']) +
        3 * np.exp(-M / beta_masses['t']) +
        0.963 * np.exp(-M / beta_masses['g']) +
        np.exp(-M / beta_masses['e']) +
        np.exp(-M / beta_masses['w']) +
        np.exp(-M / beta_masses['z']) +
        np.exp(-M / beta_masses['h'])
    )
    return result

def Mdot(M):
    """Mass evolution equation dM/dt"""
    return -5.34e25 * f(M) / (M * M)

# Step 3: Sample initial masses
mu = 1e15  # Characteristic mass in grams
sigma = 0.5
num_samples = 1000
initial_masses = sample_lognormal_masses(mu, sigma, num_samples)

# Step 4: Solve the mass evolution equation for each sampled mass
def evolve_mass(M0):
    """Solve the mass evolution equation for a given initial mass M0"""
    def dMdt(t, M):
        return Mdot(M[0])
    
    solution = solve_ivp(
        dMdt,
        t_span=(0, 1e17),  # Large time span to evolve until near present
        y0=[M0],
        method='RK45',
        rtol=1e-5,
        atol=1e-5
    )
    return solution.t, solution.y[0]

evolved_masses = []
for M0 in initial_masses:
    _, M_evolved = evolve_mass(M0)
    evolved_masses.append(M_evolved[-1])  # Final mass at the end of evolution

evolved_masses = np.array(evolved_masses)

# Step 5: Plot initial vs. evolved mass distribution
plt.figure(figsize=(12, 6))

plt.hist(initial_masses, bins=50, alpha=0.5, label="Initial Masses")
plt.hist(evolved_masses, bins=50, alpha=0.5, label="Evolved Masses")
plt.xlabel("Mass (grams)")
plt.ylabel("Frequency")
plt.title("Initial vs. Evolved PBH Mass Distribution")
plt.legend()
plt.show()
