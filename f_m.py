import numpy as np
import matplotlib.pyplot as plt
import math

# Constants
hbar = 1.0545718e-27  # erg·s
c = 2.99792458e10     # cm/s
G = 6.67430e-8        # cm³/g·s²

def gev_to_grams(energy_gev):
    energy_ergs = energy_gev * 1.60218e-3
    mass_grams = energy_ergs / (c ** 2)
    
    return mass_grams

def f(M):
    """
    Calculate f(M) with M in grams
    """
    # particle masses in GeV
    masses_gev = {
        'mu': 0.10566,          # muon
        'u': 0.34,              # up quark
        'd': 0.34,              # down quark
        's': 0.96,              # strange quark
        'c': 1.28,              # charm quark
        'T': 1.7768,            # tau
        'b': 4.18,              # bottom quark
        't': 173.1,             # top quark
        'g': 0.650,             # gluon
        'w': 80.433,            # W boson
        'z': 91.19,             # Z boson
        'h': 124.07,            # Higgs boson
    }
    
    # Convert GeV to grams
    masses = {key: gev_to_grams(value) for key, value in masses_gev.items()}
    
    # Spin values for each particle
    spins = {
        'mu': '1/2', 'u': '1/2', 'd': '1/2', 's': '1/2', 'c': '1/2',
        'T': '1/2', 'b': '1/2', 't': '1/2', 'g': '1',
        'w': '1', 'z': '1', 'h': '0'
    }
    
    # Calculate beta masses based on the paper's values
    # Planck mass in grams
    planck_mass = np.sqrt(hbar * c / G)
    
    # beta_coefficient based on spin (these values need verification against the paper)
    beta_coef = {
        '0': 2.66,
        '1/2': 4.53,
        '1': 6.04,
        '2': 9.56
    }
    
    # Calculate beta masses for each particle
    beta_masses = {}
    for particle, mass in masses.items():
        spin = spins[particle]
        beta_masses[particle] = (planck_mass**2 * beta_coef[spin]) / mass
    
    # Base constant from the original equation
    base = 1.569
    
    # Calculate f(M) using the same structure as the original code
    result = (
        base
        + 0.569 * (
            np.exp(-M / beta_masses['mu'])
            + 3 * np.exp(-M / beta_masses['u'])
            + 3 * np.exp(-M / beta_masses['d'])
            + 3 * np.exp(-M / beta_masses['s'])
            + 3 * np.exp(-M / beta_masses['c'])
            + np.exp(-M / beta_masses['T'])
            + 3 * np.exp(-M / beta_masses['b'])
            + 3 * np.exp(-M / beta_masses['t'])
            + 0.963 * np.exp(-M / beta_masses['g'])
        )  
        + 0.36 * np.exp(-M / beta_masses['w'])
        + 0.18 * np.exp(-M / beta_masses['z'])
        + 0.267 * np.exp(-M / beta_masses['h'])
    )
    return result

# Define a range for M in logarithmic space
M_values = np.logspace(9, 20, 10000)  # M values from 10^9 to 10^20
f_values = f(M_values)

# Plot the function using logarithmic scales
plt.figure(figsize=(12, 8))
plt.semilogx(M_values, f_values, label='$f(M)$', color='blue', linewidth=2)

# Customize the plot
plt.xlabel('$M$ (Grams)', fontsize=14)
plt.ylabel('$f(M)$', fontsize=14)
plt.title('Plot of $f(M)$', fontsize=16)
plt.grid(True, which="both", ls="-", alpha=0.4)
plt.legend(fontsize=12)

# Customize y-axis ticks
plt.yticks(np.arange(2, 16, 2))  # Set y-axis ticks at intervals of 2, up to 14

# Display the plot
plt.show()