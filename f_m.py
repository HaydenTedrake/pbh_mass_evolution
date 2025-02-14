import numpy as np
import matplotlib.pyplot as plt
import math

# Constants
hbar = 1.0545718e-27  # erg·s
c = 2.99792458e10  # cm/s
G = 6.67430e-8  # cm³/g·s²

def f(M):
    """
    Calculate f(M) with M in grams
    """
    masses = {
        'mu': 1.883531627e-28,  # muon
        'u': 3.8e-30,           # up quark
        'd': 7.6e-30,           # down quark
        's': 2.9e-28,           # strange quark
        'c': 2.3e-27,           # charm quark
        'T': 3.16754e-27,       # tau
        'b': 7.9e-27,           # bottom quark
        't': 3.1e-25,           # top quark
        'g': 0.0,               # gluon (massless)
        'w': 1.433e-25,         # W boson
        'z': 1.625e-25,         # Z boson
        'h': 3.732e-25          # Higgs boson
    }

    beta_0 = 2.66
    beta_1/2 = 4.53
    beta_1 = 6.04
    beta_2 = 9.56

    base = 1.569

    def beta_masses(mass, spin):
        """Calculate hbar * c^3 / (8 * pi * G * mass) and return in grams."""
        return (hbar * c**3) / (8 * math.pi * G * mass)
    
    result = {}
    # # in grams from Table I
    # beta_masses = {
    #     'mu': 4.53e14,     # muon
    #     'u': 1.6e14,       # up quark
    #     'd': 1.6e14,       # down quark
    #     's': 9.6e13,       # strange quark
    #     'c': 2.56e13,      # charm quark
    #     'T': 2.68e13,      # tau
    #     'b': 9.07e12,      # bottom quark
    #     't': 0.24e12,      # top quark (unobserved at time of paper)
    #     'g': 1.1e14,       # gluon (effective mass)
    #     'w': 7.97e11,      # W boson
    #     'z': 7.01e11,      # Z boson
    #     'h': 2.25e11       # Higgs boson
    # }
    
    # # Base constant from the original equation
    # base = 1.569
    
    # # Detailed calculation following the exact equation
    # result = (
    #     base
    #     + 0.569 * (
    #         np.exp(-M / beta_masses['mu'])
    #         + 3 * np.exp(-M / beta_masses['u'])
    #         + 3 * np.exp(-M / beta_masses['d'])
    #         + 3 * np.exp(-M / beta_masses['s'])
    #         + 3 * np.exp(-M / beta_masses['c'])
    #         + np.exp(-M / beta_masses['T'])
    #         + 3 * np.exp(-M / beta_masses['b'])
    #         + 3 * np.exp(-M / beta_masses['t'])
    #         + 0.963 * np.exp(-M / beta_masses['g'])
    #     )  
    #     + 0.36 * np.exp(-M / beta_masses['w'])
    #     + 0.18 * np.exp(-M / beta_masses['z'])
    #     + 0.267 * np.exp(-M / beta_masses['h'])
    # )
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
