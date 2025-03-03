import numpy as np
import matplotlib.pyplot as plt
import math

# Constants
hbar = 1.0545718e-27  # erg·s
c = 2.99792458e10  # cm/s
G = 6.67430e-8  # cm³/g·s²

def gev_to_grams(energy_gev):
    # Constants
    c = 2.998e8              # Speed of light in m/s
    eV_per_joule = 6.242e18  # 1 Joule = 6.242 x 10^18 eV
    gev_per_ev = 1e-9        # 1 GeV = 10^9 eV
    
    # Convert energy from GeV to Joules
    energy_joules = energy_gev / (eV_per_joule * gev_per_ev)
    
    # Convert energy to mass in grams via E = mc^2  =>  m = E / c^2
    mass_grams = energy_joules / (c ** 2)
    
    return mass_grams

def old_f(M):
    """
    Calculate f(M) with M in grams
    """
    # in grams from Table I
    beta_masses = {
        'mu': 4.53e14,     # muon
        'u': 1.6e14,       # up quark
        'd': 1.6e14,       # down quark
        's': 9.6e13,       # strange quark
        'c': 2.56e13,      # charm quark
        'T': 2.68e13,      # tau
        'b': 9.07e12,      # bottom quark
        't': 0.24e12,      # top quark (unobserved at time of paper)
        'g': 1.1e14,       # gluon (effective mass)
        'w': 7.97e11,      # W boson
        'z': 7.01e11,      # Z boson
        'h': 2.25e11       # Higgs boson
    }
    
    # Base constant from the original equation
    base = 1.569
    
    # Detailed calculation following the exact equation
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

def f(M):
    """
    Calculate f(M) with M in grams
    """
    # particle masses in GeV
    masses = {
        'mu': gev_to_grams(0.10566),          # muon
        'u': gev_to_grams(0.34),              # up quark
        'd': gev_to_grams(0.34),              # down quark
        's': gev_to_grams(0.96),              # strange quark
        'c': gev_to_grams(1.28),              # charm quark
        'T': gev_to_grams(1.7768),            # tau
        'b': gev_to_grams(4.18),              # bottom quark
        't': gev_to_grams(173.1),             # top quark
        'g': gev_to_grams(0.650),             # gluon
        'w': gev_to_grams(80.433),            # W boson
        'z': gev_to_grams(91.19),             # Z boson
        'h': gev_to_grams(124.07),            # Higgs boson
    }

    beta_values = {
        '0': 2.66,
        '1/2': 4.53,
        '1': 6.04,
        '2': 9.56
    }

    base = 1.569

    def beta_masses(mass, spin):
        """Calculate hbar * c^3 / (8 * pi * G * mass) and return in grams."""
        return (hbar * c**3) / (8 * math.pi * G * mass) * beta_values[spin]

    result = (
        base
        + 0.569 * (
            np.exp(-M / beta_masses(masses['mu'], '1/2'))
            + 3 * np.exp(-M / beta_masses(masses['u'], '1/2'))
            + 3 * np.exp(-M / beta_masses(masses['d'], '1/2'))
            + 3 * np.exp(-M / beta_masses(masses['s'], '1/2'))
            + 3 * np.exp(-M / beta_masses(masses['c'], '1/2'))
            + np.exp(-M / beta_masses(masses['T'], '1/2'))
            + 3 * np.exp(-M / beta_masses(masses['b'], '1/2'))
            + 3 * np.exp(-M / beta_masses(masses['t'], '1/2'))
            + 0.963 * np.exp(-M / beta_masses(masses['g'], '1'))
        )  
        + 0.36 * np.exp(-M / beta_masses(masses['w'], '1'))
        + 0.18 * np.exp(-M / beta_masses(masses['z'], '1'))
        + 0.267 * np.exp(-M / beta_masses(masses['h'], '0'))
    )
    return result

# Define a range for M in logarithmic space
M_values = np.logspace(0, 10, 10000)  # M values from 10^9 to 10^20
f_values = f(M_values)
print(f_values)


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