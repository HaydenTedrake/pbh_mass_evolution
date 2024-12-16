import numpy as np
import matplotlib.pyplot as plt

def f(M):
    # Translate black hole masses into GeV
    # Masses in GeV from Table I
    masses = {
        'mu': 0.106,     # muon
        'd': 0.34,       # down quark2
        's': 0.5,        # strange quark
        'c': 1.87,       # charm quark
        'T': 1.78,       # tau
        'b': 5.28,       # bottom quark
        't': 100,        # top quark (unobserved at time of paper)
        'g': 0.6         # gluon (effective mass)
    }
    
    # Beta values (assuming s=1/2 and s=1 as mentioned)
    beta_half = 4.53
    beta_one = 6.04
    
    # Base constant from the original equation
    base = 1.569
    
    # Detailed calculation following the exact equation
    result = base + 0.569 * (
        np.exp(-M / (beta_half * masses['mu'])) +
        3 * np.exp(-M / (beta_half * masses['d'])) +
        3 * np.exp(-M / (beta_half * masses['s'])) +
        3 * np.exp(-M / (beta_half * masses['c'])) +
        np.exp(-M / (beta_half * masses['T'])) +
        3 * np.exp(-M / (beta_half * masses['b'])) +
        3 * np.exp(-M / (beta_half * masses['t'])) +
        0.963 * np.exp(-M / (beta_one * masses['g']))
    )
    return result

# Define a range for M in logarithmic space
M_values = np.logspace(0, 15, 500)  # M values from 10^0 to 10^15
f_values = f(M_values)

# Plot the function using logarithmic scales
plt.figure(figsize=(12, 8))
plt.loglog(M_values, f_values, label='$f(M)$', color='blue', linewidth=2)

# Customize the plot
plt.xlabel('$M$ (GeV)', fontsize=14)
plt.ylabel('$f(M)$', fontsize=14)
plt.title('Plot of $f(M)$', fontsize=16)
plt.grid(True, which="both", ls="-", alpha=0.4)
plt.legend(fontsize=12)

# Customize y-axis ticks
plt.yticks(np.arange(2, 16, 2))  # Set y-axis ticks at intervals of 2, up to 14
plt.ylim(-1, 14)  # Set y-axis limits to match the desired range

# Display the plot
plt.show()
