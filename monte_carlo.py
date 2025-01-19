import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from scipy.integrate import solve_ivp

age_of_universe = 4.35e17  # in seconds

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

# # Create a histogram of the sampled masses
# plt.figure(figsize=(10, 6))
# plt.hist(np.log10(sampled_masses), bins=100, color="skyblue", edgecolor="black")
# plt.xlabel('log10(mass) [g]')
# plt.ylabel('Density')
# plt.xlim(11, 19)
# plt.grid(True, alpha=0.3)
# plt.title('Mass Distribution')
# plt.show()

def f(M):
    """
    Calculate f(M) with M in grams
    M_grams: black hole mass in grams
    """
    # in grams from Table I
    beta_masses = {
        'mu': 4.53e14,     # muon
        'd': 1.6e14,       # down quark
        's': 9.6e13,        # strange quark
        'c': 2.56e13,       # charm quark
        'T': 2.68e13,       # tau
        'b': 9.07e12,       # bottom quark
        't': 0.24e12,        # top quark (unobserved at time of paper)
        'g': 1.1e14,         # gluon (effective mass)
        'e': 9.42e16,         # electron
        'w': 7.97e11,         # W boson
        'z': 7.01e11,         # Z boson
        'h': 2.25e11          # Higgs boson
    }
    
    # Base constant from the original equation
    base = 1.569
    
    # Detailed calculation following the exact equation
    result = base + 0.569 * (
        np.exp(-M / (beta_masses['mu'])) +
        3 * np.exp(-M / (beta_masses['d'])) +
        3 * np.exp(-M / (beta_masses['s'])) +
        3 * np.exp(-M / (beta_masses['c'])) +
        np.exp(-M / (beta_masses['T'])) +
        3 * np.exp(-M / (beta_masses['b'])) +
        3 * np.exp(-M / (beta_masses['t'])) +
        0.963 * np.exp(-M / (beta_masses['g'])) +
        np.exp(-M / (beta_masses['e'])) +
        np.exp(-M / (beta_masses['w'])) +
        np.exp(-M / (beta_masses['z'])) +
        np.exp(-M / (beta_masses['h']))
    )
    
    return result

def Mdot(M):
    return -5.34e25 * f(M) / (M * M)

# I have the masses ass sampled_masses as a numpy array, now i need to evolve each one of them over the age of the universe

def evolve(masses):
    def dMdt(t, M):
        return Mdot(M[0])
    
    # Set up solver parameters
    rtol = 1e-5
    atol = 1e-5
    
    # Use solve_ivp with adaptive step size
    solution = solve_ivp(
        dMdt,
        t_span=(0, age_of_universe),
        y0=masses,
        method='RK45',
        rtol=rtol,
        atol=atol,
        #max_step=dt if dt is not None else np.inf,
        #first_step=100000,  # this was a simple test from Russ (don't trust it!)
        dense_output=True
    )

    return solution.y[0], solution.t

# Evolve the masses and save states
mass_history, times = evolve(sampled_masses)

# Create the animation from saved states
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel('log10(mass) [g]')
ax.set_ylabel('Count')
ax.set_xlim(11, 19)
ax.grid(True, alpha=0.3)

# Update function for animation
def animate(frame):
    ax.clear()  # Clear the current plot
    ax.set_xlabel('log10(mass) [g]')
    ax.set_ylabel('Count')
    ax.set_xlim(11, 19)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'PBH Masses Evolved (Time = {times[frame]:.2f} seconds)')
    ax.hist(
        np.log10(mass_history[frame]),
        bins=100,
        color="skyblue",
        edgecolor="black",
        alpha=0.7
    )

# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=50, interval=200)

# # Save the animation as a video
# ani.save("mass_distribution_evolution.mp4", writer="ffmpeg", fps=10)
plt.show()
