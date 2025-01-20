import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import math
from scipy.integrate import solve_ivp

age_of_universe = 4.35e17  # in seconds

# Parameters
N = 100  # Number of samples
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
# plt.ylabel('Count')
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

def evolve(masses, n_time_points=10):
    """
    Evolve an array of black hole masses over time.
    """
    times = np.geomspace(1, age_of_universe, n_time_points)
    mass_history = np.zeros((len(masses), n_time_points))
    
    for i, initial_mass in enumerate(masses):
        def dMdt(t, M):
            return Mdot(M[0])
        
        def event_mass_threshold(t, M):
            return M[0] - 1e9
        
        event_mass_threshold.terminal = True
        event_mass_threshold.direction = -1
        
        solution = solve_ivp(
            dMdt,
            t_span=(times[0], times[-1]),
            y0=[initial_mass],
            method='RK45',
            t_eval=times,
            events=event_mass_threshold,
            rtol=1e-6,
            atol=1e-6
        )
        
        mass_history[i, :len(solution.t)] = solution.y[0]
        
        if len(solution.t) < n_time_points:
            mass_history[i, len(solution.t):] = 1e9
            
    return mass_history, times

# Evolve the masses
mass_history, times = evolve(sampled_masses)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate initial distribution to set y-axis limits
initial_masses = mass_history[:, 0]
hist_initial, bins_initial = np.histogram(np.log10(initial_masses[initial_masses > 1e9]), bins=50)
y_max = np.max(hist_initial) * 1.1  # Add 10% margin

def animate(frame):
    ax.clear()
    
    # Plot initial distribution in red with transparency
    ax.hist(np.log10(initial_masses[initial_masses > 1e9]), 
            bins=50, color="red", alpha=0.3, edgecolor="darkred", label="Initial Distribution")
    
    # Plot current distribution in blue
    masses_at_time = mass_history[:, frame]
    ax.hist(np.log10(masses_at_time[masses_at_time > 1e9]), 
            bins=50, color="skyblue", alpha=0.7, edgecolor="black", label="Current Distribution")
    
    ax.set_xlabel('log10(mass) [g]')
    ax.set_ylabel('Count')
    ax.set_xlim(11, 19)
    ax.set_ylim(0, y_max)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f'Mass Distribution (Time: {times[frame]:.2e} seconds)')

# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=len(times), interval=100)

# writer = PillowWriter(fps=15)
# ani.save("black_hole_evolution.gif", writer=writer)

# print("Animation saved successfully!")
# plt.close()
plt.show()
