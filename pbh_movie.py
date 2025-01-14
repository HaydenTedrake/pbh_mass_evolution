import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.special import gamma

# Constants
age_of_universe = 4.35e17  # seconds
rho_DM = 0.403  # GeV/cm^3 (from paper)

def f(M):
    """Calculate f(M) with M in grams"""
    beta_masses = {
        'mu': 4.53e14,     # muon
        'd': 1.6e14,       # down quark
        's': 9.6e13,       # strange quark
        'c': 2.56e13,      # charm quark
        'T': 2.68e13,      # tau
        'b': 9.07e12,      # bottom quark
        't': 0.24e12,      # top quark
        'g': 1.1e14,       # gluon
        'e': 9.42e16,      # electron
        'w': 7.97e11,      # W boson
        'z': 7.01e11,      # Z boson
        'h': 2.25e11       # Higgs boson
    }
    
    base = 1.569
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

def lognormal_mass_function(M, mu, sigma, rho_DM):
    """Lognormal PBH mass function"""
    psi = (rho_DM / (np.sqrt(2 * np.pi) * sigma * M)) * \
          np.exp(-(np.log(M/mu))**2 / (2 * sigma**2))
    return psi

def calculate_mass_at_time(M0, t):
    """Analytical solution for mass at time t"""
    mass_cubed = (-16.02e25 * 1.0 * t + M0**3)
    return np.cbrt(np.maximum(mass_cubed, 0))

# Generate mass points following lognormal distribution
def generate_masses_from_distribution(n_points, mu, sigma):
    # Create mass range for sampling
    M = np.logspace(12, 16, 10000)
    
    # Calculate probability distribution
    prob = lognormal_mass_function(M, mu, sigma, rho_DM)
    prob = prob / np.sum(prob)  # Normalize
    
    # Generate random masses following the distribution
    return np.random.choice(M, size=n_points, p=prob)

# Setup parameters
n_pbhs = 1000  # number of PBHs to simulate
mu = 1e15      # location parameter
sigma = 2      # width parameter

# Generate initial masses using lognormal distribution
initial_masses = generate_masses_from_distribution(n_pbhs, mu, sigma)

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Mass (g)')
ax.set_ylabel('Number of PBHs')
ax.set_title('PBH Mass Distribution Evolution')

# Number of frames for the animation
n_frames = 100
times = np.linspace(0, age_of_universe, n_frames)

def update(frame):
    ax.clear()
    
    # Calculate masses at current time
    current_masses = calculate_mass_at_time(initial_masses, times[frame])
    
    # Remove evaporated black holes (mass < 1e9)
    current_masses = current_masses[current_masses > 1e9]
    
    if len(current_masses) > 0:
        # Create histogram
        ax.hist(current_masses, bins=np.logspace(9, 16, 50), 
                alpha=0.6, density=True)
        
        # Plot the theoretical distribution scaled to match histogram
        M_plot = np.logspace(9, 16, 1000)
        current_dist = lognormal_mass_function(M_plot, mu, sigma, rho_DM)
        ax.plot(M_plot, current_dist/np.max(current_dist), 'r-', 
                label='Initial Distribution (scaled)')
        
        # Set scales and labels
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Mass (g)')
        ax.set_ylabel('Number of PBHs (normalized)')
        ax.set_title(f'PBH Mass Distribution at {times[frame]/age_of_universe:.2%} of Universe Age')
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend()
        
        # Set consistent axis limits
        ax.set_xlim(1e9, 1e16)
        ax.set_ylim(1e-20, 1e1)

# Create animation
anim = FuncAnimation(fig, update, frames=n_frames, 
                    interval=100, repeat=False)

plt.show()
