import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.special import gamma

# Constants
age_of_universe = 4.35e17  # seconds

def f(M):
    """Calculate f(M) with M in grams"""
    beta_masses = {
        'mu': 4.53e14, 'd': 1.6e14, 's': 9.6e13, 'c': 2.56e13,
        'T': 2.68e13, 'b': 9.07e12, 't': 0.24e12, 'g': 1.1e14,
        'e': 9.42e16, 'w': 7.97e11, 'z': 7.01e11, 'h': 2.25e11
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

def critical_collapse_mass_function(M, Mp):
    """Critical collapse PBH mass function"""
    nu = 0.36  # critical exponent for radiation fluid
    psi = (1 / (nu * Mp * gamma(nu + 1))) * \
          (M/Mp)**(1/nu) * \
          np.exp(-(M/Mp)**(1/nu))
    return psi

def calculate_mass_at_time(M0, t):
    """Analytical solution for mass at time t"""
    mass_cubed = (-16.02e25 * 1.0 * t + M0**3)
    return np.cbrt(np.maximum(mass_cubed, 0))

# Generate initial masses using critical collapse distribution
n_pbhs = 10000  # increased number for better statistics
Mp = 5e14  # peak mass for critical collapse
mass_range = np.logspace(12, 16, 1000)
probabilities = critical_collapse_mass_function(mass_range, Mp)
probabilities = probabilities / np.sum(probabilities)  # normalize
initial_masses = np.random.choice(mass_range, size=n_pbhs, p=probabilities)

# Set up the figure and histogram
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
        hist, bins, _ = ax.hist(current_masses, bins=np.logspace(9, 16, 50), 
                               alpha=0.6, density=True)
        
        # Also plot the theoretical distribution at t=0 for comparison
        if frame == 0:
            theoretical = critical_collapse_mass_function(bins[:-1], Mp)
            theoretical = theoretical / np.max(theoretical) * np.max(hist)
            ax.plot(bins[:-1], theoretical, 'r-', label='Theoretical Initial')
            ax.legend()
        
        # Set scales and labels
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Mass (g)')
        ax.set_ylabel('Number of PBHs (normalized)')
        ax.set_title(f'PBH Mass Distribution at {times[frame]/age_of_universe:.2%} of Universe Age')
        ax.grid(True, which="both", ls="-", alpha=0.2)
        
        # Set consistent axis limits
        ax.set_xlim(1e9, 1e16)
        ax.set_ylim(1e-20, 1e1)

# Create animation
anim = FuncAnimation(fig, update, frames=n_frames, 
                    interval=100, repeat=False)

plt.show()
