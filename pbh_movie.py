import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Constants from your original code
age_of_universe = 4.35e17  # seconds

def f(M):
    """Simplified f(M) for performance"""
    return 1.569  # base value without exponential terms

def Mdot(M):
    """Mass loss rate"""
    return -5.34e25 * f(M) / (M * M)

def calculate_mass_at_time(M0, t):
    """Analytical solution for mass at time t"""
    mass_cubed = (-16.02e25 * 1.0 * t + M0**3)
    return np.cbrt(np.maximum(mass_cubed, 0))

# Set up initial distribution
n_pbhs = 1000  # number of PBHs to simulate
mean_log_mass = np.log(5e14)
std_log_mass = 0.5
initial_masses = np.exp(np.random.normal(mean_log_mass, std_log_mass, n_pbhs))

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
    
    if len(current_masses) > 0:  # Check if there are any black holes left
        # Create histogram
        ax.hist(current_masses, bins=np.logspace(9, 16, 50), 
                alpha=0.6, density=True)
        
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

# Save animation (optional)
# anim.save('pbh_evolution.gif', writer='pillow')

plt.show()
