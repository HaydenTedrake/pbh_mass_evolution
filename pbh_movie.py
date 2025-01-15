import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation, PillowWriter

# Step 1: Define the functions
def f(M):
    """Calculate f(M) with M in grams."""
    beta_masses = {
        'mu': 4.53e14, 'd': 1.6e14, 's': 9.6e13, 'c': 2.56e13, 'T': 2.68e13,
        'b': 9.07e12, 't': 0.24e12, 'g': 1.1e14, 'e': 9.42e16, 'w': 7.97e11,
        'z': 7.01e11, 'h': 2.25e11
    }
    base = 1.569
    result = base + 0.569 * (
        np.exp(-M / beta_masses['mu']) + 3 * np.exp(-M / beta_masses['d']) +
        3 * np.exp(-M / beta_masses['s']) + 3 * np.exp(-M / beta_masses['c']) +
        np.exp(-M / beta_masses['T']) + 3 * np.exp(-M / beta_masses['b']) +
        3 * np.exp(-M / beta_masses['t']) + 0.963 * np.exp(-M / beta_masses['g']) +
        np.exp(-M / beta_masses['e']) + np.exp(-M / beta_masses['w']) +
        np.exp(-M / beta_masses['z']) + np.exp(-M / beta_masses['h'])
    )
    return result

def Mdot(M):
    """Mass evolution equation dM/dt."""
    return -5.34e25 * f(M) / (M * M)

def evolve_mass(M0, t_end):
    """Solve the mass evolution equation for a given initial mass M0 up to time t_end."""
    def dMdt(t, M):
        return Mdot(M[0])

    solution = solve_ivp(
        dMdt,
        t_span=(0, t_end),
        y0=[M0],
        method='RK45',
        rtol=1e-5,
        atol=1e-5
    )
    return solution.t, solution.y[0]

# Step 2: Sample initial masses
mu = 1e15  # Characteristic mass in grams
sigma = 0.5
num_samples = 1000
initial_masses = np.random.lognormal(mean=np.log(mu), sigma=sigma, size=num_samples)

# Step 3: Set up the animation
fig, ax = plt.subplots(figsize=(10, 6))
bins = np.logspace(10, 18, 50)  # Logarithmic bins for mass distribution
ax.set_xscale('log')
ax.set_xlim(1e10, 1e18)
ax.set_ylim(0, 80)
ax.set_xlabel("Mass (grams)")
ax.set_ylabel("Frequency")
ax.set_title("Evolution of PBH Mass Distribution")

hist_data = ax.hist(initial_masses, bins=bins, alpha=0.5, label="PBH Mass Distribution")[0]
line, = ax.plot([], [], color='r', label="Current Time Step")
ax.legend()

# Time points for animation (logarithmic spacing for faster early evolution)
time_points = np.logspace(10, 17.64, 100)  # Up to 4.35e17 seconds (age of the universe)

def update(frame):
    """Update function for animation."""
    t_end = time_points[frame]
    evolved_masses = []
    for M0 in initial_masses:
        _, M_evolved = evolve_mass(M0, t_end)
        evolved_masses.append(M_evolved[-1])
    evolved_masses = np.array(evolved_masses)
    
    ax.clear()
    ax.hist(evolved_masses, bins=bins, alpha=0.5, label="PBH Mass Distribution")
    ax.set_xscale('log')
    ax.set_xlim(1e10, 1e18)
    ax.set_ylim(0, 80)
    ax.set_xlabel("Mass (grams)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"PBH Mass Distribution at t = {t_end:.2e} seconds")
    ax.legend()

# Create the animation
anim = FuncAnimation(fig, update, frames=len(time_points), interval=100)

# # Save the animation as a GIF
# writer = PillowWriter(fps=10)
# anim.save("pbh_mass_evolution.gif", writer=writer)

plt.show()
