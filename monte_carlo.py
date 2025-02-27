import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp
from scipy.special import gamma

# -------------------------
# CONFIGURATION & PARAMETERS
# -------------------------
mass_function_choice = 'critical_collapse'  # options: 'lognormal' or 'critical_collapse'
age_of_universe = 4.35e17  # in seconds (4.35e17 ~ 13.8 billion years in seconds)
N = 1000  # number of black holes to sample/evolve

# Lognormal parameters (if using 'lognormal')
sigma = 2.0       # standard deviation
mu = 10e15         # mean of the lognormal distribution (grams)

# Critical collapse parameters (if using 'critical_collapse')
Mp = 10e15         # characteristic mass scale (grams)
nu = 0.35

# -------------------------
# MASS FUNCTION DEFINITIONS
# -------------------------

def lognormal_pdf(mass, mu, sigma):
    """
    Lognormal PDF:
    p(m) = (1 / [m * sigma * sqrt(2*pi)]) * exp( -0.5 * [ln(m/mu)/sigma]^2 )
    """
    return (
        1.0 / (mass * sigma * np.sqrt(2.0 * np.pi)) 
        * np.exp(-0.5 * ( (np.log(mass / mu) / sigma) ** 2 ))
    )

def critical_collapse_pdf(mass, Mp):
    """
    Critical collapse mass function:
    p(m) = (1 / [nu * Mp * Gamma(1+nu)]) * (m/Mp)^(1/nu) * exp( -(m/Mp)^(1/nu) )
    """
    coeff = 1.0 / (nu * Mp * gamma(1 + nu))
    exponent = (mass / Mp)**(1.0 / nu)
    return coeff * exponent * np.exp(-exponent)

# -------------------------
# SAMPLING THE DISTRIBUTION
# -------------------------
# We'll define the mass range (for building the PDF and CDF).
masses = np.logspace(11, 19, 10000)  # from 1e11 g to 1e19 g

# Select the PDF based on user choice
if mass_function_choice == 'lognormal':
    pdf_values = lognormal_pdf(masses, mu, sigma)
elif mass_function_choice == 'critical_collapse':
    pdf_values = critical_collapse_pdf(masses, Mp)
else:
    raise ValueError("Invalid mass_function_choice. Use 'lognormal' or 'critical_collapse'.")

# Normalize the PDF so it integrates to 1 across this discrete grid
pdf_normalized = pdf_values / np.sum(pdf_values)

# Compute the CDF via cumulative sum
cdf_values = np.cumsum(pdf_normalized)
cdf_values /= cdf_values[-1]  # normalize so last entry is exactly 1

# Inverse transform sampling:
#   1. Generate N random numbers in [0, 1].
#   2. Use np.interp(...) to map them to the masses array via the CDF.
random_values = np.random.rand(N)
sampled_masses = np.interp(random_values, cdf_values, masses)

# -------------------------
# HAWKING RADIATION MODEL
# -------------------------

def f(M):
    """
    Effective degrees of freedom factor from (Halzen, Zas, et al. style) or the referenced paper.
    This is your f(M) in grams (M is in grams).
    The table of beta_masses is used for exponent cutoffs.
    """
    beta_masses = {
        'mu': 4.53e14,  
        'u': 1.6e14,
        'd': 1.6e14,
        's': 9.6e13,
        'c': 2.56e13,
        'T': 2.68e13,
        'b': 9.07e12,
        't': 0.24e12,
        'g': 1.1e14,
        'w': 7.97e11,
        'z': 7.01e11,
        'h': 2.25e11
    }
    base = 1.569
    # Summation in piecewise form
    return (
        base
        + 0.569 * (
            np.exp(-M / beta_masses['mu'])
            + 3*np.exp(-M / beta_masses['u'])
            + 3*np.exp(-M / beta_masses['d'])
            + 3*np.exp(-M / beta_masses['s'])
            + 3*np.exp(-M / beta_masses['c'])
            + np.exp(-M / beta_masses['T'])
            + 3*np.exp(-M / beta_masses['b'])
            + 3*np.exp(-M / beta_masses['t'])
            + 0.963*np.exp(-M / beta_masses['g'])
        )
        + 0.36*np.exp(-M / beta_masses['w'])
        + 0.18*np.exp(-M / beta_masses['z'])
        + 0.267*np.exp(-M / beta_masses['h'])
    )

def Mdot(M):
    """
    The time derivative of black hole mass in grams/second:
    dM/dt = - (constant * f(M)) / (M^2)
    """
    return -5.34e25 * f(M) / (M*M)

# -------------------------
# ODE SOLVER: EVOLVE MASSES
# -------------------------
def evolve(masses, n_time_points=500):
    """
    Evolve an array of black hole masses over a geometric grid of times up to 'age_of_universe'.
    Returns:
      mass_history: shape (len(masses), n_time_points)
      times: shape (n_time_points,)
    """
    # Time points from 1 second to age_of_universe on a log scale
    times = np.geomspace(1, age_of_universe, n_time_points)
    mass_history = np.zeros((len(masses), n_time_points))
    
    for i, initial_mass in enumerate(masses):
        def dMdt(t, M):
            # Single-variable ODE function for solve_ivp
            return Mdot(M[0])
        
        def event_mass_threshold(t, M):
            # We'll stop the integration if mass < 1e9 g
            return M[0] - 1e9
        event_mass_threshold.terminal = True
        event_mass_threshold.direction = -1
        
        # Solve the ODE from times[0] to times[-1] with provided events
        solution = solve_ivp(
            dMdt,
            t_span=(times[0], times[-1]),
            y0=[initial_mass],
            t_eval=times,
            events=event_mass_threshold,
            method='RK45',
            rtol=1e-6,
            atol=1e-6
        )
        
        # Fill mass_history up to the length of solution.t
        mass_history[i, :len(solution.t)] = solution.y[0]
        
        # Once mass hits 1e9 or the solver stops, keep it at 1e9 for the remainder
        if len(solution.t) < n_time_points:
            mass_history[i, len(solution.t):] = 1e9
            
    return mass_history, times

# Evolve the masses
mass_history, times = evolve(sampled_masses)

# ----------------------------------
# MATPLOTLIB PLOTTING WITH SLIDER
# ----------------------------------
# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.25)  # Make room for slider at bottom

# Calculate initial distribution to set y-axis limits
initial_masses = mass_history[:, 0]
hist_initial, bins_initial = np.histogram(
    np.log10(initial_masses[initial_masses > 1e9]), 
    bins=50
)
y_max = np.max(hist_initial) * 1.1  # Add 10% margin

# Start with the earliest time index
current_index = 0
masses_at_time = mass_history[:, current_index]

# Plot initial distribution in red
ax.hist(
    np.log10(initial_masses[initial_masses > 1e9]),
    bins=50, color="red", alpha=0.3, edgecolor="darkred",
    label="Initial Distribution"
)

# Plot current distribution in blue
ax.hist(
    np.log10(masses_at_time[masses_at_time > 1e9]),
    bins=50, color="skyblue", alpha=0.7, edgecolor="black",
    label="Current Distribution"
)

ax.set_xlabel('log10(mass) [g]')
ax.set_ylabel('Count')
ax.set_xlim(11, 19)
ax.set_ylim(0, y_max)
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title(f'Mass Distribution (Time: {times[current_index]:.2e} seconds)')

# Create an axis for the slider
slider_ax = plt.axes([0.15, 0.1, 0.7, 0.03])  # [left, bottom, width, height]

# Build the slider for time index
time_slider = Slider(
    ax=slider_ax,
    label='Time',
    valmin=0,
    valmax=len(times) - 1,
    valinit=current_index,
    valstep=1  # step through discrete time indices
)

def update(val):
    # Convert slider value to integer index
    idx = int(val)
    
    # Clear the axes for fresh plot
    ax.clear()
    
    # Always plot the initial distribution in red
    ax.hist(
        np.log10(initial_masses[initial_masses > 1e9]),
        bins=50, color="red", alpha=0.3, edgecolor="darkred",
        label="Initial Distribution"
    )
    
    # Plot the distribution at the new time index in blue
    masses_at_time = mass_history[:, idx]
    ax.hist(
        np.log10(masses_at_time[masses_at_time > 1e9]),
        bins=50, color="skyblue", alpha=0.7, edgecolor="black",
        label="Current Distribution"
    )
    
    # Restore the same axes limits and styling
    ax.set_xlabel('log10(mass) [g]')
    ax.set_ylabel('Count')
    ax.set_xlim(11, 19)
    ax.set_ylim(0, y_max)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f'Mass Distribution (Time: {times[idx]:.2e} seconds)')
    
    # Redraw the figure
    fig.canvas.draw_idle()

# Register the slider update function
time_slider.on_changed(update)

plt.show()