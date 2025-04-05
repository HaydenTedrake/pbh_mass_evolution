import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp
from scipy.special import gamma

hbar = 1.0545718e-27  # erg·s
c = 2.99792458e10     # cm/s
G = 6.67430e-8        # cm³/g·s²

def gev_to_grams(energy_gev):
    energy_ergs = energy_gev * 1.60218e-3
    mass_grams = energy_ergs / (c ** 2)
    return mass_grams

# -------------------------
# CONFIGURATION & PARAMETERS
# -------------------------
mass_function_choice = 'critical_collapse'  # options: 'lognormal' or 'critical_collapse'
age_of_universe = 4.35e17  # in seconds (4.35e17 ~ 13.8 billion years in seconds)
N = 100  # number of black holes to sample

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
masses = np.logspace(-5, 20, 10000)  # from 10e-5 g to 10e19 g

# # Select the PDF based on user choice
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

m_low = 10e-5
m_high = 10e19

cdf_low = np.interp(m_low, masses, cdf_values)
cdf_high = np.interp(m_high, masses, cdf_values)

# Compute probability in range
# *******remember to account for this probability later on
probability = cdf_high - cdf_low

# Inverse transform sampling:
#   1. Generate N random numbers in [0, 1].
#   2. Use np.interp(...) to map them to the masses array via the CDF.
random_values = np.random.rand(N)
sampled_masses = np.clip(np.interp(random_values, cdf_values, masses), m_low, m_high)

# -------------------------
# HAWKING RADIATION MODEL
# -------------------------

def f(M):
    """Calculate f(M) with M in grams."""
    
    masses_gev = {'mu': 0.10566, 'u': 0.34, 'd': 0.34, 's': 0.96, 'c': 1.28, 'T': 1.7768,
                  'b': 4.18, 't': 173.1, 'g': 0.650, 'w': 80.433, 'z': 91.19, 'h': 124.07}
    spins = {'mu': '1/2', 'u': '1/2', 'd': '1/2', 's': '1/2', 'c': '1/2', 'T': '1/2',
             'b': '1/2', 't': '1/2', 'g': '1', 'w': '1', 'z': '1', 'h': '0'}
    beta_coef = {'0': 2.66, '1/2': 4.53, '1': 6.04, '2': 9.56}
    
    masses = {p: gev_to_grams(m) for p, m in masses_gev.items()}
    beta_masses = lambda p: (hbar * c) / (8 * math.pi * G * masses[p]) * beta_coef[spins[p]]

    exp_terms = lambda *particles: sum(np.exp(-M / beta_masses(p)) for p in particles)

    return (1.569 + 0.569 * (exp_terms('mu') + 3 * exp_terms('u', 'd', 's', 'c', 'b', 't') + 
                             exp_terms('T') + 0.963 * exp_terms('g')) +
            0.36 * exp_terms('w') + 0.18 * exp_terms('z') + 0.267 * exp_terms('h'))

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
plt.subplots_adjust(bottom=0.25)

initial_masses = mass_history[:, 0]
hist_initial, bins_initial = np.histogram(
    np.log10(initial_masses[initial_masses > 1e9]), 
    bins=50
)
y_max = np.max(hist_initial) * 1.1

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
ax.set_title(f'Mass Distribution (Time: {times[current_index]:.2e} s)')

slider_ax = plt.axes([0.15, 0.1, 0.7, 0.03])
time_slider = Slider(
    ax=slider_ax,
    label='Time',
    valmin=0,
    valmax=len(times) - 1,
    valinit=current_index,
    valstep=1
)

def update(val):
    idx = int(val)
    ax.clear()
    
    ax.hist(
        np.log10(initial_masses[initial_masses > 1e9]),
        bins=50, color="red", alpha=0.3, edgecolor="darkred",
        label="Initial Distribution"
    )
    
    masses_at_time = mass_history[:, idx]
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
    ax.set_title(f'Mass Distribution (Time: {times[idx]:.2e} s)')
    fig.canvas.draw_idle()

time_slider.on_changed(update)
plt.show()

# ------------------------------------------------------
# AFTER EVOLUTION: CONSTRUCT & NORMALIZE THE FINAL MF
# ------------------------------------------------------

# Grab final masses
final_masses = mass_history[:, -1]
final_masses = final_masses[final_masses > 1e9]

# Histogram in log10-space
num_bins = 50
log_bins = np.linspace(9, 19, num_bins + 1)  # from 1e9 g to 1e19 g
hist_vals, _ = np.histogram(np.log10(final_masses), bins=log_bins)

# Normalize so that the total "area" under the histogram in log-space is 1
hist_vals = hist_vals.astype(float)
massfunc_log_norm = hist_vals / hist_vals.sum()  # fraction per log-bin

# Bin centers in log-space, then convert to linear mass
log_centers = 0.5 * (log_bins[:-1] + log_bins[1:])
mass_centers = 10**log_centers

# dn/dM = (1/M) * (normalized distribution in log M)
dn_dM = massfunc_log_norm / mass_centers

# Compute the widths in linear mass space for each log-bin
mass_edges = 10**log_bins
bin_widths = np.diff(mass_edges)

# Build a discrete cumulative distribution: n(M) = ∫ (dn/dM) dM, from the lowest to M
cdf = np.cumsum(dn_dM * bin_widths)

def number_density_up_to(M):
    """
    Returns the cumulative integral of dn/dM from the smallest bin
    up to mass = M. This effectively is the fraction of black holes
    with mass < M in this final snapshot, if you interpret the total = 1.
    """
    # If M is below our lowest bin, the integral is zero
    if M < mass_edges[0]:
        return 0.0
    # If above our highest bin, return full integral
    if M >= mass_edges[-1]:
        return cdf[-1]
    
    # Otherwise, find which bin M falls into
    idx = np.searchsorted(mass_edges, M) - 1
    # fraction of that bin to integrate
    partial_fraction = (M - mass_edges[idx]) / bin_widths[idx]
    partial_bin_area = dn_dM[idx] * (M - mass_edges[idx])
    
    # sum up everything below idx + partial of the idx-th bin
    return (cdf[idx-1] + partial_bin_area) if idx > 0 else partial_bin_area

# Example usage: get fraction up t o 1e14 g
example_mass = 1.0e14
fraction_below_1e14 = number_density_up_to(example_mass)
print(f"Fraction of final BHs with M < {example_mass:.1g} g = {fraction_below_1e14:.4f}")

# Similarly, you can get fraction up to 1e18 g, or any other mass
example_mass2 = 1.0e18
fraction_below_1e18 = number_density_up_to(example_mass2)
print(f"Fraction of final BHs with M < {example_mass2:.1g} g = {fraction_below_1e18:.4f}")