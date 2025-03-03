import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp

age_of_universe = 4.35e17  # in seconds
hbar = 1.0545718e-27  # erg·s
c = 2.99792458e10     # cm/s
G = 6.67430e-8        # cm³/g·s²

def gev_to_grams(energy_gev):
    energy_ergs = energy_gev * 1.60218e-3
    mass_grams = energy_ergs / (c ** 2)
    return mass_grams

def joules_to_gev(joules):
    # Constants
    eV_per_joule = 6.242e18  # 1 Joule = 6.242 x 10^18 eV
    gev_per_ev = 1e-9        # 1 GeV = 10^9 eV
    
    # Convert Joules to GeV
    gev = joules * eV_per_joule * gev_per_ev
    
    return gev

def grams_to_gev(grams):
    # Constants
    c = 2.998e8           # Speed of light in m/s
    eV_per_joule = 6.242e18  # 1 Joule = 6.242 x 10^18 eV
    gev_per_ev = 1e-9        # 1 GeV = 10^9 eV
    
    # Convert mass to energy using E = mc²
    energy_joules = grams * (c ** 2)
    
    # Convert energy from Joules to GeV using our previous function
    energy_gev = energy_joules * eV_per_joule * gev_per_ev
    
    return energy_gev

def kelvin_to_gev(temperature_kelvin):
    # Constants
    k_b = 1.380649e-23    # Boltzmann constant in J/K
    eV_per_joule = 6.242e18  # 1 Joule = 6.242 x 10^18 eV
    gev_per_ev = 1e-9        # 1 GeV = 10^9 eV
    
    # Convert temperature to energy using E = k_b * T
    energy_joules = k_b * temperature_kelvin
    
    # Convert to GeV using our previous conversion factors
    energy_gev = energy_joules * eV_per_joule * gev_per_ev
    
    return energy_gev

def bh_temperature_in_GeV(mass_g):
    # Constants
    hbar_eV_s = 6.582e-16  # Reduced Planck constant in eV·s
    c = 3.0e8  # Speed of light in m/s
    G = 6.67e-11  # Gravitational constant in m^3·kg^-1·s^-2
    g_to_kg = 1e-3  # Conversion factor from grams to kilograms
    
    # Convert mass from grams to kilograms
    mass_kg = mass_g * g_to_kg
    
    # MacGibbon's equation (temperature in eV)
    temperature_eV = (hbar_eV_s * c**3) / (8 * math.pi * G * mass_kg)
    
    # Convert eV to GeV (1 GeV = 10^9 eV)
    temperature_GeV = temperature_eV / 1e9
    
    return temperature_GeV

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
    return -5.34e25 * f(M) / (M * M)

def solve_Mdot(M0, target_mass=1e9):
    """
    Solve the mass evolution equation using scipy's solve_ivp with adaptive step size.
    
    Args:
        M0 (float): Initial mass at t = 0 in grams
        explosion_time (float): Total integration time in seconds
        dt (float, optional): Maximum time step for integration
    
    Returns:
        times (ndarray): Time steps
        masses (ndarray): Mass values corresponding to each time step
    """
    def dMdt(t, M):
        return Mdot(M[0])
    
    def event_mass_threshold(t, M):
        return M[0] - target_mass
    
    event_mass_threshold.terminal = True  # Stop integration when event occurs
    event_mass_threshold.direction = -1   # Only trigger when crossing from above

    explosion_time = np.inf

    # Use solve_ivp with adaptive step size
    solution = solve_ivp(
        dMdt,
        t_span=(0, explosion_time),
        y0=[M0],
        method='RK45',
        rtol=1e-5,
        atol=1e-5,
        events=event_mass_threshold,
        dense_output=True
    )

    explosion_time = solution.t[-1]
    explosion_mass = solution.y[0][-1]

    print(f"Explosion time: {explosion_time} s")
    print(f"Explosion mass: {explosion_mass} g")

    formation_time = explosion_time - age_of_universe
    if formation_time < solution.t[0]:
        print("Warning: Extrapolation beyond computed range.")
        M0_exploding_now = None  # Or use another method to estimate it
    else:
        M0_exploding_now = solution.sol(formation_time)[0]

    # M0_exploding_3moago = solution.sol(explosion_time-age_of_universe+7884e3)[0]

    print(f"Formation mass of a PBH exploding now: {M0_exploding_now} g")
    # print(f"Formation mass difference of a PBH exploding now and a PBH exploding 3 months ago: {M0_exploding_now - M0_exploding_3moago} g")
    
    return solution.t, solution.y[0], explosion_time

def PBHDemo(explosion_x, M0, x, target_mass=1e9):
    """
    Simulates and plots the mass evolution from the formation (initial) mass, M0,
    until the mass reaches the target mass. We treat the time when the mass reaches
    the target mass as the explosion time, and set that position as explosion_x. 
    Then, assuming a constant velocity and a linear trajectory, we back out what
    the mass was when the PBH passed through x.
    
    Args:
        explosion_x (float): Explosion position in km
        M0 (float): Initial mass in grams
        x (float): Position in km
        target_mass (float, optional): Target mass for explosion time calculation in grams
        dt (float, optional): Maximum time step for integration
    """
    # Calculate parameters
    displacement = np.abs(x - explosion_x)  # in km
    boundary_time = displacement / 220  # (km/s)
    
    # Solve using improved method
    times_numerical, masses_numerical, explosion_time = solve_Mdot(M0, target_mass)
    # print("finished solve_Mdot")

    rough_estimate = (M0**3 - target_mass**3) / (16.02e25 * 1.0)
    # print(f"Rough estimate: {rough_estimate} s")
    if explosion_time is None:
        print("Could not determine explosion time. Using fallback calculation.")
        explosion_time = rough_estimate
    
    # Shift times by explosion time
    times_numerical_shifted = times_numerical - explosion_time

    def exp_time_points(T, num_points=1000000):
        exp_space = np.exp(np.linspace(0, 5, num_points))
        return T - T * (exp_space - exp_space[0]) / (exp_space[-1] - exp_space[0])
    
    # Analytical solution
    t_analytical = exp_time_points(rough_estimate)

    def MassAnalytical_vectorized(M0, t):
        Mass_cubed = (-16.02e25 * 1.0 * t + np.power(M0, 3))
        Mass = np.cbrt(np.maximum(Mass_cubed, 0))  # Avoid negative masses
        return Mass

    M_analytical = MassAnalytical_vectorized(M0, t_analytical)
    
    # Find the index closest to -boundary_time
    boundary_time_idx = np.abs(times_numerical_shifted - (-boundary_time)).argmin()
    mass_at_negative_boundary_time = masses_numerical[boundary_time_idx]

    plot = True
    if plot:  # enable/disable plotting
        # Create the plot with logarithmic scales
        plt.figure(figsize=(12, 8))
        
        # Plot analytical solution in blue
        plt.plot(t_analytical - explosion_time, M_analytical, 'b-', label='Analytical Solution', alpha=0.8)

        # Plot numerical solution
        plt.semilogy(times_numerical_shifted, masses_numerical, 'r--', label='Numerical Solution', linewidth=2)
        
        # Highlight M(-boundary_time)
        plt.scatter(-boundary_time, mass_at_negative_boundary_time, color='green', label=f"M at target x ≈ {mass_at_negative_boundary_time:.2e} g", zorder=5)
        
        # Customize the plot
        plt.xlabel("Time relative to explosion time (s)")
        plt.ylabel("PBH Mass (g)")
        plt.title(f"PBH Mass Evolution (M₀ = {M0:.2e} g)")
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        
        # Add some key information as text
        # info_text = f"Explosion Time: {explosion_time:.2e} s"
        info_text = f"Explosion Time: {explosion_time:.2e} s"
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.show()
    
    return times_numerical_shifted, masses_numerical, mass_at_negative_boundary_time

# Parameters
explosion_x = 0
M0 = 4e16
x = 22000000
target_mass = 1e9

times_shifted, masses, M_at_negative_boundary = PBHDemo(explosion_x, M0, x, target_mass)
print(f"If a PBH explodes at x=0, a PBH with M0 {M0} g will have a mass of {M_at_negative_boundary} g at {x} meters")
