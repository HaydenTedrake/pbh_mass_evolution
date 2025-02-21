import rebound
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm  # Import progress bar

# Constants
age_of_universe = 4.35e17  # Universe age in seconds
c = 2.998e8  # Speed of light in m/s

def f(M):
    """
    Compute f(M) from MacGibbon's equation.
    """
    beta_masses = {
        'mu': 4.53e14, 'u': 1.6e14, 'd': 1.6e14, 's': 9.6e13,
        'c': 2.56e13, 'T': 2.68e13, 'b': 9.07e12, 't': 0.24e12,
        'g': 1.1e14, 'w': 7.97e11, 'z': 7.01e11, 'h': 2.25e11
    }
    base = 1.569
    result = (
        base + 0.569 * (
            np.exp(-M / beta_masses['mu'])
            + 3 * np.exp(-M / beta_masses['u'])
            + 3 * np.exp(-M / beta_masses['d'])
            + 3 * np.exp(-M / beta_masses['s'])
            + 3 * np.exp(-M / beta_masses['c'])
            + np.exp(-M / beta_masses['T'])
            + 3 * np.exp(-M / beta_masses['b'])
            + 3 * np.exp(-M / beta_masses['t'])
            + 0.963 * np.exp(-M / beta_masses['g'])
        )  
        + 0.36 * np.exp(-M / beta_masses['w'])
        + 0.18 * np.exp(-M / beta_masses['z'])
        + 0.267 * np.exp(-M / beta_masses['h'])
    )
    return result

def Mdot(M):
    """ Compute dM/dt using f(M). """
    return -5.34e25 * f(M) / (M * M)

def integrate_pbh_mass(M0, target_mass=1e9):
    """
    Uses REBOUND to integrate PBH mass evolution with a progress bar.
    
    Args:
        M0 (float): Initial mass in grams
        target_mass (float): Target mass at explosion
        
    Returns:
        times (numpy array): Time values
        masses (numpy array): Corresponding mass values
        explosion_time (float): Time when PBH reaches target mass
    """
    sim = rebound.Simulation()
    sim.integrator = "IAS15"  # High-accuracy adaptive timestep
    sim.add(m=M0, x=0, y=0, z=0, vx=0, vy=0, vz=0)

    times, masses = [], []
    total_steps = int(1e6)  # Estimated number of integration steps
    progress = tqdm(total=total_steps, desc="Integrating PBH Mass", unit="step")

    def mass_evolution(sim_pointer):
        """ Custom function to evolve mass over time """
        sim = sim_pointer.contents
        p = sim.particles[0]  # Access the PBH
        M = p.m
        dM = Mdot(M) * sim.dt

        dM = max(dM, -0.001 * M)  # Limit how much mass can change per step
        p.m = max(M + dM, target_mass)  # Ensure mass doesn't go negative

        times.append(sim.t)
        masses.append(p.m)
        
        progress.update(1)  # Update progress bar

        if p.m <= target_mass:
            sim.exit_condition = 1  # Stops integration
            progress.close()  # Close the progress bar

    sim.additional_forces = mass_evolution
    sim.dt = 1e-1000  
    sim.integrate(1e18)  

    explosion_time = times[-1]
    progress.close()  # Close the progress bar
    print(f"Explosion time: {explosion_time:.2e} s")
    
    return np.array(times), np.array(masses), explosion_time

def PBHDemo(explosion_x, M0, x, target_mass=1e9):
    """
    Simulates PBH mass evolution with REBOUND and plots results.
    
    Args:
        explosion_x (float): Explosion position in km
        M0 (float): Initial mass in grams
        x (float): Observation position in km
        target_mass (float): Mass when PBH explodes
    """
    displacement = np.abs(x - explosion_x)  # in km
    boundary_time = displacement / 220  # (km/s)
    
    times_numerical, masses_numerical, explosion_time = integrate_pbh_mass(M0, target_mass)

    def exp_time_points(T, num_points=1000000):
        exp_space = np.exp(np.linspace(0, 5, num_points))
        return T - T * (exp_space - exp_space[0]) / (exp_space[-1] - exp_space[0])
    
    t_analytical = exp_time_points(explosion_time)

    def MassAnalytical(M0, t):
        Mass_cubed = (-16.02e25 * 1.0 * t + np.power(M0, 3))
        return np.cbrt(np.maximum(Mass_cubed, 0))

    M_analytical = MassAnalytical(M0, t_analytical)

    boundary_time_idx = np.abs(times_numerical - (-boundary_time)).argmin()
    mass_at_boundary = masses_numerical[boundary_time_idx]

    plt.figure(figsize=(12, 8))
    plt.plot(t_analytical - explosion_time, M_analytical, 'b-', label='Analytical Solution', alpha=0.8)
    plt.semilogy(times_numerical - explosion_time, masses_numerical, 'r--', label='Numerical Solution', linewidth=2)
    plt.scatter(-boundary_time, mass_at_boundary, color='green', label=f"M at x ≈ {mass_at_boundary:.2e} g", zorder=5)
    
    plt.xlabel("Time relative to explosion time (s)")
    plt.ylabel("PBH Mass (g)")
    plt.title(f"PBH Mass Evolution (M₀ = {M0:.2e} g)")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()

    plt.text(0.02, 0.98, f"Explosion Time: {explosion_time:.2e} s", transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.show()
    
    return times_numerical, masses_numerical, mass_at_boundary

# Parameters
explosion_x = 0
M0 = 4e16
x = 22000000
target_mass = 1e9

times, masses, M_at_x = PBHDemo(explosion_x, M0, x, target_mass)
print(f"If a PBH explodes at x=0, a PBH with M0 {M0} g will have a mass of {M_at_x} g at {x} meters")
