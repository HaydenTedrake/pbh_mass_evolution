import rebound
import numpy as np
import matplotlib.pyplot as plt

# Constants
age_of_universe = 4.35e17  # in seconds

def f(M):
    """
    Calculate f(M) with M in grams
    """
    beta_masses = {
        'mu': 4.53e14,     # muon
        'u': 1.6e14,       # up quark
        'd': 1.6e14,       # down quark
        's': 9.6e13,        # strange quark
        'c': 2.56e13,       # charm quark
        'T': 2.68e13,       # tau
        'b': 9.07e12,       # bottom quark
        't': 0.24e12,        # top quark (unobserved at time of paper)
        'g': 1.1e14,         # gluon (effective mass)
        'w': 7.97e11,         # W boson
        'z': 7.01e11,         # Z boson
        'h': 2.25e11          # Higgs boson
    }

    base = 1.569
    result = (
        base
        + 0.569 * (
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
    """ Compute dM/dt """
    return -5.34e25 * f(M) / (M * M)

def integrate_pbh_mass(M0, target_mass=1e9):
    """
    Uses REBOUND to integrate the mass evolution of a PBH.
    
    Args:
        M0 (float): Initial mass in grams
        target_mass (float): Target mass for explosion
    
    Returns:
        times, masses (arrays)
    """
    sim = rebound.Simulation()
    sim.integrator = "IAS15"  # Adaptive, high-accuracy integrator
    
    # Use a dummy particle to hold mass information
    sim.add(m=M0, x=0, y=0, z=0, vx=0, vy=0, vz=0)

    times = []
    masses = []

    def mass_evolution(sim):
        """ Custom force function to evolve mass """
        p = sim.particles[0]  # The PBH particle
        M = p.m
        dM = Mdot(M) * sim.dt
        p.m = max(M + dM, target_mass)  # Avoid negative mass

        # Store data
        times.append(sim.t)
        masses.append(p.m)

        # Stop when target mass is reached
        if p.m <= target_mass:
            sim.exit_condition = 1

    sim.additional_forces = mass_evolution

    # Run the simulation for a long time (adaptive steps)
    sim.dt = 1e13  # Set initial time step (adaptive)
    sim.integrate(1e18)  # Run until mass reaches target

    return np.array(times), np.array(masses)

# Parameters
M0 = 4e16  # Initial PBH mass in grams
target_mass = 1e9  # Mass when PBH explodes

# Integrate
times, masses = integrate_pbh_mass(M0, target_mass)

# Plot the results
plt.figure(figsize=(10, 6))
plt.semilogy(times, masses, label="PBH Mass Evolution")
plt.xlabel("Time (s)")
plt.ylabel("Mass (g)")
plt.title("Primordial Black Hole Mass Evolution using REBOUND")
plt.legend()
plt.grid()
plt.show()
