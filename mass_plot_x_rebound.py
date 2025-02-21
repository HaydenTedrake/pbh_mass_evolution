import rebound
import numpy as np
import matplotlib.pyplot as plt

def Mdot(M):
    """ Compute dM/dt for mass evolution """
    return -5.34e25 / (M * M)  # Simplified version

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
    sim.integrator = "IAS15"  # High-accuracy adaptive integrator

    # Add a dummy particle (PBH mass tracking)
    sim.add(m=M0, x=0, y=0, z=0, vx=0, vy=0, vz=0)

    times = []
    masses = []

    def mass_evolution(sim_pointer):
        """ Custom function to evolve mass over time """
        sim = sim_pointer.contents  # Access simulation contents
        p = sim.particles[0]  # Correct way to access particles
        M = p.m
        dM = Mdot(M) * sim.dt
        p.m = max(M + dM, target_mass)  # Ensure mass doesn't go negative

        # Debugging info
        print(f"Time: {sim.t:.2e}, Mass: {p.m:.2e}, dM: {dM:.2e}")

        # Store values
        times.append(sim.t)
        masses.append(p.m)

        # Stop when target mass is reached
        if p.m <= target_mass:
            sim.exit_condition = 1  # Stops integration when mass threshold is hit

    sim.additional_forces = mass_evolution

    # Adjust timestep and run for longer
    sim.dt = 1e9  # Reduce dt to prevent instability
    sim.integrate(1e18)  # Run until explosion

    return np.array(times), np.array(masses)

# Parameters
M0 = 4e16  # Initial PBH mass in grams
target_mass = 1e9  # Mass when PBH explodes

# Integrate
times, masses = integrate_pbh_mass(M0, target_mass)

# Plot
plt.figure(figsize=(10, 6))
plt.semilogy(times, masses, label="PBH Mass Evolution")
plt.xlabel("Time (s)")
plt.ylabel("Mass (g)")
plt.title("Primordial Black Hole Mass Evolution using REBOUND")
plt.legend()
plt.grid()
plt.show()
