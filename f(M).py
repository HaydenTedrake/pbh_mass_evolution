import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import root_scalar

# Physical Constants
c = 2.997924580 * 10**8          # speed of light, m/s
mE = 9.109383702 * 10**-31       # electron mass, kg
NA = 6.022140760 * 10**23        # Avogadro's number, 1/mole
e = 1.602176634 * 10**-19        # electron charge, C
hbar = 1.054571818 * 10**-34     # reduced Planck constant, J s
alpha = 7.297352569 * 10**-3     # fine structure constant, dimensionless
e0 = 8.854187813 * 10**-12       # electric constant, A s/m V
Ryd = 13.60569312                # Rydberg energy, eV
mP = 1.672621924 * 10**-27       # proton mass, kg
pi = math.pi

mu = 0
sigma = 1


def f(M):
    """Approximate of Carr's f(m) function

    Args:
        M: current pbh mass

    Returns:
        f(M): number of particles that can be emitted
    """
    M_log = np.log10(M)
    out = np.where((M_log >= 14) & (M_log <= 17), np.power(M, -2.0/3.0) * np.power(10, 34.0/3.0), 
                   np.where(M_log < 14, 100, 1))
    return out


def Mdot(M):
    '''Compute the mass as a function of time
    '''
    out = -5.34e25*f(M)/M**2
    return out


def solve_Mdot(M0, explosion_time, dt=10):
    """
    Numerically solve the mass evolution equation backwards in time using RK4.

    Args:
        M0 (float): Initial mass at t = 0 in grams.
        explosion_time (float): Total integration time in seconds.
        dt (float): Time step for integration.

    Returns:
        times (list): Time steps (negative, integrating backwards).
        masses (list): Mass values corresponding to each time step.
    """
    # Initialize variables
    t = 0
    M = M0
    times = [t]
    masses = [M]

    # Integrate using RK4
    while t < explosion_time * 1.25:
        k1 = Mdot(M)
        k2 = Mdot(M + 0.5 * dt * k1)
        k3 = Mdot(M + 0.5 * dt * k2)
        k4 = Mdot(M + dt * k3)
        
        dM = (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        M += dM
        t += dt
        
        masses.append(max(M, 0))
        times.append(t)
    
    return times, masses

def find_explosion_time(M0, target_mass=1e9):
    """
    Find the time at which the PBH mass reaches the target mass.

    Args:
        M0 (float): Initial mass at t = 0 in grams.
        target_mass (float): Target mass in grams.

    Returns:
        explosion_time (float): Time at which the PBH mass reaches the target mass.
    """
    def mass_at_time(t):
        '''Helper function to compute mass at time t
        '''
        dt = t/1000
        current_mass = M0

        for _ in range(1000):
            k1 = Mdot(current_mass)
            k2 = Mdot(current_mass + 0.5 * dt * k1)
            k3 = Mdot(current_mass + 0.5 * dt * k2)
            k4 = Mdot(current_mass + dt * k3)

            current_mass += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            if current_mass < target_mass:
                return current_mass - target_mass
        
        return current_mass - target_mass
    
    rough_estimate = (M0**3 - target_mass**3) / (16.02e25 * f(M0))

    try:
        result = root_scalar(
            mass_at_time, 
            bracket=[rough_estimate*0.1, rough_estimate*2], 
            method='brentq'
        )
        return result.root
    except ValueError:
        print("Failed to find explosion time.")
        return None


def PBHDemo(explosion_x, M0, x, dt=1000):
    # Calculate common parameters
    displacement = x - explosion_x  # in km
    boundary_time = displacement / 220  # (km/s), boundary_time in seconds
    explosion_time = find_explosion_time(M0, target_mass=1e9)
    # if explosion_time is not None:
    #     print(f"Initial mass: {M0} g")
    #     print(f"Explosion time: {explosion_time} s")
    
    # Numerical solution
    times_numerical, masses_numerical = solve_Mdot(M0=M0, explosion_time=explosion_time)
    
    # Create the combined plot
    plt.figure(figsize=(10, 6))

    # Plot numerical solution in red
    plt.plot(times_numerical, masses_numerical, 'r--', label='Numerical Solution', alpha=0.8)
    
    # Customize the plot
    plt.xlabel("Time (s)")
    plt.ylabel("PBH Mass (g)")
    plt.title("PBH Mass Evolution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Display the plot
    plt.show()

PBHDemo(explosion_x=0, M0=1e11, x=2200)
