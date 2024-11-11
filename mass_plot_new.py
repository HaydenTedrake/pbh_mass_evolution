import numpy as np
import matplotlib.pyplot as plt
# from pydrake.all import (
#     DiagramBuilder,
#     LeafSystem,
#     LogVectorOutput,
#     ResetIntegratorFromFlags,
#     Simulator,
# )
import math
from scipy.integrate import quad


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


def constant_f(M):
    out = 10
    return out


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
    '''Compute the mass as a function of time'''
    out = -1e26*f(M)/M**2
    return -out


def solve_Mdot_with_constant(M0, boundary_time, dt=0.1):
    """
    Numerically solve the mass evolution equation with a constant to enforce M(t=0) = M0.

    Args:
        M0 (float): Initial PBH mass in grams.
        boundary_time (float): Total integration time in seconds.
        dt (float): Time step for integration.

    Returns:
        times (list): Time steps.
        masses (list): Mass values corresponding to each time step.
    """
    # Initial conditions
    t = 0
    M = M0
    
    # Adjust the constant to satisfy M(t=0) = M0
    C = 0  # Start with zero, iteratively adjust
    
    # Iterative approach to find C
    tolerance = 1e-6
    while True:
        # Solve one step forward to check M(t=0)
        M_test = M + (Mdot(M) + C) * dt
        if abs(M_test - M0) < tolerance:
            break
        C += (M0 - M_test) / dt  # Adjust C incrementally

    # Now solve numerically with the correct C
    times = [t]
    masses = [M]

    while t <= boundary_time and M > 0:
        dM = (Mdot(M) + C) * dt
        M += dM
        t += dt
        masses.append(max(M, 0))  # Ensure mass doesn't go negative
        times.append(t)
    
    return times, masses


def MassAnalytical(M, M0, t):
    """Compute Mass as a function of time."""

    Mass_cubed = (-3e26 * constant_f(M) * t + np.power(M0, 3))

    Mass = np.cbrt(Mass_cubed)
    if Mass < 0: 
        Mass = 0
    return Mass


def PBHDemoAnalytical(explosion_x, M0, x, dt=0.1):
    M = 1
    displacement = x-explosion_x # in km
    boundary_time = displacement / 220 #(km/s), boundary_time in seconds
    explosion_time = (np.power(M0, 3) - 1e27) / (3e26 * constant_f(M))

    mass_value = MassAnalytical(M, M0=M0, t=explosion_time-boundary_time)

    increments = int(boundary_time / dt)

    t = [explosion_time-boundary_time + i * dt for i in range(increments)]
    M = [MassAnalytical(M, M0=M0, t=ti) for ti in t]
    
    # Plot the results
    plt.plot(t, M)
    plt.xlabel("Time (s)")
    plt.ylabel("PBH Mass (g)")
    plt.title("PBH Mass vs. Time")

    # Mark the mass value at x
    plt.plot(explosion_time-boundary_time, mass_value, 'ro')  # Red dot
    plt.text(explosion_time-boundary_time, mass_value, f'({boundary_time:.2f}, {mass_value:.2e})', fontsize=12, ha='right')

    plt.show()
    return mass_value


def PBHDemoNumerical(explosion_x, M0, x, dt=0.1):
    """
    Demo for numerical solution of PBH mass evolution with constant adjustment.
    
    Args:
        explosion_x (float): Location of PBH explosion in km.
        M0 (float): Initial PBH mass in grams.
        x (float): Target location in km.
        dt (float): Time step for integration.
    
    Returns:
        mass_value (float): Final mass value at the target location.
    """
    displacement = x-explosion_x # in km
    boundary_time = displacement / 220 #(km/s), boundary_time in seconds

    # Solve the mass evolution numerically
    times, masses = solve_Mdot_with_constant(M0, boundary_time, dt)
    
    # Final mass value
    mass_value = masses[-1]
    
    # Plot the results
    plt.plot(times, masses)
    plt.xlabel("Time (s)")
    plt.ylabel("PBH Mass (g)")
    plt.title("PBH Mass Evolution with Constant Adjustment")
    
    # Mark the final mass
    plt.plot(boundary_time, mass_value, 'ro')  # Red dot
    plt.text(boundary_time, mass_value, f'({boundary_time:.2f}, {mass_value:.2e})', fontsize=12, ha='right')
    
    plt.show()
    return mass_value


# mass_value = PBHDemoAnalytical(explosion_x=0, M0=1e20, x=1100)
# print(f"Final mass value: {mass_value:.2e}")

mass_value = PBHDemoNumerical(explosion_x=0, M0=1e20, x=1100)
print(f"Final mass value: {mass_value:.2e}")
