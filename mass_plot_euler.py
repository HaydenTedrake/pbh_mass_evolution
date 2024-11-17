import numpy as np
import matplotlib.pyplot as plt
import math

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


constant_fM=1


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
    out = -5.34e25*f(M)/M**2
    return out


def solve_Mdot(M0, boundary_time, dt=0.1):
    """
    Numerically solve the mass evolution equation backwards in time.

    Args:
        explosion_mass (float): The mass at the endpoint (e.g., 1e9 grams).
        M0 (float): Initial mass at t = 0 in grams.
        boundary_time (float): Total integration time in seconds.
        dt (float): Time step for integration.

    Returns:
        times (list): Time steps (negative, integrating backwards).
        masses (list): Mass values corresponding to each time step.
    """
    # calculate the constant C using M0 at t=0
    C = (1e26 * f(M0) / M0**2)

    # initialize variables
    t = 0
    M = M0
    times = [t]
    masses = [M]

    # integrate backwards
    while t > -boundary_time:
        dM = -(-5.34e25 * f(M) / M**2) * dt
        M += dM
        t += dt
        masses.append(max(M, 0))  # ensure mass does not go negative
        times.append(abs(t))
    
    return times[::-1], masses[::-1]


def MassAnalytical(M0, t):
    """Compute Mass as a function of time."""

    Mass_cubed = (-16.02e25 * constant_fM * t + np.power(M0, 3))
    Mass = np.cbrt(Mass_cubed)
    if Mass < 0: 
        Mass = 0
    return Mass


def PBHDemoAnalytical(explosion_x, M0, x, dt=1000):
    displacement = x-explosion_x # in km
    boundary_time = displacement / 220 #(km/s), boundary_time in seconds
    explosion_time = (np.power(M0, 3) - 1e27) / (16.02e25 * constant_fM)

    t = [i * dt for i in range(int(explosion_time/dt*1.25))]
    M = [MassAnalytical(M0=M0, t=ti) for ti in t]

    
    # Plot the results
    plt.plot(t, M)
    plt.xlabel("Time (s)")
    plt.ylabel("PBH Mass (g)")
    plt.title("PBH Mass vs. Time")

    plt.show()


def PBHDemoNumerical(explosion_x, M0, x):
    """
    Demo for numerical solution of PBH mass evolution integrating backwards.

    Args:
        explosion_x (float): Location of PBH explosion in km.
        M0 (float): Initial PBH mass at t=0 in grams.
        x (float): Target location in km.
        dt (float): Time step for integration.

    Returns:
        mass_value (float): Mass value at the starting time (boundary_time).
    """
    displacement = x-explosion_x # in km
    boundary_time = displacement / 220 #(km/s), boundary_time in seconds

    # solve the mass evolution numerically
    times, masses = solve_Mdot(M0=M0, boundary_time=boundary_time)
    
    # Plot the results
    plt.plot(times, masses)
    plt.xlabel("Time (s)")
    plt.ylabel("PBH Mass (g)")
    plt.title("PBH Mass vs. Time")
    
    plt.show()


PBHDemoAnalytical(explosion_x=0, M0=1e11, x=2200)
