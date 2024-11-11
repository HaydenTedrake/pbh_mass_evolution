import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import (
    DiagramBuilder,
    LeafSystem,
    LogVectorOutput,
    ResetIntegratorFromFlags,
    Simulator,
)
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
    # for now set f(M) to be 10
    out = 10
    return out


def MdotAnalytical(M, M0):
    """Compute Mdot as a function of M."""
    Mdot_cubed = (-3e26 * f(M) * np.power(M0, 3))
    Mdot = np.cbrt(Mdot_cubed)
    if M < 0: 
        Mdot = 0
    return -Mdot

# this is not being used anymore since we are doing euler by hand
# class PBHMass(LeafSystem):
#     """Boilerplate to define the simple Drake system."""

#     def __init__(self, M0):
#         LeafSystem.__init__(self)
#         self.M0 = M0
#         state_index = self.DeclareContinuousState(1)
#         self.DeclareStateOutputPort("M", state_index)

#     def DoCalcTimeDerivatives(self, context, derivatives):
#         M = context.get_continuous_state_vector().CopyToVector()
#         derivatives.get_mutable_vector().SetFromVector(MdotAnalytical(M, self.M0))


def solve_Mdot(MdotAnalytical, boundary_time, M0, dt):
    # initial conditions
    ti = 0
    Mi = 1e9

    t=[ti]
    M=[Mi]

    while t[-1] <= boundary_time:
        Mdot = MdotAnalytical(M[-1], M0)

        M1 = Mi + Mdot*dt
        t1 = ti + dt

        M.append(M1)
        t.append(t1)

        Mi=M1
        ti=t1

    return M, t


def PBHDemo(explosion_x, M0, x, dt=0.1):
    displacement = x-explosion_x # in km
    boundary_time = displacement / 220 #(km/s), boundary_time in seconds
    
    M, t = solve_Mdot(MdotAnalytical, boundary_time, M0, dt)

    mass_value = M[-1]
    
    # Plot the results
    plt.plot(t, M)
    plt.xlabel("Time (s)")
    plt.ylabel("PBH Mass (g)")
    plt.title("PBH Mass vs. Time")

    # Mark the mass value at x
    plt.plot(boundary_time, mass_value, 'ro')  # Red dot
    plt.text(boundary_time, mass_value, f'({boundary_time:.2f}, {mass_value:.2e})', fontsize=12, ha='right')

    plt.show()


PBHDemo(explosion_x=0, M0=1e23, x=2200)
