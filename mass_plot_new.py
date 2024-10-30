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
    return out

def dynamics(M):
    """Compute Mdot as a function of M."""
    Mdot = -1e26 * f(M) * np.power(M, -2)  # Carr's function
    Mdot[M < 0] = 0
    return -Mdot

class PBHMass(LeafSystem):
    """Boilerplate to define the simple Drake system."""

    def __init__(self):
        LeafSystem.__init__(self)
        state_index = self.DeclareContinuousState(1)
        self.DeclareStateOutputPort("M", state_index)

    def DoCalcTimeDerivatives(self, context, derivatives):
        M = context.get_continuous_state_vector().CopyToVector()
        derivatives.get_mutable_vector().SetFromVector(dynamics(M))


def PBHDemo():
    builder = DiagramBuilder()
    sys = builder.AddSystem(PBHMass())
    logger = LogVectorOutput(sys.get_output_port(), builder)
    diagram = builder.Build()

    simulator = Simulator(diagram)
    # Choose the numerical integration scheme:
    #   https://drake.mit.edu/doxygen_cxx/group__integrators.html
    # Runge-Kutta 3 is a basic error-controlled integrator.
    ResetIntegratorFromFlags(simulator, scheme="explicit_euler", max_step_size=0.1)
    # ResetIntegratorFromFlags(simulator, scheme="runge_kutta3", max_step_size=0.1)
    context = simulator.get_mutable_context()

    M0 = 10e9  # initial conditions
    context.SetContinuousState([M0])

    simulator.AdvanceTo(boundary_time=10)

    log = logger.FindLog(context)
    plt.plot(log.sample_times(), log.data().T)

PBHDemo()
