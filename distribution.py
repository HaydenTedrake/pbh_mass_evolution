import numpy as np
from pydrake.all import (
    LeafSystem,
    DiagramBuilder,
    PyPlotVisualizer,
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


def initial_conditions(generator, num_samples):
    """Generate samples for random initial conditions, M(0).

    Args:
        generator: A numpy random number generator.
        num_samples: The number of samples to generate.
    """
    return generator.lognormal(mean=mu, sigma=sigma, size=num_samples)


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
    """Compute Mdot as a function of M.

    Args:
        M: a numpy vector, with one element per sample.

    Returns:
        Mdot: the element-wise time derivative of M.
    """
    Mdot = -M
    return -Mdot


class PBHMass(LeafSystem):
    """Defines a Drake system with the vectorized dynamics.

    This is boilerplate that you probably don't need to modify (unless you start integrating systems with more variables per sample).
    """

    def __init__(self, num_samples):
        LeafSystem.__init__(self)
        state_index = self.DeclareContinuousState(num_samples)
        self.DeclareStateOutputPort("y", state_index)

    def DoCalcTimeDerivatives(self, context, derivatives):
        M = context.get_continuous_state_vector().CopyToVector()
        derivatives.get_mutable_vector().SetFromVector(dynamics(M))


class HistogramVisualizer(PyPlotVisualizer):
    """A simple visualizer that plots a histogram of the input vector.

    This is boilerplate that you probably don't need to modify."""

    def __init__(
        self, num_samples, bins, xlim, ylim, draw_time_step, figsize=(6, 4), show=True
    ):
        PyPlotVisualizer.__init__(self, draw_time_step, figsize=figsize, show=show)
        self.DeclareVectorInputPort(f"x", num_samples)
        self.num_samples = num_samples
        self.bins = bins
        self.data = [0] * num_samples
        self.scale = 10
        self.limits = xlim
        self.ax.set_xlim(xlim)
        self.ax.axis("auto")
        self.ax.set_ylim(ylim)
        self.patches = None

    def draw(self, context):
        if self.patches:
            [p.remove() for p in self.patches]
        self.data = self.EvalVectorInput(context, 0).value()
        count, bins, self.patches = self.ax.hist(
            self.data,
            bins=self.bins,
            range=self.limits,
            density=False,
            weights=[self.scale / self.num_samples] * self.num_samples,
            facecolor="b",
        )
        self.ax.set_title("t = " + str(context.get_time()))

def PBHDemo():
    num_samples = 1000  # The number of samples to simulate
    generator = np.random.default_rng(seed=42)

    builder = DiagramBuilder()
    sys = builder.AddSystem(PBHMass(num_samples))
    visualizer = builder.AddSystem(
        HistogramVisualizer(
            num_samples=num_samples,
            bins=100,
            xlim=[-2, 2],
            ylim=[-1, 2],
            draw_time_step=0.25,
            show=True,
        )
    )
    builder.Connect(sys.get_output_port(), visualizer.get_input_port())
    diagram = builder.Build()

    simulator = Simulator(diagram)
    # Choose the numerical integration scheme:
    #   https://drake.mit.edu/doxygen_cxx/group__integrators.html
    # Runge-Kutta 3 is a basic error-controlled integrator.
    # explicit_euler is a simple, fixed-step integrator.
    ResetIntegratorFromFlags(simulator, scheme="runge_kutta3", max_step_size=0.1)
    context = simulator.get_mutable_context()
    context.SetContinuousState(initial_conditions(generator, num_samples))

    movie_filename = None
    # movie_filename = "pbh_mass.html"  # uncomment this to save a movie to file. Note: It's also easy to save an mp4 with a small change to the code below.
    if movie_filename:
        print("simulating... ", end=" ")
        visualizer.start_recording()

    simulator.AdvanceTo(boundary_time=10)
    if movie_filename:
        print("done.\ngenerating animation...")
        ani = visualizer.get_recording_as_animation()
        with open(movie_filename, "w") as f:
            f.write(ani.to_jshtml())


if __name__ == "__main__":
    PBHDemo()
    input("Press Enter to close the window.")
