import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

# Physical Constants (using numpy for consistent array operations)
c = np.float64(2.997924580e8)          # speed of light, m/s
mE = np.float64(9.109383702e-31)       # electron mass, kg
NA = np.float64(6.022140760e23)        # Avogadro's number, 1/mole
e = np.float64(1.602176634e-19)        # electron charge, C
hbar = np.float64(1.054571818e-34)     # reduced Planck constant, J s
alpha = np.float64(7.297352569e-3)     # fine structure constant
e0 = np.float64(8.854187813e-12)       # electric constant, A s/m V
Ryd = np.float64(13.60569312)          # Rydberg energy, eV
mP = np.float64(1.672621924e-27)       # proton mass, kg

def f(M):
    """f(m) function"""    
    return np.float64(1)

def Mdot(M):
    """Mass evolution function"""
    return -5.34e25 * f(M) / (M * M)

def find_explosion_time(M0, target_mass=1e9, max_iterations=100, precision=1e-10):
    """
    Find the time at which the PBH mass reaches the target mass.

    Args:
        M0 (float): Initial mass at t = 0 in grams.
        target_mass (float): Target mass in grams.

    Returns:
        explosion_time (float): Time at which the PBH mass reaches the target mass.
    """
    def mass_at_time(t):
        """Helper function to compute mass at time t"""
        dt = t / 1000
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

    # Rough estimate for the explosion time
    rough_estimate = (M0**3 - target_mass**3) / (16.02e25 * f(M0))
    lower_bound = rough_estimate * 0.5
    upper_bound = rough_estimate * 2

    for _ in range(max_iterations):
        try:
            result = root_scalar(
                mass_at_time,
                bracket=[lower_bound, upper_bound],
                method='brentq',
                xtol=precision
            )
            final_mass = mass_at_time(result.root)
            
            if abs(final_mass) < precision * target_mass:
                return result.root
        except ValueError:
            lower_bound *= 0.5
            upper_bound *= 2.0

    print("Warning: Precise explosion time calculation did not converge.")
    return rough_estimate

def solve_Mdot(M0, explosion_time, target_mass=1e9, dt=None):
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

    # Set up solver parameters
    rtol = 1e-6
    atol = 1e-6
    
    # Use solve_ivp with adaptive step size
    solution = solve_ivp(
        dMdt,
        t_span=(0, explosion_time),
        y0=[M0],
        method='RK45',
        rtol=rtol,
        atol=atol,
        max_step=dt if dt is not None else np.inf,
        events=event_mass_threshold
    )
    
    return solution.t, solution.y[0]

def MassAnalytical(M0, t):
    """Compute Mass as a function of time."""

    Mass_cubed = (-16.02e25 * f(M0) * t + np.power(M0, 3))
    Mass = np.cbrt(Mass_cubed)
    if Mass < 0: 
        Mass = 0
    return Mass

def PBHDemoError(explosion_x, M0, x, target_mass=1e9, dt=100):
    '''Plot the error between the analytical and numerical solutions over time'''
    # Calculate explosion time
    explosion_time = find_explosion_time(M0, target_mass=target_mass)
    
    if explosion_time is None:
        print("Could not determine explosion time. Using fallback calculation.")
        explosion_time = (np.power(M0, 3) - np.power(target_mass, 3)) / (16.02e25 * f(M0))
    
    # Analytical solution
    t_analytical = np.arange(0, explosion_time, 10)
    M_analytical = np.array([MassAnalytical(M0=M0, t=ti) for ti in t_analytical])
    
    # Solve using numerical method
    times_numerical, masses_numerical = solve_Mdot(M0, explosion_time, target_mass, dt=dt)
    
    # Interpolate the numerical solution to align with analytical time points
    masses_numerical_interp = np.interp(t_analytical, times_numerical, masses_numerical)
    
    # Compute the absolute and relative errors
    abs_error = np.abs(M_analytical - masses_numerical_interp)
    rel_error = np.abs((M_analytical - masses_numerical_interp) / M_analytical) * 100
    
    # Plot the errors
    plt.figure(figsize=(12, 8))
    
    # Absolute error
    plt.subplot(2, 1, 1)
    plt.plot(t_analytical, abs_error, label='Absolute Error', color='blue', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Absolute Error (g)')
    plt.title('Absolute Error Between Analytical and Numerical Solutions')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    # Relative error
    plt.subplot(2, 1, 2)
    plt.semilogy(t_analytical, rel_error, label='Relative Error (%)', color='red', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Relative Error (%)')
    plt.title('Relative Error Between Analytical and Numerical Solutions')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

PBHDemoError(explosion_x=0, M0=1e11, x=1e6, target_mass=1e9)
