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

    print(f"M0: {M0}, Target Mass: {target_mass}")
    print(f"f(M0): {f(M0)}")

    # Rough estimate for the explosion time
    rough_estimate = (M0**3 - target_mass**3) / (16.02e25 * f(M0))
    lower_bound = rough_estimate * 0.1
    upper_bound = rough_estimate * 2

    # Step 2: Expand the interval systematically if bracketing fails
    max_attempts = 10
    expansion_factor = 2  # How much to expand the bounds each step

    for attempt in range(max_attempts):
        lower_value = mass_at_time(lower_bound)
        upper_value = mass_at_time(upper_bound)
        print(f"Attempt {attempt}: Lower = {lower_value}, Upper = {upper_value}")

        if lower_value * upper_value < 0:
            print("Root found within expanded interval.")
            break  # Root is bracketed
        else:
            lower_bound /= expansion_factor
            upper_bound *= expansion_factor
    else:
        print("Failed to bracket root after maximum attempts.")
        return None  # Return None if bracketing fails

    # Use root_scalar to find the explosion time
    try:
        result = root_scalar(
            mass_at_time,
            bracket=[lower_bound, upper_bound],
            method='brentq'
        )
        return result.root
    except ValueError as e:
        print(f"Root finding failed: {e}")
        return None


def solve_Mdot(M0, explosion_time, dt=None):
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
        return Mdot(M)

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
        max_step=dt if dt is not None else np.inf
    )
    
    return solution.t, solution.y[0]

def MassAnalytical(M0, t):
    """Compute Mass as a function of time."""

    Mass_cubed = (-16.02e25 * f(M0) * t + np.power(M0, 3))
    Mass = np.cbrt(Mass_cubed)
    if Mass < 0: 
        Mass = 0
    return Mass

def PBHDemo(explosion_x, M0, x, dt=100):
    """
    Improved version of PBH demonstration with better numerical integration
    and plotting capabilities.
    
    Args:
        explosion_x (float): Explosion position in km
        M0 (float): Initial mass in grams
        x (float): Position in km
        dt (float, optional): Maximum time step for integration
    """
    # Calculate parameters
    displacement = x - explosion_x  # in km
    boundary_time = displacement / 220  # (km/s)
    explosion_time = find_explosion_time(M0)  # Using the new function
    scale_num = M0/2
    
    if explosion_time is None:
        print("Could not determine explosion time. Using fallback calculation.")
        explosion_time = (np.power(M0, 3) - 1e27) / (16.02e25 * f(M0))
    
    # Analytical solution
    t_analytical = [i * dt for i in range(int(explosion_time/dt))]
    M_analytical = [MassAnalytical(M0=M0, t=ti) for ti in t_analytical]

    # Solve using improved method
    times_numerical, masses_numerical = solve_Mdot(M0, explosion_time=explosion_time, dt=dt)
    
    # Create the plot with logarithmic scales
    plt.figure(figsize=(12, 8))
    
    # Plot analytical solution in blue
    plt.plot(t_analytical, M_analytical, 'b-', label='Analytical Solution', alpha=0.8)

    # Plot numerical solution
    plt.semilogy(times_numerical, masses_numerical, 'r--', label='Numerical Solution', linewidth=2)
    
    # Customize the plot
    plt.xlabel("Time (s)")
    plt.ylabel("PBH Mass (g)")
    plt.title(f"PBH Mass Evolution (M₀ = {M0:.2e} g): Analytical vs Numerical Solution")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    # Add some key information as text
    info_text = f"Explosion Time: {explosion_time} s"
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.show()
    
    return times_numerical, masses_numerical

# Example usage
PBHDemo(explosion_x=0, M0=1e11, x=1e6)
