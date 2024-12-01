import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from typing import Tuple, List

def f(M):
    return 1.0

def Mdot(M):
    return -5.34e25 * f(M) / (M * M)

def solve_Mdot(M0, target_mass=1e9, dt=None):
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
    rtol = 1e-5
    atol = 1e-5
    
    explosion_time = np.inf

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

    explosion_time = solution.t[-1]
    explosion_mass = solution.y[0][-1]

    print(f"Explosion time: {explosion_time}")
    print(f"Explosion mass: {explosion_mass}")

    rough_estimate = (M0**3 - target_mass**3) / (16.02e25 * f(M0))
    print(f"Rough estimate: {rough_estimate}")
    
    return solution.t, solution.y[0], explosion_time

def PBHDemo(explosion_x, M0, x, target_mass=1e9, dt=100):
    """
    Simulates and plots the mass evolution from the formation (initial) mass, M0,
    until the mass reaches the target mass. We treat the time when the mass reaches
    the target mass as the explosion time, and set that position as explosion_x. 
    Then, assuming a constant velocity and a linear trajectory, we back out what
    the mass was when the PBH passed through x.
    
    Args:
        explosion_x (float): Explosion position in km
        M0 (float): Initial mass in grams
        x (float): Position in km
        target_mass (float, optional): Target mass for explosion time calculation in grams
        dt (float, optional): Maximum time step for integration
    """
    # Calculate parameters
    displacement = np.abs(x - explosion_x)  # in km
    boundary_time = displacement / 220  # (km/s)
    
    # Solve using improved method
    times_numerical, masses_numerical, explosion_time = solve_Mdot(M0, target_mass, dt=dt)

    if explosion_time is None:
        print("Could not determine explosion time. Using fallback calculation.")
        explosion_time = (np.power(M0, 3) - np.power(target_mass, 3)) / (16.02e25 * f(1))
    
    # Shift times by explosion time
    times_numerical_shifted = times_numerical - explosion_time

    # Analytical solution
    t_analytical = np.arange(0, explosion_time, 10)

    def MassAnalytical_vectorized(M0, t):
        Mass_cubed = (-16.02e25 * f(1) * t + np.power(M0, 3))
        Mass = np.cbrt(np.maximum(Mass_cubed, 0))  # Avoid negative masses
        return Mass

    M_analytical = MassAnalytical_vectorized(M0, t_analytical)
    
    mask_analytical = M_analytical >= target_mass
    t_analytical = t_analytical[mask_analytical]
    M_analytical = M_analytical[mask_analytical]
    
    # Find the index closest to -boundary_time
    boundary_time_idx = np.abs(times_numerical_shifted - (-boundary_time)).argmin()
    mass_at_negative_boundary_time = masses_numerical[boundary_time_idx]

    # Interpolate to find M(-boundary_time)
    interpolation_function = interp1d(
        times_numerical_shifted, 
        masses_numerical, 
        kind='linear', 
        bounds_error=False, 
        fill_value="extrapolate"
    )
    mass_at_negative_boundary_time = interpolation_function(-boundary_time)
    
    # Create the plot with logarithmic scales
    plt.figure(figsize=(12, 8))
    
    # Plot analytical solution in blue
    plt.plot(t_analytical - explosion_time, M_analytical, 'b-', label='Analytical Solution', alpha=0.8)

    # Plot numerical solution
    plt.semilogy(times_numerical_shifted, masses_numerical, 'r--', label='Numerical Solution', linewidth=2)
    
    # Highlight M(-boundary_time)
    plt.scatter(-boundary_time, mass_at_negative_boundary_time, color='green', label=f"M at target x ≈ {mass_at_negative_boundary_time:.2e} g", zorder=5)
    
    # Customize the plot
    plt.xlabel("Time relative to explosion time (s)")
    plt.ylabel("PBH Mass (g)")
    plt.title(f"PBH Mass Evolution (M₀ = {M0:.2e} g)")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    # Add some key information as text
    # info_text = f"Explosion Time: {explosion_time:.2e} s"
    info_text = f"Explosion Time: {explosion_time} s"
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.show()
    
    return times_numerical_shifted, masses_numerical, mass_at_negative_boundary_time

# Example usage with custom target mass
times_shifted, masses, M_at_negative_boundary = PBHDemo(explosion_x=0, M0=1e11, x=22000, target_mass=1e9)
print(f"M at target x: {M_at_negative_boundary} g")
