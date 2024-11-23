import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d

def f(M):
    return 1.0

def Mdot(M):
    if M <= 0:
        raise ValueError("Mass must be positive.")
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
        def dMdt(t, M):
            return Mdot(M)

        # Event to stop integration when reaching the target mass
        def event_reach_target(t, M):
            return M[0] - target_mass

        event_reach_target.terminal = True
        event_reach_target.direction = -1

        # Solve the ODE
        solution = solve_ivp(
            dMdt,
            [0, 1e15],  # Extend time span to very large values
            [M0],
            method='RK45',
            events=event_reach_target,
            max_step=1e7  # Allow large steps
        )


        # Debugging: Check solver output
        print(f"Solver status for t={t}: Success={solution.success}, Events={solution.t_events}")

        # Check if the solver stopped due to the event
        if solution.t_events[0].size > 0:
            print(f"Event triggered at t={solution.t_events[0][0]} with mass {solution.y_events[0][0]}")
            return solution.y_events[0][0] - target_mass

        # If no event was triggered, return the final mass deviation
        final_mass = solution.y[0][-1]
        print(f"Final mass at t={t}: {final_mass}")
        return final_mass - target_mass


    # Rough estimate for the explosion time
    rough_estimate = (M0**3 - target_mass**3) / (16.02e25 * f(M0))
    print(f"Rough estimate: {rough_estimate}")
    lower_bound = max(rough_estimate * 0.1, 1e-10)  # Ensure non-zero lower bound
    print(f"Lower bound: {lower_bound}")
    upper_bound = rough_estimate * 10
    print(f"Upper bound: {upper_bound}")

    for _ in range(max_iterations):
        try:
            print(f"Trying bounds: {lower_bound}, {upper_bound}")
            result = root_scalar(
                mass_at_time,
                bracket=[lower_bound, upper_bound],
                method='brentq',
                xtol=precision
            )
            print(f"Result: {result}")
            final_mass = mass_at_time(result.root)
            print(f"Final mass: {final_mass}")
            
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

    Mass_cubed = (-16.02e25 * f(1) * t + np.power(M0, 3))
    Mass = np.cbrt(Mass_cubed)
    if Mass < 0: 
        Mass = 0
    return Mass

def PBHDemo(explosion_x, M0, x, target_mass=1e9, dt=100):
    """
    Improved version of PBH demonstration with better numerical integration
    and plotting capabilities.
    
    Args:
        explosion_x (float): Explosion position in km
        M0 (float): Initial mass in grams
        x (float): Position in km
        target_mass (float, optional): Target mass for explosion time calculation in grams
        dt (float, optional): Maximum time step for integration
    """
    M0 = M0
    # Calculate parameters
    displacement = x - explosion_x  # in km
    boundary_time = displacement / 220  # (km/s)
    explosion_time = find_explosion_time(M0, target_mass=target_mass)  # Using target_mass
    
    if explosion_time is None:
        print("Could not determine explosion time. Using fallback calculation.")
        explosion_time = (np.power(M0, 3) - np.power(target_mass, 3)) / (16.02e25 * f(1))
    
    # Analytical solution
    t_analytical = np.arange(0, explosion_time, 10)
    M_analytical = np.array([MassAnalytical(M0=M0, t=ti) for ti in t_analytical])
    
    mask_analytical = M_analytical >= target_mass
    t_analytical = t_analytical[mask_analytical]
    M_analytical = M_analytical[mask_analytical]
    
    # Solve using improved method
    times_numerical, masses_numerical = solve_Mdot(M0, explosion_time, target_mass, dt=dt)
    
    # Shift times by explosion time
    times_numerical_shifted = times_numerical - explosion_time
    
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
    info_text = f"Explosion Time: {explosion_time:.2e} s"
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.show()
    
    return times_numerical_shifted, masses_numerical, mass_at_negative_boundary_time

# Example usage with custom target mass
times_shifted, masses, M_at_negative_boundary = PBHDemo(explosion_x=0, M0=1e11, x=2200, target_mass=1e9)
