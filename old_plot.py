import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from typing import Tuple, List
import warnings

def f(M):
    return 1.0

def Mdot(M):
    return -5.34e25 * f(M) / (M * M)

def find_explosion_time(
    M0: float,
    target_mass: float = 1e9,
    rtol: float = 1e-6,
    safety_factor: float = 0.9,
    min_step: float = 1e-20,
    max_step: float = 1e10
) -> float:
    """
    Find PBH explosion time using adaptive time stepping.
    
    Args:
        M0: Initial mass in grams
        target_mass: Target mass in grams
        rtol: Relative tolerance for integration
        safety_factor: Safety factor for step size adjustment
        min_step: Minimum allowed time step
        max_step: Maximum allowed time step
    
    Returns:
        float: Explosion time
    """
    rough_estimate = (M0**3 - target_mass**3) / (16.02e25 * f(M0))
    print(f"Rough estimate: {rough_estimate}")

    def rk4_step(t: float, M: float, dt: float) -> Tuple[float, float, float]:
        """
        Single RK4 step with error estimate.
        Returns (new_mass, error_estimate, actual_step)
        """
        # Full step
        k1 = Mdot(M)
        k2 = Mdot(M + 0.5 * dt * k1)
        k3 = Mdot(M + 0.5 * dt * k2)
        k4 = Mdot(M + dt * k3)
        
        M_new = M + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Half steps for error estimate
        dt_half = dt/2.0
        k1_half = k1
        k2_half = Mdot(M + 0.5 * dt_half * k1_half)
        k3_half = Mdot(M + 0.5 * dt_half * k2_half)
        k4_half = Mdot(M + dt_half * k3_half)
        
        M_half = M + (dt_half/6.0) * (k1_half + 2*k2_half + 2*k3_half + k4_half)
        
        # Second half step
        k1_half2 = Mdot(M_half)
        k2_half2 = Mdot(M_half + 0.5 * dt_half * k1_half2)
        k3_half2 = Mdot(M_half + 0.5 * dt_half * k2_half2)
        k4_half2 = Mdot(M_half + dt_half * k3_half2)
        
        M_half2 = M_half + (dt_half/6.0) * (k1_half2 + 2*k2_half2 + 2*k3_half2 + k4_half2)
        
        error = abs(M_new - M_half2)
        return M_new, error, dt

    def adaptive_integrate() -> float:
        """
        Perform adaptive integration until target mass is reached.
        """
        t = 0.0
        M = M0
        dt = min(max_step, abs(M/Mdot(M)) * 0.01)  # Initial step based on characteristic time
        
        times: List[float] = [t]
        masses: List[float] = [M]
        
        while M > target_mass:
            # Prevent too small steps
            if dt < min_step:
                raise RuntimeError(f"Step size {dt} below minimum {min_step}")
                
            # Try step
            M_new, error, actual_dt = rk4_step(t, M, dt)
            
            # Relative error
            rel_error = error / M if M != 0 else error
            
            # Accept or reject step based on error
            if rel_error <= rtol:
                t += actual_dt
                M = M_new
                times.append(t)
                masses.append(M)
                
                # Break if we've reached target
                if M <= target_mass:
                    break
            
            # Adjust step size using PI controller
            dt_new = safety_factor * dt * (rtol/rel_error)**0.2
            dt = min(max_step, max(min_step, dt_new))
            
            # Additional safety checks
            if not np.isfinite(M) or not np.isfinite(dt):
                raise RuntimeError("Non-finite values encountered")
                
            if len(times) > 1000000:  # Prevent infinite loops
                raise RuntimeError("Too many steps")
        
        # Interpolate to find exact crossing time
        if len(times) >= 2:
            idx = next(i for i, m in enumerate(masses) if m <= target_mass)
            if idx > 0:
                t1, t2 = times[idx-1:idx+1]
                M1, M2 = masses[idx-1:idx+1]
                return t1 + (t2 - t1) * (M1 - target_mass)/(M1 - M2)
        
        return times[-1]

    try:
        return adaptive_integrate()
    except Exception as e:
        warnings.warn(f"Adaptive integration failed: {str(e)}. Using analytical estimate.")
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
    rtol = 1e-5
    atol = 1e-5
    
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
    # Calculate parameters
    displacement = x - explosion_x  # in km
    boundary_time = displacement / 220  # (km/s)
    explosion_time = find_explosion_time(M0, target_mass=target_mass)  # Using target_mass
    
    if explosion_time is None:
        print("Could not determine explosion time. Using fallback calculation.")
        explosion_time = (np.power(M0, 3) - np.power(target_mass, 3)) / (16.02e25 * f(1))
    
    def dMdt(t, M):
        return Mdot(M[0])
    
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
    
    # Solve using improved method
    times_numerical, masses_numerical = solve_Mdot(M0, explosion_time, target_mass, dt=dt)
    
    # Shift times by explosion time
    times_numerical_shifted = times_numerical - explosion_time
    
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
