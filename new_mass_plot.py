import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

age_of_universe = 4.35e17  # in seconds

def joules_to_gev(joules):
    # Constants
    eV_per_joule = 6.242e18  # 1 Joule = 6.242 x 10^18 eV
    gev_per_ev = 1e-9        # 1 GeV = 10^9 eV
    
    # Convert Joules to GeV
    gev = joules * eV_per_joule * gev_per_ev
    
    return gev

def f(M):
    # Masses in GeV from Table I
    masses = {
        'mu': 0.106,     # muon
        'd': 0.34,       # down quark
        's': 0.5,        # strange quark
        'c': 1.87,       # charm quark
        'T': 1.78,       # tau
        'b': 5.28,       # bottom quark
        't': 100,        # top quark (unobserved at time of paper)
        'g': 0.6         # gluon (effective mass)
    }
    
    # Beta values (assuming s=1/2 and s=1 as mentioned)
    beta_half = 4.53
    beta_one = 6.04
    
    # Base constant from the original equation
    base = 1.569
    
    # Detailed calculation following the exact equation
    result = base + 0.569 * (
        np.exp(-M / (beta_half * masses['mu'])) +
        3 * np.exp(-M / (beta_half * masses['d'])) +
        3 * np.exp(-M / (beta_half * masses['s'])) +
        3 * np.exp(-M / (beta_half * masses['c'])) +
        np.exp(-M / (beta_half * masses['T'])) +
        3 * np.exp(-M / (beta_half * masses['b'])) +
        3 * np.exp(-M / (beta_half * masses['t'])) +
        0.963 * np.exp(-M / (beta_one * masses['g']))
    )
    
    return result

def Mdot(M):
    return -5.34e25 * f(M) / (M * M)

def solve_Mdot(M0, target_mass=1e9):
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
        #max_step=dt if dt is not None else np.inf,
        #first_step=100000,  # this was a simple test from Russ (don't trust it!)
        events=event_mass_threshold,
        dense_output=True
    )

    explosion_time = solution.t[-1]
    explosion_mass = solution.y[0][-1]

    print(f"Explosion time: {explosion_time} s")
    print(f"Explosion mass: {explosion_mass} g")

    return solution.t, solution.y[0], explosion_time

def PBHDemo(M0, explosion_coordinate, velocity, boundary_time, target_mass=1e9):
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
    vx, vy, vz = velocity
    explosion_x, explosion_y, explosion_z = explosion_coordinate
    
    # Solve using improved method
    times_numerical, masses_numerical, explosion_time = solve_Mdot(M0, target_mass)
    # print("finished solve_Mdot")

    rough_estimate = (M0**3 - target_mass**3) / (16.02e25 * 1.0)
    print(f"Rough estimate: {rough_estimate} s")
    if explosion_time is None:
        print("Could not determine explosion time. Using fallback calculation.")
        explosion_time = rough_estimate
    
    # Shift times by explosion time
    times_numerical_shifted = times_numerical - explosion_time

    def exp_time_points(T, num_points=1000000):
        exp_space = np.exp(np.linspace(0, 5, num_points))
        return T - T * (exp_space - exp_space[0]) / (exp_space[-1] - exp_space[0])
    
    # Analytical solution
    t_analytical = exp_time_points(rough_estimate)

    def MassAnalytical_vectorized(M0, t):
        Mass_cubed = (-16.02e25 * 1.0 * t + np.power(M0, 3))
        Mass = np.cbrt(np.maximum(Mass_cubed, 0))  # Avoid negative masses
        return Mass

    M_analytical = MassAnalytical_vectorized(M0, t_analytical)
    
    # Find the index closest to -boundary_time
    boundary_time_idx = np.abs(times_numerical_shifted - (-boundary_time)).argmin()
    mass_at_negative_boundary_time = masses_numerical[boundary_time_idx]
    
    # Calculate position at -boundary_time
    time_to_boundary = -boundary_time
    displacement_x = time_to_boundary * vx
    displacement_y = time_to_boundary * vy
    displacement_z = time_to_boundary * vz
    position_x = explosion_x + displacement_x
    position_y = explosion_y + displacement_y
    position_z = explosion_z + displacement_z
    position = (position_x, position_y, position_z)

    plot = True
    if plot:  # enable/disable plotting
        # Create the plot with logarithmic scales
        plt.figure(figsize=(12, 8))
        
        # Plot analytical solution in blue
        plt.plot(t_analytical - explosion_time, M_analytical, 'b-', label='Analytical Solution', alpha=0.8)

        # Plot numerical solution
        plt.semilogy(times_numerical_shifted, masses_numerical, 'r--', label='Numerical Solution', linewidth=2)
        
        # Highlight M(-boundary_time)
        plt.scatter(-boundary_time, mass_at_negative_boundary_time, color='green', label=f"M at target t ≈ {mass_at_negative_boundary_time:.2e} g" , zorder=5)
        
        plt.annotate(
            f"Position:\n({position_x:.2f}, {position_y:.2f}, {position_z:.2f})",
            xy=(-boundary_time, mass_at_negative_boundary_time),
            xytext=(-boundary_time * 0.8, mass_at_negative_boundary_time * 1.2),
            arrowprops=dict(facecolor='black', arrowstyle="->"),
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        )

        # Customize the plot
        plt.xlabel("Time relative to explosion time (s)")
        plt.ylabel("PBH Mass (g)")
        plt.title(f"PBH Mass Evolution (M₀ = {M0:.2e} g)")
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        
        # Add some key information as text
        # info_text = f"Explosion Time: {explosion_time:.2e} s"
        info_text = f"Explosion Time: {explosion_time:.2e} s"
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.show()
    
    return times_numerical_shifted, masses_numerical, mass_at_negative_boundary_time, position

# Example usage with custom target mass
times_shifted, masses, M_at_negative_boundary, position = PBHDemo(
    M0=1e15, 
    explosion_coordinate=(0, 0, 0), 
    velocity=(220, 0, 0), 
    boundary_time=100000, 
    target_mass=1e9
)

print(f"The PBH has mass of {M_at_negative_boundary} g with position {position} km at target time before the explosion.")
