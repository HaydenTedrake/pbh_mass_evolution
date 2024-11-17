import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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

constant_fM = np.float64(1)

@jit(nopython=True)
def f(M):
    """Vectorized and JIT-compiled version of Carr's f(m) function"""
    M_log = np.log10(M)
    if M_log >= 14 and M_log <= 17:
        return np.power(M, -2.0/3.0) * np.power(10, 34.0/3.0)
    elif M_log < 14:
        return 100
    else:
        return 1

@jit(nopython=True)
def Mdot(M):
    """JIT-compiled version of mass evolution function"""
    return -5.34e25 * constant_fM / (M * M)

def solve_Mdot_improved(M0, explosion_time, dt=None):
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
        t_span=(0, explosion_time * 1.25),
        y0=[M0],
        method='RK45',
        rtol=rtol,
        atol=atol,
        max_step=dt if dt is not None else np.inf
    )
    
    return solution.t, solution.y[0]

def PBHDemo_improved(explosion_x, M0, x, dt=1000):
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
    explosion_time = (np.power(M0, 3) - 1e27) / (16.02e25 * constant_fM)
    
    # Solve using improved method
    times_numerical, masses_numerical = solve_Mdot_improved(M0=M0, explosion_time=explosion_time, dt=dt)
    
    # Create the plot with logarithmic scales
    plt.figure(figsize=(12, 8))
    
    # Plot numerical solution
    plt.semilogy(times_numerical, masses_numerical, 'r-', label='Numerical Solution', linewidth=2)
    
    # Customize the plot
    plt.xlabel("Time (s)")
    plt.ylabel("PBH Mass (g)")
    plt.title(f"PBH Mass Evolution (M₀ = {M0:.2e} g)")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    # Add some key information as text
    info_text = f"Initial Mass: {M0:.2e} g\nExplosion Time: {explosion_time:.2e} s"
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.show()
    
    return times_numerical, masses_numerical

PBHDemo_improved(explosion_x=0, M0=1e11, x=1e6)
