import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def Mdot(t, M):
    """ Compute dM/dt for mass evolution """
    return -5.34e25 / (M * M)  # Hawking radiation mass loss

def integrate_pbh_mass(M0, target_mass=1e9):
    """
    Uses SciPy's solve_ivp to integrate the mass evolution of a PBH.
    
    Args:
        M0 (float): Initial mass in grams
        target_mass (float): Target mass for explosion
    
    Returns:
        times, masses (arrays)
    """
    # Define the stopping event (when M reaches target_mass)
    def event(t, M):
        return M[0] - target_mass  # Extract scalar value from list
    event.terminal = True  # Stop when target_mass is reached
    event.direction = -1    # Only trigger when mass decreases past target_mass

    # Solve the differential equation
    t_span = [0, 1e18]  # Start from t=0 and integrate forward
    sol = solve_ivp(Mdot, t_span, [M0], method='RK45', events=event, max_step=1e15)

    return sol.t, sol.y[0]

# Parameters
M0 = 4e14  # Initial PBH mass in grams (adjusted to match previous graph)
target_mass = 5.93e10  # Target mass when PBH explodes

# Integrate
times, masses = integrate_pbh_mass(M0, target_mass)

# Shift time relative to explosion
explosion_time = times[-1]
times_relative = times - explosion_time

# Plot
plt.figure(figsize=(10, 6))
plt.semilogy(times_relative, masses, 'r--', label="Numerical Solution")
plt.axvline(0, color='black', linestyle=':', label="Explosion Time")
plt.xlabel("Time relative to explosion time (s)")
plt.ylabel("PBH Mass (g)")
plt.title(f"PBH Mass Evolution (M₀ = {M0:.2e} g)")
plt.legend()
plt.grid()
plt.show()
