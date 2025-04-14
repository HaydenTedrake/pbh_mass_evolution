import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
from scipy.integrate import solve_ivp

# Parameters
D = 1.0
nx, ny, nz = 50, 50, 50
dx = dy = dz = 1.0
dt = 0.1
steps = 100
total_time = steps * dt
t_eval = np.linspace(0, total_time, steps)

# Center point for the delta source
cx, cy, cz = nx // 2, ny // 2, nz // 2

# Laplacian using np.roll
def laplacian(u):
    return (
        np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
        np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) +
        np.roll(u, 1, axis=2) + np.roll(u, -1, axis=2) -
        6 * u
    ) / dx**2

# RHS function for solve_ivp
def dudt(t, u_flat, kind="log", alpha=2):
    u = u_flat.reshape((nx, ny, nz))
    du = D * laplacian(u)
    # Inject source at center
    du[cx, cy, cz] += source_function(t, kind=kind, alpha=alpha)
    return du.ravel()

# Time-dependent source function
def source_function(t, kind="log", alpha=2):
    if kind == "log":
        return np.log(t + 1) * 1e-4
    elif kind == "power":
        return (t ** alpha) * 1e-6
    elif kind == "impulse":
        return 1e-3 if np.isclose(t, 5.0, atol=0.1) else 0.0
    else:
        return 0.0

# Initial condition: delta function at the center
u0 = np.zeros((nx, ny, nz))
u0[cx, cy, cz] = 1.0 / (dx * dy * dz)
u0_flat = u0.ravel()

# Solve the system
rhs = lambda t, u_flat: dudt(t, u_flat, kind="power", alpha=2)
sol = solve_ivp(rhs, (0, total_time), u0_flat, t_eval=t_eval, method='RK23')

# Precompute the slices to animate
slices = [np.maximum(sol.y[:, i].reshape((nx, ny, nz))[:, :, nz//2], 1e-6) for i in range(steps)]

# Set up plot
fig, ax = plt.subplots()
vmax = np.max(sol.y)
im = ax.imshow(slices[0], origin='lower', cmap='hot', norm=LogNorm(vmin=1e-6, vmax=vmax))
plt.colorbar(im, ax=ax)

# Update function for animation
def update(frame):
    im.set_data(slices[frame])
    ax.set_title(f"Step {frame} - Time {sol.t[frame]:.2f}")
    return [im]

ani = animation.FuncAnimation(fig, update, frames=steps, interval=100, blit=False)
plt.show()
