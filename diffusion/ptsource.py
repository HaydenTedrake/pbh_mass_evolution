import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm

# Parameters
D = 1.0
nx, ny, nz = 50, 50, 50
dx = dy = dz = 1.0
dt = 0.1
steps = 100

# Stability check
assert dt < dx**2 / (6 * D), "Time step too large for stability!"

# Initialize concentration field
u = np.zeros((nx, ny, nz))
cx, cy, cz = nx // 2, ny // 2, nz // 2
u[cx, cy, cz] = 1.0 / (dx * dy * dz)

# Laplacian function
def laplacian(u):
    return (
        np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
        np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) +
        np.roll(u, 1, axis=2) + np.roll(u, -1, axis=2) -
        6 * u
    ) / dx**2

# Set up plot
fig, ax = plt.subplots()
# Use LogNorm for better visibility of small values
im = ax.imshow(u[:, :, nz//2], origin='lower', cmap='hot',
               norm=LogNorm(vmin=1e-6, vmax=1.0))
plt.colorbar(im, ax=ax)

# Animation update
def update(frame):
    global u
    u += D * dt * laplacian(u)
    im.set_data(u[:, :, nz//2])
    ax.set_title(f"Step {frame} - Time {frame*dt:.2f}")
    return [im]

ani = animation.FuncAnimation(fig, update, frames=steps, interval=100)
plt.show()