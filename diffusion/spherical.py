import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.animation as animation

# Parameters
D = 1.0
R = 5.0          # Max radius (half domain size)
N = 100          # Radial steps
dr = R / N
dt = 0.001
t_final = 0.5
steps = int(t_final / dt)

r = np.linspace(dr, R, N)  # skip r = 0
u = np.zeros(N)

# Source functions
def source_power(t, alpha=2):
    return t**alpha

# Choose one
source_type = "power"

# Construct Crank-Nicolson matrices
diagonal = np.zeros(N)
upper = np.zeros(N - 1)
lower = np.zeros(N - 1)

for i in range(1, N - 1):
    ri = r[i]
    diagonal[i] = 1 + dt * D * (2 / dr**2)
    upper[i] = -dt * D * ((1 / dr**2) + (1 / (2 * ri * dr)))
    lower[i - 1] = -dt * D * ((1 / dr**2) - (1 / (2 * ri * dr)))

# Dirichlet BC at r=dr (u=0) and r=R (u=0)
diagonal[0] = 1.0
diagonal[-1] = 1.0
upper[0] = 0.0
lower[-1] = 0.0

A = diags([lower, diagonal, upper], offsets=[-1, 0, 1]).tocsc()
B = diags([-lower, 2 - diagonal, -upper], offsets=[-1, 0, 1]).tocsc()

# Time stepping
snapshots = []
timesteps = []

for step in range(steps):
    t = step * dt

    # Source
    if source_type == "power":
        s_val = source_power(t)
    else:
        s_val = 0.0

    source = np.zeros(N)
    source[0] = s_val / dr  # approximate delta at center

    rhs = B @ u + dt * source
    u = spsolve(A, rhs)

    if step % 2 == 0:
        # Reconstruct 2D from radial solution for visualization
        grid_size = N
        x = np.linspace(-R, R, grid_size)
        y = np.linspace(-R, R, grid_size)
        X, Y = np.meshgrid(x, y)
        R_grid = np.sqrt(X**2 + Y**2)

        u2d = np.interp(R_grid.ravel(), r, u, left=0, right=0)
        u2d = u2d.reshape((grid_size, grid_size))

        snapshots.append(u2d)
        timesteps.append(step)

# Plot animation
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(snapshots[0], cmap='viridis', extent=[-R, R, -R, R], origin='lower')
ax.set_title("Spherical Diffusion: Center Slice")
fig.colorbar(im)

def update(frame):
    im.set_array(snapshots[frame])
    ax.set_title(f"Time step {timesteps[frame]}")
    return [im]

ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=50, blit=False)
plt.show()