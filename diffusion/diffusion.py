import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve
import time
import matplotlib.animation as animation

# Parameters
D = 1.0       # Diffusion coefficient
L = 10.0      # Domain size
N = 11        # Keep this moderate for testing
dx = L/(N-1)  # Spatial step
dt = 0.005    # Time step
t_final = 0.5 # Final time

def source_logarithmic(t):
    return np.log1p(t)

def source_power(t, alpha=2):
    return t**alpha

def source_impulse(t, t0=5.0, width=0.1):
    return np.exp(-((t - t0)**2) / (2 * width**2))

def source_delta(t, t0=0.0):
    return 1.0 if np.isclose(t, t0, atol=1e-8) else 0.0

print(f"Running 3D diffusion with {N}x{N}x{N} grid...")

# Create grid
x = y = z = np.linspace(0, L, N)
center = N//2

# Initial condition (delta function)
u = np.zeros((N, N, N))
u[center, center, center] = 1.0/(dx**3)

# 1D Laplacian with Dirichlet BCs
diagonals = [-2*np.ones(N), np.ones(N-1), np.ones(N-1)]
L1D = diags(diagonals, [0, 1, -1], shape=(N, N))/(dx**2)
L1D = L1D.tolil()
L1D[0,:] = 0; L1D[0,0] = 1
L1D[-1,:] = 0; L1D[-1,-1] = 1
L1D = L1D.tocsc()

# 3D Laplacian via Kronecker products
I = eye(N, format='csc')
L3D = (kron(kron(I, I), L1D) + 
      kron(kron(I, L1D), I) + 
      kron(kron(L1D, I), I))

# Crank-Nicolson matrices
A = eye(N**3, format='csc') - 0.5*dt*D*L3D
B = eye(N**3, format='csc') + 0.5*dt*D*L3D

# Time stepping
u_flat = u.flatten()
center_idx = center * N * N + center * N + center  # 1D index of the center cell
steps = int(t_final/dt)
start_time = time.time()

# Store snapshots of the center xy-plane
snapshots = []
timesteps = []

for step in range(steps):
    t = step * dt

    # Choose one of the source types:
    source_type = "delta"  # change this to try others: "log", "power", "impulse"

    if source_type == "log":
        source_value = source_logarithmic(t)
    elif source_type == "power":
        source_value = source_power(t, alpha=2)
    elif source_type == "impulse":
        source_value = source_impulse(t, t0=5.0, width=0.1)
    elif source_type == "delta":
        source_value = source_delta(t, t0=0.0)
    else:
        source_value = 0.0

    source_vec = np.zeros(N**3)
    source_vec[center_idx] = source_value / dx**3  # normalize for spatial delta

    rhs = B @ u_flat + dt * source_vec  # inject source

    u_flat = spsolve(A, rhs)

    if step % 2 == 0:
        u_snap = u_flat.reshape((N, N, N))
        snapshots.append(u_snap[:, :, center])
        timesteps.append(step)

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(snapshots[0], cmap='viridis', extent=[0, L, 0, L])
ax.set_title("Diffusion: center xy-plane")
fig.colorbar(im)

def update(frame):
    im.set_array(snapshots[frame])
    ax.set_title(f"Time step {timesteps[frame]}")
    return [im]

ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=50, blit=False)
plt.show()