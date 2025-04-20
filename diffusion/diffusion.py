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
steps = int(t_final/dt)
start_time = time.time()

# Store snapshots of the center xy-plane
snapshots = []
timesteps = []

for step in range(steps):
    u_flat = spsolve(A, B.dot(u_flat))
    
    if step % 2 == 0:
        u_snap = u_flat.reshape((N, N, N))
        snapshots.append(u_snap[:, :, center])  # or change to desired slice
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