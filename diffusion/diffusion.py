import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve
import time

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

for step in range(steps):
    if step % 10 == 0:
        elapsed = time.time() - start_time
        remaining = (steps-step)*(elapsed/(step+1)) if step > 0 else 0
        print(f"Step {step+1}/{steps} | Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s")
    
    u_flat = spsolve(A, B.dot(u_flat))

print("Simulation completed!")
u_final = u_flat.reshape((N, N, N))

# Visualization
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.imshow(u_final[center, :, :], extent=[0, L, 0, L], cmap='viridis')
plt.colorbar()
plt.title('yz-plane slice')

plt.subplot(122)
plt.imshow(u_final[:, :, center], extent=[0, L, 0, L], cmap='viridis')
plt.colorbar()
plt.title('xy-plane slice')

plt.tight_layout()
plt.show()