import numpy as np
import matplotlib.pyplot as plt

# Grid size
Nr, Ntheta, Nphi = 21, 21, 21

# Grid spacing
dr = 10 / (Nr - 1)
dtheta = np.pi / (Ntheta - 1)
dphi = 2 * np.pi / (Nphi - 1)

# Time step
dt = 1e-4
D = 1.0  # diffusion coefficient

# Coordinate arrays
r_vals = np.linspace(0, 10, Nr)
theta_vals = np.linspace(0, np.pi, Ntheta)
phi_vals = np.linspace(0, 2 * np.pi, Nphi)

# Source as a function of time
def source_func(t):
    grid = np.zeros((Nr, Ntheta, Nphi))
    grid[r, theta, phi] = 1/dt
    return grid

# Initialize u with zeros and one at the source
u = source_func(0, 0, 0, 0)

# Time stepping function
def update(u, t):
    new_u = np.copy(u)
    for i in range(1, Nr - 1):
        r = r_vals[i]
        for j in range(1, Ntheta - 1):
            theta = theta_vals[j]
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            for k in range(1, Nphi - 1):
                laplacian = (u[i+1, j, k] - 2*u[i, j, k] + u[i-1, j, k]) / dr**2 \
                    + 2/r * (u[i+1, j, k] - u[i-1, j, k]) / dr \
                    + 1/(r**2*sin_theta) * (cos_theta * (u[i, j+1, k] - u[i, j-1, k])/ dtheta \
                    + sin_theta * (u[i, j+1, k] - 2*u[i, j, k] + u[i, j-1, k])/ dtheta**2) \
                    + 1/((r*sin_theta)**2) * (u[i, j, k+1] - 2*u[i, j, k] + u[i, j, k-1])/ dphi**2

                new_u[i, j, k] += dt * D * laplacian
    
    new_u += dt * source_func(t)
    return new_u

# Run simulation for a few steps
steps = 500
for n in range(1, steps+1):
    u = update(u, t=n*dt)

theta_idx = Ntheta//2  # This is at theta = pi/2 (equatorial plane)

r_grid, phi_grid = np.meshgrid(r_vals, phi_vals, indexing='ij')
x = r_grid * np.cos(phi_grid)
y = r_grid * np.sin(phi_grid)

plt.contourf(x, y, u[:, theta_idx, :])
plt.axis('equal')
plt.colorbar(label='u')
plt.title("θ = π/2")
plt.xlabel('x')
plt.ylabel('y')
plt.show()