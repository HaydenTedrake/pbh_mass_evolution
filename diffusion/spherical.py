import numpy as np
import matplotlib.pyplot as plt

# Parameters
D = 1.0       # Diffusion coefficient
R = 10.0      # Maximum radius
N = 100       # Number of radial points
dr = R / N    # Radial step
dt = 0.0005   # Time step
t_final = 0.1 # Final time
steps = int(t_final / dt)

r = np.linspace(dr, R, N)  # Avoid r=0 to prevent singularity
u = np.zeros_like(r)

# Source options
def source_logarithmic(t): return np.log1p(t)
def source_power(t, alpha=2): return t**alpha
def source_impulse(t, t0=0.025, width=0.005): return np.exp(-((t - t0)**2) / (2 * width**2))
def source_delta(t, t0=0.0): return 1.0 if np.isclose(t, t0, atol=1e-8) else 0.0

# Choose source type
source_type = "power"

# Visualization storage
snapshots = []
snapshot_times = []

def laplacian_spherical(u, r, dr):
    dudr = np.zeros_like(u)
    dudr[1:-1] = (u[2:] - u[:-2]) / (2 * dr)
    term = r**2 * dudr
    d_term = np.zeros_like(u)
    d_term[1:-1] = (term[2:] - term[:-2]) / (2 * dr)
    return d_term / r**2

for step in range(steps):
    t = step * dt

    if source_type == "log":
        source_value = source_logarithmic(t)
    elif source_type == "power":
        source_value = source_power(t, alpha=2)
    elif source_type == "impulse":
        source_value = source_impulse(t)
    elif source_type == "delta":
        source_value = source_delta(t)
    else:
        source_value = 0.0

    source = np.zeros_like(r)
    source[0] = source_value / dr  # source at center, normalized

    u += dt * (D * laplacian_spherical(u, r, dr) + source)

    if step % 20 == 0:
        snapshots.append(u.copy())
        snapshot_times.append(t)

# Plot snapshots
for idx, u_snap in enumerate(snapshots):
    plt.plot(r, u_snap, label=f"t={snapshot_times[idx]:.3f}")

plt.xlabel("Radius")
plt.ylabel("Concentration")
plt.title("Spherically Symmetric Diffusion")
plt.legend()
plt.grid()
plt.show()