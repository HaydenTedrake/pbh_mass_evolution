import numpy as np
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

# List of energy levels and corresponding .bin files
energy_levels = [100, 125.893, 158.489, 199.526, 251.189, 316.228, 398.107, 501.187, 794.328, 1000, 1258.93, 1584.89, 1995.26, 2511.89, 3162.28]  # Example energy levels in GeV
bin_files = ["bigSolution4-1.bin", "bigSolution4-2.bin", "bigSolution4-3.bin", "bigSolution4-4.bin", "bigSolution4-5.bin", "bigSolution4-6.bin", "bigSolution4-7.bin", "bigSolution4-8.bin", "bigSolution4-10.bin", "bigSolution4-11.bin", "bigSolution4-12.bin", "bigSolution4-13.bin", "bigSolution4-14.bin", "bigSolution4-15.bin", "bigSolution4-16.bin"]

# Dictionary to store interpolators for each energy
interpolators = {}

# Define the grid
t_values = np.arange(-10000, 10001, 10)
x_values = np.arange(-10, 11, 2)
y_values = np.arange(-10, 11, 2)
z_values = np.arange(-10, 11, 2)
grid_points = np.array([(t, x, y, z) for t in t_values for x in x_values for y in y_values for z in z_values])

# Iterate over energy levels and construct interpolators with a progress bar
for E, file in tqdm(zip(energy_levels, bin_files), total=len(energy_levels), desc="Building Interpolators"):
    data = np.fromfile(file, dtype=np.float64)
    
    if len(data) != len(grid_points):
        raise ValueError(f"Mismatch for {file}: Expected {len(grid_points)} values, got {len(data)}")

    t_grid, x_grid, y_grid, z_grid = grid_points.T

    # Create and store 4D interpolator
    interpolators[E] = RegularGridInterpolator((t_values, x_values, y_values, z_values), data.reshape(len(t_values), len(x_values), len(y_values), len(z_values)))

print("All interpolators have been built successfully!")

value_at_point = interpolators[100]((-5000, 0, 0, 0))
print(f"Value at E=100, (t=-5000, x=0, y=0, z=0): {value_at_point}")
