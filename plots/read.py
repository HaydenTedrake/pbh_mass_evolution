import numpy as np

# Read the binary file as an array of 64-bit floats
file_path = "plots/bin(2)/bigSolution21-22.bin"
data = np.fromfile(file_path, dtype=np.float64)

# Print the first 100 values
print(data[:100])

t_values = np.arange(-10000, 10001, 2)
x_values = np.arange(-10, 11, 2)
y_values = np.arange(-10, 11, 2)
z_values = np.arange(-10, 11, 2)

reshaped_data = data.reshape(len(t_values), len(x_values), len(y_values), len(z_values))

print(reshaped_data[:2, :2, :2, :2])
