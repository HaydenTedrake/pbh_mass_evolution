import numpy as np

# Read the binary file as an array of 64-bit floats
file_path = "test.bin"  # Replace with your actual file path
data = np.fromfile(file_path, dtype=np.float64)

# Print the first 100 values
print(data[:100])
