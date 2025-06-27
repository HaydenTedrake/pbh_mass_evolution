import numpy as np
import matplotlib.pyplot as plt

# Read the binary file as an array of 64-bit floats
number = 22
bin_files = [f"plots/bin(2)/bigSolution{number}-{i}.bin" for i in [22, 24, 26, 28, 30, 32, 34, 36]]
for file_path in bin_files:
    data = np.fromfile(file_path, dtype=np.float64)

    # # Print the first 100 values
    # print(data[:1000])

    t_values = np.arange(-10000, 10001, 2)
    x_values = np.arange(-10, 11, 2)
    y_values = np.arange(-10, 11, 2)
    z_values = np.arange(-10, 11, 2)

    #reshaped_data = data.reshape(len(t_values), len(x_values), len(y_values), len(z_values))
    try:
        print(f"{file_path}")
        reshaped_data = data.reshape(-1, 11, 11, 11)
        N = data.size // (11 * 11 * 11)
        print("N =", N)
    except:
        print(f"error")