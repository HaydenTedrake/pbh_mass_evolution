import pandas as pd
import itertools
from tqdm import tqdm

# Define the range of values for each variable
t_values = range(-10000, 10001)  # Time varies the fastest
x_values = range(-10, 11)
y_values = range(-10, 11)
z_values = range(-10, 11)

# Compute total number of combinations
total_combinations = len(x_values) * len(y_values) * len(z_values) * len(t_values)

# Open CSV file for writing
csv_filename = "combinations.csv"
with open(csv_filename, "w") as f:
    # Write header
    f.write("t,x,y,z\n")

    # Iterate through the combinations in the specified order with a progress bar
    with tqdm(total=total_combinations, desc="Generating CSV") as pbar:
        for x, y, z in itertools.product(x_values, y_values, z_values):
            for t in t_values:  # Time varies the fastest
                f.write(f"{t},{x},{y},{z}\n")
                pbar.update(1)

print(f"CSV file '{csv_filename}' has been generated successfully.")
