import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load flux data
df = pd.read_csv('flux2122.csv')
a = df['0.0000000000'].values

n = len(a)  # 20000

# Load M_ext.csv
df_M_ext = pd.read_csv("M_ext.csv")
M_ext_csv = df_M_ext.values
bandwidth = 50
v = M_ext_csv[:, 0][:bandwidth+1]
M_ext = np.zeros((n, n), dtype=np.float32)

for i in range(n):
    for j in range(max(0, i-bandwidth), i+1):
        M_ext[i, j] = v[i-j]

# Compute g = M_ext @ a
g = M_ext @ a

# Plot the result
plt.figure(figsize=(12, 6))
plt.plot(g, label='g')
plt.xlabel('Time index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.title("Response g = M_ext @ a")
plt.show()

indices_to_plot = [0, 200, 400, 600, 800] 
colors = ['blue', 'orange', 'green', 'red', 'purple']
plt.figure(figsize=(10, 6))
for idx, color in zip(indices_to_plot, colors):
    plt.plot(M_ext[:, idx][:1000], color=color, label=f'Column {idx}')
plt.xlabel("Row index (i)")
plt.ylabel("Amplitude")
plt.title("M_ext")
plt.legend()
plt.grid(True)
plt.show()