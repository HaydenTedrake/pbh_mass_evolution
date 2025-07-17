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
    for j in range(i, min(i+bandwidth+1, n)):
        M_ext[i, j] = v[j-i]

# Compute g = M_ext @ a
g = M_ext @ a

# Plot the result
plt.figure(figsize=(18, 5))

# Full range
plt.subplot(1, 3, 1)
plt.plot(g, label='g')
plt.xlabel('Time index')
plt.ylabel('Amplitude')
plt.title("Full range")
plt.xlim(0, len(g))
plt.grid(True)

# Zoom linear
plt.subplot(1, 3, 2)
plt.plot(g, label='g')
plt.xlabel('Time index')
plt.ylabel('Amplitude')
plt.title("Zoom: 9000–10000")
plt.xlim(9000, 10000)
plt.grid(True)

# Zoom log
plt.subplot(1, 3, 3)
plt.plot(g, label='g')
plt.xlabel('Time index')
plt.ylabel('Amplitude (log)')
plt.title("Zoom: 9000–10000 (log)")
plt.xlim(9000, 10000)
plt.yscale('log')
plt.grid(True)

plt.tight_layout()
plt.show()

# indices_to_plot = [0, 200, 400, 600, 800] 
# colors = ['blue', 'orange', 'green', 'red', 'purple']
# plt.figure(figsize=(10, 6))
# for idx, color in zip(indices_to_plot, colors):
#     plt.plot(M_ext[:, idx][:1000], color=color, label=f'Column {idx}')
# plt.xlabel("Row index (i)")
# plt.ylabel("Amplitude")
# plt.title("M_ext")
# plt.legend()
# plt.grid(True)
# plt.show()

df_new_M_ext = pd.DataFrame(M_ext)
df_new_M_ext.to_csv("new_M_ext.csv", index=False)
print("done")