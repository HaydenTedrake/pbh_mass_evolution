import pandas as pd
import matplotlib.pyplot as plt

# Load and parse the data
filename = 'summer/cutoff_rigidity.txt'
with open(filename, 'r') as f:
    lines = f.readlines()

data = []
for line in lines:
    if line.strip().startswith('Year') or not line.strip():
        continue
    line = line.replace(',', '.')
    parts = line.strip().split()
    if len(parts) >= 5:
        try:
            year = int(parts[0])
            lat = float(parts[1])
            lon = float(parts[2])
            stoermer = float(parts[3])
            vertical = float(parts[4])
            data.append((year, lat, lon, stoermer, vertical))
        except ValueError:
            continue

df = pd.DataFrame(data, columns=['Year', 'Lat', 'Lon', 'Stoermer', 'Vertical'])

# Filter and plot: Longitude vs Vertical Rigidity
combos = [(2011, -51.6), (2011, 51.6), (2022, -51.6), (2022, 51.6)]

for year, lat in combos:
    subset = df[(df['Year'] == year) & (df['Lat'] == lat)]
    plt.figure()
    plt.plot(subset['Lon'], subset['Vertical'], marker='o', label='Effective Vertical Cutoff')
    plt.title(f'Effective Vertical Cutoff vs Longitude\nYear: {year}, Latitude: {lat}°')
    plt.xlabel('Longitude (°)')
    plt.ylabel('Effective Vertical Cutoff Rigidity (GV)')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()