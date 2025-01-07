import math
import matplotlib.pyplot as plt

def bh_temperature_in_GeV(mass_g):
    # Constants
    hbar_eV_s = 6.582e-16  # Reduced Planck constant in eV·s
    c = 3.0e8  # Speed of light in m/s
    G = 6.67e-11  # Gravitational constant in m^3·kg^-1·s^-2
    g_to_kg = 1e-3  # Conversion factor from grams to kilograms
    
    # Convert mass from grams to kilograms
    mass_kg = mass_g * g_to_kg
    
    # MacGibbon's equation (temperature in eV)
    temperature_eV = (hbar_eV_s * c**3) / (8 * math.pi * G * mass_kg)
    
    # Convert eV to GeV (1 GeV = 10^9 eV)
    temperature_GeV = temperature_eV / 1e9
    
    return temperature_GeV

# Masses of the 17 Standard Model particles in GeV/c^2 (approximate values)
# Format: {"particle_name": mass_in_GeV}
particle_masses = {
    "up quark": 2.3e-3,
    "down quark": 4.8e-3,
    "charm quark": 1.28,
    "strange quark": 0.095,
    "top quark": 173,
    "bottom quark": 4.18,
    "electron": 0.000511,
    "muon": 0.106,
    "tau": 1.78,
    "electron neutrino": 1e-10,  # upper bound
    "muon neutrino": 1.9e-10,    # upper bound
    "tau neutrino": 1.8e-10,     # upper bound
    "photon": 0,  # massless particle
    "gluon": 0,   # massless particle
    "Z boson": 91.2,
    "W boson": 80.4,
    "Higgs boson": 125
}

# Convert particle masses from GeV/c^2 to grams
GeV_to_kg = 1.783e-27  # 1 GeV/c^2 = 1.783e-27 kg
kg_to_g = 1e3  # 1 kg = 1000 g

particle_masses_in_g = {name: mass * GeV_to_kg * kg_to_g for name, mass in particle_masses.items()}

# Calculate black hole temperatures for each particle mass
particle_temperatures = {name: bh_temperature_in_GeV(mass) for name, mass in particle_masses_in_g.items() if mass > 0}

# Filter out massless particles for plotting
filtered_masses = {name: mass for name, mass in particle_masses.items() if mass > 0}

# Extract mass and temperature values for plotting
x_values = list(filtered_masses.values())
y_values = list(particle_temperatures.values())

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, color='blue')
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Mass (GeV/c²)")
plt.ylabel("Black Hole Temperature (GeV)")
plt.title("Temperature of Black Holes Corresponding to Standard Model Particles")
plt.grid(True, which="both", ls="--")

# Annotate points with particle names
for particle, mass in filtered_masses.items():
    plt.annotate(particle, (mass, particle_temperatures[particle]), fontsize=8, ha='right')

plt.show()

