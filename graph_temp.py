import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter

def bh_temp_gev(mass_g):
    """Calculate black hole temperature in GeV for a given mass in grams"""
    return 1.06e13 / mass_g

def gev_to_grams(mass_gev):
    """Convert mass from GeV to grams using E = mc²"""
    gev_to_joule = 1.602176634e-10  # 1 GeV in Joules
    c = 2.998e8  # Speed of light in m/s
    
    # E = mc² -> m = E/c²
    mass_kg = (mass_gev * gev_to_joule) / (c * c)
    return mass_kg * 1000  # Convert kg to g

# Standard Model particles with their masses in GeV
particles = {
    'Electron neutrino': 1e-9,  # Approximate upper limit
    'Muon neutrino': 1e-9,     # Approximate upper limit
    'Tau neutrino': 1e-9,      # Approximate upper limit
    'Electron': 0.000511,
    'Muon': 0.1057,
    'Tau': 1.777,
    'Up quark': 0.002,
    'Down quark': 0.005,
    'Charm quark': 1.275,
    'Strange quark': 0.095,
    'Top quark': 173.0,
    'Bottom quark': 4.18,
    'Photon': 0,               # Massless
    'Gluon': 0,               # Massless
    'Z boson': 91.1876,
    'W boson': 80.379,
    'Higgs boson': 125.18
}

# Create lists for plotting, excluding massless particles
masses_gev = []
temps_gev = []
names = []
particle_types = []

for particle, mass in particles.items():
    if mass > 0:  # Skip massless particles
        masses_gev.append(mass)
        temps_gev.append(bh_temp_gev(gev_to_grams(mass)))
        names.append(particle)
        
        # Determine particle type for color coding
        if 'neutrino' in particle.lower():
            particle_types.append('Leptons (Neutrinos)')
        elif particle in ['Electron', 'Muon', 'Tau']:
            particle_types.append('Leptons (Charged)')
        elif 'quark' in particle.lower():
            particle_types.append('Quarks')
        elif 'boson' in particle.lower():
            particle_types.append('Bosons')
        else:
            particle_types.append('Other')

# Create the plot
plt.figure(figsize=(15, 10))

# Create a color map for particle types
unique_types = list(set(particle_types))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
color_map = dict(zip(unique_types, colors))

# Plot each point with different colors per particle type
for i in range(len(masses_gev)):
    plt.scatter(masses_gev[i], temps_gev[i], 
               color=color_map[particle_types[i]], 
               s=100, 
               label=particle_types[i] if particle_types[i] not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.annotate(names[i], 
                (masses_gev[i], temps_gev[i]),
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=8)

# Set log scale for both axes
plt.xscale('log')
plt.yscale('log')

# Labels and title
plt.xlabel('Particle Mass (GeV)', fontsize=12)
plt.ylabel('Black Hole Temperature (GeV)', fontsize=12)
plt.title('Standard Model Particles:\nEquivalent Black Hole Temperatures vs Particle Masses', 
          fontsize=14, pad=20)

# Add grid
plt.grid(True, which="both", ls="-", alpha=0.2)

# Add legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()
