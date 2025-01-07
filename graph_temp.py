import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter

def particle_temp_gev(mass_gev):
    """Convert particle mass to equivalent temperature in GeV"""
    return mass_gev  # E = mc², and we're already in GeV

def bh_mass_for_temp(temp_gev):
    """Calculate black hole mass in grams that would have given temperature in GeV
    Using MacGibbon equation T = 1.06e13/M_g solved for M_g"""
    return 1.06e13 / temp_gev

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
bh_masses_g = []
temps_gev = []
names = []
particle_types = []

for particle, mass_gev in particles.items():
    if mass_gev > 0:  # Skip massless particles
        temp_gev = particle_temp_gev(mass_gev)
        temps_gev.append(temp_gev)
        bh_mass_g = bh_mass_for_temp(temp_gev)
        bh_masses_g.append(bh_mass_g)
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
for i in range(len(bh_masses_g)):
    plt.scatter(bh_masses_g[i], temps_gev[i], 
               color=color_map[particle_types[i]], 
               s=100, 
               label=particle_types[i] if particle_types[i] not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.annotate(f'{names[i]}\n(BH: {bh_masses_g[i]:.2e}g)', 
                (bh_masses_g[i], temps_gev[i]),
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=8)

# Set log scale for both axes
plt.xscale('log')
plt.yscale('log')

# Labels and title
plt.xlabel('Black Hole Mass (g)', fontsize=12)
plt.ylabel('Temperature (GeV)', fontsize=12)
plt.title('Standard Model Particles:\nParticle Temperatures and Equivalent Black Hole Masses', 
          fontsize=14, pad=20)

# Add grid
plt.grid(True, which="both", ls="-", alpha=0.2)

# Add legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()

# Print out the values for verification
print("\nParticle temperatures and equivalent black hole masses:")
for i in range(len(names)):
    print(f"{names[i]}:")
    print(f"  Temperature: {temps_gev[i]:.6e} GeV")
    print(f"  BH Mass: {bh_masses_g[i]:.6e} g")
