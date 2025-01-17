import numpy as np
import matplotlib.pyplot as plt

def pdf(mass, mu, sigma):
    # Lognormal probability density function (PDF)
    pdf = (1 / (mass * sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(mass / mu) / sigma) ** 2))
    return pdf

def run_pdf(Mdict, N):
    rng = np.random.default_rng()
    if Mdict['type'] == 'normal':
        masses = rng.normal(Mdict['mean'],Mdict['sigma'],N)
        values = pdf(masses, Mdict['mean'], Mdict['sigma'])
    return masses, values

Mdict = {}
Mdict['type'] = 'normal'
Mdict['mean'] = 10**15
Mdict['sigma'] = 2

N = 100000

# Get both masses and their PDF values
masses, pdf_values = run_pdf(Mdict, N)

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(masses, pdf_values)
plt.xscale('log')
plt.xlim(10**11, 10**19)
plt.ylim(0, 220)

plt.xlabel('Mass [g]')
plt.ylabel('PDF Value')
plt.title('Mass Distribution')
plt.show()
