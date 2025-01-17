import numpy as np
import matplotlib.pyplot as plt

def pdf(mass, mu, sigma):
    # Lognormal probability density function (PDF)
    pdf = (1 / (mass * sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(mass / mu) / sigma) ** 2))
    # pdf = mass**2
    return pdf

# # Plotting the initial mass distribution
# plt.figure(figsize=(10, 6))
# plt.plot(log_mass, pdf, color='salmon')
# plt.fill_between(log_mass, pdf, color='salmon', alpha=0.5)
# plt.title("Initial Mass Distribution")
# plt.xlabel("log10(Mass) [g]")
# plt.ylabel("Probability Density")
# plt.grid(True)
# plt.show()

def run_pdf(Mdict, N):
    rng = np.random.default_rng()
    if Mdict['type'] == 'normal':
        M = rng.normal(Mdict['mean'], Mdict['sigma'], N)
    value = pdf(M, Mdict['mean'], Mdict['sigma'])
    return value


Mdict = {}
Mdict['type'] = 'normal'
Mdict['mean'] = 10**15
Mdict['sigma'] = 2

N = 100000

value = run_pdf(Mdict, N)

# plt.figure()
# plt.hist(value, 200)
# plt.show()

# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(value, bins=np.logspace(11, 19, 100), density=True)
plt.xscale('log')
plt.xlim(10**11, 10**19)
plt.ylim(0, 220)

plt.title('Mass Distribution')
plt.show()
