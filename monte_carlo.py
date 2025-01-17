import numpy as np
import matplotlib.pyplot as plt

def pdf(mass, mu, sigma):
    # Lognormal probability density function (PDF)
    pdf = (1 / (mass * sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(mass / mu) / sigma) ** 2))
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

def calc_ptarget(value, value_target):
    Ntarget = np.count_nonzero(np.logical_and(value >= value_target[0], value <= value_target[1]))
    Ptarget = Ntarget / len(value)
    return Ptarget

Mdict = {}
Mdict['type'] = 'normal'
Mdict['mean'] = 10**15
Mdict['sigma'] = 2

N = 10000

value = run_pdf(Mdict, N)

value_target = (10**10,10**12)
Ptarget = calc_ptarget(value, value_target)

plt.figure()
plt.hist(value, 200)
plt.show()
