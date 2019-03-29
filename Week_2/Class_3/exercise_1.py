import numpy as np
import matplotlib.pyplot as plt


def my_gauss(x, mu, sigma):
    result = 1/(np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    return result


mu = 0.2
sigma = 0.1
N = 50

s = np.random.normal(mu, sigma, N)

binwidth = 0.01

mu_scan = np.arange(0.05, 0.3 + binwidth, binwidth)
sigma_scan = np.arange(0.05, 0.3 + binwidth, binwidth)


LLH = np.zeros((len(mu_scan), len(sigma_scan)))


for i, mu in enumerate(mu_scan, 0):
    for j, sigma in enumerate(sigma_scan, 0):

        LLH[i, j] = np.sum(np.log(my_gauss(s, mu, sigma)))


plt.imshow(LLH, cmap='hot')
plt.show()
