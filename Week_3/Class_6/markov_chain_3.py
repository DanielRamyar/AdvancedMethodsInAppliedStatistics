import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.stats import binom

mu = 0
sigma = 0.3

a = 5
b = 17
x = 0.1

n = 100
k = 66


keept = np.array([])

for i in np.arange(2000):

    x_new = x + np.random.normal(mu, sigma, 1)

    if x_new < 0:
        x_new = x + np.random.normal(mu, sigma, 1)

    r = (beta.pdf(x_new, a, b) * binom.pmf(k, n, x_new) /
         (beta.pdf(x, a, b) * binom.pmf(k, n, x)))
    if r > 1:
        keept = np.append(keept, x_new)
        x = x_new
    else:
        keep_or_not = np.random.uniform()
        if r > keep_or_not:
            keept = np.append(keept, x_new)
            x = x_new
        else:
            keept = np.append(keept, x)
            x = x

binwidth = 0.01
n_bins = np.arange(min(keept), max(keept) + binwidth, binwidth)
plt.hist(keept, bins=n_bins)
plt.show()

