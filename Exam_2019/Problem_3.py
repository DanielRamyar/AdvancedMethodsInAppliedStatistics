import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import chisquare, chi2, binom, beta
from scipy import integrate
from scipy.optimize import minimize

plt.style.use('bmh')

fname = 'Exam_2019_Problem3a.txt'
data = np.loadtxt(fname)

sorted_data = np.sort(data)
index = len(sorted_data)*(1-0.0455)

print(sorted_data[math.ceil(index)])


binwidth = 0.8
n_bins = np.arange(0, max(data) + binwidth, binwidth)
plt.hist(data, bins=n_bins, density=True)

print('Threshold value ', chi2.isf(0.0455, 5))
plt.close()
###############################################################################
a = 89
b = 45


n = 5
k = 2
p = np.arange(0, 1.001, 0.001)

normz1 = integrate.quad(lambda p: binom.pmf(k, n, p), 0, 1)[0]
normz2 = integrate.quad(lambda p: beta.pdf(p, a, b), 0, 1)[0]
normz3 = integrate.quad(lambda p: beta.pdf(p, a, b) * binom.pmf(k, n, p), 0, 1)[0]
estimate = integrate.quad(lambda p: p*beta.pdf(p, a, b) * binom.pmf(k, n, p) / normz3, 0, 1)[0]
estimate2 = integrate.quad(lambda p: p**2 * beta.pdf(p, a, b) * binom.pmf(k, n, p) / normz3, 0, 1)[0]

plt.plot(p, beta.pdf(p, a, b) / normz2, 'k', label='prior')
plt.plot(p, binom.pmf(k, n, p) / normz1 , 'b', label='likelyhood')
plt.plot(p, beta.pdf(p, a, b) * binom.pmf(k, n, p) / normz3, 'r', label='posterior')
plt.legend()
plt.show()


print(estimate2 - estimate ** 2)
print('Mean of posterior: ', estimate ** 2)
