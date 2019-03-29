import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.stats import binom


a = 5
b = 17
theta = np.linspace(0, 1, 1000)

n = 100
k = 66


plt.plot(theta, beta.pdf(theta, a, b), 'k', label='prior')
plt.plot(theta, binom.pmf(k, n, theta) * 10, 'b', label='likelyhood')
plt.plot(theta, beta.pdf(theta, a, b) * binom.pmf(k, n, theta) * 10000, 'r', label='posterior')
plt.legend()
plt.show()
