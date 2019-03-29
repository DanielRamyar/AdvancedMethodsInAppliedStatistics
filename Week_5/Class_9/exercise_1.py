import numpy as np
import matplotlib.pyplot as plt


sigma = 1

for i in np.arange(0, 11, 0.05):
    mu = i
    s = np.random.normal(mu, sigma, 100)

    percentile = s[(s >= (i - sigma)) & (s <= (i + sigma))]
    x = np.ones(len(percentile)) * i

    plt.plot(x, percentile, 'r')

plt.xlabel('True Value')
plt.ylabel('Observed Values')
plt.show()
