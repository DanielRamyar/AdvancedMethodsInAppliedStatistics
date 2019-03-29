import random
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

random.seed(142)

# N is the number of throws per test
N = 100
number_of_tests = 10000
pi = np.empty((1, 0))
counter = 1
# Fill my arrays with random numbers
for i in range(number_of_tests):

    x = np.empty((1, 0))
    y = np.empty((1, 0))
    x_missed = np.empty((1, 0))
    y_missed = np.empty((1, 0))

    for i in range(N):

        x_random = random.random()
        y_random = random.random()
        if (np.sqrt(x_random**2 + y_random**2)) < 1:
            x = np.append(x, x_random)
            y = np.append(y, y_random)

        else:
            x_missed = np.append(x_missed, x_random)
            y_missed = np.append(y_missed, y_random)

    pi = np.append(pi, len(x) / N * 4)

binwidth = 0.1
n_bins = np.arange(min(pi), max(pi) + binwidth, binwidth)
plt.hist(pi, bins=n_bins, density='True')

lnspc = np.linspace(min(pi), max(pi), len(pi))
# lets try the normal distribution first
m, s = stats.norm.fit(pi)  # get mean and standard deviation
pdf_g = stats.norm.pdf(lnspc, m, s)  # now get theoretical values in our interval
plt.plot(lnspc, pdf_g, label="Norm")  # plot it

print(stats.chisquare(pi, f_exp=pdf_g))
# plt.show()
