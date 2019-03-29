import random
import matplotlib.pyplot as plt
import numpy as np

random.seed(142)

# N is the number of throws per test and A is area
N = 100
number_of_tests = 1000
A = np.empty((1, 0))

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

    A = np.append(A, len(x) / N * 4 * 5.2 ** 2)

binwidth = 3
n_bins = np.arange(min(A), max(A) + binwidth, binwidth)
plt.figure(1)
plt.hist(A, bins=n_bins)


binwidth = 1
n_bins = np.arange(min(A), max(A) + binwidth, binwidth)
plt.figure(2)
plt.hist(A, bins=n_bins)


binwidth = 0.1
n_bins = np.arange(min(A), max(A) + binwidth, binwidth)
plt.figure(3)
plt.hist(A, bins=n_bins)
plt.show()
